# -*- coding: utf-8 -*-
"""후보집합 분해 실험 — "게이팅 신호가 아니라 후보 집합"이라는 주장의 근거 표.

각 설정에서 LOYO(나머지 연도 튜닝 -> held-out 적용)로 9모델 평균 정확도를 낸다.
게이팅 신호를 정교화하는 축과 후보를 넓히는 축을 분리해 기여를 비교한다.

  [신호 축]  관련도 T만 / SC C만 / T×C 둘 다
  [후보 축]  plain만 / +gate / +always_dense / +graph계열

사용: ablation.py <pred_dir> <maxrel_dir> [delta]
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

PRED = sys.argv[1]
MRD = sys.argv[2]
DELTA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
YEARS = ["2022", "2023", "2024", "2025"]
THRS = [x / 20 for x in range(21)]
CONFS = [0.0, 0.67, 1.0]

sc = lambda t: Counter(t).most_common(1)[0][1] / len(t) if t else 1.0
data = defaultdict(lambda: defaultdict(dict))
for f in glob.glob(f"{PRED}/RES_*.json"):
    m = re.match(r"(.+)_(\d{4})_(base|rag|graph)$", os.path.basename(f)[4:-5])
    if not m:
        continue
    tag, y, c = m.groups()
    d = json.load(open(f, encoding="utf-8"))
    if d.get("stage") != 5:
        continue
    data[tag][y][c] = {str(r["id"]): (r["predicted"], r["correct_answer"], sc(r.get("trials")))
                       for r in d["results"]}
MR = {}
for pref, k in (("maxrel", "rag"), ("maxrel_graph", "graph")):
    MR[k] = {y: (json.load(open(f"{MRD}/{pref}_{y}.json", encoding="utf-8"))
                 if os.path.exists(f"{MRD}/{pref}_{y}.json") else {}) for y in YEARS}


def pick(tag, y, cd):
    base = data[tag][y].get("base", {})
    if cd[0] == "plain":
        return [int(base[q][0] == base[q][1]) for q in sorted(base)]
    s = cd[1]
    aug = data[tag][y].get(s, {})
    mr = MR[s].get(y, {})
    out = []
    for q in sorted(base):
        bp, g, _ = base[q]
        r = aug.get(q)
        if cd[0] == "always":
            pr = r[0] if r else bp
        else:
            _, _, T, C = cd
            m = mr.get(q, 0.0)
            pr = r[0] if (r and m > 0 and m >= T and r[2] >= C) else bp
        out.append(int(pr == g))
    return out


def vec(tag, cd, ys):
    v = []
    for y in ys:
        v += pick(tag, y, cd)
    return v


def build(tag, signal, cand_level):
    """signal: 'T'|'C'|'TC'   cand_level: 0=plain만 1=+gate 2=+always 3=+graph"""
    c = [("plain",)]
    if cand_level == 0:
        return c
    srcs = ["rag"] if cand_level < 3 else ["rag", "graph"]
    for s in srcs:
        if not any(data[tag].get(y, {}).get(s) for y in YEARS):
            continue
        if signal == "T":
            c += [("gate", s, T, 0.0) for T in THRS]
        elif signal == "C":
            c += [("gate", s, 0.0, C) for C in CONFS]
        else:
            c += [("gate", s, T, C) for T in THRS for C in CONFS]
        if cand_level >= 2:
            c.append(("always", s))
    return c


def loyo(tag, signal, lvl):
    out = []
    for y in YEARS:
        ys = [z for z in YEARS if z != y]
        pv = vec(tag, ("plain",), ys)
        pa = sum(pv) / len(pv)
        bc, ba = ("plain",), pa
        for cd in build(tag, signal, lvl):
            v = vec(tag, cd, ys)
            a = sum(v) / len(v)
            if a > ba:
                bc, ba = cd, a
        if ba - pa < DELTA:
            bc = ("plain",)
        out += pick(tag, y, bc)
    return 100 * sum(out) / len(out)


tags = sorted(data)
SETTINGS = [
    ("plain (증강 없음)",                 "TC", 0),
    ("신호=T만,  후보=plain/gate",        "T",  1),
    ("신호=C만,  후보=plain/gate",        "C",  1),
    ("신호=T×C, 후보=plain/gate  [T1]",   "TC", 1),
    ("신호=T×C, +always_dense   [T2]",    "TC", 2),
    ("신호=T×C, +graph계열      [T3]",    "TC", 3),
    ("신호=T만,  +always_dense",          "T",  2),
    ("신호=C만,  +always_dense",          "C",  2),
]
print(f"후보집합 분해 실험 (LOYO held-out, δ={DELTA}, 9모델 평균)")
print(f"{'설정':34s} | {'정확도':>7s} | {'vs plain':>9s} | {'후보 수':>7s}")
print("-" * 72)
base_acc = None
rows = []
for name, sig, lvl in SETTINGS:
    accs = [loyo(t, sig, lvl) for t in tags]
    a = sum(accs) / len(accs)
    if base_acc is None:
        base_acc = a
    ncand = len(build(tags[0], sig, lvl))
    rows.append((name, a, a - base_acc, ncand))
    print(f"{name:34s} | {a:6.2f}% | {a-base_acc:+8.2f}p | {ncand:7d}")
print("-" * 72)
d = dict((r[0], r[1]) for r in rows)
print("\n[기여 분해]")
print(f"  신호 정교화 (T만 -> T×C, 후보 고정)      : {d['신호=T×C, 후보=plain/gate  [T1]'] - d['신호=T만,  후보=plain/gate']:+.2f}p")
print(f"  후보 확장   (gate -> +always, 신호 고정) : {d['신호=T×C, +always_dense   [T2]'] - d['신호=T×C, 후보=plain/gate  [T1]']:+.2f}p")
print(f"  후보 확장   (+graph계열)                 : {d['신호=T×C, +graph계열      [T3]'] - d['신호=T×C, +always_dense   [T2]']:+.2f}p")
print(f"  신호 최소화+후보 확장 (T만 +always)      : {d['신호=T만,  +always_dense'] - base_acc:+.2f}p  <- 신호가 단순해도 후보만 넓히면 대부분 회수")
json.dump([{"setting": r[0], "acc": r[1], "delta_vs_plain": r[2], "n_cand": r[3]} for r in rows],
          open("ablation_result.json", "w"), ensure_ascii=False, indent=1)
