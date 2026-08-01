# -*- coding: utf-8 -*-
"""논문에 들어갈 수치 일괄 산출 — 5.2 / 5.6 / 5.7 / 5.8 및 부속 분석.

산출 항목
  A. 파싱 실패·chance 보정 (규칙 6.5: raw / 응답분 / chance보정 3종 보고)
  B. 2025 형식 효과 (4지 862 vs 5지 276, chance 보정 후 비교)
  C. 게이트 Rescue / Damage 분해 + HPR·BRR
  D. Oracle 천장과 남은 여지
  E. 선행연구 대리 비교 (생성 전 게이트 = SC 미사용, 관련도만)
  F. 과목별 게이트 효과 (n<40 은 소표본 표시)
  G. 오류 분석 — 구제/훼손 문항의 특성

사용: paper_numbers.py <pred_dir> <maxrel_dir> <ktm_data_dir>
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

PRED, MRD, DATA = sys.argv[1], sys.argv[2], sys.argv[3]
YEARS = ["2022", "2023", "2024", "2025"]
THRS = [x / 20 for x in range(21)]
CONFS = [0.0, 0.67, 1.0]
DELTA = 0.005
DISP = {"qwen2_5_7b": "Qwen2.5-7B", "exaone3_5_7_8b": "EXAONE-3.5-7.8B", "solar_10_7b": "SOLAR-10.7B",
        "mistral_7b": "Mistral-7B", "huatuogpt-o1-7b": "HuatuoGPT-o1-7B", "biomistral-7b": "BioMistral-7B",
        "medllama2-7b": "MedLLaMA2-7B", "openbiollm-8b": "OpenBioLLM-8B", "medgemma-4b": "MedGemma-4B"}

sc = lambda t: Counter(t).most_common(1)[0][1] / len(t) if t else 1.0
raw = defaultdict(lambda: defaultdict(dict))     # tag -> year -> cond -> [record]
for f in glob.glob(f"{PRED}/RES_*.json"):
    m = re.match(r"(.+)_(\d{4})_(base|rag|graph)$", os.path.basename(f)[4:-5])
    if not m:
        continue
    tag, y, c = m.groups()
    d = json.load(open(f, encoding="utf-8"))
    if d.get("stage") != 5:
        continue
    raw[tag][y][c] = d["results"]

qmeta = {}
for y in YEARS:
    for q in json.load(open(f"{DATA}/{y}.json", encoding="utf-8")):
        qmeta[str(q["id"])] = {"year": y, "subject": (q.get("subject") or "").strip(),
                               "nch": len(q.get("choices_kr") or [])}

MR = {}
for pref, k in (("maxrel", "rag"), ("maxrel_graph", "graph")):
    MR[k] = {y: (json.load(open(f"{MRD}/{pref}_{y}.json", encoding="utf-8"))
                 if os.path.exists(f"{MRD}/{pref}_{y}.json") else {}) for y in YEARS}

D = defaultdict(lambda: defaultdict(dict))
for tag in raw:
    for y in YEARS:
        for c, recs in raw[tag][y].items():
            D[tag][y][c] = {str(r["id"]): (r["predicted"], r["correct_answer"], sc(r.get("trials")))
                            for r in recs}
tags = [t for t in D if any(D[t][y].get("rag") for y in YEARS)]
tags.sort(key=lambda t: -sum(1 for y in YEARS for q, v in D[t][y].get("base", {}).items() if v[0] == v[1]))


def pick(tag, y, cd):
    base = D[tag][y].get("base", {})
    if cd[0] == "plain":
        return {q: int(base[q][0] == base[q][1]) for q in base}
    s = cd[1]; aug = D[tag][y].get(s, {}); mr = MR[s].get(y, {})
    out = {}
    for q in base:
        bp, g, _ = base[q]; r = aug.get(q)
        if cd[0] == "always":
            pr = r[0] if r else bp
        else:
            _, _, T, C = cd; mm = mr.get(q, 0.0)
            pr = r[0] if (r and mm > 0 and mm >= T and r[2] >= C) else bp
        out[q] = int(pr == g)
    return out


def flat(tag, cd, ys=YEARS):
    o = {}
    for y in ys:
        o.update(pick(tag, y, cd))
    return o


def cands(tag, signal="TC", lvl=3):
    c = [("plain",)]
    srcs = ["rag"] if lvl < 3 else ["rag", "graph"]
    for s in srcs:
        if not any(D[tag][y].get(s) for y in YEARS):
            continue
        if signal == "T":
            c += [("gate", s, T, 0.0) for T in THRS]
        else:
            c += [("gate", s, T, C) for T in THRS for C in CONFS]
        if lvl >= 2:
            c.append(("always", s))
    return c


def loyo(tag, signal="TC", lvl=3):
    out = {}
    for y in YEARS:
        ys = [z for z in YEARS if z != y]
        pv = flat(tag, ("plain",), ys); pa = sum(pv.values()) / len(pv)
        bc, ba = ("plain",), pa
        for cd in cands(tag, signal, lvl):
            v = flat(tag, cd, ys); a = sum(v.values()) / len(v)
            if a > ba:
                bc, ba = cd, a
        if ba - pa < DELTA:
            bc = ("plain",)
        out.update(pick(tag, y, bc))
    return out


print("=" * 96)
print("[A] 파싱 실패 · chance 보정 (규칙 6.5)")
print("=" * 96)
print(f"{'모델':17s} | {'raw':>7s} {'무응답%':>7s} {'응답분':>7s} {'chance보정':>9s}")
print("-" * 60)
for t in tags:
    n = c = nu = 0
    ch_sum = 0.0
    for y in YEARS:
        for q, (p, g, _) in D[t][y].get("base", {}).items():
            n += 1
            if p is None:
                nu += 1
            else:
                c += int(p == g)
            ch_sum += 1.0 / max(2, qmeta.get(q, {}).get("nch", 4))
    rawa = 100 * c / n
    ansd = 100 * c / max(1, n - nu)
    chance = 100 * ch_sum / n
    corr = 100 * (rawa / 100 - chance / 100) / (1 - chance / 100)
    print(f"{DISP.get(t,t):17s} | {rawa:6.2f}% {100*nu/n:6.2f}% {ansd:6.2f}% {corr:8.2f}%")

print()
print("=" * 96)
print("[B] 2025 형식 효과 — 4지(862문항) vs 5지(276문항)")
print("=" * 96)
print(f"{'모델':17s} | {'4지 raw':>8s} {'5지 raw':>8s} {'차이':>7s} | {'4지 보정':>8s} {'5지 보정':>8s} {'보정차':>7s}")
print("-" * 84)
for t in tags:
    agg = {4: [0, 0], 5: [0, 0]}
    for y in YEARS:
        for q, (p, g, _) in D[t][y].get("base", {}).items():
            k = qmeta.get(q, {}).get("nch", 4)
            if k not in agg:
                continue
            agg[k][0] += int(p == g); agg[k][1] += 1
    a4 = 100 * agg[4][0] / agg[4][1]; a5 = 100 * agg[5][0] / agg[5][1]
    c4 = 100 * (a4 / 100 - .25) / .75; c5 = 100 * (a5 / 100 - .20) / .80
    print(f"{DISP.get(t,t):17s} | {a4:7.2f}% {a5:7.2f}% {a5-a4:+6.2f}p | {c4:7.2f}% {c5:7.2f}% {c5-c4:+6.2f}p")

print()
print("=" * 96)
print("[C] 게이트 Rescue / Damage 분해  [D] Oracle 천장")
print("=" * 96)
print(f"{'모델':17s} | {'plain':>6s} {'게이트':>6s} | {'구제':>5s} {'훼손':>5s} {'순증':>5s} | "
      f"{'구제율':>6s} {'훼손율':>6s} | {'oracle':>7s} {'여지':>6s}")
print("-" * 96)
tot = defaultdict(int)
for t in tags:
    pv = flat(t, ("plain",)); gv = loyo(t)
    resc = sum(1 for q in pv if pv[q] == 0 and gv[q] == 1)
    dmg = sum(1 for q in pv if pv[q] == 1 and gv[q] == 0)
    npw = sum(1 for q in pv if pv[q] == 0); npr = sum(1 for q in pv if pv[q] == 1)
    # oracle: 문항별 plain/rag/graph 중 하나라도 맞으면 정답
    orc = 0
    for y in YEARS:
        base = D[t][y].get("base", {})
        for q, (bp, g, _) in base.items():
            ok = int(bp == g)
            for s in ("rag", "graph"):
                r = D[t][y].get(s, {}).get(q)
                if r and r[0] == g:
                    ok = 1
            orc += ok
    n = len(pv)
    ap, ag, ao = 100 * sum(pv.values()) / n, 100 * sum(gv.values()) / n, 100 * orc / n
    tot["resc"] += resc; tot["dmg"] += dmg; tot["n"] += n
    print(f"{DISP.get(t,t):17s} | {ap:5.1f}% {ag:5.1f}% | {resc:5d} {dmg:5d} {resc-dmg:+5d} | "
          f"{100*resc/max(1,npw):5.1f}% {100*dmg/max(1,npr):5.1f}% | {ao:6.1f}% {ao-ag:+5.1f}p")
print("-" * 96)
print(f"{'합계':17s} | 구제 {tot['resc']}, 훼손 {tot['dmg']}, 순증 {tot['resc']-tot['dmg']:+d} "
      f"(문항 {tot['n']:,})")

print()
print("=" * 96)
print("[E] 선행연구 대리 비교 — 생성 전 게이트(관련도만) vs 생성 후 선택(관련도×SC)")
print("=" * 96)
print(f"{'모델':17s} | {'plain':>6s} | {'생성전(T만)':>11s} | {'생성후(T×C)':>11s} | {'차이':>6s}")
print("-" * 66)
sA = sB = sP = 0
for t in tags:
    pv = flat(t, ("plain",)); n = len(pv)
    a = loyo(t, "T", 3); b = loyo(t, "TC", 3)
    ap, aa, ab = 100 * sum(pv.values()) / n, 100 * sum(a.values()) / n, 100 * sum(b.values()) / n
    sP += ap; sA += aa; sB += ab
    print(f"{DISP.get(t,t):17s} | {ap:5.1f}% | {aa:10.1f}% | {ab:10.1f}% | {ab-aa:+5.2f}p")
k = len(tags)
print("-" * 66)
print(f"{'평균':17s} | {sP/k:5.2f}% | {sA/k:10.2f}% | {sB/k:10.2f}% | {(sB-sA)/k:+5.2f}p")

print()
print("=" * 96)
print("[F] 과목별 게이트 효과 (9모델 합산, n<40 은 소표본 주의)")
print("=" * 96)
subj = defaultdict(lambda: [0, 0, 0])     # subject -> [plain_correct, gate_correct, n]
for t in tags:
    pv = flat(t, ("plain",)); gv = loyo(t)
    for q in pv:
        s = qmeta.get(q, {}).get("subject", "")
        if not s:
            continue
        subj[s][0] += pv[q]; subj[s][1] += gv[q]; subj[s][2] += 1
print(f"{'과목':18s} {'n(문항)':>8s} | {'plain':>7s} {'게이트':>7s} {'Δ':>7s}")
print("-" * 56)
for s, (pc, gc, n) in sorted(subj.items(), key=lambda kv: -(kv[1][1] - kv[1][0]) / kv[1][2]):
    per_model_n = n // len(tags)
    flag = "  ← 소표본" if per_model_n < 40 else ""
    print(f"{s:18s} {per_model_n:8d} | {100*pc/n:6.1f}% {100*gc/n:6.1f}% {100*(gc-pc)/n:+6.2f}p{flag}")

print()
print("=" * 96)
print("[G] 오류 분석 — 구제/훼손 문항의 특성")
print("=" * 96)
rq = Counter(); dq = Counter(); rn = Counter(); dn = Counter()
for t in tags:
    pv = flat(t, ("plain",)); gv = loyo(t)
    for q in pv:
        s = qmeta.get(q, {}).get("subject", "?"); nch = qmeta.get(q, {}).get("nch", 4)
        if pv[q] == 0 and gv[q] == 1:
            rq[s] += 1; rn[nch] += 1
        elif pv[q] == 1 and gv[q] == 0:
            dq[s] += 1; dn[nch] += 1
print("구제 상위 과목:", ", ".join(f"{k} {v}" for k, v in rq.most_common(5)))
print("훼손 상위 과목:", ", ".join(f"{k} {v}" for k, v in dq.most_common(5)))
print(f"보기 수별 — 구제: {dict(rn)} / 훼손: {dict(dn)}")
tr, td = sum(rq.values()), sum(dq.values())
print(f"구제:훼손 = {tr}:{td} = {tr/max(1,td):.2f}:1")
