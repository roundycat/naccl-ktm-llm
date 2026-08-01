# -*- coding: utf-8 -*-
"""T3 — 후보집합을 검색 방식(dense RAG / GraphRAG)에 걸쳐 통합.

T1: {plain, gate_dense(T,C)}                          … 기존
T2: {plain, gate_dense(T,C), always_dense}            … always-inject 추가
T3: {plain,
     gate_dense(T,C),  always_dense,
     gate_graph(T,C),  always_graph}                  … 검색 방식까지 후보에 포함

LOYO(나머지 연도로 선택 → held-out 적용)는 동일하게 유지하므로 test 누수가 없다.
DELTA(abstain 마진)는 인자로 조절 가능하며 기본값은 tune_gate1 과 같은 0.015.

사용: tune_gate3.py <pred_dir> <maxrel_dir> [delta]
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
MR_DIR = sys.argv[2] if len(sys.argv) > 2 else PRED_DIR
DELTA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.015

YEARS = ["2022", "2023", "2024", "2025"]
THRS = [x / 20 for x in range(0, 21)]
CONFS = [0.0, 0.67, 1.0]
DISPLAY = {"qwen2_5_7b": "Qwen2.5-7B", "exaone3_5_7_8b": "EXAONE-3.5-7.8B", "solar_10_7b": "SOLAR-10.7B",
           "mistral_7b": "Mistral-7B", "huatuogpt-o1-7b": "HuatuoGPT-o1-7B", "biomistral-7b": "BioMistral-7B",
           "medllama2-7b": "MedLLaMA2-7B", "openbiollm-8b": "OpenBioLLM-8B", "medgemma-4b": "MedGemma-4B"}


def sc_conf(t):
    return Counter(t).most_common(1)[0][1] / len(t) if t else 1.0


# ------------------------------------------------------------------ 적재
data = defaultdict(lambda: defaultdict(dict))          # tag -> year -> cond -> {qid: (pred, gold, sc)}
for f in glob.glob(os.path.join(PRED_DIR, "RES_*.json")):
    m = re.match(r"(.+)_(\d{4})_(base|rag|graph)$", os.path.basename(f)[4:-5])
    if not m:
        continue
    tag, year, cond = m.groups()
    d = json.load(open(f, encoding="utf-8"))
    if d.get("stage") != 5:
        continue
    data[tag][year][cond] = {str(r["id"]): (r["predicted"], r["correct_answer"], sc_conf(r.get("trials")))
                             for r in d.get("results", [])}

MR = {}
for pref, key in (("maxrel", "rag"), ("maxrel_graph", "graph")):
    MR[key] = {y: (json.load(open(os.path.join(MR_DIR, f"{pref}_{y}.json"), encoding="utf-8"))
                   if os.path.exists(os.path.join(MR_DIR, f"{pref}_{y}.json")) else {}) for y in YEARS}


# ------------------------------------------------------------------ 후보 정의
# cand = ("plain",) | ("always", src) | ("gate", src, T, C)     src in {"rag","graph"}
def pick(tag, year, cand):
    base = data[tag][year].get("base", {})
    out = []
    if cand[0] == "plain":
        for q in sorted(base):
            out.append(int(base[q][0] == base[q][1]))
        return out
    src = cand[1]
    aug = data[tag][year].get(src, {})
    mr = MR[src].get(year, {})
    for q in sorted(base):
        bp, gold, _ = base[q]
        r = aug.get(q)
        if cand[0] == "always":
            pred = r[0] if r is not None else bp
        else:
            _, _, T, C = cand
            m = mr.get(q, 0.0)
            pred = r[0] if (r is not None and m > 0 and m >= T and r[2] >= C) else bp
        out.append(int(pred == gold))
    return out


def vec(tag, cand, years):
    v = []
    for y in years:
        v += pick(tag, y, cand)
    return v


def candidates(tag, level):
    """level: 1=T1, 2=T2, 3=T3"""
    c = [("plain",)]
    srcs = ["rag"] if level < 3 else ["rag", "graph"]
    for s in srcs:
        if not any(data[tag].get(y, {}).get(s) for y in YEARS):
            continue
        c += [("gate", s, T, C) for T in THRS for C in CONFS]
        if level >= 2:
            c.append(("always", s))
    return c


def best(tag, years, level):
    pv = vec(tag, ("plain",), years)
    pa = sum(pv) / len(pv) if pv else 0.0
    bc, ba = ("plain",), pa
    for cd in candidates(tag, level):
        v = vec(tag, cd, years)
        a = sum(v) / len(v) if v else 0.0
        if a > ba:
            bc, ba = cd, a
    if ba - pa < DELTA:
        return ("plain",)
    return bc


def loyo(tag, level):
    out, chosen = [], {}
    for y in YEARS:
        cd = best(tag, [z for z in YEARS if z != y], level)
        chosen[y] = cd[0] if cd[0] != "gate" else f"gate_{cd[1]}"
        out += pick(tag, y, cd)
    return out, chosen


# ------------------------------------------------------------------ 통계
def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0
    from math import comb
    return min(1.0, sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n * 2)


def boot(a, b, iters=2000, seed=7):
    d = [y - x for x, y in zip(a, b)]
    n, st, out = len(d), seed, []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            st = (1103515245 * st + 12345) & 0x7FFFFFFF
            s += d[st % n]
        out.append(100 * s / n)
    out.sort()
    return out[int(.025 * iters)], out[int(.975 * iters)]


# ------------------------------------------------------------------ 실행
tags = [t for t in data if any(data[t].get(y, {}).get("rag") or data[t].get(y, {}).get("graph") for y in YEARS)]
tags.sort(key=lambda t: -sum(vec(t, ("plain",), YEARS)) / len(vec(t, ("plain",), YEARS)))

print(f"DELTA(abstain 마진) = {DELTA}")
print(f"{'모델':17s} | {'plain':>6s} | {'T1':>6s} {'T2':>6s} {'T3':>6s} | {'ΔT3':>7s} {'p':>8s} {'95% CI':>16s} | T3 선택")
print("-" * 122)
S = defaultdict(float)
rep = {}
for tag in tags:
    pv = vec(tag, ("plain",), YEARS)
    v1, _ = loyo(tag, 1)
    v2, _ = loyo(tag, 2)
    v3, ch3 = loyo(tag, 3)
    n = len(pv)
    ap, a1, a2, a3 = (100 * sum(x) / n for x in (pv, v1, v2, v3))
    p = mcnemar(pv, v3)
    lo, hi = boot(pv, v3)
    S["p"] += ap; S["1"] += a1; S["2"] += a2; S["3"] += a3
    rep[tag] = {"plain": ap, "T1": a1, "T2": a2, "T3": a3, "delta_T3": a3 - ap,
                "mcnemar_p": p, "ci95": [lo, hi], "choice": ch3}
    star = "*" if p < 0.05 else " "
    print(f"{DISPLAY.get(tag, tag):17s} | {ap:5.1f}% | {a1:5.1f}% {a2:5.1f}% {a3:5.1f}% | {a3-ap:+6.2f}p "
          f"{p:8.4f}{star} [{lo:+6.2f},{hi:+6.2f}] | {','.join(sorted(set(ch3.values())))}")
k = len(tags)
print("-" * 122)
print(f"{'9모델 평균':17s} | {S['p']/k:5.2f}% | {S['1']/k:5.2f}% {S['2']/k:5.2f}% {S['3']/k:5.2f}% | "
      f"{(S['3']-S['p'])/k:+6.2f}p")
print(f"\nT3 - T2 = {(S['3']-S['2'])/k:+.2f}%p    T3 - T1 = {(S['3']-S['1'])/k:+.2f}%p")
json.dump(rep, open(f"tune3_delta{DELTA}.json", "w"), ensure_ascii=False, indent=1)
