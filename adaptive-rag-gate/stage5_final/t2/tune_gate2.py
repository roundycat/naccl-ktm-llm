# -*- coding: utf-8 -*-
"""T2 — LOYO 후보집합에 always-inject 추가.

기존(tune_gate1): 후보 = {plain, gate(T,C)}  … gate 는 분류기 주입 문항(m>0)만 건드림
                  → always-inject 를 선택지로 갖지 못해 무게이트에 진다 (CLAUDE.md 4.3)
T2(이 파일):     후보 = {plain, gate(T,C), always-inject}
                  → 모델별로 LOYO 가 셋 중 최선을 held-out 에 적용

모든 델타에 McNemar 검정 + 부트스트랩 95% CI 를 붙인다 (CLAUDE.md 6.3).
동일 문항 집합에서만 비교하며 결측 연도를 분모에서 빼지 않는다 (6.4).

사용: tune_gate2.py <pred_dir> <maxrel_dir> <maxrel_prefix> <rag|graph> [out.json]
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
MR_DIR = sys.argv[2] if len(sys.argv) > 2 else PRED_DIR
MR_PREFIX = sys.argv[3] if len(sys.argv) > 3 else "maxrel"
RAGCOND = sys.argv[4] if len(sys.argv) > 4 else "rag"
OUT = sys.argv[5] if len(sys.argv) > 5 else f"tune2_{RAGCOND}.json"

YEARS = ["2022", "2023", "2024", "2025"]
THRS = [x / 20 for x in range(0, 21)]
CONFS = [0.0, 0.67, 1.0]
DELTA = 0.015          # tune_gate1 과 동일한 abstain 마진(공정 비교 위해 유지)
ALWAYS = "__always__"  # 후보 식별자

DISPLAY = {"qwen2_5_7b": "Qwen2.5-7B", "exaone3_5_7_8b": "EXAONE-3.5-7.8B", "solar_10_7b": "SOLAR-10.7B",
           "mistral_7b": "Mistral-7B", "huatuogpt-o1-7b": "HuatuoGPT-o1-7B", "biomistral-7b": "BioMistral-7B",
           "medllama2-7b": "MedLLaMA2-7B", "openbiollm-8b": "OpenBioLLM-8B", "medgemma-4b": "MedGemma-4B"}


def sc_conf(trials):
    if not trials:
        return 1.0
    c = Counter(trials)
    return c.most_common(1)[0][1] / len(trials)


# ------------------------------------------------------------------ 데이터 적재
data = defaultdict(lambda: defaultdict(dict))
for f in glob.glob(os.path.join(PRED_DIR, "RES_*.json")):
    b = os.path.basename(f)[4:-5]
    m = re.match(r"(.+)_(\d{4})_(base|rag|graph)$", b)
    if not m:
        continue
    tag, year, cond = m.groups()
    if cond == "base":
        pass
    elif cond == RAGCOND:
        cond = "rag"
    else:
        continue
    d = json.load(open(f, encoding="utf-8"))
    if d.get("stage") != 5:
        continue
    data[tag][year][cond] = {str(r["id"]): (r["predicted"], r["correct_answer"], sc_conf(r.get("trials")))
                             for r in d.get("results", [])}

maxrel = {}
for y in YEARS:
    p = os.path.join(MR_DIR, f"{MR_PREFIX}_{y}.json")
    maxrel[y] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


# ------------------------------------------------------------------ 선택 규칙
def pick(tag, year, cand):
    """cand = ALWAYS 또는 (T, C). 문항별 정오 리스트를 (qid 순서 고정) 반환."""
    base = data[tag][year].get("base", {})
    rag = data[tag][year].get("rag", {})
    mr = maxrel.get(year, {})
    out = []
    for qid in sorted(base):
        bp, gold, _ = base[qid]
        r = rag.get(qid)
        if cand is ALWAYS:
            pred = r[0] if r is not None else bp
        else:
            T, C = cand
            pred = r[0] if (r is not None and mr.get(qid, 0.0) > 0
                            and mr.get(qid, 0.0) >= T and r[2] >= C) else bp
        out.append(int(pred == gold))
    return out


def acc_over(tag, cand, years):
    v = []
    for y in years:
        v += pick(tag, y, cand)
    return (sum(v) / len(v) if v else 0.0), v


def plain_vec(tag, years):
    v = []
    for y in years:
        base = data[tag].get(y, {}).get("base", {})
        v += [int(base[q][0] == base[q][1]) for q in sorted(base)]
    return v


def ungated_vec(tag, years):
    """무게이트: 증강답이 있으면 무조건 증강답(= ALWAYS 와 동일)."""
    return acc_over(tag, ALWAYS, years)[1]


# ------------------------------------------------------------------ 후보 탐색
def best_cand(tag, years, include_always, margin=True):
    pv = plain_vec(tag, years)
    pa = sum(pv) / len(pv) if pv else 0.0
    best, best_acc = None, pa                      # plain 이 기본 후보
    cands = [(T, C) for T in THRS for C in CONFS]
    if include_always:
        cands.append(ALWAYS)
    for cd in cands:
        a, _ = acc_over(tag, cd, years)
        if a > best_acc:
            best, best_acc = cd, a
    if best is None:
        return None
    if margin and best_acc - pa < DELTA:
        return None                                # 이득 미미 → plain 으로 abstain
    return best


def loyo(tag, include_always):
    """각 연도를 나머지 연도로 튜닝 → held-out 적용. 문항별 정오 벡터 반환."""
    avail = [y for y in YEARS if data[tag].get(y, {}).get("rag")]
    vec, chosen = [], {}
    for y in YEARS:
        cd = best_cand(tag, [z for z in avail if z != y], include_always) if (y in avail and len(avail) >= 2) else None
        chosen[y] = "always" if cd is ALWAYS else ("plain" if cd is None else f"T={cd[0]:.2f},C={cd[1]:.2f}")
        base = data[tag].get(y, {}).get("base", {})
        if cd is None:
            vec += [int(base[q][0] == base[q][1]) for q in sorted(base)]
        else:
            vec += pick(tag, y, cd)
    return vec, chosen


# ------------------------------------------------------------------ 통계
def mcnemar(a, b):
    """a,b: 같은 문항 순서의 정오 벡터. 양측 이항검정 p."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    from math import comb
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p), n01, n10


def boot_ci(a, b, iters=2000, seed=12345):
    """페어드 부트스트랩 95% CI (b - a)."""
    n = len(a)
    d = [y - x for x, y in zip(a, b)]
    st = seed
    out = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            st = (1103515245 * st + 12345) & 0x7FFFFFFF
            s += d[st % n]
        out.append(100.0 * s / n)
    out.sort()
    return out[int(0.025 * iters)], out[int(0.975 * iters)]


# ------------------------------------------------------------------ 실행
tags = [t for t in data if any(data[t].get(y, {}).get("rag") for y in YEARS)]
tags.sort(key=lambda t: -sum(plain_vec(t, YEARS)) / max(1, len(plain_vec(t, YEARS))))

print(f"조건: {RAGCOND}   (후보집합 T2 = plain / gate(T,C) / always-inject)")
print(f"{'모델':17s} | {'plain':>6s} {'무게이트':>7s} | {'게이트 T1':>8s} | {'게이트 T2':>8s} {'ΔT2-plain':>9s} {'p':>8s} {'95% CI':>16s} | T2선택")
print("-" * 132)
rep, sums = {}, defaultdict(float)
for tag in tags:
    pv = plain_vec(tag, YEARS)
    uv = ungated_vec(tag, YEARS)
    v1, _ = loyo(tag, include_always=False)
    v2, ch2 = loyo(tag, include_always=True)
    n = len(pv)
    ap, au, a1, a2 = (100 * sum(x) / n for x in (pv, uv, v1, v2))
    p, n01, n10 = mcnemar(pv, v2)
    lo, hi = boot_ci(pv, v2)
    picks = sorted(set(ch2.values()))
    rep[tag] = {"n": n, "plain": ap, "ungated": au, "gate_T1": a1, "gate_T2": a2,
                "delta_T2": a2 - ap, "mcnemar_p": p, "n01": n01, "n10": n10,
                "ci95": [lo, hi], "loyo_choice": ch2}
    for k, v in (("plain", ap), ("ungated", au), ("t1", a1), ("t2", a2)):
        sums[k] += v
    star = "*" if p < 0.05 else " "
    print(f"{DISPLAY.get(tag, tag):17s} | {ap:5.1f}% {au:6.1f}% | {a1:7.1f}% | {a2:7.1f}% {a2-ap:+8.2f}p "
          f"{p:8.4f}{star} [{lo:+6.2f},{hi:+6.2f}] | {','.join(picks)}")

k = len(tags)
print("-" * 132)
print(f"{'9모델 평균':17s} | {sums['plain']/k:5.2f}% {sums['ungated']/k:6.2f}% | {sums['t1']/k:7.2f}% | "
      f"{sums['t2']/k:7.2f}% {(sums['t2']-sums['plain'])/k:+8.2f}p")
print(f"\nT1 = 기존 후보집합(plain/gate)   T2 = plain/gate/always-inject")
print(f"T2 - T1 = {(sums['t2']-sums['t1'])/k:+.2f}%p,   T2 - 무게이트 = {(sums['t2']-sums['ungated'])/k:+.2f}%p")
json.dump(rep, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"저장: {OUT}")
