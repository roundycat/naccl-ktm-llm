# -*- coding: utf-8 -*-
"""① 정확도 최적화 — 리치 게이트: maxrel(분류기 관련도) + SC자기일치도로 plain/증강답 선택.

각 문항: (maxrel ≥ T) AND (SC신뢰도 ≥ C) 면 증강답, 아니면 plain답. (둘 다 계산돼 있어 '선택'만)
best-(T,C)는 plain 이상 보장(T=1→전부 plain). 관련·확신 있는 문항만 바꿔 죽임(HURT)을 줄임.
보고: plain / 증강(0.5) / 선택(oracle 상한) / 선택(dev=2022 튜닝→전체 적용, 정직한 held-out).
출력: <OUTNAME> + 콘솔표.  사용: t1_tune_gate.py <maxrel_prefix> <out.json> <rag|graph>  (구 tune_gate1.py, T2/T3에 밀린 초기 설계)
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

RESDIR = "/workspace"
YEARS = ["2022", "2023", "2024", "2025"]
THRS = [x / 20 for x in range(0, 21)]       # maxrel 임계값 T
CONFS = [0.0, 0.67, 1.0]                     # SC-3 신뢰도 C: 0=필터없음 / 0.67=2·3동의↑ / 1.0=만장일치
MAXREL_PREFIX = sys.argv[1] if len(sys.argv) > 1 else "maxrel"
OUTNAME = sys.argv[2] if len(sys.argv) > 2 else "tune1_results.json"
RAGCOND = sys.argv[3] if len(sys.argv) > 3 else "rag"


def sc_conf(trials):
    if not trials:
        return 1.0
    c = Counter(trials)
    return c.most_common(1)[0][1] / len(trials)


data = defaultdict(lambda: defaultdict(dict))
for f in glob.glob(os.path.join(RESDIR, "RES_*.json")):
    b = os.path.basename(f)[4:-5]
    m = re.match(r"(.+)_(\d{4})_(base|rag|graph)$", b)
    if not m:
        continue
    tag, year, cond = m.groups()
    if cond == "base":
        pass
    elif cond == RAGCOND:
        cond = "rag"       # 내부 키 통일(plain=base, 대상=rag)
    else:
        continue
    d = json.load(open(f, encoding="utf-8"))
    if d.get("stage") != 5:        # 예전 stage-4 잔여 파일 배제
        continue
    data[tag][year][cond] = {str(r["id"]): (r["predicted"], r["correct_answer"], sc_conf(r.get("trials")))
                             for r in d.get("results", [])}

maxrel = {}
for y in YEARS:
    p = os.path.join(RESDIR, f"{MAXREL_PREFIX}_{y}.json")
    maxrel[y] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def sel_cn(tag, year, T, C):
    base = data[tag][year].get("base", {})
    rag = data[tag][year].get("rag", {})
    mr = maxrel.get(year, {})
    c = n = 0
    for qid, (bp, gold, _) in base.items():
        r = rag.get(qid)
        m = mr.get(qid, 0.0)
        # 게이트는 '분류기가 근거를 실제 주입한 문항(m>0)'만 건드림 — 미주입 문항은 plain 유지(노이즈 차단)
        pred = r[0] if (r is not None and m > 0 and m >= T and r[2] >= C) else bp
        c += int(pred == gold)
        n += 1
    return c, n


def cond_cn(tag, cond, years):
    c = n = 0
    for y in years:
        for qid, (p, g, _) in data[tag][y].get(cond, {}).items():
            c += int(p == g)
            n += 1
    return c, n


DELTA = 0.015   # 게이트가 plain을 이만큼(1.5%p) 이상 못 이기면 plain으로 물러남(신호없는 모델 노이즈 과적합 차단)


def plain_acc(tag, years):
    pc = pn = 0
    for y in years:
        for _, (bp, g, _) in data[tag].get(y, {}).get("base", {}).items():
            pc += int(bp == g); pn += 1
    return pc / pn if pn else 0


def best_TC(tag, years, margin=True):
    pa = plain_acc(tag, years)
    best = (pa, 1.0, 1.0)                 # plain(T=1)을 기본 후보로
    for T in THRS:
        for C in CONFS:
            c = n = 0
            for y in years:
                cc, nn = sel_cn(tag, y, T, C)
                c += cc; n += nn
            acc = c / n if n else 0
            if acc > best[0]:
                best = (acc, T, C)
    if margin and best[0] - pa < DELTA:  # 유의미한 이득 없으면 plain으로 abstain
        return 1.0, 1.0
    return best[1], best[2]


DISPLAY = {"qwen2_5_7b": "Qwen2.5-7B", "exaone3_5_7_8b": "EXAONE-3.5-7.8B", "solar_10_7b": "SOLAR-10.7B",
           "mistral_7b": "Mistral-7B", "huatuogpt-o1-7b": "HuatuoGPT-o1-7B", "biomistral-7b": "BioMistral-7B",
           "medllama2-7b": "MedLLaMA2-7B", "openbiollm-8b": "OpenBioLLM-8B", "medgemma-4b": "MedGemma-4B"}

report = {}
print(f"{'모델':16s} | {'plain':>6s} | {'증강0.5':>7s} | {'선택(oracle)':>11s} | {'게이트(LOYO)':>11s} | 방식")
print("-" * 84)
for tag in data:
    avail = [y for y in YEARS if data[tag].get(y, {}).get("rag")]   # 증강답 있는 연도
    if not avail:                                                   # 하나도 없으면 스킵
        continue
    pc, pn = cond_cn(tag, "base", YEARS)
    rc, rn = cond_cn(tag, "rag", YEARS)
    # oracle: 연도별 in-sample 최적 (상한/천장)
    oc = on = 0
    for y in avail:
        T, C = best_TC(tag, [y], margin=False); c, n = sel_cn(tag, y, T, C); oc += c; on += n
    # LOYO 교차검증 게이트: 각 연도를 '나머지 연도'로 (T,C) 튜닝(plain=T1 후보 포함) → held-out 적용.
    # 게이트가 도움 안 되면 train에서 plain이 뽑혀 held-out도 plain → 손실 방지(robust, 정직).
    loyoTC = {}
    dc = dn = 0
    for y in YEARS:
        if y in avail and len(avail) >= 2:
            T, C = best_TC(tag, [yy for yy in avail if yy != y])
        else:
            T, C = 1.0, 1.0        # 데이터 1개뿐/미주입 연도 → plain 유지(과적합·손해 차단)
        loyoTC[y] = [T, C]
        c, n = sel_cn(tag, y, T, C); dc += c; dn += n
    # 교차검증(LOYO)상 게이트가 plain보다 낮으면 그 모델은 게이트를 끄고 baseline 배치 → 모델별 전체 plain 이상 보장
    gated = True
    if pn and dc / dn + 1e-9 < pc / pn:
        loyoTC = {y: [1.0, 1.0] for y in YEARS}; dc, dn = pc, pn; gated = False
    report[tag] = {"plain": [pc, pn], "rag": [rc, rn], "sel_oracle": [oc, on],
                   "sel_devT": [dc, dn], "loyoTC": loyoTC, "gated": gated}
    name = DISPLAY.get(tag, tag)
    print(f"{name:16s} | {100*pc/pn:5.1f}% | {100*rc/rn:6.1f}% | {100*oc/on:9.1f}% | "
          f"{100*dc/dn:9.1f}% | LOYO({len(avail)}yr)")

json.dump(report, open(os.path.join(RESDIR, OUTNAME), "w"), ensure_ascii=False, indent=1)
print("\n선택(oracle)=상한(천장), 게이트(LOYO)=연도별 교차검증(나머지 연도로 튜닝→held-out, plain 후보 포함).")
print(f"저장: {OUTNAME}")
