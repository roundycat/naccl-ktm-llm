# -*- coding: utf-8 -*-
"""논문 5.1 Run-to-Run Variance.

동일 조건(stage 5, SC-3, 동일 프롬프트)을 서로 다른 실행에서 두 번 생성한 결과를 비교한다.
  run A = ktm_results/          (팀 baseline, 원 실행)
  run B = ktm_results_plain_v2/ (본 연구에서 동일 스택으로 재생성)

두 실행의 차이는 조건 차이가 아니라 표집 분산(temperature>0 인 SC)에서만 온다.
따라서 이 차이가 곧 '어떤 개입 효과가 유의하려면 넘어야 하는 노이즈 바닥'이다.

사용: variance.py <dirA> <dirB>
"""
import glob
import json
import os
import sys
from collections import defaultdict
from math import comb

A, B = sys.argv[1], sys.argv[2]
DISP = {"qwen2.5-7b": "Qwen2.5-7B", "exaone3.5-7.8b": "EXAONE-3.5-7.8B", "solar-10.7b": "SOLAR-10.7B",
        "mistral-7b": "Mistral-7B", "huatuogpt-o1-7b": "HuatuoGPT-o1-7B", "biomistral-7b": "BioMistral-7B",
        "medllama2-7b": "MedLLaMA2-7B", "openbiollm-8b": "OpenBioLLM-8B", "medgemma-4b": "MedGemma-4B"}


def load(root):
    out = defaultdict(dict)          # model -> qid -> (pred, gold)
    for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        name = os.path.basename(p)[:-5]
        if name.startswith("_"):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for r in d.get("results", []):
            out[name][str(r["id"])] = (r["predicted"], r["correct_answer"])
    return out


ra, rb = load(A), load(B)
models = sorted(set(ra) & set(rb), key=lambda m: -sum(1 for v in ra[m].values() if v[0] == v[1]))


def mcnemar(x, y):
    n01 = sum(1 for a, b in zip(x, y) if a == 0 and b == 1)
    n10 = sum(1 for a, b in zip(x, y) if a == 1 and b == 0)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2), n01, n10


print("=" * 100)
print("[5.1] Run-to-Run Variance — 동일 조건 두 실행의 차이 (= 노이즈 바닥)")
print("=" * 100)
print(f"{'모델':17s} | {'run A':>7s} {'run B':>7s} {'|Δ|':>6s} | {'불일치문항':>9s} {'불일치율':>8s} | {'p':>8s}")
print("-" * 84)
tot_dis = tot_n = 0
deltas = []
for m in models:
    common = sorted(set(ra[m]) & set(rb[m]))
    if not common:
        continue
    va = [int(ra[m][q][0] == ra[m][q][1]) for q in common]
    vb = [int(rb[m][q][0] == rb[m][q][1]) for q in common]
    aa, ab = 100 * sum(va) / len(va), 100 * sum(vb) / len(vb)
    dis = sum(1 for q in common if ra[m][q][0] != rb[m][q][0])
    p, n01, n10 = mcnemar(va, vb)
    tot_dis += dis; tot_n += len(common)
    deltas.append(abs(aa - ab))
    print(f"{DISP.get(m,m):17s} | {aa:6.2f}% {ab:6.2f}% {abs(aa-ab):5.2f}p | "
          f"{dis:9d} {100*dis/len(common):7.1f}% | {p:8.4f}{'*' if p<0.05 else ' '}")
print("-" * 84)
mean_d = sum(deltas) / len(deltas)
mx = max(deltas)
print(f"{'평균':17s} | {'':7s} {'':7s} {mean_d:5.2f}p | {tot_dis:9d} {100*tot_dis/tot_n:7.1f}% |")
print(f"{'최대':17s} | {'':7s} {'':7s} {mx:5.2f}p |")
print()
print(f"■ 노이즈 바닥: 동일 조건 재실행 시 모델별 정확도가 평균 {mean_d:.2f}%p (최대 {mx:.2f}%p) 변동")
print(f"■ 문항 단위로는 {100*tot_dis/tot_n:.1f}% 의 문항에서 예측이 뒤바뀜 ({tot_dis:,}/{tot_n:,})")
print()
print("해석: 개입 효과가 이 수준을 넘지 못하면 표집 분산과 구별할 수 없다.")
json.dump({"mean_abs_delta": mean_d, "max_abs_delta": mx,
           "disagree_rate": 100 * tot_dis / tot_n, "disagree_n": tot_dis, "n": tot_n},
          open("variance_result.json", "w"), ensure_ascii=False, indent=1)
print("저장: variance_result.json")
