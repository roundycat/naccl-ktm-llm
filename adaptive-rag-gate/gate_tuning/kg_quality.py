# -*- coding: utf-8 -*-
"""논문 4.2.3 Graph Quality Validation + 4.2.4 Leakage Control + 5.5 Graph Coverage.

4.2.3 구조 검증
  - 유형 규칙 위반: 주치(처방->증상), 구성(처방->약재), 계통(처방->계통) 등
  - 고아 엣지(존재하지 않는 노드 참조), 자기루프, 중복 엣지
  - 동명 분열(name collision): 같은 name_ko 가 서로 다른 id 로 갈라진 비율

4.2.4 누수 통제
  - 검색 근거가 정답 보기를 그대로 담고 있으면 정답 누수.
  - containment(보기, 근거) = 보기 문자열이 근거에 포함되는지 / 최대 포함률
  - 정답 보기와 오답 보기의 containment 분포를 비교해 누수 여부를 판정.

5.5 커버리지
  - 시드 0개 문항 비율, 그래프 근거 없는 문항 비율

사용: kg_quality.py <kg_dir> <ktm_data_dir> <graphrag_ctx_dir>
"""
import json
import os
import sys
from collections import Counter, defaultdict

KG, DATA, CTX = sys.argv[1], sys.argv[2], sys.argv[3]
YEARS = ["2022", "2023", "2024", "2025"]

nodes = {}
for l in open(f"{KG}/kg_all_nodes.jsonl", encoding="utf-8"):
    n = json.loads(l); nodes[n["id"]] = n
edges = [json.loads(l) for l in open(f"{KG}/kg_all_edges.jsonl", encoding="utf-8")]

print("=" * 78)
print("[4.2.3] Graph Quality Validation")
print("=" * 78)
print(f"노드 {len(nodes):,} / 엣지 {len(edges):,}")

# --- 유형 규칙
RULE = {"주치": ("처방", ("증상", "변증")), "구성": ("처방", ("약재",)),
        "계통": ("처방", ("계통",)), "귀경": ("약재", ("장부",)),
        "팔강귀속": (None, None), "지표": (None, ("증상지표",)), "포함": (None, None)}
viol = Counter(); orphan = 0; selfloop = 0
seen = set(); dup = 0
for e in edges:
    s, d, t = e.get("src"), e.get("dst"), e.get("type")
    if s not in nodes or d not in nodes:
        orphan += 1; continue
    if s == d:
        selfloop += 1
    k = (s, d, t)
    if k in seen:
        dup += 1
    seen.add(k)
    exp = RULE.get(t)
    if exp and exp[0] and nodes[s]["type"] != exp[0]:
        viol[f"{t}: src가 {nodes[s]['type']}(기대 {exp[0]})"] += 1
    if exp and exp[1] and nodes[d]["type"] not in exp[1]:
        viol[f"{t}: dst가 {nodes[d]['type']}(기대 {'/'.join(exp[1])})"] += 1
tot_struct = orphan + selfloop + dup + sum(viol.values())
print(f"  고아 엣지(미존재 노드 참조) : {orphan}")
print(f"  자기루프                   : {selfloop}")
print(f"  중복 엣지                  : {dup}")
print(f"  유형 규칙 위반             : {sum(viol.values())}")
for k, v in viol.most_common(6):
    print(f"      - {k}: {v}")
print(f"  ▶ 구조 오류율 = {tot_struct}/{len(edges)} = {100*tot_struct/len(edges):.3f}%")

# --- 동명 분열
byname = defaultdict(list)
for n in nodes.values():
    byname[(n["type"], n.get("name_ko", ""))].append(n["id"])
split = {k: v for k, v in byname.items() if len(v) > 1}
n_split_nodes = sum(len(v) for v in split.values())
print(f"\n  동명 분열: {len(split):,}개 이름이 2개 이상 id 보유 "
      f"({n_split_nodes:,}노드 = 전체의 {100*n_split_nodes/len(nodes):.1f}%)")
for k, v in sorted(split.items(), key=lambda kv: -len(kv[1]))[:5]:
    print(f"      - {k[0]}/{k[1]}: {len(v)}개 id")
by_type = Counter(k[0] for k in split)
print(f"  유형별 분열 이름 수: {dict(by_type)}")

# --- 5.5 커버리지
print()
print("=" * 78)
print("[5.5] Graph Coverage")
print("=" * 78)
cov = {}
tot_q = tot_g = 0
for y in YEARS:
    ctx = json.load(open(f"{CTX}/graphrag_context_{y}.json", encoding="utf-8"))
    qs = json.load(open(f"{DATA}/{y}.json", encoding="utf-8"))
    has_graph = sum(1 for v in ctx.values() if v and "[그래프 근거" in v)
    empty = sum(1 for v in ctx.values() if not v)
    cov[y] = (len(qs), has_graph, empty)
    tot_q += len(qs); tot_g += has_graph
    print(f"  {y}: {len(qs)}문항 | 그래프근거 {has_graph} ({100*has_graph/len(qs):.1f}%) | 근거 전무 {empty}")
print(f"  ▶ 전체: {tot_g}/{tot_q} = {100*tot_g/tot_q:.1f}% 에 그래프 근거 존재 "
      f"(seed 0 문항 = {100*(tot_q-tot_g)/tot_q:.1f}%)")

# --- 4.2.4 누수 통제
print()
print("=" * 78)
print("[4.2.4] Leakage Control — 근거가 정답 보기를 담고 있는가")
print("=" * 78)


def containment(choice, ctx):
    """보기 문자열이 근거에 통째로 등장하면 1.0, 아니면 최장 공통 부분열 비율 근사."""
    c = choice.strip()
    if not c:
        return 0.0
    if c in ctx:
        return 1.0
    # 부분 포함률: 보기를 2-gram 으로 쪼개 근거에 나타나는 비율
    grams = [c[i:i + 2] for i in range(len(c) - 1)] or [c]
    return sum(1 for g in grams if g in ctx) / len(grams)


rows = []
for y in YEARS:
    ctx = json.load(open(f"{CTX}/graphrag_context_{y}.json", encoding="utf-8"))
    qs = json.load(open(f"{DATA}/{y}.json", encoding="utf-8"))
    gold_c, dist_c = [], []
    for q in qs:
        c = ctx.get(str(q["id"]), "")
        if not c:
            continue
        ch = q.get("choices_kr") or []
        gi = q["correct_answer"] - 1
        for i, opt in enumerate(ch):
            v = containment(opt, c)
            (gold_c if i == gi else dist_c).append(v)
    if not gold_c:
        continue
    gmax, dmax = max(gold_c), max(dist_c)
    gmean, dmean = sum(gold_c) / len(gold_c), sum(dist_c) / len(dist_c)
    gfull = sum(1 for v in gold_c if v >= 1.0)
    dfull = sum(1 for v in dist_c if v >= 1.0)
    rows.append((y, gmean, gmax, gfull, len(gold_c), dmean, dmax, dfull, len(dist_c)))
    print(f"  {y}: 정답보기 평균 {gmean:.3f} 최대 {gmax:.3f} 완전포함 {gfull}/{len(gold_c)} | "
          f"오답보기 평균 {dmean:.3f} 최대 {dmax:.3f} 완전포함 {dfull}/{len(dist_c)}")
if rows:
    G = [r[1] for r in rows]; D = [r[5] for r in rows]
    print(f"\n  ▶ 정답 평균 containment {sum(G)/len(G):.3f} vs 오답 {sum(D)/len(D):.3f} "
          f"(차이 {sum(G)/len(G)-sum(D)/len(D):+.3f})")
    print(f"  ▶ 최대 containment: 정답 {max(r[2] for r in rows):.3f} / 오답 {max(r[6] for r in rows):.3f}")
    print("  해석: 정답과 오답의 containment 가 비슷하면 근거가 정답을 직접 노출하지 않는다는 뜻.")
json.dump({"struct_error_rate": 100 * tot_struct / len(edges),
           "orphan": orphan, "selfloop": selfloop, "dup": dup,
           "type_violation": sum(viol.values()),
           "name_split_names": len(split), "name_split_nodes": n_split_nodes,
           "name_split_pct": 100 * n_split_nodes / len(nodes),
           "coverage_pct": 100 * tot_g / tot_q, "seed0_pct": 100 * (tot_q - tot_g) / tot_q,
           "leakage_rows": rows}, open("kg_quality_result.json", "w"), ensure_ascii=False, indent=1)
print("\n저장: kg_quality_result.json")
