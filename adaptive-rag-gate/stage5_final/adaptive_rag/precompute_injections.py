# -*- coding: utf-8 -*-
"""RAG 주입을 미리 1회 계산해 저장(분류기+BGE만 사용, LLM 무관).

주입은 질문에만 의존(모델 무관)하므로 한 번 계산해 모든 모델이 재사용한다.
특히 vLLM으로 도메인 모델을 서빙할 때 GPU를 분류기와 나눠 쓰지 않아도 되게 한다.

사용: python3 precompute_injections.py <corpus> <classifier> <data.json> <out.json> [top_k] [thr] [max]
출력: {질문id: 주입블록문자열}  (주입 없으면 "")
"""
import json
import sys
from adaptive_rag import AdaptiveRAG

corpus, clf, data, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
top_k = int(sys.argv[5]) if len(sys.argv) > 5 else 20
thr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
mx = int(sys.argv[7]) if len(sys.argv) > 7 else 5

rag = AdaptiveRAG(corpus, clf, top_k=top_k, threshold=thr, max_inject=mx)
qs = json.load(open(data, encoding="utf-8"))

inj, n = {}, 0
for q in qs:
    query = q["question_kr"] + " " + " ".join(q["choices_kr"])
    block, docs = rag.augment(query)
    inj[q["id"]] = block
    if block:
        n += 1

json.dump(inj, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"주입 사전계산: {n}/{len(qs)} 문항 주입 → {out}")
