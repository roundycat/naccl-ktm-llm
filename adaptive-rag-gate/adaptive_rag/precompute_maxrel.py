# -*- coding: utf-8 -*-
"""① 튜닝용 — KTM 문항별 '최대 관련도'(top-20 후보 중 분류기 최고점) 저장.
이 점수로 나중에 plain/RAG 선택 임계값을 스윕한다(재실행 없이).
사용: python3 precompute_maxrel.py <corpus> <classifier> <KTM_data.json> <out.json>
"""
import json
import sys
import numpy as np
from adaptive_rag import AdaptiveRAG

corpus, clf, data, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rag = AdaptiveRAG(corpus, clf, top_k=20, threshold=0.0, max_inject=20)
qs = json.load(open(data, encoding="utf-8"))
res = {}
for q in qs:
    query = q["question_kr"] + " " + " ".join(q["choices_kr"])
    q_emb = rag.embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
    sims = rag.doc_emb @ q_emb
    top_idx = np.argsort(-sims)[:20]
    probs = rag._gate_scores(query, [rag.doc_texts[i] for i in top_idx])
    res[str(q["id"])] = float(max(probs)) if len(probs) else 0.0
json.dump(res, open(out, "w"), ensure_ascii=False)
print(f"maxrel {len(res)} -> {out}", flush=True)
