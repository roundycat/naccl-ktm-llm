# -*- coding: utf-8 -*-
"""게이트용 — 각 문항의 GraphRAG 근거에 대해 분류기 관련도 점수.
sel: 점수≥T면 graph답, 아니면 plain답. 근거 없으면 0(→plain).
사용: python3 precompute_maxrel_graph.py <classifier> <KTM_data_dir> <inj_prefix> <out_prefix>
"""
import json
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

clf_path, ktm_dir, inj_prefix, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
tok = AutoTokenizer.from_pretrained(clf_path)
dev = "cuda" if torch.cuda.is_available() else "cpu"
clf = AutoModelForSequenceClassification.from_pretrained(clf_path).to(dev).eval()


@torch.no_grad()
def score_batch(pairs):
    out = []
    for i in range(0, len(pairs), 32):
        b = pairs[i:i + 32]
        enc = tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=256,
                  padding=True, return_tensors="pt", return_token_type_ids=False).to(dev)
        out.extend(F.softmax(clf(**enc).logits, dim=-1)[:, 1].cpu().tolist())
    return out


for year in ["2022", "2023", "2024", "2025"]:
    qs = json.load(open(os.path.join(ktm_dir, f"{year}.json"), encoding="utf-8"))
    inj = json.load(open(f"{inj_prefix}_{year}.json", encoding="utf-8"))
    ids, pairs = [], []
    res = {}
    for q in qs:
        qid = str(q["id"])
        block = inj.get(qid, "") or ""
        if block.strip():
            ids.append(qid)
            pairs.append((q["question_kr"] + " " + " ".join(q["choices_kr"]), block))
        else:
            res[qid] = 0.0
    scores = score_batch(pairs) if pairs else []
    for qid, sc in zip(ids, scores):
        res[qid] = float(sc)
    json.dump(res, open(f"{out_prefix}_{year}.json", "w"), ensure_ascii=False)
    print(f"[maxrel_graph] {year}: {len(res)}문항 ({len(ids)}개 근거有) -> {out_prefix}_{year}.json", flush=True)
print("MAXREL_GRAPH DONE", flush=True)
