# -*- coding: utf-8 -*-
"""qwen LoRA 분류기로 KTM 문항별 maxrel 계산 (4연도 한 번에, 모델 1회 로드).
사용: python3 qwen_maxrel.py <corpus_noleak.jsonl> <base_model> <adapter> <KTM_data_dir> <out_prefix>
출력: <out_prefix>_2022.json ... 2025.json
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

corpus_path, base_model, adapter, ktm_dir, out_prefix = sys.argv[1:6]

docs = [json.loads(l)["doc_text"] for l in open(corpus_path, encoding="utf-8") if l.strip()]
doc_emb = np.load(corpus_path + ".bge_m3.npy")
embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")

tok = AutoTokenizer.from_pretrained(base_model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForSequenceClassification.from_pretrained(
    base_model, num_labels=2, torch_dtype=torch.bfloat16, device_map="cuda")
base.config.pad_token_id = tok.pad_token_id
clf = PeftModel.from_pretrained(base, adapter).eval()
print("[qwen] 분류기 로드 완료", flush=True)


@torch.no_grad()
def scores(q, cands):
    enc = tok([q] * len(cands), cands, truncation=True, max_length=256,
              padding=True, return_tensors="pt").to("cuda")
    return F.softmax(clf(**enc).logits, dim=-1)[:, 1].float().cpu().numpy()


for year in ["2022", "2023", "2024", "2025"]:
    qs = json.load(open(os.path.join(ktm_dir, f"{year}.json"), encoding="utf-8"))
    res = {}
    for q in qs:
        query = q["question_kr"] + " " + " ".join(q["choices_kr"])
        q_emb = embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
        sims = doc_emb @ q_emb
        top_idx = np.argsort(-sims)[:20]
        pr = scores(query, [docs[i] for i in top_idx])
        res[str(q["id"])] = float(max(pr)) if len(pr) else 0.0
    out = f"{out_prefix}_{year}.json"
    json.dump(res, open(out, "w"), ensure_ascii=False)
    print(f"[qwen] {year}: maxrel {len(res)} -> {out}", flush=True)
print("QWEN MAXREL DONE", flush=True)
