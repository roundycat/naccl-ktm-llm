# -*- coding: utf-8 -*-
"""어댑티브 RAG — 우리가 학습한 관련성 분류기를 게이트로 사용.

2단계:
  1) 1차 리트리버(BGE-m3 임베딩): 코퍼스에서 질문과 유사한 후보 top-K 검색.
  2) 게이트(우리 BERT 크로스인코더): (질문, 각 후보) 관련성 확률 P(relevant) 판정.
     임계값 이상만 통과 → 확률 내림차순 상위 N개를 '참고 지식'으로 주입.

'adaptive'인 이유: 통과 문서가 없으면(전부 무관) 아무것도 주입하지 않는다
→ 무관한 지식으로 프롬프트를 오염시키지 않음. 이게 분류기의 핵심 가치.

의존성(박스에서 실행): torch, transformers, sentence-transformers, numpy.
분류기 경로 예: /workspace/bert_clf/best  (klue/roberta 크로스인코더)
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class AdaptiveRAG:
    def __init__(self, corpus_path: str, classifier_path: str,
                 embed_model: str = "BAAI/bge-m3", device: str | None = None,
                 top_k: int = 20, threshold: float = 0.5, max_inject: int = 5,
                 emb_cache: str | None = None, max_len: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k, self.threshold, self.max_inject, self.max_len = top_k, threshold, max_inject, max_len

        # 코퍼스
        self.docs = [json.loads(l) for l in open(corpus_path, encoding="utf-8") if l.strip()]
        self.doc_texts = [d["doc_text"] for d in self.docs]

        # 1차 리트리버 (BGE-m3) — 코퍼스 임베딩은 1회 만들어 캐시
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embed_model, device=self.device)
        self.emb_cache = emb_cache or (corpus_path + ".bge_m3.npy")
        self.doc_emb = self._load_or_build_embeddings()

        # 게이트 (우리 분류기)
        self.tok = AutoTokenizer.from_pretrained(classifier_path)
        self.clf = AutoModelForSequenceClassification.from_pretrained(classifier_path).to(self.device).eval()

    def _load_or_build_embeddings(self) -> np.ndarray:
        if os.path.exists(self.emb_cache):
            return np.load(self.emb_cache)
        print(f"[RAG] 코퍼스 임베딩 생성 {len(self.doc_texts)}개 (1회, 캐시→{self.emb_cache})", flush=True)
        emb = self.embedder.encode(self.doc_texts, batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=True).astype(np.float32)
        np.save(self.emb_cache, emb)
        return emb

    @torch.no_grad()
    def _gate_scores(self, question: str, cand_texts: list[str]) -> np.ndarray:
        """크로스인코더로 (질문, 후보) 관련성 확률. return_token_type_ids=False (klue/roberta)."""
        enc = self.tok([question] * len(cand_texts), cand_texts, truncation=True,
                       max_length=self.max_len, padding=True, return_tensors="pt",
                       return_token_type_ids=False).to(self.device)
        logits = self.clf(**enc).logits
        return F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    def retrieve(self, query: str) -> list[dict]:
        """query(문제+보기)로 후보 검색 → 게이트 통과 문서 리스트(관련도 포함)."""
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        sims = self.doc_emb @ q_emb
        top_idx = np.argsort(-sims)[:self.top_k]
        cand_texts = [self.doc_texts[i] for i in top_idx]
        probs = self._gate_scores(query, cand_texts)
        keep = [(int(top_idx[j]), float(probs[j])) for j in range(len(top_idx))
                if probs[j] >= self.threshold]
        keep.sort(key=lambda x: -x[1])
        keep = keep[:self.max_inject]
        return [{**self.docs[i], "relevance": p} for i, p in keep]

    @staticmethod
    def format_injection(docs: list[dict]) -> str:
        if not docs:
            return ""
        lines = ["[참고 지식 — 아래는 이 문제와 관련된 한의학 지식입니다]"]
        for d in docs:
            lines.append(f"- {d['doc_text']}")
        return "\n".join(lines)

    def augment(self, query: str) -> tuple[str, list[dict]]:
        """(주입블록 문자열, 통과문서들) 반환. 통과 없으면 ('', [])."""
        docs = self.retrieve(query)
        return self.format_injection(docs), docs
