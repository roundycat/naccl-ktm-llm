"""
rag_retriever.py
================
rag_index.py로 만든 FAISS 인덱스를 로드해서 검색 기능을 제공한다.
tkm_pipeline.py가 --rag 옵션 사용 시 이 모듈을 import한다.

단독 테스트:
    python rag_retriever.py --query "골절 환자 침 치료"
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np


class RAGRetriever:
    def __init__(self, index_dir: str, model_name: Optional[str] = None, device: str = "cpu"):
        """
        device는 기본값 "cpu"로 둔다. 실제 벤치마크 실행 중에는 vLLM 서버가
        이미 GPU 메모리 대부분(gpu_memory_utilization)을 점유하고 있어서,
        임베딩 모델까지 GPU에 올리면 OOM이 난다. 검색 시점엔 질문 텍스트
        하나만 임베딩하면 되므로 CPU로도 충분히 빠르다.
        (대량 인덱싱을 하는 rag_index.py는 vLLM 없이 단독 실행되므로 GPU를 써도 무방)
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)

        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        self.chunks: list[dict] = []
        with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        self.model = SentenceTransformer(model_name or meta["model_name"], device=device)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        vec = self.model.encode([query], normalize_embeddings=True)
        vec = np.asarray(vec, dtype="float32")
        scores, idxs = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["_score"] = float(score)
            results.append(chunk)
        return results

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        """검색된 청크들을 프롬프트에 넣기 좋은 텍스트 블록으로 변환."""
        blocks = []
        for c in chunks:
            title = c.get("document_title", "")
            section = " > ".join(c.get("section_path", []))
            text = c.get("text_raw") or c.get("text_normalized") or ""
            blocks.append(f"[{title} - {section}]\n{text}")
        return "\n\n---\n\n".join(blocks)


def main():
    parser = argparse.ArgumentParser(description="RAG 인덱스 검색 테스트")
    parser.add_argument("--index-dir", default="km_rag/index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    retriever = RAGRetriever(args.index_dir)
    results = retriever.retrieve(args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        section = " > ".join(r.get("section_path", []))
        print(f"\n[{i}] score={r['_score']:.4f}  {r.get('document_title')} > {section}")
        print((r.get("text_raw", "") or "")[:300])


if __name__ == "__main__":
    main()
