"""
rag_index.py
============
km_rag/*.jsonl 에 있는 RAG 청크들을 로컬 임베딩 모델로 임베딩해서
FAISS 인덱스로 만든다. (한 번만 실행하면 됨. 청크 내용이 바뀔 때만 재실행)

설치:
    pip install sentence-transformers faiss-cpu --break-system-packages

실행:
    python rag_index.py
    python rag_index.py --rag-dir km_rag --output-dir km_rag/index --model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def load_chunks(rag_dir: str) -> list[dict]:
    chunks = []
    for path in sorted(glob.glob(os.path.join(rag_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
    return chunks


def build_index(rag_dir: str, output_dir: str, model_name: str, batch_size: int = 64) -> None:
    from sentence_transformers import SentenceTransformer
    import faiss

    chunks = load_chunks(rag_dir)
    if not chunks:
        raise RuntimeError(f"{rag_dir} 안에서 .jsonl 청크를 찾지 못했습니다.")
    print(f"청크 {len(chunks)}개 로드됨 ({rag_dir})")

    texts = [
        c.get("embedding_text") or c.get("text_raw") or c.get("text_normalized") or ""
        for c in chunks
    ]

    print(f"임베딩 모델 로드 중: {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # 정규화해두면 내적(IP) = 코사인 유사도
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
    with open(os.path.join(output_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"model_name": model_name, "n_chunks": len(chunks), "dim": dim},
            f, ensure_ascii=False, indent=2,
        )

    print(f"인덱스 저장 완료: {output_dir} (청크 {len(chunks)}개, 차원 {dim})")


def main():
    parser = argparse.ArgumentParser(description="km_rag 청크를 임베딩해서 FAISS 인덱스 생성")
    parser.add_argument("--rag-dir", default="km_rag", help="RAG jsonl 청크들이 있는 디렉토리")
    parser.add_argument("--output-dir", default="km_rag/index", help="인덱스 저장 위치")
    parser.add_argument("--model", default="BAAI/bge-m3", help="sentence-transformers 임베딩 모델명")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_index(args.rag_dir, args.output_dir, args.model, args.batch_size)


if __name__ == "__main__":
    main()
