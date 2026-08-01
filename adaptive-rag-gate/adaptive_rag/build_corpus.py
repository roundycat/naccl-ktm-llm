# -*- coding: utf-8 -*-
"""어댑티브 RAG 코퍼스 빌더.

우리 관련성 분류기가 (질문, 문서) 관련성을 판정할 '문서' 후보 집합을 만든다.
소스는 GraphRAG 코퍼스(분류기가 학습한 바로 그 지식):
  - 처방_rag_chunks.jsonl   (처방명 + 계통 + 주치 + 구성)
  - 한의학용어_rag_chunks.jsonl (용어 + 정의)
  - 해설_rag_chunks.jsonl   (기출 해설)

각 문서의 doc_text 는 분류기 학습 때의 포맷을 최대한 그대로 재현한다
  - 처방:  "{이름} (계통: {계통}). 주치: {주치}"
  - 용어:  "{용어}: {정의}"
  - 해설:  "{해설}"[:400]

출력: adaptive_rag/corpus.jsonl  — {id, type, name, doc_text}
"""
from __future__ import annotations
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "graphrag", "data")
OUT = os.path.join(HERE, "corpus.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _line_after(text: str, key: str) -> str:
    """'key: 값' 형태 줄에서 값만 추출(없으면 '')."""
    m = re.search(rf"{key}\s*[:：]\s*(.+)", text)
    return m.group(1).strip() if m else ""


def build():
    docs = []

    # ── 처방: "이름 (계통: 계통). 주치: 주치" ──
    p = os.path.join(DATA, "처방_rag_chunks.jsonl")
    for c in load_jsonl(p):
        text = c.get("text", "")
        meta = c.get("metadata", {}) or {}
        # 첫 줄: "가감내고환(加減內固丸) [간계내과]"
        head = text.splitlines()[0] if text else ""
        name = meta.get("처방명") or head.split("(")[0].split("[")[0].strip()
        gyetong = meta.get("계통") or ""
        if not gyetong:
            mt = re.search(r"\[([^\]]+)\]", head)
            gyetong = mt.group(1) if mt else ""
        juchi = _line_after(text, "주치")
        doc_text = f"{name} (계통: {gyetong}). 주치: {juchi}".strip()
        docs.append({"id": c.get("id"), "type": "처방", "name": name, "doc_text": doc_text})

    # ── 용어: "용어: 정의" ──
    p = os.path.join(DATA, "한의학용어_rag_chunks.jsonl")
    for c in load_jsonl(p):
        text = c.get("text", "")
        meta = c.get("metadata", {}) or {}
        term = meta.get("term") or (text.splitlines()[0].split("(")[0].split("[")[0].strip() if text else "")
        jeongui = _line_after(text, "정의") or text
        doc_text = f"{term}: {jeongui}".strip()
        docs.append({"id": c.get("id"), "type": "용어", "name": term, "doc_text": doc_text})

    # ── 해설: 본문[:400] ──
    p = os.path.join(DATA, "해설_rag_chunks.jsonl")
    for c in load_jsonl(p):
        body = (c.get("해설") or c.get("text") or "").strip()
        if not body:
            continue
        did = c.get("id") or f"hae-{c.get('idx')}"
        docs.append({"id": did, "type": "해설", "name": f"해설#{c.get('idx')}",
                     "doc_text": body[:400]})

    with open(OUT, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    by_type = {}
    for d in docs:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    print(f"코퍼스 {len(docs)}문서 → {OUT}")
    print(f"  유형별: {by_type}")
    # 샘플 출력
    for t in ("처방", "용어", "해설"):
        ex = next((d for d in docs if d["type"] == t), None)
        if ex:
            print(f"  [{t}] {ex['doc_text'][:90]}...")


if __name__ == "__main__":
    build()
