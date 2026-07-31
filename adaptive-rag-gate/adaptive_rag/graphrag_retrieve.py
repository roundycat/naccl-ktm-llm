# -*- coding: utf-8 -*-
"""GraphRAG 검색기 — 지식그래프 순회로 질문별 후보 처방 생성.

기존 graphrag/build_evidence.py 의 그래프 로직을 재사용한다:
  - 보기(선택지)에 등장하는 처방 → KG에서 주치·구성·계통 조회
  - 질문에 등장하는 증상 → 그 증상을 주치로 갖는 처방 후보 랭킹(증상→처방 그래프)

반환하는 후보 처방의 doc_text 는 **분류기 학습 때와 동일 포맷**
("{처방명} (계통: {계통}). 주치: {주치}") 이라, 우리 BERT 분류기가 그대로 게이트할 수 있다.

이게 밀집(BGE) 검색과의 유일한 차이: 후보를 '그래프'로 뽑는다. 게이트·주입은 동일.
"""
from __future__ import annotations
import json
import os


class GraphRAGRetriever:
    def __init__(self, data_dir: str):
        self.rx_by_name: dict[str, dict] = {}
        self.sym_to_rx: dict[str, set] = {}
        # 처방 청크: 처방명 → 메타(주치증상·구성약재·계통…). 다판본은 리스트 필드 union.
        for c in self._load(os.path.join(data_dir, "처방_rag_chunks.jsonl")):
            m = c["metadata"]; nm = m["처방명"]
            if nm not in self.rx_by_name:
                self.rx_by_name[nm] = {**m, "주치증상": list(m.get("주치증상", [])),
                                       "구성약재": list(m.get("구성약재", []))}
            else:
                dst = self.rx_by_name[nm]
                for f in ("주치증상", "구성약재"):
                    have = set(dst.get(f, []))
                    for v in m.get(f, []):
                        if v not in have:
                            have.add(v); dst.setdefault(f, []).append(v)
            for s in m.get("주치증상", []):
                self.sym_to_rx.setdefault(s, set()).add(nm)
        # 노드: 증상/변증 이름(시드 추출용)
        self.sym_names = []
        for n in self._load(os.path.join(data_dir, "kg_all_nodes.jsonl")):
            if n.get("type") in ("증상", "변증") and len(n.get("name_ko", "")) >= 2:
                self.sym_names.append(n["name_ko"])
        self.sym_names = sorted(set(self.sym_names), key=len, reverse=True)  # 긴 이름 우선
        self.STOP = {"한다"}

    @staticmethod
    def _load(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _doc_text(self, nm: str, m: dict) -> str:
        gyetong = m.get("계통", "")
        juchi = ", ".join(m.get("주치증상", [])[:8])
        return f"{nm} (계통: {gyetong}). 주치: {juchi}"

    @staticmethod
    def _parse_rx_names(options):
        names = []
        for o in options:
            nm = o.split("(")[0].strip()
            names.extend([x.strip() for x in nm.split(" 합 ")] if " 합 " in nm else [nm])
        return names

    def candidates(self, question: str, options: list[str], topn: int = 6) -> list[dict]:
        """그래프에서 후보 처방 뽑기 → [{name, doc_text, source}]. 중복 제거."""
        cands: dict[str, dict] = {}
        # ① 보기 처방 (선택지에 등장한 처방을 KG에서 조회)
        for nm in self._parse_rx_names(options):
            m = self.rx_by_name.get(nm)
            if m:
                cands[nm] = {"name": nm, "doc_text": self._doc_text(nm, m), "source": "보기"}
        # ② 증상→처방 (질문 증상으로 주치 처방 랭킹)
        seeds = [s for s in self.sym_names if s not in self.STOP and s in question]
        score: dict[str, int] = {}
        for s in seeds:
            for rx in self.sym_to_rx.get(s, ()):
                score[rx] = score.get(rx, 0) + 1
        for rx, _ in sorted(score.items(), key=lambda kv: -kv[1])[:topn]:
            if rx not in cands:
                cands[rx] = {"name": rx, "doc_text": self._doc_text(rx, self.rx_by_name[rx]),
                             "source": "증상"}
        return list(cands.values())
