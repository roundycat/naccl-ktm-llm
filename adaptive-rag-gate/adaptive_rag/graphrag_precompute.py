# -*- coding: utf-8 -*-
"""1,138문항용 GraphRAG 근거 사전계산 — bigse0u1/step3 로직을 Neo4j·Chroma 없이 이식.

원본 대응:
  name_index()/_symptom_nodes_and_texts()  -> 증상·변증 노드 색인 (동일)
  extract_seeds()                          -> 질문 텍스트 직접 매칭 (동일)
  GRAPH_Q (Cypher)                         -> 주치 엣지 역탐색 + 매칭 수 정렬 top6 + 구성 조회 (동일 의미)
  vector_retrieve()                        -> bge-m3 코사인 top-k (Chroma 대체, 동일 임베딩 모델)
  build_context()                          -> 동일 포맷

원본과 다른 점(문서화 대상):
  - 시드 추출에 LLM(extract_seeds_llm)을 쓰지 않고 직접 매칭을 쓴다.
    원본 docstring 이 LLM 추출을 "구어체 질문에 적합"으로 명시하는데, 국시 문항은
    교과서 문어체라 직접 매칭이 적합하고 결정론적이다(재현성 확보).

출력: graphrag_context_{year}.json  =  {문항ID: 근거텍스트}
사용: graphrag_precompute.py <kg_dir> <ktm_data_dir> <out_dir> [top_k]
"""
import json
import os
import re
import sys
from collections import defaultdict

KG = sys.argv[1]
DATA = sys.argv[2]
OUT = sys.argv[3]
TOPK = int(sys.argv[4]) if len(sys.argv) > 4 else 5
YEARS = ["2022", "2023", "2024", "2025"]
STOP_NAMES = {"한다"}          # 원본과 동일: '~한다' 종결어미 충돌 방지

os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ KG 적재
print("[1/5] KG 적재", flush=True)
nodes = {}
for line in open(f"{KG}/kg_all_nodes.jsonl", encoding="utf-8"):
    n = json.loads(line)
    nodes[n["id"]] = n
edges = []
for line in open(f"{KG}/kg_all_edges.jsonl", encoding="utf-8"):
    edges.append(json.loads(line))
print(f"   노드 {len(nodes):,} / 엣지 {len(edges):,}", flush=True)
from collections import Counter
print(f"   노드 유형: {dict(Counter(n['type'] for n in nodes.values()))}", flush=True)
print(f"   엣지 유형: {dict(Counter(e.get('type') for e in edges))}", flush=True)

# 주치: 처방 -> 증상 / 구성: 처방 -> 약재
주치 = defaultdict(set)          # rx_id -> {symptom_id}
구성 = defaultdict(list)         # rx_id -> [herb name]
주치_rev = defaultdict(set)      # symptom_id -> {rx_id}
for e in edges:
    t = e.get("type")
    s, o = e.get("src") or e.get("source"), e.get("dst") or e.get("target")
    if s is None or o is None:
        continue
    if t == "주치":
        주치[s].add(o); 주치_rev[o].add(s)
    elif t == "구성":
        구성[s].append(nodes.get(o, {}).get("name_ko", ""))

# 자기검증: 주치 인접리스트가 비어 있으면 스키마 오독이므로 즉시 중단
if not 주치_rev:
    raise SystemExit("!! 주치 엣지 파싱 실패 — 엣지 키 이름 확인 필요")
print(f"   주치 역인덱스 {len(주치_rev):,}개 증상 / 구성 {len(구성):,}개 처방", flush=True)

# 증상·변증 색인 (원본 name_index 와 동일 조건)
sym_index = [(n["name_ko"], n["id"]) for n in nodes.values()
             if n.get("type") in ("증상", "변증") and len(n.get("name_ko", "")) >= 2
             and n.get("name_ko") not in STOP_NAMES]
sym_index.sort(key=lambda x: -len(x[0]))     # 긴 이름 우선 매칭
print(f"   증상·변증 색인 {len(sym_index):,}개", flush=True)


def extract_seeds(text):
    """질문 텍스트에 등장하는 증상·변증 노드 id 목록 (원본 extract_seeds 와 동일한 직접 매칭)."""
    seeds, names, seen = [], [], set()
    for nm, nid in sym_index:
        if nid in seen:
            continue
        if nm in text:
            seeds.append(nid); names.append(nm); seen.add(nid)
    return seeds, names


def graph_retrieve(seeds, limit=6):
    """GRAPH_Q 이식: 시드 증상을 주치로 갖는 처방을 매칭 수로 정렬해 top-N."""
    if not seeds:
        return []
    sset = set(seeds)
    score = {}
    for sid in sset:
        for rx in 주치_rev.get(sid, ()):
            score[rx] = score.get(rx, 0) + 1
    top = sorted(score.items(), key=lambda kv: -kv[1])[:limit]
    rows = []
    for rx, sc in top:
        n = nodes.get(rx, {})
        matched = [nodes.get(s, {}).get("name_ko", "") for s in (주치.get(rx, set()) & sset)]
        rows.append({"처방": n.get("name_ko", ""), "한자": n.get("name_hanja", ""),
                     "계통": n.get("계통", "") or "", "matched": [m for m in matched if m],
                     "약재": [h for h in 구성.get(rx, []) if h][:8], "score": sc})
    return rows


# ------------------------------------------------------------------ 벡터 검색 (Chroma 대체)
print("[2/5] 처방 청크 적재 + 임베딩", flush=True)
chunk_docs = []
for fn in ("처방_rag_chunks_clinical.jsonl", "처방_rag_chunks.jsonl"):
    p = f"{KG}/{fn}"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            o = json.loads(line)
            d = o.get("text") or o.get("document") or o.get("content") or ""
            if d:
                chunk_docs.append(d)
        break
print(f"   청크 {len(chunk_docs):,}개", flush=True)

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # 임베딩은 CPU 로만
import numpy as np
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("BAAI/bge-m3", device="cpu")   # vLLM 이 GPU 를 점유하므로 CPU 고정
CACHE = f"{OUT}/chunk_emb.npy"
if os.path.exists(CACHE):
    M = np.load(CACHE)
    print(f"   임베딩 캐시 재사용 {M.shape}", flush=True)
else:
    M = emb.encode(chunk_docs, normalize_embeddings=True, batch_size=32,
                   show_progress_bar=False).astype("float32")
    np.save(CACHE, M)
    print(f"   임베딩 완료 {M.shape}", flush=True)


def vector_retrieve(question, k=TOPK):
    q = emb.encode([question], normalize_embeddings=True).astype("float32")[0]
    sims = M @ q
    idx = np.argsort(-sims)[:k]
    return [chunk_docs[i] for i in idx]


def build_context(graph_rows, chunks):
    """원본 build_context 와 동일 포맷."""
    lines = []
    if graph_rows:
        lines.append("[그래프 근거: 증상에 부합하는 처방]")
        for g in graph_rows:
            lines.append(f"- {g['처방']}({g['한자']}) [{g['계통']}] "
                         f"| 부합 증상: {', '.join(g['matched'])} "
                         f"| 구성: {', '.join(g['약재'])}")
    if chunks:
        lines.append("\n[본문 근거]")
        for doc in chunks:
            lines.append(f"- {doc}")
    return "\n".join(lines)


# ------------------------------------------------------------------ 문항별 사전계산
print("[3/5] 문항별 근거 생성", flush=True)
stats = {"total": 0, "with_graph": 0, "seeds": 0}
for y in YEARS:
    qs = json.load(open(f"{DATA}/{y}.json", encoding="utf-8"))
    out = {}
    ng = 0
    for q in qs:
        text = q["question_kr"] + " " + " ".join(q.get("choices_kr") or [])
        seeds, names = extract_seeds(text)
        rows = graph_retrieve(seeds)
        chunks = vector_retrieve(q["question_kr"])
        out[str(q["id"])] = build_context(rows, chunks)
        stats["total"] += 1; stats["seeds"] += len(seeds)
        if rows:
            ng += 1; stats["with_graph"] += 1
    json.dump(out, open(f"{OUT}/graphrag_context_{y}.json", "w"), ensure_ascii=False)
    print(f"   {y}: {len(qs)}문항, 그래프근거 있음 {ng} ({100*ng/len(qs):.1f}%)", flush=True)

print(f"[4/5] 합계: {stats['total']}문항, 그래프근거 {stats['with_graph']} "
      f"({100*stats['with_graph']/stats['total']:.1f}%), 평균 시드 {stats['seeds']/stats['total']:.2f}개", flush=True)
print(f"[5/5] 저장 위치: {OUT}/graphrag_context_*.json", flush=True)
