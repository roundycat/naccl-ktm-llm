# tkm_pipeline.py 에 --rag-context-file 추가.
# 사전계산된 {문항ID: 근거텍스트} 를 rag_context 로 주입한다.
# 기존 FAISS 경로(--rag / rag_retriever)는 손대지 않으므로 km_rag 실행에 영향 없음.
# 목적: GraphRAG 근거를 km_rag·plain 과 '완전히 동일한 프롬프팅'으로 평가하기 위함.
import re
import sys

P = "/workspace/bench/ktm-llm-benchmark/tkm_pipeline.py"
s = open(P, encoding="utf-8").read()
if "_RAG_CONTEXT_MAP" in s:
    print("[patch] 이미 적용됨"); sys.exit(0)

# 1) 전역 맵 + evaluate 의 _build 훅
old_build = """    def _build(q: Question) -> tuple[str, str]:
        rag_context = None
        if rag_retriever is not None:
            chunks = rag_retriever.retrieve(q.question_kr, top_k=rag_top_k)
            rag_context = RAGRetriever.format_context(chunks) if chunks else None"""
new_build = """    def _build(q: Question) -> tuple[str, str]:
        rag_context = None
        if _RAG_CONTEXT_MAP:                       # 사전계산 근거(GraphRAG 등) 우선
            rag_context = _RAG_CONTEXT_MAP.get(str(q.id)) or None
        elif rag_retriever is not None:
            chunks = rag_retriever.retrieve(q.question_kr, top_k=rag_top_k)
            rag_context = RAGRetriever.format_context(chunks) if chunks else None"""
if old_build not in s:
    print("[patch] !! _build 블록 불일치"); sys.exit(1)
s = s.replace(old_build, new_build)

# 2) 전역 선언 (evaluate 정의 앞)
anchor = "\ndef evaluate("
if anchor not in s:
    print("[patch] !! evaluate 정의 못 찾음"); sys.exit(1)
s = s.replace(anchor, "\n_RAG_CONTEXT_MAP: dict = {}   # --rag-context-file 로 채워짐\n\n\ndef evaluate(", 1)

# 3) CLI 인자
old_arg = '        "--rag-top-k", type=int, default=3,'
if old_arg not in s:
    print("[patch] !! --rag-top-k 인자 못 찾음"); sys.exit(1)
s = s.replace(old_arg,
              '        "--rag-context-file", default=None,\n'
              '        help="사전계산된 {문항ID: 근거} JSON. 지정 시 FAISS 검색 대신 이 값을 주입",\n'
              '    )\n'
              '    parser.add_argument(\n'
              '        "--rag-top-k", type=int, default=3,', 1)

# 4) main 에서 로드 — args 파싱 직후 삽입
m = re.search(r"\n(\s*)args = parser\.parse_args\(\)\n", s)
if not m:
    print("[patch] !! args 파싱부 못 찾음"); sys.exit(1)
ind = m.group(1)
ins = (f"\n{ind}args = parser.parse_args()\n"
       f"{ind}if getattr(args, 'rag_context_file', None):\n"
       f"{ind}    import json as _json\n"
       f"{ind}    globals()['_RAG_CONTEXT_MAP'] = _json.load(open(args.rag_context_file, encoding='utf-8'))\n"
       f"{ind}    _n = sum(1 for v in _RAG_CONTEXT_MAP.values() if v)\n"
       f"{ind}    print(f'[RAG] 사전계산 근거 로드 {{args.rag_context_file}}: "
       f"{{_n}}/{{len(_RAG_CONTEXT_MAP)}} 주입', flush=True)\n")
s = s[:m.start()] + ins + s[m.end():]

open(P, "w", encoding="utf-8").write(s)
import ast
ast.parse(open(P, encoding="utf-8").read())
print("[patch] --rag-context-file 추가 + 문법 검증 통과")
for k in ("_RAG_CONTEXT_MAP", "--rag-context-file", "사전계산 근거 로드"):
    print(f"   마커 '{k}': {s.count(k)}회")
