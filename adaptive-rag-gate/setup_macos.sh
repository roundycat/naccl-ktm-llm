#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# KTM-LLM — macOS 개발/연구 환경 셋업 스크립트
#   원본 프로젝트는 Windows(D: 드라이브)+Ollama 환경이었다. 이 스크립트는
#   macOS(Apple Silicon)에서 동일 연구를 이어가기 위한 셋업을 자동화한다.
#
#   사용법:  bash setup_macos.sh           # 기본(용어 RAG 트랙) 셋업
#            bash setup_macos.sh --graphrag # + GraphRAG 무거운 의존성까지
#
#   전제:    uv, python3.12(또는 3.13), Homebrew 가 설치돼 있어야 한다.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'; NC=$'\033[0m'
info(){ echo "${GREEN}▶ $*${NC}"; }
warn(){ echo "${YELLOW}⚠ $*${NC}"; }
err(){  echo "${RED}✖ $*${NC}"; }

WITH_GRAPHRAG=0
[[ "${1:-}" == "--graphrag" ]] && WITH_GRAPHRAG=1

# 1) 파이썬 가상환경 ----------------------------------------------------------
if [[ ! -d .venv ]]; then
  info "가상환경(.venv) 생성 — uv, Python 3.12"
  uv venv --python 3.12 .venv
else
  info ".venv 이미 존재 — 재사용"
fi
PY="$ROOT/.venv/bin/python"

# 2) 기본 의존성(용어 RAG 트랙: training/) -----------------------------------
info "기본 의존성 설치 (openai, dotenv, tqdm, requests, bs4)"
uv pip install --python .venv -r requirements.txt

# 3) .env ---------------------------------------------------------------------
if [[ ! -f .env ]]; then
  info ".env 생성 (.env.example 복사) — 값은 직접 채우세요"
  cp .env.example .env
else
  info ".env 이미 존재 — 유지"
fi

# 4) Ollama (로컬 LLM 추론 런타임) --------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama 미설치. 로컬 모델 평가에는 필수입니다."
  echo "    설치:  brew install ollama"
  echo "    구동:  ollama serve   (별도 터미널, 또는 macOS 앱 실행)"
else
  info "Ollama 감지됨: $(ollama --version 2>/dev/null | head -1)"
fi

# 5) 디스크 여유 점검(원본 프로젝트의 고질 이슈) ------------------------------
FREE_GB=$(df -g "$ROOT" | awk 'NR==2{print $4}')
if [[ "${FREE_GB:-0}" -lt 20 ]]; then
  warn "디스크 여유 ${FREE_GB}GB — 부족합니다."
  echo "    Ollama 모델은 개당 4~6GB(qwen2.5:7b≈4.7GB, exaone3.5:7.8b≈4.8GB)."
  echo "    필요 시 OLLAMA_MODELS 를 외장/여유 볼륨으로 지정하세요:"
  echo "      export OLLAMA_MODELS=/Volumes/<여유디스크>/ollama-models"
else
  info "디스크 여유 ${FREE_GB}GB — OK"
fi

# 6) (옵션) GraphRAG 무거운 스택 ----------------------------------------------
if [[ "$WITH_GRAPHRAG" == "1" ]]; then
  warn "GraphRAG 스택 설치 — torch/sentence-transformers 포함(수 GB, 디스크 확인)"
  uv pip install --python .venv -r graphrag/requirements_graphrag.txt
  echo "    원본 GraphRAG(bigse0u1/)는 추가로 Neo4j 가 필요합니다:"
  echo "      brew install neo4j && neo4j start   (또는 docker run neo4j)"
  echo "      export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PW=<pw>"
fi

echo
info "셋업 완료. 다음 단계는 CONTINUE_HERE.md 참조."
echo "  즉시 실행 가능(Ollama 불요):  $PY training/report.py"
echo "  로컬 평가(Ollama 필요):        $PY training/evaluate.py --all --limit 10"
