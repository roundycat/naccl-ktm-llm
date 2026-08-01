#!/usr/bin/env bash
#
# 설치/경로/vLLM 서빙이 제대로 됐는지 문항 5개로 빠르게 확인한다 (GPU 필요, 전체
# 108개 예측 재생성인 reproduce_full.sh와 달리 몇 분 안에 끝나야 함).
#
# 사전 준비: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --served-model-name qwen2.5-7b
#           (다른 모델로 스모크 테스트하고 싶으면 아래 MODEL/REPO_ID를 바꿀 것)
#
# 사용법:
#   cd adaptive-rag-gate && ./scripts/smoke_test.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # adaptive-rag-gate/ 로 이동

MODEL="qwen2.5-7b"
API_BASE="http://localhost:8000/v1"
DATA="../baseline/KTM_data/2022.json"
LIMIT=5
TMP_DIR="$(mktemp -d)"

if ! curl -sf "${API_BASE}/models" > /dev/null 2>&1; then
  echo "에러: ${API_BASE} 에서 vLLM 서버가 응답하지 않습니다."
  echo "먼저 실행: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --served-model-name ${MODEL}"
  exit 1
fi

echo "=== 1) baseline (RAG 없음), 문항 ${LIMIT}개 ==="
python3 adaptive_rag/run_rag.py --model "openai/${MODEL}" --api-base "$API_BASE" --api-key sk-dummy \
  --data "$DATA" --stage 4 --limit "$LIMIT" --output "${TMP_DIR}/smoke_base.json"

echo ""
echo "=== 2) 어댑티브 RAG (사전계산 주입 사용), 문항 ${LIMIT}개 ==="
python3 adaptive_rag/run_rag.py --model "openai/${MODEL}" --api-base "$API_BASE" --api-key sk-dummy \
  --data "$DATA" --stage 4 --limit "$LIMIT" --rag \
  --injections-file results/gate/rag_injections_2022.json \
  --output "${TMP_DIR}/smoke_rag.json"

echo ""
echo "=== 3) 저장된 결과만으로 헤드라인 수치 재생성 (GPU 불필요) ==="
./scripts/reproduce_metrics.sh > /dev/null && echo "OK — reproduce_metrics.sh 정상 동작"

echo ""
echo "스모크 테스트 통과. 임시 출력: ${TMP_DIR}"
