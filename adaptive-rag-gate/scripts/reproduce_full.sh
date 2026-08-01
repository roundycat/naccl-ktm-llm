#!/usr/bin/env bash
#
# results/predictions/ 의 108개 파일(9모델 x 4개년 x {base,rag,graph})을 처음부터
# 다시 만든다. GPU 필요, 모델당 수십 분 단위로 오래 걸림.
#
# ⚠️ 주의: 이 스크립트는 run_all_models.sh 의 서빙 패턴과 results/predictions/의
# 실제 파일명 규칙(RES_{tag}_{year}_{cond}.json)을 근거로 재구성한 것이다.
# 원래 이 108개를 만든 스크립트 자체는 저장소에 남아있지 않아(scripts/cluster/의
# chain*.sh 는 이 중 GraphRAG 일부만 다룸), 전체를 처음부터 실행해서 끝까지
# 검증하지는 못했다 — 실행 전 모델 1개·연도 1개로 먼저 테스트해볼 것을 권장한다.
#
# rag/graph 조건은 이미 계산되어 있는 주입 파일(results/gate/rag_injections_<year>.json,
# data/graphrag_ctx/graphrag_context_<year>.json)을 그대로 사용한다 — 즉 검색·게이트
# 자체를 다시 계산하지 않고, "이미 결정된 근거를 각 모델에 다시 주입해서 답을 다시
# 생성"하는 것만 재현한다. 주입 자체(관련도 분류기 추론)를 처음부터 재현하려면
# adaptive_rag/precompute_injections.py, adaptive_rag/graphrag_precompute.py 참고.
#
# 사용법:
#   cd adaptive-rag-gate && ./scripts/reproduce_full.sh

set -uo pipefail
cd "$(dirname "$0")/.."   # adaptive-rag-gate/ 로 이동

YEARS=(2022 2023 2024 2025)
STAGE=5
N_TRIALS=3
PORT=8000
API_BASE="http://localhost:${PORT}/v1"
PRED_DIR="results/predictions"
GATE_DIR="results/gate"
GRAPH_CTX_DIR="data/graphrag_ctx"
BENCH_DATA_DIR="../baseline/KTM_data"
CHAT_TPL_DIR="../baseline/chat_templates"

mkdir -p "$PRED_DIR"

# tag(=RES_ 파일명에 쓰인 이름) \t repo_id \t vllm 추가옵션
# repo_id·옵션은 ../baseline/models/{general,domain/*}.txt 에서 그대로 가져옴.
MODELS=(
  "qwen2_5_7b	Qwen/Qwen2.5-7B-Instruct	"
  "exaone3_5_7_8b	LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct	--trust-remote-code"
  "solar_10_7b	upstage/SOLAR-10.7B-Instruct-v1.0	"
  "mistral_7b	mistralai/Mistral-7B-Instruct-v0.3	"
  "huatuogpt-o1-7b	FreedomIntelligence/HuatuoGPT-o1-7B	"
  "biomistral-7b	BioMistral/BioMistral-7B	--chat-template ${CHAT_TPL_DIR}/generic.jinja"
  "medllama2-7b	llSourcell/medllama2_7b	--chat-template ${CHAT_TPL_DIR}/generic.jinja"
  "openbiollm-8b	aaditya/Llama3-OpenBioLLM-8B	--chat-template ${CHAT_TPL_DIR}/generic.jinja"
  "medgemma-4b	google/medgemma-4b-it	"
)

wait_for_server() {
  local pid="$1" deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    curl -sf "${API_BASE}/models" > /dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

for entry in "${MODELS[@]}"; do
  IFS=$'\t' read -r tag repo_id extra_args <<< "$entry"

  all_done=true
  for year in "${YEARS[@]}"; do
    for cond in base rag graph; do
      [[ -f "${PRED_DIR}/RES_${tag}_${year}_${cond}.json" ]] || all_done=false
    done
  done
  if [[ "$all_done" == true ]]; then
    echo "[$tag] 12개(4년x3조건) 이미 있음 — 건너뜀"
    continue
  fi

  echo "=================================================="
  echo "[$tag] vLLM 서빙 시작 ($repo_id)"
  echo "=================================================="
  # shellcheck disable=SC2086
  vllm serve "$repo_id" --port "$PORT" --served-model-name "$tag" $extra_args > "/tmp/${tag}_vllm.log" 2>&1 &
  vllm_pid=$!

  if ! wait_for_server "$vllm_pid"; then
    echo "[$tag] 서버 기동 실패 (로그: /tmp/${tag}_vllm.log) — 건너뜀"
    kill -9 "$vllm_pid" 2>/dev/null; wait "$vllm_pid" 2>/dev/null
    continue
  fi

  for year in "${YEARS[@]}"; do
    data_file="${BENCH_DATA_DIR}/${year}.json"
    [[ -f "$data_file" ]] || { echo "  [$year] $data_file 없음 — 건너뜀"; continue; }
    tcache="/tmp/${tag}_${year}_translation_cache.json"

    for cond in base rag graph; do
      out="${PRED_DIR}/RES_${tag}_${year}_${cond}.json"
      [[ -f "$out" ]] && { echo "  [$year/$cond] 이미 있음 — 건너뜀"; continue; }

      rag_args=()
      case "$cond" in
        base) : ;;
        rag)  rag_args=(--rag --injections-file "${GATE_DIR}/rag_injections_${year}.json") ;;
        graph) rag_args=(--rag --injections-file "${GRAPH_CTX_DIR}/graphrag_context_${year}.json") ;;
      esac

      echo "  [$year/$cond] 시작"
      python3 adaptive_rag/run_rag.py --model "openai/${tag}" --api-base "$API_BASE" --api-key sk-dummy \
        --data "$data_file" --stage "$STAGE" --n-trials "$N_TRIALS" \
        --translation-cache "$tcache" "${rag_args[@]}" --output "$out" \
        || echo "  [$year/$cond] 실패 (모델/데이터 확인 필요)"
    done
  done

  kill "$vllm_pid" 2>/dev/null; wait "$vllm_pid" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null
  sleep 5
done

echo "=================================================="
echo "완료. ./scripts/reproduce_metrics.sh 로 헤드라인 수치 재생성할 것."
echo "=================================================="
