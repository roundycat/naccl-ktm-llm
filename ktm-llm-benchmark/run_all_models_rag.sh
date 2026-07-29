#!/usr/bin/env bash
#
# run_all_models.sh와 동일한 구조지만, tkm_pipeline.py에 --rag를 추가해서
# km_rag(표준임상진료지침 기반 RAG) 검색 결과를 프롬프트에 넣고 벤치마크한다.
# 결과는 ktm_results/가 아니라 ktm_results_rag/에 따로 저장되어 RAG 없는
# 결과(stage5)와 나란히 비교할 수 있다.
#
# 비교 축:
#   models/general.txt              국가별 범용 LLM
#   models/domain/tcm.txt           중국의학(TCM) 도메인 특화 LLM
#   models/domain/western_med.txt   서양의학 도메인 특화 LLM
#
# 사전 준비:
#   pip install vllm litellm sentence-transformers faiss-cpu --break-system-packages
#   export HF_TOKEN="hf_..."   # 게이트된 모델(Llama, Gemma 계열 등) 접근용
#   python3 rag_index.py       # km_rag/index/ 가 먼저 만들어져 있어야 함
#
# 사용법:
#   chmod +x run_all_models_rag.sh
#   ./run_all_models_rag.sh

set -uo pipefail

RAG_INDEX_DIR="km_rag/index"
if [[ ! -f "${RAG_INDEX_DIR}/index.faiss" ]]; then
  echo "에러: ${RAG_INDEX_DIR}/index.faiss 가 없습니다. 먼저 'python3 rag_index.py'를 실행하세요."
  exit 1
fi

# category_label:models_file 쌍의 목록. category_label은 ktm_results_rag/, logs_rag/ 아래 하위 경로로 쓰인다.
CATEGORIES=(
  "general:models/general.txt"
  "domain/tcm:models/domain/tcm.txt"
  "domain/western_med:models/domain/western_med.txt"
)

# 연도별 문항 데이터 (KTM_data/<year>.json). 모델 하나를 서빙해둔 상태에서
# 연도를 바꿔가며 여러 번 벤치마크한다 (모델을 매번 다시 받는 낭비를 피하기 위함).
YEARS=(2022 2023 2024 2025)

STAGE=5
N_TRIALS=3
RAG_TOP_K=3
MAX_WORKERS=24   # 문항 간 병렬화까지 적용됨 (tkm_pipeline.py 참고). vLLM이 admission을
                 # 알아서 관리하므로 클라이언트 쪽 동시 요청 수를 넉넉히 높여도 안전함
PORT=8000
API_BASE="http://localhost:${PORT}/v1"
API_KEY="sk-dummy"
READY_TIMEOUT=900          # vLLM 서버 기동 대기 최대 시간(초)
GPU_COOLDOWN=10            # 모델 종료 후 GPU 메모리 정리 대기 시간(초)
CLEANUP_HF_CACHE=true      # 모델 하나 끝날 때마다 그 모델의 HuggingFace 캐시(가중치)를 삭제할지
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"

RESULT_ROOT="ktm_results_rag"
LOG_ROOT="logs_rag"

SUMMARY_FILE="${RESULT_ROOT}/_run_summary.tsv"
mkdir -p "$RESULT_ROOT"
echo -e "category\tmodel\tyear\tstatus\tstarted_at\tfinished_at" > "$SUMMARY_FILE"

wait_for_server() {
  local vllm_pid="$1"
  local deadline=$((SECONDS + READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "  vLLM 프로세스가 이미 종료됨 (인증 실패 등) — 타임아웃까지 안 기다리고 바로 다음으로 넘어감"
      return 1
    fi
    sleep 5
  done
  return 1
}

# 디스크 공간 확보용: 모델 하나가 끝나면 그 모델의 HuggingFace 캐시(가중치, 보통 수십 GB)를 지운다.
cleanup_hf_cache() {
  local repo_id="$1"
  [[ "$CLEANUP_HF_CACHE" != true ]] && return
  local cache_name="models--${repo_id//\//--}"
  local cache_path="${HF_CACHE_DIR}/${cache_name}"
  if [[ -d "$cache_path" ]]; then
    local freed
    freed=$(du -sh "$cache_path" 2>/dev/null | cut -f1)
    rm -rf "$cache_path"
    echo "  캐시 삭제: $cache_path (${freed:-?} 확보)"
  fi
}

# vLLM은 APIServer 프로세스와 별개로 EngineCore라는 자식 프로세스를 띄우는데,
# 부모(vllm_pid)만 kill하면 이 자식이 고아로 남아 GPU 메모리를 계속 붙잡고
# 있는 경우가 있다 (실제로 겪음). 그래서 정상 종료 시도 후에 vllm 관련
# 프로세스를 한 번 더 싹 정리(pkill)해서 다음 모델이 GPU를 확실히 쓸 수 있게 한다.
kill_vllm_tree() {
  local vllm_pid="$1"
  kill "$vllm_pid" 2>/dev/null
  wait "$vllm_pid" 2>/dev/null
  sleep 2
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -f "vllm serve" 2>/dev/null
  sleep 1
}

# Ctrl+C(INT)나 예기치 않은 종료(TERM/EXIT) 시에도 지금 처리 중이던 모델의
# vLLM 프로세스와 (다운로드 도중이었을 수도 있는) 캐시를 정리한다.
CURRENT_VLLM_PID=""
CURRENT_REPO_ID=""

cleanup_on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM   # 정리 중 또 시그널 걸려서 재진입하는 것 방지

  if [[ -n "$CURRENT_VLLM_PID" ]] && kill -0 "$CURRENT_VLLM_PID" 2>/dev/null; then
    echo ""
    echo "[중단 감지] vLLM 프로세스(PID $CURRENT_VLLM_PID) 종료 중..."
    kill_vllm_tree "$CURRENT_VLLM_PID"
  fi
  if [[ -n "$CURRENT_REPO_ID" ]]; then
    echo "[중단 감지] 진행 중이던 모델 캐시 정리: $CURRENT_REPO_ID"
    cleanup_hf_cache "$CURRENT_REPO_ID"
  fi

  exit "$exit_code"
}
trap cleanup_on_exit EXIT INT TERM

run_category() {
  local category="$1" models_file="$2"
  local log_dir="${LOG_ROOT}/${category}"
  mkdir -p "$log_dir"

  if [[ ! -f "$models_file" ]]; then
    echo "[$category] $models_file 없음 — 건너뜀"
    return
  fi

  while IFS=$'\t' read -r name repo_id extra_args; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue   # 빈 줄/주석 건너뛰기

    if [[ "$repo_id" == "???" ]]; then
      echo "[$category/$name] repo id가 아직 '???'로 비어있음 — $models_file 을 채운 뒤 다시 실행하세요. 건너뜀."
      for year in "${YEARS[@]}"; do
        echo -e "${category}\t${name}\t${year}\tSKIPPED_NO_REPO_ID\t-\t-" >> "$SUMMARY_FILE"
      done
      continue
    fi

    # 이미 4개 연도 결과가 전부 있으면 서버조차 띄우지 않고 통째로 건너뛴다.
    all_years_done=true
    for year in "${YEARS[@]}"; do
      [[ -f "${RESULT_ROOT}/${category}/${year}/${name}.json" ]] || { all_years_done=false; break; }
    done
    if [[ "$all_years_done" == true ]]; then
      echo "[$category/$name] 이미 4개 연도 결과 다 있음 — 건너뜀"
      for year in "${YEARS[@]}"; do
        echo -e "${category}\t${name}\t${year}\tALREADY_DONE\t-\t-" >> "$SUMMARY_FILE"
      done
      continue
    fi

    echo "=================================================="
    echo "[$category/$name] 시작 ($repo_id) [RAG 켜짐, top_k=${RAG_TOP_K}]"
    echo "=================================================="

    vllm_log="$log_dir/${name}_vllm.log"

    # 1) vLLM 서버 기동 (백그라운드, 모델당 한 번만 — 연도별로 다시 띄우지 않음)
    # shellcheck disable=SC2086
    vllm serve "$repo_id" \
      --port "$PORT" \
      --served-model-name "$name" \
      $extra_args \
      > "$vllm_log" 2>&1 &
    vllm_pid=$!
    CURRENT_VLLM_PID="$vllm_pid"
    CURRENT_REPO_ID="$repo_id"

    # 2) 서버 준비될 때까지 대기
    if ! wait_for_server "$vllm_pid"; then
      echo "[$category/$name] vLLM 서버가 ${READY_TIMEOUT}초 안에 기동되지 않음 — 건너뜀 (로그: $vllm_log)"
      kill_vllm_tree "$vllm_pid"
      cleanup_hf_cache "$repo_id"
      CURRENT_VLLM_PID=""
      CURRENT_REPO_ID=""
      for year in "${YEARS[@]}"; do
        echo -e "${category}\t${name}\t${year}\tSERVER_TIMEOUT\t-\t-" >> "$SUMMARY_FILE"
      done
      continue
    fi
    echo "[$category/$name] 서버 준비 완료, 연도별 벤치마크 시작"

    # 3) 서버를 띄워둔 채로 연도를 바꿔가며 반복 실행
    for year in "${YEARS[@]}"; do
      data_file="KTM_data/${year}.json"
      if [[ ! -f "$data_file" ]]; then
        echo "  [$year] $data_file 없음 — 건너뜀"
        echo -e "${category}\t${name}\t${year}\tSKIPPED_NO_DATA\t-\t-" >> "$SUMMARY_FILE"
        continue
      fi

      result_dir="${RESULT_ROOT}/${category}/${year}"
      mkdir -p "$result_dir"
      run_log="$log_dir/${name}_${year}_run.log"
      output_json="$result_dir/${name}.json"

      if [[ -f "$output_json" ]]; then
        echo "  [$year] 이미 결과 있음 — 건너뜀 ($output_json)"
        echo -e "${category}\t${name}\t${year}\tALREADY_DONE\t-\t-" >> "$SUMMARY_FILE"
        continue
      fi

      started_at=$(date +"%Y-%m-%d %H:%M:%S")

      echo "  [$year] 시작"
      python3 tkm_pipeline.py \
        --model "openai/${name}" \
        --api-base "$API_BASE" \
        --api-key "$API_KEY" \
        --data "$data_file" \
        --stage "$STAGE" \
        --n-trials "$N_TRIALS" \
        --max-workers "$MAX_WORKERS" \
        --rag \
        --rag-index-dir "$RAG_INDEX_DIR" \
        --rag-top-k "$RAG_TOP_K" \
        --output "$output_json" \
        > "$run_log" 2>&1
      run_status=$?
      finished_at=$(date +"%Y-%m-%d %H:%M:%S")

      if [[ $run_status -eq 0 ]]; then
        echo "  [$year] 완료 -> $output_json"
        echo -e "${category}\t${name}\t${year}\tOK\t${started_at}\t${finished_at}" >> "$SUMMARY_FILE"
      else
        echo "  [$year] 벤치마크 실행 실패 (로그: $run_log)"
        echo -e "${category}\t${name}\t${year}\tRUN_FAILED\t${started_at}\t${finished_at}" >> "$SUMMARY_FILE"
      fi
    done

    # 4) 모든 연도 끝난 뒤 서버 종료 + GPU 메모리 정리 대기 + 다음 모델을 위한 디스크 공간 확보
    kill_vllm_tree "$vllm_pid"
    sleep "$GPU_COOLDOWN"
    cleanup_hf_cache "$repo_id"
    CURRENT_VLLM_PID=""
    CURRENT_REPO_ID=""

  done < "$models_file"
}

for entry in "${CATEGORIES[@]}"; do
  category="${entry%%:*}"
  models_file="${entry#*:}"
  run_category "$category" "$models_file"
done

echo "=================================================="
echo "전체 완료. 요약: $SUMMARY_FILE"
echo "=================================================="
