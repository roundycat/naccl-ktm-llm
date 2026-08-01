#!/bin/bash
# GraphRAG stage5 + stage0 — plain_v2 다음, stage4/규모사다리 앞에 실행.
# 사전계산된 정본 KG 근거를 --rag-context-file 로 주입하므로 프롬프팅·SC·파싱이
# plain / km_rag RAG 와 완전히 동일하다 -> stage 만 바꾼 대조로 프롬프팅 교란을 분리할 수 있다.
set -uo pipefail
B=/workspace/bench/ktm-llm-benchmark
VENV=/root/venv
CTX=/workspace/graphrag_ctx_canon
export PATH="$VENV/bin:$PATH"
export HF_HOME=/workspace/dhf HF_TOKEN="$(cat /workspace/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0

MODELS=(
  "general|qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct|"
  "general|exaone3.5-7.8b|LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct|--trust-remote-code"
  "general|solar-10.7b|upstage/SOLAR-10.7B-Instruct-v1.0|"
  "general|mistral-7b|mistralai/Mistral-7B-Instruct-v0.3|"
  "domain/tcm|huatuogpt-o1-7b|FreedomIntelligence/HuatuoGPT-o1-7B|"
  "domain/western_med|medgemma-4b|google/medgemma-4b-it|"
  "domain/western_med|biomistral-7b|BioMistral/BioMistral-7B|--chat-template chat_templates/generic.jinja"
  "domain/western_med|medllama2-7b|llSourcell/medllama2_7b|--chat-template chat_templates/generic.jinja --max-model-len 4096"
  "domain/western_med|openbiollm-8b|aaditya/Llama3-OpenBioLLM-8B|--chat-template chat_templates/generic.jinja"
)
YEARS=(2022 2023 2024 2025)
PORT=8000
API="http://localhost:${PORT}/v1"

echo "===== QUEUE_GRAPHRAG 대기 시작 $(date '+%m-%d %H:%M:%S') ====="
# ---- 앞 단계(RAG, plain_v2) 종료 대기
idle=0
while (( idle < 3 )); do
  if pgrep -f "[w]atchdog.sh" > /dev/null || pgrep -f "[a]fter_rag_plain.sh" > /dev/null \
     || pgrep -f "[r]un_all_models" > /dev/null; then idle=0; else idle=$((idle+1)); fi
  echo "[gq $(date '+%H:%M:%S')] rag=$(find $B/ktm_results_rag -name '*.json' 2>/dev/null|wc -l|tr -d ' ')/36" \
       "plain_v2=$(find $B/ktm_results_plain_v2 -name '*.json' 2>/dev/null|wc -l|tr -d ' ')/36 idle=$idle"
  sleep 180
done
echo "[gq] 앞 단계 종료 감지 $(date '+%H:%M:%S')"

# ---- 사전계산 근거 확인
for y in "${YEARS[@]}"; do
  [[ -f "$CTX/graphrag_context_${y}.json" ]] || { echo "!! $CTX/graphrag_context_${y}.json 없음 — 중단"; exit 1; }
done
echo "[gq] 근거 파일 4개 확인"
cd "$B" || exit 1

kill_vllm() { pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 6; }

for entry in "${MODELS[@]}"; do
  IFS='|' read -r cat name repo extra <<< "$entry"
  # 이 모델의 stage5·stage0 이 이미 다 있으면 건너뜀
  done5=0; done0=0
  for y in "${YEARS[@]}"; do
    [[ -f "ktm_results_graphrag/${cat}/${y}/${name}.json" ]] && done5=$((done5+1))
    [[ -f "ktm_results_graphrag_stage0/${cat}/${y}/${name}.json" ]] && done0=$((done0+1))
  done
  if (( done5 == 4 && done0 == 4 )); then echo "[gq] $name 이미 완료 — 건너뜀"; continue; fi

  echo "=================================================="
  echo "[gq] $name 서빙 시작 $(date '+%H:%M:%S')"
  kill_vllm
  # shellcheck disable=SC2086
  "$VENV/bin/vllm" serve "$repo" --served-model-name "$name" --port $PORT $extra \
      > "/workspace/gq_${name}_vllm.log" 2>&1 &
  V=$!
  ok=0
  for _ in $(seq 1 240); do
    kill -0 $V 2>/dev/null || break
    curl -sf "$API/models" >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  if (( ok != 1 )); then
    echo "[gq] !! $name 서빙 실패 — 기록 후 다음 모델"; tail -5 "/workspace/gq_${name}_vllm.log"
    echo -e "graphrag\t$name\tALL\tSERVER_FAIL" >> /workspace/gq_failures.tsv
    kill_vllm; continue
  fi
  echo "[gq] $name 서빙 완료 → 생성 $(date '+%H:%M:%S')"

  for stage in 5 0; do
    [[ $stage == 5 ]] && root=ktm_results_graphrag  || root=ktm_results_graphrag_stage0
    [[ $stage == 5 ]] && ntr=3 || ntr=1
    for y in "${YEARS[@]}"; do
      out="${root}/${cat}/${y}/${name}.json"
      [[ -f "$out" ]] && { echo "  [s${stage} ${y}] 이미 있음"; continue; }
      mkdir -p "$(dirname "$out")"
      echo "  [s${stage} ${y}] 시작 $(date '+%H:%M:%S')"
      "$VENV/bin/python" -u tkm_pipeline.py --model "openai/${name}" \
        --data "KTM_data/${y}.json" --stage $stage --n-trials $ntr \
        --api-base "$API" --api-key sk-dummy --max-workers 24 \
        --rag-context-file "$CTX/graphrag_context_${y}.json" \
        --output "$out" > "/workspace/gq_${name}_s${stage}_${y}.log" 2>&1 \
        && echo "  [s${stage} ${y}] 완료" \
        || { echo "  [s${stage} ${y}] 실패"; echo -e "graphrag\t$name\t$y\tstage$stage\tRUN_FAIL" >> /workspace/gq_failures.tsv; }
    done
  done
  kill_vllm
  # 가중치 캐시 정리(쿼터 보호)
  d="/workspace/dhf/hub/models--${repo//\//--}"
  [[ -d "$d" ]] && rm -rf "$d" && echo "[gq] 캐시 삭제 $name"
done

echo "===== QUEUE_GRAPHRAG DONE $(date '+%m-%d %H:%M:%S') ====="
echo "stage5 : $(find ktm_results_graphrag -name '*.json' 2>/dev/null|wc -l|tr -d ' ')/36"
echo "stage0 : $(find ktm_results_graphrag_stage0 -name '*.json' 2>/dev/null|wc -l|tr -d ' ')/36"
[[ -f /workspace/gq_failures.tsv ]] && { echo "실패 기록:"; cat /workspace/gq_failures.tsv; }
