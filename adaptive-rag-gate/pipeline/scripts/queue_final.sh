#!/bin/bash
# 남은 작업 전부를 하나의 프로세스에서 순차 실행 (동시 실행 충돌 재발 방지).
#   1) GraphRAG 큐 종료 대기
#   2) Qwen2.5-7B GraphRAG 실패분 재실행 (stage4 큐가 vLLM 을 죽여 Connection error 발생했던 건)
#   3) HuatuoGPT-o1-7B km_rag RAG 재시도 (가중치 JSON 손상 건)
#   4) stage 4 잔여분
#   5) 규모 사다리 Qwen 14B/32B (AWQ)
set -uo pipefail
B=/workspace/bench/ktm-llm-benchmark
V=/root/venv
CTX=/workspace/graphrag_ctx_canon
export PATH="$V/bin:$PATH"
export HF_HOME=/workspace/dhf HF_TOKEN="$(cat /workspace/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0
YEARS=(2022 2023 2024 2025); API=http://localhost:8000/v1
cd "$B" || exit 1

kill_vllm(){ pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 6; }

serve(){  # $1=repo $2=name $3=extra
  kill_vllm
  # shellcheck disable=SC2086
  "$V/bin/vllm" serve "$1" --served-model-name "$2" --port 8000 $3 > "/workspace/qf_$2_vllm.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 300); do kill -0 $SP 2>/dev/null || break
    curl -sf "$API/models" >/dev/null 2>&1 && return 0; sleep 5; done
  echo "  !! $2 서빙 실패"; tail -5 "/workspace/qf_$2_vllm.log"; return 1
}

runq(){ # $1=out $2=name $3=data $4=stage $5=ntrials $6=extra_args
  [[ -f "$1" ]] && { echo "    이미 있음: $1"; return 0; }
  mkdir -p "$(dirname "$1")"
  # shellcheck disable=SC2086
  "$V/bin/python" -u tkm_pipeline.py --model "openai/$2" --data "$3" --stage "$4" \
      --n-trials "$5" --api-base "$API" --api-key sk-dummy --max-workers 24 $6 \
      --output "$1" > "/workspace/qf_$(basename "$1" .json)_s$4_$(basename "$3" .json).log" 2>&1 \
    && echo "    완료 $1" || { echo "    실패 $1"; echo -e "$2\t$3\tstage$4\tFAIL" >> /workspace/qf_failures.tsv; }
}

echo "===== QUEUE_FINAL 대기 시작 $(date '+%m-%d %H:%M:%S') ====="
idle=0
while (( idle < 3 )); do
  if pgrep -f "[q]ueue_graphrag.sh" >/dev/null || pgrep -f "[r]etry_huatuo.sh" >/dev/null \
     || pgrep -f "[r]un_all_models" >/dev/null; then idle=0; else idle=$((idle+1)); fi
  echo "[qf $(date '+%H:%M:%S')] graphrag=$(find ktm_results_graphrag -name '*.json' 2>/dev/null|wc -l|tr -d ' ')/36" \
       "stage0=$(find ktm_results_graphrag_stage0 -name '*.json' 2>/dev/null|wc -l|tr -d ' ')/36 idle=$idle"
  sleep 120
done
echo "[qf] 선행 작업 종료 확인 $(date '+%H:%M:%S')"

# ---------------------------------------------------- 2) Qwen GraphRAG 재실행
echo "===== [1/4] Qwen2.5-7B GraphRAG 재실행 $(date '+%H:%M:%S') ====="
need=0
for y in "${YEARS[@]}"; do
  [[ -f "ktm_results_graphrag/general/$y/qwen2.5-7b.json" ]] || need=1
  [[ -f "ktm_results_graphrag_stage0/general/$y/qwen2.5-7b.json" ]] || need=1
done
if (( need )); then
  if serve "Qwen/Qwen2.5-7B-Instruct" "qwen2.5-7b" ""; then
    for y in "${YEARS[@]}"; do
      runq "ktm_results_graphrag/general/$y/qwen2.5-7b.json" qwen2.5-7b "KTM_data/$y.json" 5 3 \
           "--rag-context-file $CTX/graphrag_context_$y.json"
      runq "ktm_results_graphrag_stage0/general/$y/qwen2.5-7b.json" qwen2.5-7b "KTM_data/$y.json" 0 1 \
           "--rag-context-file $CTX/graphrag_context_$y.json"
    done
  fi
  kill_vllm
else echo "  이미 완료"; fi

# ---------------------------------------------------- 3) HuatuoGPT km_rag RAG
echo "===== [2/4] HuatuoGPT-o1-7B km_rag RAG 재시도 $(date '+%H:%M:%S') ====="
rm -rf /workspace/dhf/hub/models--FreedomIntelligence--HuatuoGPT-o1-7B
if serve "FreedomIntelligence/HuatuoGPT-o1-7B" "huatuogpt-o1-7b" ""; then
  for y in "${YEARS[@]}"; do
    runq "ktm_results_rag/domain/tcm/$y/huatuogpt-o1-7b.json" huatuogpt-o1-7b "KTM_data/$y.json" 5 3 \
         "--rag --rag-index-dir km_rag/index --rag-top-k 3"
  done
fi
kill_vllm

# ---------------------------------------------------- 4) stage 4 잔여
echo "===== [3/4] stage 4 잔여 $(date '+%H:%M:%S') ====="
declare -a M=(
 "general|qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct|"
 "general|exaone3.5-7.8b|LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct|--trust-remote-code"
 "general|solar-10.7b|upstage/SOLAR-10.7B-Instruct-v1.0|"
 "general|mistral-7b|mistralai/Mistral-7B-Instruct-v0.3|"
 "domain/tcm|huatuogpt-o1-7b|FreedomIntelligence/HuatuoGPT-o1-7B|"
 "domain/western_med|medgemma-4b|google/medgemma-4b-it|"
 "domain/western_med|biomistral-7b|BioMistral/BioMistral-7B|--chat-template chat_templates/generic.jinja"
 "domain/western_med|medllama2-7b|llSourcell/medllama2_7b|--chat-template chat_templates/generic.jinja --max-model-len 4096"
 "domain/western_med|openbiollm-8b|aaditya/Llama3-OpenBioLLM-8B|--chat-template chat_templates/generic.jinja")
for e in "${M[@]}"; do
  IFS='|' read -r cat name repo extra <<< "$e"
  d=0; for y in "${YEARS[@]}"; do [[ -f "ktm_results_stage4/$cat/$y/$name.json" ]] && d=$((d+1)); done
  (( d == 4 )) && { echo "  $name 완료됨"; continue; }
  echo "  --- $name ---"
  serve "$repo" "$name" "$extra" || continue
  for y in "${YEARS[@]}"; do runq "ktm_results_stage4/$cat/$y/$name.json" "$name" "KTM_data/$y.json" 4 1 ""; done
  kill_vllm
  rm -rf "/workspace/dhf/hub/models--${repo//\//--}"
done

# ---------------------------------------------------- 5) 규모 사다리
echo "===== [4/4] 규모 사다리 $(date '+%H:%M:%S') ====="
for e in "qwen2.5-14b|Qwen/Qwen2.5-14B-Instruct-AWQ|--max-model-len 4096 --gpu-memory-utilization 0.92" \
         "qwen2.5-32b|Qwen/Qwen2.5-32B-Instruct-AWQ|--max-model-len 4096 --gpu-memory-utilization 0.95"; do
  IFS='|' read -r name repo extra <<< "$e"
  echo "  --- $name ---"
  serve "$repo" "$name" "$extra" || continue
  for y in "${YEARS[@]}"; do
    runq "ktm_results_scale/scale/$y/$name.json"     "$name" "KTM_data/$y.json" 5 3 ""
    runq "ktm_results_scale_rag/scale/$y/$name.json" "$name" "KTM_data/$y.json" 5 3 \
         "--rag --rag-index-dir km_rag/index --rag-top-k 3"
  done
  kill_vllm
  rm -rf "/workspace/dhf/hub/models--${repo//\//--}"
done

echo "===== QUEUE_FINAL DONE $(date '+%m-%d %H:%M:%S') ====="
for d in ktm_results_rag ktm_results_plain_v2 ktm_results_graphrag ktm_results_graphrag_stage0 \
         ktm_results_stage4 ktm_results_scale ktm_results_scale_rag; do
  printf "  %-30s %s\n" "$d" "$(find $d -name '*.json' 2>/dev/null|wc -l|tr -d ' ')"
done
[[ -f /workspace/qf_failures.tsv ]] && { echo "실패:"; cat /workspace/qf_failures.tsv; }
