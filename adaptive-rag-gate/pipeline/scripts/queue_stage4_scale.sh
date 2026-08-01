#!/bin/bash
# plain_v2 까지 끝난 뒤 실행되는 3·4단계 예약.
#   3) stage 4 (SC 없음: STAGE=4, N_TRIALS=1) — CLAUDE.md 8(a), 논문 6.2 McNemar p 확보용
#   4) 규모 사다리 Qwen 7B/14B/32B — "전부 7~8B" 비판 방어. 14B/32B 는 AWQ 4bit.
# STAGE/N_TRIALS 를 바꾸는 것은 stage4 전용 사본에서만 하며 원본 스크립트는 건드리지 않는다.
set -uo pipefail
B=/workspace/bench/ktm-llm-benchmark
VENV=/root/venv
export PATH="$VENV/bin:$PATH"
export HF_HOME=/workspace/dhf HF_TOKEN="$(cat /workspace/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0

echo "===== QUEUE(stage4+scale) 대기 시작 $(date '+%m-%d %H:%M:%S') ====="

# ---- 앞 단계(RAG, plain_v2)가 모두 끝날 때까지 대기
idle=0
while (( idle < 3 )); do
  if pgrep -f "[a]fter_rag_plain.sh" > /dev/null || pgrep -f "[w]atchdog.sh" > /dev/null \
     || pgrep -f "[r]un_all_models" > /dev/null; then
    idle=0
  else
    idle=$(( idle + 1 ))
  fi
  echo "[queue $(date '+%H:%M:%S')] rag=$(find "$B/ktm_results_rag" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/36" \
       "plain_v2=$(find "$B/ktm_results_plain_v2" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/36 idle=$idle"
  sleep 180
done
echo "[queue] 앞 단계 종료 감지 $(date '+%H:%M:%S')"
cd "$B" || exit 1

FULL9() {
  printf 'qwen2.5-7b\tQwen/Qwen2.5-7B-Instruct\n' > models/general.txt
  printf 'exaone3.5-7.8b\tLGAI-EXAONE/EXAONE-3.5-7.8B-Instruct\t--trust-remote-code\n' >> models/general.txt
  printf 'solar-10.7b\tupstage/SOLAR-10.7B-Instruct-v1.0\n' >> models/general.txt
  printf 'mistral-7b\tmistralai/Mistral-7B-Instruct-v0.3\n' >> models/general.txt
  printf 'huatuogpt-o1-7b\tFreedomIntelligence/HuatuoGPT-o1-7B\n' > models/domain/tcm.txt
  printf 'medgemma-4b\tgoogle/medgemma-4b-it\n' > models/domain/western_med.txt
  printf 'biomistral-7b\tBioMistral/BioMistral-7B\t--chat-template chat_templates/generic.jinja\n' >> models/domain/western_med.txt
  printf 'medllama2-7b\tllSourcell/medllama2_7b\t--chat-template chat_templates/generic.jinja --max-model-len 4096\n' >> models/domain/western_med.txt
  printf 'openbiollm-8b\taaditya/Llama3-OpenBioLLM-8B\t--chat-template chat_templates/generic.jinja\n' >> models/domain/western_med.txt
}

# ---------------------------------------------------------------- 3) stage 4
echo "===== [3/4] STAGE 4 시작 $(date '+%H:%M:%S') ====="
FULL9
sed -e 's/^STAGE=5/STAGE=4/' -e 's/^N_TRIALS=3/N_TRIALS=1/' \
    -e 's|ktm_results/|ktm_results_stage4/|g' -e 's|mkdir -p ktm_results$|mkdir -p ktm_results_stage4|' \
    -e 's|logs/|logs_stage4/|g' \
    run_all_models.sh > run_all_models_stage4.sh
chmod +x run_all_models_stage4.sh
echo "[3/4] 설정 확인:"; grep -nE '^STAGE=|^N_TRIALS=|ktm_results_stage4' run_all_models_stage4.sh | head -4 | sed 's/^/   /'
./run_all_models_stage4.sh
echo "[3/4] stage4 완료: $(find ktm_results_stage4 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/36"

# ---------------------------------------------------------------- 4) 규모 사다리
echo "===== [4/4] 규모 사다리 시작 $(date '+%H:%M:%S') ====="
# 7B 는 이미 general 에 있으므로 14B/32B 만 추가로 돌린다 (AWQ 4bit, 24GB 에 적재)
printf 'qwen2.5-14b\tQwen/Qwen2.5-14B-Instruct-AWQ\t--max-model-len 4096 --gpu-memory-utilization 0.92\n' > models/scale.txt
printf 'qwen2.5-32b\tQwen/Qwen2.5-32B-Instruct-AWQ\t--max-model-len 4096 --gpu-memory-utilization 0.95\n' >> models/scale.txt

for mode in plain rag; do
  if [[ $mode == plain ]]; then
    src=run_all_models.sh;      root=ktm_results_scale;     logd=logs_scale
  else
    src=run_all_models_rag.sh;  root=ktm_results_scale_rag; logd=logs_scale_rag
  fi
  out="run_scale_${mode}.sh"
  sed -e "s|ktm_results_rag/|${root}/|g" -e "s|ktm_results/|${root}/|g" \
      -e "s|mkdir -p ktm_results_rag$|mkdir -p ${root}|" -e "s|mkdir -p ktm_results$|mkdir -p ${root}|" \
      -e "s|logs_rag/|${logd}/|g" -e "s|logs/|${logd}/|g" \
      -e 's|^CATEGORIES=(|CATEGORIES=(\n  "scale:models/scale.txt"|' \
      -e 's|^  "general:models/general.txt"||' \
      -e 's|^  "domain/tcm:models/domain/tcm.txt"||' \
      -e 's|^  "domain/western_med:models/domain/western_med.txt"||' \
      "$src" > "$out"
  chmod +x "$out"
  echo "[4/4] ${mode} 카테고리 확인:"; grep -A3 '^CATEGORIES=(' "$out" | head -5 | sed 's/^/   /'
  ./"$out"
  echo "[4/4] ${mode} 완료: $(find ${root} -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/8"
done

echo "===== QUEUE DONE $(date '+%m-%d %H:%M:%S') ====="
for d in ktm_results_rag ktm_results_plain_v2 ktm_results_stage4 ktm_results_scale ktm_results_scale_rag; do
  printf "  %-24s %s\n" "$d" "$(find $d -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
done
