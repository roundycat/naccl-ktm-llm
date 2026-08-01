#!/bin/bash
# RAG 벤치마크 완주를 감지한 뒤, 동일한 vLLM 스택·동일 설정으로 plain(stage5)을 재생성한다.
# 목적: 기존 ktm_results/ 는 별도 실행에서 온 것이라 증강 조건과 표집 분산이 섞여 있다.
#       같은 환경에서 뽑은 짝맞춤 baseline 을 만들어 게이트 델타의 신뢰도를 확보한다.
# 기존 ktm_results/ 는 절대 덮어쓰지 않는다 (CLAUDE.md 3절 검증 수치 보존).
set -uo pipefail
B=/workspace/bench/ktm-llm-benchmark
VENV=/root/venv
export PATH="$VENV/bin:$PATH"
export HF_HOME=/workspace/dhf HF_TOKEN="$(cat /workspace/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0

echo "===== AFTER_RAG_PLAIN 대기 시작 $(date '+%m-%d %H:%M:%S') ====="

# ---- RAG 실행이 끝날 때까지 대기 (연속 3회 미검출이어야 종료로 간주)
idle=0
while (( idle < 3 )); do
  if pgrep -f "[w]atchdog.sh" > /dev/null || pgrep -f "[r]un_all_models_rag" > /dev/null; then
    idle=0
  else
    idle=$(( idle + 1 ))
  fi
  n=$(find "$B/ktm_results_rag" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
  echo "[after $(date '+%H:%M:%S')] RAG $n/36, idle=$idle"
  sleep 120
done

n=$(find "$B/ktm_results_rag" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "[after] RAG 종료 감지 — 완료 $n/36 $(date '+%H:%M:%S')"

cd "$B" || exit 1

# ---- 논문 9모델 전체 목록 복원 (watchdog 이 미완료분만 남겨놨을 수 있음)
printf 'qwen2.5-7b\tQwen/Qwen2.5-7B-Instruct\n' > models/general.txt
printf 'exaone3.5-7.8b\tLGAI-EXAONE/EXAONE-3.5-7.8B-Instruct\t--trust-remote-code\n' >> models/general.txt
printf 'solar-10.7b\tupstage/SOLAR-10.7B-Instruct-v1.0\n' >> models/general.txt
printf 'mistral-7b\tmistralai/Mistral-7B-Instruct-v0.3\n' >> models/general.txt
printf 'huatuogpt-o1-7b\tFreedomIntelligence/HuatuoGPT-o1-7B\n' > models/domain/tcm.txt
printf 'medgemma-4b\tgoogle/medgemma-4b-it\n' > models/domain/western_med.txt
printf 'biomistral-7b\tBioMistral/BioMistral-7B\t--chat-template chat_templates/generic.jinja\n' >> models/domain/western_med.txt
printf 'medllama2-7b\tllSourcell/medllama2_7b\t--chat-template chat_templates/generic.jinja --max-model-len 4096\n' >> models/domain/western_med.txt
printf 'openbiollm-8b\taaditya/Llama3-OpenBioLLM-8B\t--chat-template chat_templates/generic.jinja\n' >> models/domain/western_med.txt

# ---- plain 재실행 스크립트 생성 (출력 경로만 바꾸고 STAGE/N_TRIALS 는 손대지 않는다)
sed -e 's|ktm_results/|ktm_results_plain_v2/|g' \
    -e 's|^SUMMARY_FILE="ktm_results_plain_v2/|SUMMARY_FILE="ktm_results_plain_v2/|' \
    -e 's|mkdir -p ktm_results$|mkdir -p ktm_results_plain_v2|' \
    -e 's|logs/|logs_plain_v2/|g' \
    run_all_models.sh > run_all_models_plain_v2.sh
chmod +x run_all_models_plain_v2.sh

echo "[after] 설정 확인 (STAGE/N_TRIALS 는 기존과 동일해야 함):"
grep -nE '^STAGE=|^N_TRIALS=|^MAX_WORKERS=|^CLEANUP_HF_CACHE=|ktm_results_plain_v2' run_all_models_plain_v2.sh | head -8 | sed 's/^/   /'

echo "[after] plain 재생성 시작 $(date '+%H:%M:%S')"
./run_all_models_plain_v2.sh
echo "===== AFTER_RAG_PLAIN DONE $(date '+%m-%d %H:%M:%S') ====="
echo "plain_v2: $(find ktm_results_plain_v2 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/36"
echo "rag:      $(find ktm_results_rag -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/36"
