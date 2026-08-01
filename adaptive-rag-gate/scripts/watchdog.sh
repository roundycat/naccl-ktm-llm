#!/bin/bash
# (1) 벤치마크가 끝난 모델의 HF 가중치 캐시를 지워 볼륨 쿼터 초과를 막는다.
# (2) 본 체인이 끝나면 SERVER_TIMEOUT 으로 실패한 모델만 골라 자동 재실행한다.
set -uo pipefail
B=/workspace/bench/ktm-llm-benchmark
HUB=/workspace/dhf/hub
VENV=/root/venv
export PATH="$VENV/bin:$PATH"
export HF_HOME=/workspace/dhf HF_TOKEN="$(cat /workspace/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0

# 모델명 -> HF repo (캐시 디렉토리 계산용). bge-m3 는 검색에 계속 쓰므로 절대 지우지 않는다.
declare -A REPO=(
  [qwen2.5-7b]=Qwen/Qwen2.5-7B-Instruct
  [exaone3.5-7.8b]=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
  [solar-10.7b]=upstage/SOLAR-10.7B-Instruct-v1.0
  [mistral-7b]=mistralai/Mistral-7B-Instruct-v0.3
  [huatuogpt-o1-7b]=FreedomIntelligence/HuatuoGPT-o1-7B
  [medgemma-4b]=google/medgemma-4b-it
  [biomistral-7b]=BioMistral/BioMistral-7B
  [medllama2-7b]=llSourcell/medllama2_7b
  [openbiollm-8b]=aaditya/Llama3-OpenBioLLM-8B
)

done_count() { find "$B/ktm_results_rag" -name "$1.json" 2>/dev/null | wc -l | tr -d ' '; }

prune() {
  for name in "${!REPO[@]}"; do
    [[ "$(done_count "$name")" == "4" ]] || continue          # 4연도 다 끝난 모델만
    local d="$HUB/models--${REPO[$name]//\//--}"
    if [[ -d "$d" ]]; then
      local sz; sz=$(du -sh "$d" 2>/dev/null | cut -f1)
      rm -rf "$d" && echo "[watchdog $(date '+%H:%M:%S')] 캐시 삭제 $name ($sz)"
    fi
  done
}

echo "===== WATCHDOG START $(date '+%m-%d %H:%M:%S') ====="
# ---- 1단계: 본 체인이 도는 동안 주기적 정리
while pgrep -f chain_rag.sh > /dev/null; do
  prune
  u=$(du -sh /workspace 2>/dev/null | cut -f1)
  echo "[watchdog $(date '+%H:%M:%S')] 사용량 $u / 완료 $(find "$B/ktm_results_rag" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')개"
  sleep 180
done
prune
echo "[watchdog] 본 체인 종료 감지 $(date '+%H:%M:%S')"

# ---- 2단계: 실패(미생성) 모델 자동 재실행
cd "$B" || exit 1
MISSING=()
for name in qwen2.5-7b exaone3.5-7.8b solar-10.7b mistral-7b huatuogpt-o1-7b medgemma-4b biomistral-7b medllama2-7b openbiollm-8b; do
  [[ "$(done_count "$name")" == "4" ]] || MISSING+=("$name")
done
echo "[watchdog] 미완료 모델: ${MISSING[*]:-없음}"
[[ ${#MISSING[@]} -eq 0 ]] && { echo "===== WATCHDOG DONE (전부 완료) $(date '+%m-%d %H:%M:%S') ====="; exit 0; }

# 미완료 모델만 담은 목록으로 재실행 (카테고리 구조 유지)
declare -A CAT=(
  [qwen2.5-7b]=general [exaone3.5-7.8b]=general [solar-10.7b]=general [mistral-7b]=general
  [huatuogpt-o1-7b]=domain/tcm
  [medgemma-4b]=domain/western_med [biomistral-7b]=domain/western_med
  [medllama2-7b]=domain/western_med [openbiollm-8b]=domain/western_med
)
declare -A EXTRA=(
  [exaone3.5-7.8b]="--trust-remote-code"
  [biomistral-7b]="--chat-template chat_templates/generic.jinja"
  [medllama2-7b]="--chat-template chat_templates/generic.jinja --max-model-len 4096"
  [openbiollm-8b]="--chat-template chat_templates/generic.jinja"
)
: > models/general.txt; : > models/domain/tcm.txt; : > models/domain/western_med.txt
for name in "${MISSING[@]}"; do
  f="models/${CAT[$name]}.txt"; [[ "${CAT[$name]}" == general ]] && f="models/general.txt"
  printf '%s\t%s\t%s\n' "$name" "${REPO[$name]}" "${EXTRA[$name]:-}" >> "$f"
done
sed -i 's/^CLEANUP_HF_CACHE=false/CLEANUP_HF_CACHE=true/' run_all_models_rag.sh   # 쿼터 재발 방지
echo "[watchdog] 재실행 시작 $(date '+%H:%M:%S')"
./run_all_models_rag.sh
echo "===== WATCHDOG DONE $(date '+%m-%d %H:%M:%S') ====="
echo "최종 결과: $(find "$B/ktm_results_rag" -name '*.json' | wc -l | tr -d ' ')개 / 36"
