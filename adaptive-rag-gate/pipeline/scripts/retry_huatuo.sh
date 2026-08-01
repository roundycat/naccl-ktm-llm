#!/bin/bash
# HuatuoGPT-o1-7B RAG 재시도 — 가중치 JSON 손상으로 0/4 실패(SERVER_TIMEOUT).
# 캐시는 이미 삭제됐으므로 재다운로드하면 정상화될 가능성이 높다.
# 모든 앞 단계가 끝난 뒤에 실행(GPU 충돌 방지).
set -uo pipefail
B=/workspace/bench/ktm-llm-benchmark
V=/root/venv
CTXDIR=/workspace   # 사용 안 함(km_rag FAISS 경로 사용)
export PATH="$V/bin:$PATH"
export HF_HOME=/workspace/dhf HF_TOKEN="$(cat /workspace/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0
NAME=huatuogpt-o1-7b; REPO=FreedomIntelligence/HuatuoGPT-o1-7B; CAT=domain/tcm
echo "===== RETRY_HUATUO 대기 $(date '+%m-%d %H:%M:%S') ====="
idle=0
while (( idle < 3 )); do
  if pgrep -f "[a]fter_rag_plain.sh" >/dev/null || pgrep -f "[q]ueue_graphrag.sh" >/dev/null \
     || pgrep -f "[q]ueue_stage4_scale.sh" >/dev/null || pgrep -f "[r]un_all_models" >/dev/null \
     || pgrep -f "[v]llm serve" >/dev/null; then idle=0; else idle=$((idle+1)); fi
  echo "[rh $(date '+%H:%M:%S')] idle=$idle"
  sleep 180
done
echo "[rh] 앞 단계 종료 — 재시도 시작 $(date '+%H:%M:%S')"
cd "$B" || exit 1
rm -rf "/workspace/dhf/hub/models--${REPO//\//--}"      # 손상 캐시 제거 후 새로 받기
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 5
"$V/bin/vllm" serve "$REPO" --served-model-name "$NAME" --port 8000 > /workspace/rh_vllm.log 2>&1 &
P=$!; ok=0
for _ in $(seq 1 300); do kill -0 $P 2>/dev/null || break
  curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
if (( ok != 1 )); then echo "[rh] !! 재시도도 실패 — 기록"; tail -8 /workspace/rh_vllm.log
  echo -e "rag\t$NAME\tALL\tSERVER_FAIL_RETRY" >> /workspace/gq_failures.tsv; exit 1; fi
echo "[rh] 서빙 OK $(date '+%H:%M:%S')"
for y in 2022 2023 2024 2025; do
  out="ktm_results_rag/${CAT}/${y}/${NAME}.json"; mkdir -p "$(dirname "$out")"
  [[ -f "$out" ]] && continue
  "$V/bin/python" -u tkm_pipeline.py --model "openai/${NAME}" --data "KTM_data/${y}.json" \
    --stage 5 --n-trials 3 --api-base http://localhost:8000/v1 --api-key sk-dummy \
    --max-workers 24 --rag --rag-index-dir km_rag/index --rag-top-k 3 \
    --output "$out" > "/workspace/rh_${y}.log" 2>&1 && echo "  [$y] 완료" || echo "  [$y] 실패"
done
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null
echo "===== RETRY_HUATUO DONE $(date '+%m-%d %H:%M:%S') ====="
echo "ktm_results_rag 최종: $(find ktm_results_rag -name '*.json'|wc -l|tr -d ' ')/36"
