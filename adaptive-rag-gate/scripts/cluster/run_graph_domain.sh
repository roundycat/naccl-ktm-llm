#!/bin/bash
# 도메인 4모델 × 4연도 graph 답 (GraphRAG 근거, vLLM). medgemma는 라이선스 대기라 제외.
VLLM=/root/vllm_venv/bin/vllm
cd /workspace/adaptive_rag
export HF_HOME=/workspace/dhf HF_TOKEN=$(cat /workspace/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /workspace/.hf_token)
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0
CT=/workspace/ktm-llm-benchmark/chat_templates/generic.jinja

ready() {
  python3 - "$1" <<'PY'
import sys, urllib.request
try:
    d = urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3).read().decode()
    sys.exit(0 if sys.argv[1] in d else 1)
except Exception:
    sys.exit(1)
PY
}

run() {
  NAME=$1; HFID=$2; EXTRA=$3
  echo "##### $NAME graph start $(date +%H:%M) #####"
  $VLLM serve "$HFID" --served-model-name "$NAME" --port 8000 \
    --gpu-memory-utilization 0.9 --max-model-len 4096 $EXTRA > /workspace/vllm_g_${NAME}.log 2>&1 &
  V=$!; R=0
  for i in $(seq 1 300); do
    kill -0 $V 2>/dev/null || break
    ready "$NAME" && { R=1; break; }
    sleep 5
  done
  if [ "$R" = 1 ]; then
    for Y in 2022 2023 2024 2025; do
      python3 run_rag.py --model "openai/$NAME" --api-base http://localhost:8000/v1 --api-key sk-dummy \
        --data /workspace/ktm-llm-benchmark/KTM_data/${Y}.json --stage 4 --max-workers 16 \
        --rag --injections-file /workspace/graphrag_inj_${Y}.json --translation-cache /workspace/trans_shared_${Y}.json \
        --output /workspace/RES_${NAME}_${Y}_graph.json 2>&1 | grep -a Accuracy
    done
  else
    echo "!! $NAME 서빙 실패"; tail -5 /workspace/vllm_g_${NAME}.log
  fi
  kill $V 2>/dev/null; sleep 5; pkill -f "/root/vllm_venv" 2>/dev/null; sleep 8
  rm -rf /workspace/dhf/hub/models--* 2>/dev/null
  echo "##### $NAME done $(date +%H:%M) #####"
}

run huatuogpt-o1-7b FreedomIntelligence/HuatuoGPT-o1-7B "--trust-remote-code"
run biomistral-7b BioMistral/BioMistral-7B "--chat-template $CT"
run medllama2-7b llSourcell/medllama2_7b "--chat-template $CT"
run openbiollm-8b aaditya/Llama3-OpenBioLLM-8B "--chat-template $CT"
echo "===== GRAPH DOMAIN DONE $(date +%H:%M) ====="
