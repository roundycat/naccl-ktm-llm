#!/bin/bash
# 정본 KG(graphrag/, 7692노드·39996엣지)로 1,138문항 GraphRAG 근거 사전계산.
set -uo pipefail
export PATH=/root/venv/bin:$PATH
export HF_HOME=/workspace/dhf HF_TOKEN="$(cat /workspace/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "===== GRAPHRAG_PREP(정본) START $(date '+%m-%d %H:%M:%S') ====="
/root/venv/bin/python /workspace/graphrag_precompute.py \
  /workspace/graphrag_kg_canon \
  /workspace/bench/ktm-llm-benchmark/KTM_data \
  /workspace/graphrag_ctx_canon 5
echo "===== GRAPHRAG_PREP(정본) DONE $(date '+%m-%d %H:%M:%S') ====="
ls -la /workspace/graphrag_ctx_canon/graphrag_context_*.json 2>/dev/null
