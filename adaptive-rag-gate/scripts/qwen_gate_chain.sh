#!/bin/bash
cd /workspace/adaptive_rag
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1
export HF_TOKEN=$(cat /workspace/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /workspace/.hf_token)
echo "[qwen chain] 시작 $(date +%H:%M)"
python3 qwen_maxrel.py corpus_noleak.jsonl Qwen/Qwen2.5-7B-Instruct /workspace/qwen_clf/best /workspace/ktm-llm-benchmark/KTM_data /workspace/maxrel_qwen 2>&1
python3 tune_gate1.py maxrel_qwen tune1_qwen_results.json 2>&1 | tee /workspace/tune1_qwen.log
echo "===== QWEN GATE DONE $(date +%H:%M) ====="
