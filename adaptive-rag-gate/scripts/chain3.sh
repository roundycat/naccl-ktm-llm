#!/bin/bash
# GraphRAG 파이프라인: graph general 대기 → ollama정지 → graph domain(vLLM) → maxrel_graph → 게이트tune
echo "[chain3] graph general 완료 대기 $(date +%H:%M)"
i=0; until grep -q "GRAPH GENERAL DONE" /workspace/run_graph_general.log 2>/dev/null; do sleep 60; i=$((i+1)); [ $i -gt 150 ] && break; done
echo "[chain3] general 완료 → ollama 정지 + domain graph $(date +%H:%M)"
pkill -x ollama; sleep 8
bash /workspace/run_graph_domain.sh
echo "[chain3] maxrel_graph(BERT,CPU) + 게이트 tune $(date +%H:%M)"
cd /workspace/adaptive_rag
CUDA_VISIBLE_DEVICES= python3 precompute_maxrel_graph.py /workspace/bert_clf/best /workspace/ktm-llm-benchmark/KTM_data /workspace/graphrag_inj /workspace/maxrel_graph 2>&1 | grep -a maxrel_graph
python3 tune_gate1.py maxrel_graph tune_graph_results.json graph 2>&1 | tee /workspace/tune_graph.log
echo "===== GRAPHRAG PIPELINE DONE $(date +%H:%M) ====="
