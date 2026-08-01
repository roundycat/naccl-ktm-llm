#!/usr/bin/env bash
#
# 저장된 predictions/*.json + gate/maxrel*.json만으로 논문 헤드라인 수치(T3 게이트
# 9모델 평균 39.80% 등)를 재생성한다. GPU/vLLM/모델 다운로드 전혀 필요 없음 —
# gate_tuning/tune_gate.py가 원시 예측을 다시 집계하는 순수 통계 스크립트라서
# 몇 초 안에 끝난다.
#
# 사용법:
#   cd adaptive-rag-gate && ./scripts/reproduce_metrics.sh
#
# 결과는 콘솔에 표로 출력되고, 상세 JSON은 results/gate/tune3_delta0.005.json 으로 저장된다.
# reports/결과보고서_게이트.md 의 표(§1, §2)와 대조해서 확인할 것.

set -euo pipefail
cd "$(dirname "$0")/.."   # adaptive-rag-gate/ 로 이동

DELTA="${1:-0.005}"   # 채택된 T3 최종 설계의 abstain 마진 (기본값)

echo "=== T3 게이트 헤드라인 수치 재생성 (delta=${DELTA}) ==="
python3 gate_tuning/tune_gate.py results/predictions results/gate "$DELTA"

mv "tune3_delta${DELTA}.json" "results/gate/tune3_delta${DELTA}.json"
echo ""
echo "상세 결과: results/gate/tune3_delta${DELTA}.json"
echo "대조 대상: reports/결과보고서_게이트.md §1(핵심 결과), §2(모델별)"
