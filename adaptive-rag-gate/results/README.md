# results/ — 산출물 안내

논문에 인용할 최종 수치의 원천은 이 폴더다. 어떤 파일이 정본이고, 무엇으로 만들어졌는지 정리했다.

## predictions/ — 원시 예측 (108개)

`RES_{model}_{year}_{base|rag|graph}.json` — 9모델 × 4개년 × 3조건(baseline/밀집RAG/GraphRAG). 문항 ID·과목·정답번호만 담고 지문은 없음.

생성 명령:
```bash
python3 ../adaptive_rag/run_rag.py --model openai/<served-name> --api-base <vllm-endpoint> \
  --data ../../baseline/KTM_data/<year>.json --stage 5 --n-trials 3 \
  --rag --injections-file gate/rag_injections_<year>.json --output predictions/RES_<model>_<year>_<cond>.json
```

## gate/ — 관련도·주입·튜닝 결과

| 파일 | 내용 | 생성 명령 |
|---|---|---|
| `maxrel_<year>.json`, `maxrel_qwen_<year>.json`, `maxrel_graph_<year>.json` | 문항별 최대 관련도(top-20 후보 중 분류기 최고점). plain/RAG 선택 임계값 스윕용 | `python3 ../adaptive_rag/precompute_maxrel.py <corpus> <classifier> <KTM_data.json> <out.json>` (graph 버전은 `precompute_maxrel_graph.py`, qwen 버전은 `qwen_maxrel.py`) |
| `rag_injections_<year>.json` | 사전계산된 RAG 주입 블록 (질문id → 주입문자열), 모델 무관이라 1회만 계산해 재사용 | `python3 ../adaptive_rag/precompute_injections.py <corpus> <classifier> <data.json> <out.json> [top_k] [thr] [max]` |
| `tune_dense_results.json`, `tune_graph_results.json` | T1 설계(초기, `gate_tuning/exploratory/t1_tune_gate.py`) LOYO 튜닝 결과 | `python3 ../gate_tuning/exploratory/t1_tune_gate.py <maxrel_prefix> <out.json> <rag\|graph>` |
| `tune2_rag.json`, `tune2_graph.json` | T2 설계(`gate_tuning/exploratory/t2_tune_gate.py`) 튜닝 결과 | `python3 ../gate_tuning/exploratory/t2_tune_gate.py <pred_dir> <maxrel_dir> <maxrel_prefix> <rag\|graph> [out.json]` |

T3(채택된 최종 설계)의 튜닝 결과는 저장소에 원래 커밋되어 있지 않았다 — `gate_tuning/tune_gate.py`는 실행 시 콘솔에 표를 출력하는 형태였기 때문. `scripts/reproduce_metrics.sh`를 실행하면 이제 `gate/tune3_delta0.005.json`으로 저장된다(스크립트가 자동으로 생성 위치를 이 폴더로 옮겨줌). 논문 서술에 옮겨 적힌 39.80% 등의 수치는 `../reports/결과보고서_게이트.md`에 있으며, 그 수치의 기계 판독 가능한 원천은 이 `tune3_delta0.005.json`이다. 재현하려면:
```bash
../scripts/reproduce_metrics.sh
# 또는 직접: python3 ../gate_tuning/tune_gate.py <pred_dir> <maxrel_dir> [delta]
```

## ablation/ — ablation·그래프 품질 검증

| 파일 | 내용 | 생성 명령 |
|---|---|---|
| `ablation_output.txt`, `ablation_result.json` | 후보집합 분해 실험 — "게이팅 신호가 아니라 후보 집합이 성능을 지배한다"는 주장의 근거 표 | `python3 ../gate_tuning/ablation.py <pred_dir> <maxrel_dir> [delta]` |
| `kg_quality_result.json` | 그래프 구조 검증(오류율·동명분열) + 누수 통제(정답 보기 containment) + 커버리지 | `python3 ../gate_tuning/kg_quality.py <kg> <data> <ctx>` |

## logs/ — (더 이상 git 추적 안 함)

원격 실행 원시 로그. 비정본·디버깅용이라 `.gitignore`로 제외했다 (로컬 디스크에는 남아있음).

## 정본 요약

두 층위가 있다. **기계 판독 가능한 정본**은 `predictions/`(원시 예측)와 `scripts/reproduce_metrics.sh`가 그걸로 재생성하는 `gate/tune3_delta0.005.json`이다 — 숫자 자체는 여기서 나온다. **논문에 실제로 인용/서술할 정본**은 그 숫자에 통계적 해석(McNemar, 부트스트랩 CI)과 한계 논의를 붙인 **`../reports/결과보고서_게이트.md`**와 **`../reports/논문_게이트파트.md`**다. 즉 숫자가 맞는지 검증할 때는 JSON을, 논문에 뭐라고 쓸지 볼 때는 md를 봐야 한다 — 서로 대체재가 아니라 원시값/해석의 관계다.
