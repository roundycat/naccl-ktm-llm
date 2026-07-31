# adaptive-rag-gate — 학습형 분류기 게이트 기반 선택적 근거 증강

한의사 국가시험(2022–2025, 1,138문항, 12과목) 벤치마크에 **관련성 분류기 게이트**를 부착한 실험 코드와 결과.
팀 벤치마크(`../ktm-llm-benchmark/`)의 stage-5 프롬프트와 baseline 결과를 그대로 사용하고, 그 위에 밀집 RAG·GraphRAG 두 버전의 증강 답안을 생성해 문항 단위로 선택한다.

## 방법 요약

분류기가 검색 근거의 **관련도**를 산출하고, 관련도 ≥ T 이면서 증강 답안의 **자기일치도**(SC 3회 최빈답 비율) ≥ C 인 문항에서만 증강 답안을 채택한다. 그 외에는 baseline 답안을 유지한다.

임계값 (T, C)는 **연도 단위 교차검증(LOYO)** 으로 결정하며, 후보에 "게이트 미사용"을 포함한다. 교차검증에서 게이트가 baseline보다 낫지 않은 모델은 게이트를 끈다 → **모델 단위에서 baseline 이하로 내려가지 않는다.**

## 결과 (9모델 × 4개년 × 2버전 = 72/72 완료)

| 모델 | plain | RAG 게이트 | GraphRAG 게이트 |
|---|---:|---:|---:|
| Qwen2.5-7B | 49.4% | 49.4% | 49.4% |
| EXAONE-3.5-7.8B | 47.3% | **49.0%** | 47.3% |
| SOLAR-10.7B | 38.7% | 38.7% | **41.0%** |
| Mistral-7B | 30.7% | **33.1%** | **31.9%** |
| HuatuoGPT-o1-7B | 48.3% | 48.6% | **49.3%** |
| BioMistral-7B | 21.1% | 21.1% | 21.1% |
| MedLLaMA2-7B | 19.4% | 20.3% | **24.0%** |
| OpenBioLLM-8B | 27.1% | 27.1% | 27.1% |
| MedGemma-4B | 42.1% | 42.1% | **42.4%** |

- 과목 단위: GraphRAG 게이트가 **12과목 중 11과목에서 상승** (최대 +2.78%p, 한방생리학)
- 게이트 없는 일괄 증강은 9모델 중 3모델에서 하락(최대 −3.9%p)
- 분류기 근거 주입률: 28.9% (329/1,138)

⚠️ **해석 주의**: baseline과 증강 조건이 서로 다른 실행에서 생성되었고 SC가 temperature > 0을 사용하므로 표집 분산이 포함된다. 근거 주입이 없는 71% 문항을 대조군으로 측정한 분산이 +2.8%p로, 게이트 이득과 같은 수준이다. 자세한 내용은 `stage5_final/논문_게이트파트.md` 5절 참조.

## 디렉토리

| 경로 | 내용 |
|---|---|
| `adaptive_rag/` | 게이트 파이프라인 (`run_rag.py`, `adaptive_rag.py`, `tune_gate1.py`) |
| `graphrag/`, `bigse0u1/` | GraphRAG 지식그래프 구축·질의 |
| `training/` | 분류기 학습 (BERT 크로스인코더 96.05%, Qwen2.5 LoRA 96.65%) |
| `stage5_final/` | **최종 결과** — 보고서 2본, 원본 예측 108개, 게이트 아티팩트, 논문 초안 |
| `stage5_final/predictions/` | `RES_{model}_{year}_{base\|rag\|graph}.json` (문항 ID·과목·정답번호만, 지문 없음) |
| `stage5_final/gate/` | maxrel(관련도), 주입 파일, LOYO 임계값 |
| `answer_parse.py`, `tests/` | 통합 답안 파서 + 회귀 테스트 |

## 재현

```bash
# 1) 증강 답안 생성 (모델별·연도별)
python3 adaptive_rag/run_rag.py --model openai/<served-name> --api-base <vllm> \
  --data KTM_data/<year>.json --stage 5 --n-trials 3 \
  --rag --injections-file rag_injections_<year>.json --output RES_<...>.json

# 2) LOYO 임계값 탐색
python3 adaptive_rag/tune_gate1.py maxrel       tune_dense_results.json rag
python3 adaptive_rag/tune_gate1.py maxrel_graph tune_graph_results.json graph

# 3) 보고서 생성
python3 build_report_2ver.py <workspace> report_stage5.docx
```

## 제외된 파일

- `.env` — 로컬 설정(키는 플레이스홀더). `.env.example` 참고
- `*.npy` — BGE-m3 임베딩 캐시(98MB). `corpus.jsonl`에서 재생성
- `.venv/` — 가상환경
