# adaptive-rag-gate — 학습형 분류기 게이트 기반 선택적 근거 증강

한의사 국가시험(2022–2025, 1,138문항, 12과목) 벤치마크에 **관련성 분류기 게이트**를 부착한 실험. 팀 벤치마크(`../ktm-llm-benchmark/`)의 stage-5 프롬프트와 baseline 결과를 그대로 사용하고, 그 위에 밀집 RAG·GraphRAG 두 버전의 증강 답안을 생성해 문항 단위로 채택 여부를 결정한다.

> 이 폴더는 `adaptive-rag-gate` 브랜치가 병합되며 옛 `KTM-LLM` 저장소(`training/`, `dataset/`, `bigse0u1/` 구조)의 문서·스크립트가 함께 딸려 들어왔던 것을 정리한 상태다. 아래 내용은 **이 폴더에 실제로 있는 코드/데이터 기준**으로만 작성했다.

## 방법 요약

분류기가 검색 근거의 **관련도**를 산출하고, 관련도 ≥ T 이면서 증강 답안의 **자기일치도**(SC 3회 최빈답 비율) ≥ C 인 문항에서만 증강 답안을 채택한다. 그 외에는 baseline(plain) 답안을 유지한다.

임계값 (T, C)는 **연도 단위 교차검증(LOYO)**으로 정하며, 세 가지 설계를 순서대로 시도했다 (`gate_tuning/`에 t1→t2→t3로 남아있음):

| 설계 | 후보 집합 | 9모델 평균 | vs plain | vs 무게이트 |
|---|---|---:|---:|---:|
| plain (증강 없음) | — | 36.00% | — | −2.77p |
| 무게이트 dense RAG (always-inject) | — | 38.77% | +2.77p | — |
| T1 (`t1_tune_gate.py`) | plain, gate(T,C) | 36.51% | +0.51p | −2.26p |
| T2 (`t2_tune_gate.py`) | + always-inject | 39.50% | +3.50p | +0.73p |
| **T3 (`t3_tune_gate.py`, 채택)** | + GraphRAG 후보, δ=0.005 | **39.80%** | **+3.80p** | **+1.03p** |

T3(최종): 게이트 vs 무게이트 직접 검정(문항 풀링 10,242, 페어드) 차이 +1.03%p, McNemar p=0.0009, 부트스트랩 95% CI [+0.42, +1.62]. GraphRAG 게이트는 12과목 중 11과목에서 상승(최대 +2.78%p, 한방생리학). 모델별 수치·해석은 `paper/결과보고서_게이트.md`, 논문 서술은 `paper/논문_게이트파트.md` 참고.

**⚠️ 제출 전 반드시 확인**: 위 이득의 크기가 방법론적 한계와 비슷한 수준이다. 자세한 내용은 [`LIMITATIONS.md`](LIMITATIONS.md).

## 디렉토리 구조

| 경로 | 내용 |
|---|---|
| `adaptive_rag/` | 게이트/검색 소스 코드 — 코퍼스 빌드(`build_corpus.py`), 증강 답안 생성(`run_rag.py`, `adaptive_rag.py`), 사전계산(`precompute_*.py`, `qwen_maxrel.py`, `graphrag_precompute.py`), GraphRAG 조회(`graphrag_retrieve.py`) |
| `gate_tuning/` | 게이트 설계 T1→T2→T3 진행 과정(`t1_tune_gate.py`~`t3_tune_gate.py`) + ablation/검정 스크립트(`ablation.py`, `vs_ungated.py`, `noise.py`, `kg_quality.py`) |
| `graphrag/` | GraphRAG 지식그래프 이관 스크립트 (`SOURCES.md`에 출처 명시 — KTM-LLM `bigse0u1` 브랜치에서 이관) |
| `data/` | 생성된 중간 데이터 — `corpus.jsonl`/`corpus_noleak.jsonl`(12,168문서 코퍼스), `graphrag_ctx/`(연도별 GraphRAG 근거) |
| `results/predictions/` | 원본 예측 108개 — `RES_{model}_{year}_{base\|rag\|graph}.json` (문항 ID·과목·정답번호만, 지문 없음) |
| `results/gate/` | maxrel(관련도), 주입 파일, LOYO 임계값, T2/T3 튜닝 결과 |
| `results/ablation/` | ablation/kg_quality 등 분석 스크립트의 원시 출력 |
| `results/logs/` | 원격 실행 원시 로그 (비정본, 디버깅용) |
| `paper/` | 논문에 인용할 최종 산출물 — 결과보고서, 논문 초안, `.docx` 보고서 2본, 초기 실험(exp1: BERT게이트 vs Qwen게이트) |
| `scripts/` | 원격 GPU 실행용 운영 스크립트(체인 실행, 워치독) + `patches/`(원격 `tkm_pipeline.py` 핫픽스 이력, 참고용) |

## 재현

```bash
# 0) 사전 준비
pip install -r requirements.txt
# ../ktm-llm-benchmark/tkm_pipeline.py 를 감싸서 호출하므로 그쪽 의존성(litellm 등)도 필요

# 1) 코퍼스 빌드 (이미 data/corpus.jsonl 로 완료돼 있음 — 재생성 시)
python3 adaptive_rag/build_corpus.py

# 2) 증강 답안 생성 (모델별·연도별, vLLM 서버 필요)
python3 adaptive_rag/run_rag.py --model openai/<served-name> --api-base <vllm-endpoint> \
  --data ../ktm-llm-benchmark/KTM_data/<year>.json --stage 5 --n-trials 3 \
  --rag --injections-file results/gate/rag_injections_<year>.json --output results/predictions/RES_<...>.json

# 3) 게이트 임계값 탐색 (LOYO) — T3(채택된 최종 설계)만 실행하면 됨, T1/T2는 설계 이력
python3 gate_tuning/t3_tune_gate.py

# 4) 보고서 생성 — build_report_2ver.py가 저장소에 없어 .docx 재생성은 현재 불가.
#    수치/서술은 paper/결과보고서_게이트.md, paper/논문_게이트파트.md 를 직접 참고할 것.
```

## 알려진 재현성 공백

- `build_report_2ver.py`(위 4단계에서 `.docx` 보고서를 생성하던 스크립트)가 저장소에 존재하지 않는다. `paper/*.docx`는 현재 코드로 재생성할 수 없고, `paper/*.md`의 수치가 원천이다.
- `data/corpus.jsonl`에서 재생성해야 하는 BGE-m3 임베딩 캐시(`*.npy`, 약 98MB)는 용량 문제로 저장소에서 제외되어 있다 (원본 문서에 명시).
