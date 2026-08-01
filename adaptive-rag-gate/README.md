# adaptive-rag-gate — 학습형 분류기 게이트 기반 선택적 근거 증강

한의사 국가시험(2022–2025, 1,138문항, 12과목) 벤치마크에 **관련성 분류기 게이트**를 부착한 실험. 팀 벤치마크(`../baseline/`)의 stage-5 프롬프트와 baseline 결과를 그대로 사용하고, 그 위에 밀집 RAG·GraphRAG 두 버전의 증강 답안을 생성해 문항 단위로 채택 여부를 결정한다.

> 이 폴더는 `adaptive-rag-gate` 브랜치가 병합되며 옛 `KTM-LLM` 저장소(`training/`, `dataset/`, `bigse0u1/` 구조)의 문서·스크립트가 함께 딸려 들어왔던 것을 정리한 상태다. 아래 내용은 **이 폴더에 실제로 있는 코드/데이터 기준**으로만 작성했다.

## 방법 요약

분류기가 검색 근거의 **관련도**를 산출하고, 관련도 ≥ T 이면서 증강 답안의 **자기일치도**(SC 3회 최빈답 비율) ≥ C 인 문항에서만 증강 답안을 채택한다. 그 외에는 baseline(plain) 답안을 유지한다.

임계값 (T, C)는 **연도 단위 교차검증(LOYO)**으로 정하며, 세 가지 설계를 순서대로 시도했다 (`gate_tuning/`에 t1→t2→t3로 남아있음):

| 설계 | 후보 집합 | 9모델 평균 | vs plain | vs 무게이트 |
|---|---|---:|---:|---:|
| plain (증강 없음) | — | 36.00% | — | −2.77p |
| 무게이트 dense RAG (always-inject) | — | 38.77% | +2.77p | — |
| T1 (`gate_tuning/exploratory/t1_tune_gate.py`) | plain, gate(T,C) | 36.51% | +0.51p | −2.26p |
| T2 (`gate_tuning/exploratory/t2_tune_gate.py`) | + always-inject | 39.50% | +3.50p | +0.73p |
| **T3 (`gate_tuning/tune_gate.py`, 채택)** | + GraphRAG 후보, δ=0.005 | **39.80%** | **+3.80p** | **+1.03p** |

T3(최종): 게이트 vs 무게이트 직접 검정(문항 풀링 10,242, 페어드) 차이 +1.03%p, McNemar p=0.0009, 부트스트랩 95% CI [+0.42, +1.62]. GraphRAG 게이트는 12과목 중 11과목에서 상승(최대 +2.78%p, 한방생리학). 모델별 수치·해석은 `reports/결과보고서_게이트.md`, 논문 서술은 `reports/논문_게이트파트.md` 참고. `./scripts/reproduce_metrics.sh`로 이 표를 직접 재생성해서 대조할 수 있다.

**⚠️ 제출 전 반드시 확인**: 위 이득의 크기가 방법론적 한계와 비슷한 수준이다. 자세한 내용은 [`LIMITATIONS.md`](LIMITATIONS.md).

## 디렉토리 구조

| 경로 | 내용 |
|---|---|
| `adaptive_rag/` | 게이트/검색 소스 코드 — 코퍼스 빌드(`build_corpus.py`), 증강 답안 생성(`run_rag.py`, `adaptive_rag.py`), 사전계산(`precompute_*.py`, `qwen_maxrel.py`, `graphrag_precompute.py`), GraphRAG 조회(`graphrag_retrieve.py`) |
| `gate_tuning/` | **`tune_gate.py`** — 채택된 최종 설계(T3)의 LOYO 튜닝 + ablation/검정(`ablation.py`, `vs_ungated.py`, `noise.py`, `kg_quality.py`). `exploratory/`에 T1→T2 설계 이력(`t1_tune_gate.py`, `t2_tune_gate.py`) |
| `graphrag/` | GraphRAG 지식그래프 이관 스크립트 (`SOURCES.md`에 출처 명시 — KTM-LLM `bigse0u1` 브랜치에서 이관) |
| `data/` | 생성된 중간 데이터 — `corpus.jsonl`/`corpus_noleak.jsonl`(12,168문서 코퍼스), `graphrag_ctx/`(연도별 GraphRAG 근거, RAG 조건과 동일하게 `--injections-file`로 바로 사용 가능) |
| `results/` | 원시 산출물(예측 108개, 게이트/튜닝 결과, ablation). 상세는 [`results/README.md`](results/README.md) |
| `reports/` | 논문에 인용할 최종 산출물 — `결과보고서_게이트.md`(수치 정본), `논문_게이트파트.md`(논문 서술), 초기 실험(`exp1_bert_vs_qwen_gate/`). `.docx` 보고서는 재현 스크립트가 없어 제외(팀원이 별도 보관) — md가 정본 |
| `scripts/` | **`smoke_test.sh`**(문항 5개로 설치·경로 확인), **`reproduce_metrics.sh`**(저장된 predictions만으로 헤드라인 수치 재생성, GPU 불필요), **`reproduce_full.sh`**(108개 예측 처음부터 재생성, GPU 필요) — `cluster/`(원격 GPU 박스 실행에 쓰인 체인 스크립트, 참고용), `legacy/`(one-off 재시도·모니터링 스크립트 + `patches/` 원격 핫픽스 이력) |
| `configs/experiment.yaml` | 실제 사용된 설정값(연도·stage·n_trials·게이트 delta·9개 모델 repo id) 기록. 아직 스크립트가 이 파일을 읽지는 않음 — 참조용 |

## 재현

```bash
# 0) 사전 준비
pip install -r requirements.txt
# ../baseline/tkm_pipeline.py 를 감싸서 호출하므로 그쪽 의존성(litellm 등)도 필요

# 0-1) 설치 확인만 빠르게 (vLLM 서버 필요, 문항 5개, 몇 분 내 완료)
./scripts/smoke_test.sh

# 1) 코퍼스 빌드 (이미 data/corpus.jsonl 로 완료돼 있음 — 재생성 시)
python3 adaptive_rag/build_corpus.py

# 2) 증강 답안 생성 — 108개(9모델x4년x3조건) 전부 처음부터: scripts/reproduce_full.sh 참고
#    (GPU 필요, 모델별로 오래 걸림. 스크립트 상단 주석에 재구성 근거와 한계 명시돼 있음)
./scripts/reproduce_full.sh

# 3) 헤드라인 수치 재생성 — predictions/gate가 이미 있으면 GPU 없이 몇 초 내 완료
./scripts/reproduce_metrics.sh
# 내부적으로: python3 gate_tuning/tune_gate.py results/predictions results/gate 0.005

# 4) 보고서 — build_report_2ver.py가 저장소에 없어 .docx 재생성은 현재 불가.
#    수치/서술은 reports/결과보고서_게이트.md, reports/논문_게이트파트.md 가 정본.
```

## 환경 (재현 시 반드시 확인)

`requirements.txt`는 코드에 실제로 import된 패키지만 나열했고, 버전은 하한선(`>=`)만 있다 — 팀원이 이 실험을 실행한 정확한 환경(원격 GPU 박스)의 로그나 `pip freeze` 결과가 저장소에 남아있지 않아, 실제 사용한 버전을 **확인하지 못한 채로 추측해서 적지 않았다.** 아래 값은 반드시 원 실행자에게 확인 후 채워 넣을 것 (모르는 상태로 논문에 재현 환경을 적으면 안 됨):

| 항목 | 값 |
|---|---|
| Python | **TODO: 확인 필요** |
| CUDA | **TODO: 확인 필요** |
| GPU | **TODO: 확인 필요** (RunPod 원격 박스로 추정, 정확한 모델명 미상) |
| torch / transformers / peft / sentence-transformers 정확한 버전 | **TODO: 확인 필요** |
| vLLM 버전 | **TODO: 확인 필요** (파이썬 의존성 아님 — 별도 서버 프로세스로 실행) |
| 각 모델의 정확한 HuggingFace repo/revision | `../baseline/models/*.txt`에 repo id는 있으나 revision(커밋 해시) 고정 없음 — 원본 저장소가 업데이트되면 동일 가중치 재현 불가 |

## 알려진 재현성 공백

- `build_report_2ver.py`(`.docx` 보고서를 생성하던 스크립트)가 저장소에 존재하지 않는다. `.docx` 보고서 자체도 저장소에서 제외했으므로(팀원이 별도 보관), `reports/*.md`의 수치가 유일한 정본이다.
- `data/corpus.jsonl`에서 재생성해야 하는 BGE-m3 임베딩 캐시(`*.npy`, 약 98MB)는 용량 문제로 저장소에서 제외되어 있다 (원본 문서에 명시).
- `graphrag/data/처방_rag_chunks.jsonl` 등 GraphRAG 원본 지식 데이터가 저장소에 없어, `adaptive_rag/graphrag_retrieve.py`로 그래프 검색 자체를 처음부터 재현할 수는 없다. 단 `data/graphrag_ctx/`에 연도별로 이미 계산된 GraphRAG 근거가 있어, "이미 검색된 근거로 답변을 재생성"하는 것(`scripts/reproduce_full.sh`의 graph 조건)은 가능하다.
- 각 모델의 정확한 HuggingFace revision(커밋 해시)이 고정되어 있지 않다 — `configs/experiment.yaml`/`README.md` "환경" 섹션 참고.
