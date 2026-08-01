# KTM-LLM — 한의사 국가시험 LLM 벤치마크 & 적응형 RAG 게이트

Jang et al. (2023, PLOS Digital Health)의 5단계 누적 프롬프트 기법을 한의사 국가시험(KTM, 2022–2025)에 재현하고, 그 위에 검색 근거를 선택적으로만 채택하는 게이트를 얹어 성능을 검증하는 연구 저장소다. 두 파트로 구성된다.

## 구성

| 폴더 | 역할 | 시작점 |
|---|---|---|
| [`baseline/`](baseline/) | **기반 파이프라인.** 5단계(한자 병기 → 영어 번역 → CoT → self-consistency) 누적 프롬프트로 국가별 범용 LLM 9종을 벤치마크. RAG 없는 순수 baseline 결과가 여기서 나온다. | [`baseline/README.md`](baseline/README.md) |
| [`adaptive-rag-gate/`](adaptive-rag-gate/) | **RAG 게이트 실험.** 위 baseline 위에 밀집 RAG/GraphRAG로 검색한 근거를, 학습형 관련성 분류기 게이트(관련도 ≥ T & 자기일치도 ≥ C)로 통과한 문항에만 선택적으로 주입. 논문에 쓸 최종 수치·보고서가 여기 있다. | [`adaptive-rag-gate/README.md`](adaptive-rag-gate/README.md), 한계점은 [`adaptive-rag-gate/LIMITATIONS.md`](adaptive-rag-gate/LIMITATIONS.md) |

`adaptive-rag-gate/`는 `baseline/`를 형제 디렉터리로 참조해서 그 위에서 동작한다 (`adaptive_rag/run_rag.py`가 `../baseline/tkm_pipeline.py`를 그대로 불러와서 씀 — 파일을 복제하지 않음).

## 핵심 결과 요약

- **baseline (stage 5, RAG 없음)**: `baseline/ktm_results/`, `ktm_results_stage0/`
- **RAG 게이트 최종 (9모델 × 4개년 × 2버전, 72/72 완료)**: 게이트 39.80% vs 무게이트 38.77% vs plain 36.00% (McNemar p=0.0009) — 자세한 수치는 [`adaptive-rag-gate/reports/결과보고서_게이트.md`](adaptive-rag-gate/reports/결과보고서_게이트.md)
- **논문 제출 전 반드시 확인**: [`adaptive-rag-gate/LIMITATIONS.md`](adaptive-rag-gate/LIMITATIONS.md) 1번 — baseline과 증강 조건의 생성 시점 차이로 인한 표집 분산 이슈가 아직 해소 확인 전 상태

## 저작권 주의

`baseline/KTM_data/2022~2024.json`은 실제 국가시험 기출문제 OCR 변환본이다. public 저장소로 push하기 전 재배포 가능 여부를 확인할 것 (자세한 내용은 `baseline/README.md` 마지막 섹션).
