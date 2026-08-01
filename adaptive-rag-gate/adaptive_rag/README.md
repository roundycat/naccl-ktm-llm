# 어댑티브 RAG — 관련성 분류기를 국시 벤치마크에 붙이기

우리가 학습한 **관련성 분류기**(BERT 크로스인코더, test 96% acc)를 팀원의
**TKM 국시 벤치마크**(`../baseline/tkm_pipeline.py`)에 게이트로 끼워,
"우리 분류기가 국시 정답률을 올리는가"를 실제로 측정한다.

## 구조

```
국시 문제 → BGE-m3로 코퍼스 후보 top-K 검색      (1차 리트리버)
          → 우리 BERT 분류기가 (문제,후보) 관련성 판정  (게이트) ← 우리가 학습한 것
          → 관련 문서만 '참고 지식'으로 프롬프트에 주입
          → LLM 답변 (팀원의 stage 0~5 기법과 결합)
```

- `build_corpus.py` — GraphRAG 지식(처방/용어/해설 12,685개)을 분류기 학습 포맷으로 `../data/corpus.jsonl` 생성. **오프라인 완료.**
- `adaptive_rag.py` — 2단계 검색(BGE-m3 → BERT 게이트) 모듈.
- `run_rag.py` — 팀원 `tkm_pipeline`을 재사용(파일 수정 X)하며 RAG 주입. 같은 스크립트로 baseline/RAG 둘 다.

## 실행 (박스에서 — vLLM + 우리 분류기 필요)

```bash
# 0) 코퍼스 (이미 생성돼 있으면 생략)
python build_corpus.py

# 1) vLLM으로 대상 모델 서빙 (팀원 run_all_models.sh 방식)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 &

# 2) baseline (RAG 없음) — 팀원 기준선과 동일
python run_rag.py --model openai/Qwen/Qwen2.5-7B-Instruct \
    --api-base http://localhost:8000/v1 --api-key sk-dummy \
    --data ../baseline/KTM_data/2025.json --stage 4 \
    --output out_qwen_2025_s4_base.json

# 3) + 어댑티브 RAG (우리 분류기 게이트)
python run_rag.py --model openai/Qwen/Qwen2.5-7B-Instruct \
    --api-base http://localhost:8000/v1 --api-key sk-dummy \
    --data ../baseline/KTM_data/2025.json --stage 4 \
    --rag --corpus ../data/corpus.jsonl --classifier /workspace/bert_clf/best \
    --output out_qwen_2025_s4_rag.json
```

정답률 차이(Δ) = **우리 분류기의 국시 기여도**. exaone3.5-7.8b·qwen2.5-7b는
팀원 기준선(`../baseline/ktm_results/general/`)이 이미 있어 직접 비교 가능.

## 하이퍼파라미터
- `--rag-top-k` (기본 20): 1차 후보 수
- `--rag-threshold` (기본 0.5): 게이트 통과 확률. 높이면 정밀↑재현↓
- `--rag-max-inject` (기본 5): 주입 문서 최대 수

## 상태
- [x] 코퍼스 빌드(12,685) · RAG 모듈 · 러너 — **오프라인 완료**
- [ ] 박스에서 실행: vLLM 서빙 + 우리 분류기로 baseline vs RAG 측정 (박스 켜야 함)
