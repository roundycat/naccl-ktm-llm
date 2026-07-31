# 이어서 개발·연구하기 (macOS 온보딩)

> 이 문서는 **원본(Windows·D: 드라이브·Ollama) 프로젝트를 macOS(Apple Silicon)에서
> 이어가기 위한** 셋업·현재상태·다음할일 가이드입니다. 프로젝트 자체의 배경·결과·방법론은
> [HANDOVER.md](HANDOVER.md)(전체 요약) · [README.md](README.md)(사용법) · [EXPERIMENTS.md](EXPERIMENTS.md)(개발 로그)를 보세요.
>
> 작성 기준일 **2026-07-20**. 이 저장소를 fresh clone 해 macOS에서 검증한 결과를 반영했습니다.

---

> ### 🖥️ 권장: 원격 GPU 서버 (용량·연산 걱정 없이)
> 이 노트북은 디스크가 꽉 차(99%) 로컬 실험이 불안정합니다. **원격 GPU 박스**에서 돌리도록 준비해 뒀습니다.
> 두 가지 모드 — **① RunPod Pod**(가장 싸고 즉시, compose 없이 `bash runpod_setup.sh`; training/·graphrag/ 트랙),
> **② Vast Linux VM**(풀 compose 스택 `bash docker/bootstrap.sh`; bigse0u1 Neo4j까지). 서버 선택·단계별 절차·비용은
> **→ [REMOTE_SETUP.md](REMOTE_SETUP.md)**. (Neo4j 는 bigse0u1/ 에만 필요 — 메인 결과는 Ollama만으로 재현.)
> 아래 §2~§7은 로컬(이 Mac) 기준 설명입니다.

---

## 0. 지금 상태 (이미 준비된 것 / 남은 것)

| | 항목 | 상태 |
|---|---|---|
| ✅ | 저장소 clone (4개 브랜치 전부) | 완료 — 정본 브랜치 `main_1` 체크아웃 |
| ✅ | Python 가상환경 `.venv` (uv, Python 3.12) + 기본 의존성 | 완료 (openai·dotenv·tqdm·requests·bs4) |
| ✅ | `.env` 생성 (`.env.example` 기반, GraphRAG·provider 키까지 보강) | 완료 — **API 키 값만 채우면 됨** |
| ✅ | 코드 macOS 이식성 | 문제 없음 (하드코딩 Windows 경로 0건, 전부 상대경로/pathlib) |
| ✅ | 데이터 무결성 (517/587/9074/86, 그래프 7,692노드·39,996엣지) | 문서값과 전부 일치 |
| ✅ | 안전장치: `evaluate.py` 결과 클로버링 방지 가드 | **추가·검증 완료**(§5 참고) |
| ⬜ | **Ollama 설치** (모든 실제 평가에 필수) | 미설치 → `brew install ollama` |
| ⚠️ | **디스크 여유** | **5.7GB (98% 사용)** — 모델 다운로드에 부족(§2) |
| ⬜ | GraphRAG 무거운 스택(Neo4j·Chroma·torch) | 선택 — 필요 시 §7 |

**브랜치 상황:** 기본 브랜치 `main`은 빈 커밋(README 9바이트)뿐이고, 실제 작업은 **`main_1`(정본,
experiment 브랜치 + 원본 `bigse0u1` GraphRAG 통합)** 에 있습니다. PR #1(`experiment/local-vs-global-rag → main`)은
아직 열려 있음. 이번 macOS 셋업 변경분은 `main_1` 을 깨끗이 두려고 **`continuation/macos-setup`** 브랜치에
올리도록 준비돼 있습니다(§9 브랜치 전략).

---

## 1. 30초 요약 — 무엇부터?

```bash
# (이미 .venv·.env 준비됨) Ollama만 깔면 실제 평가 가능:
brew install ollama && brew services start ollama
for m in exaone3.5:7.8b qwen2.5:7b; do ollama pull $m; done   # ⚠️ 디스크 확인(§2)
cd /Users/jeonghamin/Desktop/naccl/model/KTM-LLM
.venv/bin/python training/evaluate.py --limit 5        # 스모크 테스트(소수 문항)
.venv/bin/python training/evaluate.py --all            # 메인: 로컬 vs 글로벌 +용어RAG (517)
.venv/bin/python training/report.py                    # → results/report.md
```
전체 셋업을 스크립트로: `bash setup_macos.sh` (GraphRAG까지: `bash setup_macos.sh --graphrag`).

---

## 2. ⚠️ 디스크 경고 (원본과 동일한 고질 이슈)

원본은 C: 드라이브가 가득 차 D:로 우회했습니다. **이 Mac도 지금 98% 사용(여유 5.7GB)** 입니다.
- Ollama 모델 1개 ≈ 4~6GB (`qwen2.5:7b`≈4.7GB, `exaone3.5:7.8b`≈4.8GB) → **한 개도 빠듯**.
- GraphRAG 스택은 torch+BGE-m3(~2GB) 추가 → 현재 여유로는 불가.

**대응:** 공간을 확보하거나, 모델을 외장/여유 볼륨에 저장하도록 지정하세요.
```bash
export OLLAMA_MODELS=/Volumes/<여유디스크>/ollama-models   # ollama serve 전에 설정
```
6개 모델 전부(§6 벤치마크 재현)는 ~30GB가 필요하니, 필요한 모델만 골라 받으세요.

---

## 3. 지금 당장 할 수 있는 일 (Ollama 없이 — 전부 검증됨 ✅)

Ollama를 깔기 전에도 진척을 낼 수 있는, **fresh venv에서 실제로 돌려 성공을 확인한** 명령들:

```bash
cd /Users/jeonghamin/Desktop/naccl/model/KTM-LLM
PY=.venv/bin/python

# 1) 결과 리포트 오프라인 재생성 (results/summary.json → report.md, 순수 파일 I/O)
$PY training/report.py

# 2) 용어 RAG 인덱스 점검 (17,579개 용어 로드, 결정론적·오프라인)
$PY -c "import sys; sys.path.insert(0,'training'); from rag import TermIndex; \
        idx=TermIndex(); print('용어수', len(idx))"

# 3) 파인튜닝용 SFT 데이터 생성 (기출/해설/용어 → data/*.jsonl, 네트워크 불요)
$PY training/prepare_data.py                 # 466 train / 51 val (번호만 타깃)
$PY training/prepare_data.py --with-rationale # 517 CoT 해설 타깃
$PY training/prepare_terminology.py          # 용어 Q&A SFT

# 4) GraphRAG 코드 파싱 확인 (무거운 의존성 없이)
$PY -m py_compile bigse0u1/step1_load_neo4j.py bigse0u1/step2_build_vectordb.py \
                  bigse0u1/step3_graphrag_query.py bigse0u1/step4_eval.py
```
> 참고: `import`(openai 로드)가 **처음 한 번은 ~19초**로 느립니다 — 디스크가 꽉 차 콜드 캐시라 그렇습니다.
> 두 번째부터는 즉시입니다.

또한 **코드만으로 되는 개선 작업**(Ollama 불요, §8 로드맵의 S/M 항목): graphrag의 깨진 재현 경로 수리,
KG 드리프트 정리, RAG 임베딩 확장 등.

---

## 4. 메인 실험 재현 (용어 RAG 트랙, Ollama 필요)

```bash
cd /Users/jeonghamin/Desktop/naccl/model/KTM-LLM && PY=.venv/bin/python

# 사전: Ollama 데몬 + 모델
brew services start ollama          # 또는: ollama serve &
ollama pull exaone3.5:7.8b ; ollama pull qwen2.5:7b
ollama ps                            # 100% GPU/응답 확인

$PY training/evaluate.py --all                 # base + 글로벌 +용어RAG (517)
$PY training/evaluate.py --all --cot --sc 5    # CoT + self-consistency (가장 강한 레버, +4%p)
$PY training/evaluate.py --all --vote 3 --rag-local --local-models exaone3.5:7.8b solar:10.7b
$PY training/report.py                          # → results/report.md
```
플래그 전체는 `--help` 또는 [HANDOVER.md §5·부록A](HANDOVER.md) 참고.

---

## 5. ⚠️ 반드시 알아둘 안전장치 (이번에 추가함)

**문제였던 것:** Ollama가 꺼진 채로 `evaluate.py`를 돌리면 모든 호출이 오류로 잡혀도 **크래시하지 않고**
정답률 0% 결과로 `results/summary.json`·`eval_*.json`·`report.md`를 **조용히 덮어써서** 논문용 재현
결과를 날렸습니다(종료코드 0이라 알아채기 어려움).

**수정:** [training/evaluate.py](training/evaluate.py)의 `save_results()`에 가드를 추가했습니다.
- 유효 응답이 **하나도 없으면**(전부 호출오류) → **저장 중단·종료코드 1**, `results/` 원본 보존.
- 일부 모델만 전부 오류면 → 그 모델만 저장 제외(덮어쓰기 방지), 나머지는 정상 저장.
- 정상 런의 동작은 **변화 없음**. (로직 3케이스 검증 완료.)

만약 예전 버전으로 이미 클로버링됐다면 `git checkout results/` 로 복구하세요.

---

## 6. 모델별 벤치마크 재현 (6모델, 디스크·시간 큼)

```bash
for m in exaone3.5:7.8b qwen2.5:7b gemma2:9b llama3.1:8b mistral:7b solar:10.7b; do ollama pull $m; done
cd graphrag && ../.venv/bin/python run_all_models.py   # 6모델 plain/graph/hae 큐 (Ollama 필요)
../.venv/bin/python aggregate_models.py                 # → MULTIMODEL_REPORT.md  ⚠️ 현재 깨짐, §8-3
```
> `run_all_models.py`는 `eval_out/grageval_*.json`이 이미 있으면 **전부 skip**(재계산 안 함).
> 실제 재생성하려면 그 파일들을 지운 뒤 돌리되, §8-3의 하드코딩 경로 문제를 먼저 고쳐야 합니다.

---

## 7. GraphRAG 무거운 스택 (원본 Neo4j+Chroma, 선택)

```bash
# 1) 무거운 의존성 (uv 사용 — .venv엔 pip이 없음). torch·BGE-m3(~2GB) 포함, 디스크 확인!
uv pip install --python .venv -r bigse0u1/requirements.txt numpy

# 2) Neo4j (docker가 이미 있음). ⚠️ 신규 설치는 비번 변경 강제 → 실제 비번 지정
docker run -d --name ktm-neo4j -p7474:7474 -p7687:7687 \
  -e NEO4J_AUTH=neo4j/ktmpassword neo4j:5-community
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PW=ktmpassword

# 3) 파이프라인 (반드시 bigse0u1/ 를 cwd로 — 상대경로)
cd bigse0u1
../.venv/bin/python step2_build_vectordb.py     # 청크 → BGE-m3 → ./chroma_db (수십 분)
../.venv/bin/python step1_load_neo4j.py         # 그래프 적재
../.venv/bin/python step3_graphrag_query.py "가슴이 두근거리고 쉽게 피로해요"
../.venv/bin/python step4_eval.py --rx-only --k 20   # 처방형 86 평가
```
LLM을 Ollama 대신 클라우드로: `export LLM_PROVIDER=openai OPENAI_API_KEY=sk-...` (openai 패키지 이미 설치됨).

---

## 8. 이번 감사에서 발견한 알려진 이슈 (다음 작업 후보)

| # | 심각도 | 위치 | 내용 | 상태 |
|---|---|---|---|---|
| 1 | ⚠️ blocker | `training/evaluate.py` | Ollama 다운 시 결과 클로버링 | ✅ **수정됨**(§5) |
| 2 | blocker | `graphrag/aggregate_models.py`, `build_evidence.py`, `run_eval_local.py` | `D:\tmp\persubj\{gold,subjects,questions_noanswer}` 하드코딩 + 미동봉 → macOS에서 `FileNotFoundError`. "순수 Python 재현"이 실제로는 안 돌아감 | ⬜ 미해결 |
| 3 | warning | `bigse0u1/data` vs `graphrag/data` | KG 사본이 드리프트(노드 +24, 엣지 +163). 정본은 `graphrag/`(bigse0u1 README 명시) | ⬜ 정리 필요 |
| 4 | warning | `scripts/` | HANDOVER가 참조하는 추출 스크립트(`fetch_terminology.py`, `discover.py` 등)·`verify_explanations_*.js`가 `main_1`엔 없음(→ `experiment/local-vs-global-rag` 브랜치에 존재). 출력물은 이미 동봉돼 평가엔 지장 없음 | ⬜ 문서/코드 불일치 |
| 5 | note | `.gitignore` + HANDOVER §3.2 | `data/finetune_*.jsonl`을 "추적 중"이라 하나 실제 미추적(재생성됨). 문구가 낡음 | ⬜ 문구 정정 |
| 6 | note | `results/` | 파일 23개(문서상 ~24) 오프바이원 | ⬜ 확인 |

**8-2 상세(재현 경로 수리):** `dataset/한의학_문제_전체.jsonl`에서 `gold.json`(idx→정답)·`subjects.json`
(idx→과목)·`questions_noanswer.jsonl`을 생성하는 작은 스크립트를 만들고, 세 파일의 `P`/`QFILE` 상수를
`graphrag/data/persubj/` 같은 이식 가능한 경로로 바꾸면, 이미 있는 `eval_out/grageval_*.json`으로
**오프라인·무설치**로 리포트를 재생성할 수 있습니다(pip 설치 불요).

---

## 9. 브랜치 전략 (권장 — 최종 결정은 사용자 몫, 아직 push/merge 안 함)

1. **새 작업은 `main_1`에서 딴 continuation 브랜치에.** 이번 셋업/가드 변경분을 위해
   `continuation/macos-setup` 브랜치가 준비돼 있습니다. `main_1`은 깨끗한 통합 지점으로 유지.
2. **`main_1`을 진짜 기본 브랜치로 승격.** `main_1`이 이미 PR #1 내용을 포함(superset)하므로,
   좁은 PR #1을 빈 `main`에 머지하지 말고(히스토리 분기됨) **`main_1 → main`을 머지/FF** 한 뒤
   PR #1은 "이미 `main_1`에 통합됨" 코멘트와 함께 닫거나 리타깃하세요.
3. `main == main_1` 이 되면 `origin/HEAD`(기본 브랜치)도 그에 맞게 조정.

> ⚠️ 위 2·3은 원격(push·PR·기본 브랜치 변경)을 건드립니다. **제가 임의로 하지 않았습니다** —
> 원하실 때 직접 실행하세요.

이번에 만든 변경분을 커밋하려면(예):
```bash
git switch continuation/macos-setup   # 이미 생성돼 있음(main_1 기준)
git add CONTINUE_HERE.md setup_macos.sh .env.example training/evaluate.py
git commit -m "macOS 온보딩 문서·셋업 스크립트 + evaluate.py 결과 클로버링 방지 가드"
```
(`.env`·`.venv/`·`data/`는 .gitignore로 제외됨.)

---

## 10. 저장소 지도

```
KTM-LLM/ (main_1)
├── training/     용어 RAG 트랙: evaluate.py(평가 본체)·rag.py(용어주입)·config.py(설정·프롬프트)·report.py
├── graphrag/     GraphRAG 정본: 보기Graph 선택적RAG, 6모델. build_evidence·run_all_models·aggregate_models·step1~4
├── bigse0u1/     원본 GraphRAG(seed 벡터화, Neo4j+Chroma). step1~4·rewrite_chunks·add_missing_rx  [보존용 스냅샷]
├── dataset/      평가 문항(517/587)·해설·KIOM 용어(9,074)
├── results/      모델별 평가 결과 JSON(논문 원자료) + report.md   ← evaluate.py 가드가 보호
├── setup_macos.sh   ← macOS 셋업 자동화(이번 추가)
├── CONTINUE_HERE.md ← 이 문서
├── HANDOVER.md · README.md · EXPERIMENTS.md   ← 프로젝트 전체 설명
└── .env / .env.example   ← API 키·엔드포인트(GraphRAG 키까지 보강)
```

---

## 11. 연구 다음 트랙 (감사 기반 우선순위)

| 우선 | 트랙 | 왜 | 첫 스텝 |
|---|---|---|---|
| S | `evaluate.py` 클로버링 가드 | 재현 결과 보호 | ✅ 완료(§5) |
| S | KG 드리프트 정리 (bigse0u1 vs graphrag) | 검색 비교의 소스 일원화 | 두 `kg_all_nodes/edges.jsonl` diff → 정본 확정(graphrag) |
| M | GraphRAG 순수Python 재현 복구 | 가장 싼 리포트 재생성 경로가 지금 안 돌아감 | §8-2 (gold/subjects/questions 유도 + 경로 이식) |
| L | 증상→처방 recall 향상 (현 12.8%) | GraphRAG 트랙의 핵심 미해결 | `step4_eval.py --rx-only` 실패를 seed/Cypher/vector 단계로 버킷팅 |
| M | 용어 RAG 의미검색 확장 (~65% 커버리지) | 주입 품질의 직접 레버 | `rag.py`에 BGE-m3 임베딩 인덱스 옵션 추가 → 히트율 A/B |
| M | **로컬 파인튜닝** (MLX/LoRA) | OpenAI 차단·유료의 대안. Apple Silicon 통합메모리로 7~8B LoRA 가능 | SFT 데이터(§3) 생성 → MLX-LM으로 exaone3.5 LoRA 베이스라인 |
| M | 정답선택 헤드(constrained decoding) | 자유서술 파싱 노이즈 제거 | `evaluate.py`에 A~E 토큰 제약 변형 추가 → 정확도 비교 |
| M | 해설 517건 전문가 감수 | LLM 생성물, 임상 주장 전 검증 필요 | 과목별 층화표본 추출 → 한의학 전문가 사인오프 |
```
