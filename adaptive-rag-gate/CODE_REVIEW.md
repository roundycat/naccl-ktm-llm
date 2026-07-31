# 전체 코드 재분석 리포트 (KTM-LLM)

> 다중 에이전트(40) 심층 정독 + 적대적 검증(CONFIRMED/PARTIAL만 수록). 작성 2026-07-22.

> 이전 [CONTINUE_HERE.md](CONTINUE_HERE.md) §8은 '실행가능성', 이 문서는 **코드 정확성·연구 방법론 타당성**.


## ✅ 적용된 수정 (2026-07-23) — 코드 로직/구조 결함 교정

아래 결함들은 코드 수정 완료(재측정은 원격 GPU 박스에서). 검증: 파서 회귀테스트 22/22 통과,
전 파일 py_compile OK, 용어RAG 누수 재현측정 140→**0**.

| # | 결함 | 수정 | 검증 |
|---|---|---|---|
| 1 | 용어RAG 정답 누수 | `rag.retrieve(exclude=보기표면형)` + `evaluate._option_surfaces` — 보기(정답후보) 자신의 정의 주입 차단 | 옵션정의 주입 140→0, 정답옵션 80→0 |
| 2·3·4·9·10 | 파서 3종 분열·원문자불능·서수/단위 오탐 | **`answer_parse.py` 단일 파서**로 통합(training/evaluate·run_eval_local·step4×2 전부 import); 앵커 last-match+부정가드, 서수(번째)·단위(mg 등)·범위(1~5) 배제, 원문자 ①~⑤ 지원 | `tests/test_answer_parse.py` 22/22 |
| 5 | 다판본 처방근거 last-wins 손실 | `build_evidence.rx_by_name` 주치/구성 **union** 병합 | 411명(주치)·246명(구성) 근거 복원 |
| 6 | bigse0u1 벡터↔Graph 컬렉션 불일치 | 그외 문항도 GraphRAG가 `COLL_RX_CLINICAL` 사용(벡터와 동일) | py_compile |
| 7 | gold/subjects/questions `D:\` 하드코딩·미동봉 | 전 경로 `PERSUBJ_DIR`(레포상대) env화 + **`derive_persubj.py`** 로 데이터셋에서 재생성 | 파생물 517/14과목/처방형86 = 문서값 일치 |
| 11·12 | step3 Cypher: 무약재 처방 누락·비결정 정렬 | `OPTIONAL MATCH 구성` + `ORDER BY score DESC, p.id` (양 step3) | py_compile |
| 17 | 과목 라벨 2종 표기 | 공백정규화(법규 30+20→50 병합) — evaluate + derive_persubj | 파생 subjects=14 |
| 18 | 저장 가드 '전부오류'만 차단 | 오류율 ≥20%면 저장 제외(신뢰불가 런 클로버링 방지) | 로직 검토 |

**보류**: #8(한자 경계가드, LOW·영향無·인덱스 지연 우려), #14/#15/#16/#19/#20(잠재/장식). 아래 원문 참조.
**⚠️ 재측정 필수**: 파서 통합·누수 차단·persubj 재생성으로 기존 수치는 무효 → 원격 박스에서 재실행해야 정답률이 확정된다. 특히 용어RAG 상승분은 누수 제거 후 재측정 시 하락 예상.

---

## TL;DR — 결과에 영향을 주는 확정 결함 2건

1. **용어 RAG 정답 누수 (HIGH, 논문 핵심주장 무효화)** — `retrieve()`가 문제+**보기 전체**로 검색하고 용어집이 정답표와 같은 KIOM 출처라, 정의형 문항에선 **정답 보기의 정의문 자체가 주입**됨. 커밋된 결과로 재현: +0.97%p 상승분이 전부 정답주입 44문항에서 나옴(정답주입 net +14, 비누수 net −5). **누수 제거 시 용어RAG는 오히려 마이너스.**

2. **다중모델 GraphRAG 리포트의 채점기가 원문자(①~⑤) 불능 (HIGH)** — `run_eval_local.parse()`가 ①~⑤ 답을 −1로 처리하고 first-digit fallback. 실패가 verbose graph/hae 모드에 몰려(solar graph 9.9%) plain-vs-RAG 델타를 편향. 게다가 세 트랙이 **서로 다른 파서 3개**라 교차 비교 불가.


## 방법론 신뢰성 평가

GOLD LEAKAGE — Mostly clean, with one decisive exception. Verified across tracks: base/rag/cot/vote prompts are built from question+options only; gold is used purely for scoring; graph/vector evidence is driven by question text and enriches ALL options symmetrically (no correct-index marker); selective-RAG routing is a fixed gold-free question-type rule; 해설 RAG uses correct leave-one-out self-exclusion with zero exact-duplicate questions. The EXCEPTION is training/rag.py term-RAG: because retrieval keys on the options and the terminology comes from the same KIOM source as the answer key, definitional MCQs inject the gold option's own defining text — and the entire reported base-vs-RAG improvement is this leak (verified against committed results). This is the single most important validity problem; the term-RAG uplift number is untrustworthy.

PARSE_ANSWER ROBUSTNESS — At-risk and inconsistent. Three tracks use three different, un-shared parsers of increasing weakness: training's 4-tier parse_answer (best, ①-⑤ + anchors, but with a first-vs-last inconsistency in tier 1 and incomplete ordinal/unit guards in tiers 2/4), run_eval_local's 정답-anchor + first-digit (circled-blind, feeds the headline multi-model report, mode-correlated -1 failures), and step4's naive first-digit (weakest). Identical model text can receive three different scores. Within-track scoring is mostly reliable at temp=0 with number-only prompts; the parser weaknesses bite hardest on verbose CoT/graph outputs and would corrupt any verbose model.

TRAIN/VAL SPLIT — Currently correct (seed 42, 0.1, byte-identical splits, 0 eval∩train overlap in default and --with-rationale, verified), but unenforced: disjointness rides on EXPLAINED staying row-aligned with FULL, the recipe is duplicated in 3 files, and no fine-tune was actually run, so this is latent hygiene rather than an active defect.

CROSS-TRACK COMPARABILITY — NOT sound; do not compare absolute accuracies across training/ vs graphrag/ vs bigse0u1/. Different parser + different prompt + (training only) a system role + (graphrag/bigse0u1) gold loaded from absent hardcoded Windows paths that make those numbers non-reproducible from the repo. bigse0u1 additionally mixes retrieval collections by question type and uses a different seed-extraction pipeline than graphrag, and its own README deprecates it for citation. Only within-track relative deltas are trustworthy.

DETERMINISM — Sound. The documented run-to-run ±1 is serving-stack batching at temp=0, not code nondeterminism: thread results are keyed by row idx, parse tiers use rfind/positional order, Counter tie-break is insertion order. The only genuinely stochastic paths are the SC/vote variants at temp=0.7 by design (and SC has no deterministic tie-break / never abstains at sc_min=0).

BOTTOM LINE ON TRUSTWORTHINESS: The 517-question set, figure filtering, 처방형 subsetting, and no-gold-leakage guarantees (except term-RAG) are solid. TRUSTWORTHY: within-track base/cot/vote accuracy ordering, the graphrag 처방형-graph benefit, and the selective-RAG-vs-plain conclusion. AT-RISK: the training term-RAG uplift (leak-driven, likely net-negative once corrected), any cross-track absolute comparison, and the graphrag/bigse0u1 numbers' reproducibility.


## 아키텍처 개요
The KTM-LLM codebase evaluates LLMs on a 587-question Korean traditional-medicine (한의학) national-board 5-choice exam (517 text-only questions after removing 70 figure items; 86 of those are prescription-type / 처방형). It contains three largely independent evaluation tracks that share the same underlying question set but do NOT share code:

1) training/ — the primary fine-tuning + scoring track. evaluate.py runs models in base/rag/cot/sc/vote variants, parses the chosen option with a 4-tier parse_answer() regex ladder, and aggregates overall + per-subject accuracy. rag.py is a dependency-free lexical "term-RAG" that indexes 9,074 KIOM terminology records (dual-indexed by Hanja substring + Korean word-boundary regex → ~17,579 keys) and injects standard definitions of surface terms found in the question+options. prepare_data.py/prepare_terminology.py/finetune.py build OpenAI SFT files and launch fine-tunes; add_explanations.py generates per-question 해설. Train/val split = seed 42, VAL_RATIO 0.1, has_figure filtered.

2) graphrag/ — a "selective-RAG" track. build_evidence.py/build_hae_evidence.py precompute per-question graph evidence (option-derived 구성/주치/계통 + symptom-matched candidates + leave-one-out 해설); run_eval_local.py scores 6 local Ollama models in plain/graph/hae modes; aggregate_models.py routes 처방형→graph, else→plain (post-hoc, gold-free) and writes MULTIMODEL_REPORT.md.

3) bigse0u1/ — the original Neo4j+Chroma GraphRAG prototype (deprecated for citation by its own README). step1–4 load a KG of 처방/증상/약재, embed chunks with BGE-m3, extract symptom seeds (vector path), do a reverse 주치 lookup + vector retrieval, and score plain/vector/GraphRAG.

Data/scoring flow: question+options → (optional) retrieval/evidence injection → LLM → free-text answer → per-track regex parser → pred==gold (1..5) → accuracy. The three tracks use three DIFFERENT parsers, three DIFFERENT prompts, and (in graphrag/bigse0u1) gold/subjects loaded from hardcoded Windows paths absent from the repo — so absolute accuracies are only safe to compare WITHIN a track, not across tracks.


## 모듈별 품질

- **[has-issues]** `training/ (evaluate.py, config.py, run_all.py)` — Primary scoring core: model calling, self-consistency/vote, 4-tier answer parser, overall+per-subject accuracy, result persistence guard.
- **[has-issues]** `training/rag.py (term-RAG)` — Deterministic dependency-free lexical terminology injector (Hanja substring + Korean word-boundary matching over 9,074 KIOM records).
- **[has-issues]** `training/ data-prep & finetune (prepare_data.py, prepare_terminology.py, finetune.py, add_explanations.py)` — Build OpenAI SFT files, generate 해설, launch/monitor fine-tunes; owns the train/val split that must stay disjoint from eval.
- **[has-issues]** `graphrag/ (build_evidence, run_eval_local, aggregate_models, step1-4)` — Selective-RAG evaluation: precomputed graph/해설 evidence, 6 local models, 처방형-vs-else routing, multi-model report.
- **[fragile]** `bigse0u1/ (step1-4, enrich/add scripts)` — Original Neo4j+Chroma GraphRAG prototype; three-way plain/vector/GraphRAG comparison. Deprecated for citation.
- **[solid]** `dataset/ + eval/ question and terminology files` — Shared 587/517/86 question set, KIOM terminology, curated 해설. Cross-track question set is consistent and figure-filtering nets out correctly.

## 확정 발견 (심각도 순, 20건)

### 1. [HIGH] Term-RAG answer leak: definitional MCQs inject the gold option's own KIOM definition, and the entire reported base-vs-RAG gain is this leak
- **파일**: `training/evaluate.py:84`  ·  **분류**: eval-validity
- **영향**: AFFECTS REPORTED RESULTS and invalidates the project's central 'global model RAG uplift' claim. retrieve() is keyed on question_text() = question + all options (rag.py:111), so for definitional items whose options ARE KIOM terms, the correct option's own definition (which paraphrases the vignette) is injected. Reproduced on committed results: 82/517 rows inject an option-term definition, 44 inject the GOLD option's. For qwen2.5:7b the +0.97%p headline gain decomposes to NET +14 on the 44 gold-injected items (16 wrong→right vs 2 right→wrong) and NET -5 on the 435 non-leaked items; llama3.1:8b shows the same signature (gold-injected +11 at 13:2, non-leaked -13). Without the leak, term-RAG is net negative.
- **수정**: Before injecting, drop any retrieved term whose surface form equals one of the five answer options (compare against parse_rx_names/option strings). Additionally exclude definitional MCQs from the RAG-uplift measurement, or report the delta separately on leaked vs non-leaked items. Retrieve on the stem only (exclude options from question_text) for the uplift metric.

### 2. [HIGH] run_eval_local.parse() (headline multi-model report scorer) is circled-numeral-blind with a first-digit fallback and mode-correlated failures
- **파일**: `graphrag/run_eval_local.py:25`  ·  **분류**: answer-parsing
- **영향**: AFFECTS REPORTED RESULTS. run_all_models.py→run_eval_local.py→aggregate_models.py produces MULTIMODEL_REPORT.md including the selective-RAG column. parse()= re.search('정답[^0-9]{0,4}([1-5])') or first-digit; it returns -1 on ①-⑤ answers and mis-grabs prompt echoes ('정답 번호(1-5): 3'→1). Hard -1 misses are mode-correlated (qwen: plain 0/517, graph 5, hae 17/517=3.3%; solar graph 51/517=9.9%), penalizing exactly the verbose 해설/graph modes and biasing the plain-vs-RAG deltas that drive the paper's conclusion. It is a THIRD parser variant distinct from step4's parse_choice and training's parse_answer, so identical model text is scored differently across tracks.
- **수정**: Replace all three parsers with a single shared module (ideally reuse training/evaluate.py parse_answer, which handles ①-⑤, the 정답 anchor, and a boundary-aware last-digit rule). Add a retry-on-unparseable pass and log/report parse-fail vs API-fail vs genuine-wrong separately rather than folding all into 'wrong'.

### 3. [MEDIUM] parse_answer regex ladder: priority-1 takes FIRST match (siblings take last), and priorities 2/4 miss ordinal/unit/digit-boundary guards
- **파일**: `training/evaluate.py:106`  ·  **분류**: correctness
- **영향**: AFFECTS REPORTED RESULTS (bounded). Priority-1 uses re.search (first '정답/답 N') while priorities 2/3/4 deliberately take the last, contradicting the docstring's 'prefer the conclusion' intent, so negation/self-correction ('정답은 3이 아니라 4'→3) is misparsed. Priority-2 ('N번') lacks the (?<![0-9]) lookbehind and 번째 guard that priority-4 has, so ordinals/dose-frequency/question-numbers hijack ('하루 3번'→3, '3번째'→3, '12번'→2). Priority-4's unit exclusion omits g/mg/kg/ml/%/주/단계/가지, leaking trailing counts as the answer. All reproduced by direct execution. Most exposed on CoT and explanatory local outputs; no_answer=0 in stored runs means every misparse is silently scored wrong, and CoT's longer text gives more trigger surface in the base-vs-CoT comparison.
- **수정**: Make priority-1 collect re.findall and take the last (prefer the final line) plus a negation guard. Add (?<![0-9]) and a 번째 exclusion to priority-2. Widen priority-4's unit blacklist (g|mg|kg|ml|cc|%|주|단계|가지|층|군|형|위). Add an adversarial Korean-output regression test corpus as a gate on parse_answer.

### 4. [MEDIUM] step4 parse_choice() grabs the first bare 1-5 digit anywhere, misreading '5지선다'/reasoning-before-answer and any ①-⑤ output
- **파일**: `graphrag/step4_eval.py:46`  ·  **분류**: correctness
- **영향**: Partially affects results — step4_eval.py in graphrag/ and bigse0u1/ is the weaker of the sibling parsers (no 정답 anchor, no circled support). In the graphrag repo it currently only writes a diagnostic file, but bigse0u1/step4_eval.py uses it for its published plain/vector/GraphRAG numbers. temp=0 + 'output only the number' prompts mask it in observed runs, but it would corrupt metrics under any verbose/reasoning model, and it makes cross-track number comparison unsafe.
- **수정**: Anchor on 정답 first and fall back to the LAST digit with unit exclusion; add ①-⑤ support. Consolidate to the shared parser above.

### 5. [MEDIUM] rx_by_name keeps only the LAST chunk per 처방명, so option evidence uses one arbitrary edition's 주치/구성 for ~31% of prescriptions
- **파일**: `graphrag/build_evidence.py:20`  ·  **분류**: data-completeness
- **영향**: AFFECTS the graded RAG input. 3094 chunks map to 1830 distinct names; 562 (31%) have >1 divergent chunk (452 differ in 주치, 364 in 구성). Last-wins omits ≥1 indication for 411 names, and in the committed per_question_evidence.jsonl 80/87 option-matched questions (92%) hit a lossy duplicate that is fed verbatim as RAG context to the scored eval. Final accuracy delta is unquantified (model may ignore evidence) but the injected option evidence is objectively incomplete for the large majority of relevant questions.
- **수정**: Change rx_by_name[nm] to accumulate/union 주치·구성·계통 across all chunks for a name (as sym_to_rx already unions), or store a list and merge indications before formatting the injection.

### 6. [MEDIUM] bigse0u1 벡터 and GraphRAG read different Chroma collections for non-처방 questions (COLL_RX_CLINICAL vs COLL_RX)
- **파일**: `bigse0u1/step4_eval.py:117`  ·  **분류**: eval-validity
- **영향**: AFFECTS the bigse0u1 벡터-vs-Graph delta for the 431/517 (83%) non-prescription questions: the two methods retrieve from different corpora (terse hanmun hani_rx vs LLM-rewritten clinical narratives), so the reported delta conflates graph augmentation with a corpus swap. The core 처방형 comparison (both COLL_RX_CLINICAL) stays valid, and the primary --rx-only run avoids the mix, but the overall/그외 aggregate is confounded. The recall metric likewise means different things across bigse0u1 and graphrag.
- **수정**: Use the same collection for 벡터 and the GraphRAG vector component within a track (pick COLL_RX_CLINICAL consistently), and hold the corpus fixed when attributing any delta to graph augmentation.

### 7. [MEDIUM] GraphRAG/bigse0u1 gold, subjects and questions loaded from hardcoded Windows paths (D:\tmp\persubj, D:\정하민\...) absent from the repo
- **파일**: `graphrag/aggregate_models.py:6`  ·  **분류**: reproducibility
- **영향**: AFFECTS auditability of every graphrag/bigse0u1 accuracy number. aggregate_models.py, run_eval_local.py, build_evidence.py, build_hae_evidence.py all read gold.json/subjects.json/questions_noanswer.jsonl (and write) at absolute D:\ paths not versioned here. The scripts crash immediately on this checkout, and reconstructing gold from the versioned answer file yields ~20% (chance) vs the report's 32-49%, proving the gold indices are tied to the absent (option-shuffled) questions file. The reported metrics cannot be regenerated or independently verified from the repository.
- **수정**: Commit gold.json/subjects.json/questions_noanswer.jsonl (or derive them deterministically from the versioned 한의학_문제.jsonl) and replace D:\ literals with repo-relative paths / an env var.

### 8. [LOW] Hanja retrieval branch uses bare substring match (no boundary guard) so compositional fragments (황련⊂黃連湯) are injected
- **파일**: `training/rag.py:83`  ·  **분류**: correctness
- **영향**: Minor precision noise; NO measured result impact. 275/1193 injected slots on the eval rows are substring-only fragments, but the longer containing compounds are not themselves dictionary terms, so no correct longer term is evicted; retrieve() sorts longest-first so fragments only fill leftover K slots. On-topic noise, not misinformation.
- **수정**: Add a non-Hanja boundary guard analogous to the Korean branch's (?<![가-힣])…(?![가-힣]) to the Hanja key match.

### 9. [LOW] parse_answer priority-3 returns the LAST circled glyph; echoing options ①..⑤ scores ⑤
- **파일**: `training/evaluate.py:114`  ·  **분류**: correctness
- **영향**: Narrow edge case, NOT observed in runs. build_prompt renders plain-digit options (never ①..⑤) and the number-only instruction routes compliant answers away from this tier; pred distributions show no skew toward 5. Latent only.
- **수정**: Prefer the circled glyph adjacent to a conclusion marker, or ignore an ascending ①..⑤ enumeration.

### 10. [LOW] parse_answer priority-1 josa class [은는이] omits 을/를/으로 and bare 답 matches 오답/응답
- **파일**: `training/evaluate.py:106`  ·  **분류**: correctness
- **영향**: Edge case, largely self-mitigated: '정답을 4번으로' still resolves via the priority-2 fallback; a wrong result needs accusative josa PLUS a later distractor N번. CoT is immune (forced terminal '정답: N'). No stored metric shown to flip.
- **수정**: Widen the josa class and require 정답 rather than bare 답, or make priority-1 last-match.

### 11. [LOW] GRAPH_Q uses non-OPTIONAL 구성 MATCH after LIMIT 6, silently dropping herbless prescriptions (also present in graphrag/step3)
- **파일**: `bigse0u1/step3_graphrag_query.py:131`  ·  **분류**: correctness
- **영향**: Real latent bug in both bigse0u1 and graphrag step3, but zero impact on reported results: only 16/3118 (bigse0u1) / 16 (graphrag) prescriptions have 0 구성 edges, and none is a gold answer anywhere in the 517-set. Sibling GRAPH_BY_NAME_Q already uses OPTIONAL MATCH.
- **수정**: Change MATCH (p)-[:REL{type:'구성'}]->(h) to OPTIONAL MATCH to mirror the sibling query.

### 12. [LOW] Reverse 주치 lookup ORDER BY score DESC LIMIT 6 has no deterministic tiebreaker
- **파일**: `bigse0u1/step3_graphrag_query.py:130`  ·  **분류**: reproducibility
- **영향**: Real nondeterminism nit (Neo4j resolves equal-score ties arbitrarily), but the finding's 'drops the correct 처방' claim is misattributed: in all 11 observed plain-ok/rag-fail cases the gold prescription has score 0 (no matching 주치 edge), so it is never in the tie pool. The tie shuffle only reorders wrong candidates; graph rows are supplementary context to an LLM that sees the options. No reported answer flips.
- **수정**: Add a deterministic secondary sort key (e.g. p.id) for reproducibility; the real recall gap is graph coverage, addressed by seed/edge enrichment, not tiebreaking.

### 13. [LOW] Parse/API failures (-1) are silently scored wrong and cluster in graph/hae modes
- **파일**: `graphrag/run_eval_local.py:75`  ·  **분류**: eval-validity
- **영향**: Real transparency wart but does NOT bias the headline conclusions: all graph-mode -1s fall in 그외 (non-처방형), zero in 처방형, so the 처방형 graph benefit and the selective-RAG-vs-plain delta are byte-identical with -1s excluded. Only the descriptive '그외 Δ' is inflated (~2.6%p for solar). Central claims survive.
- **수정**: Retry on parse failure, log failure type, and report parse-fail/API-fail/genuine-wrong separately; reduce max_tokens (1536) or force a bare-number grammar.

### 14. [LOW] recall@ctx metric uses an unbounded substring test, counting superstring prescriptions as hits
- **파일**: `bigse0u1/step4_eval.py:122`  ·  **분류**: eval-validity
- **영향**: Inflates only the secondary retrieval-recall diagnostic (not the primary accuracy). 17/85 처방형 gold names are proper substrings of other prescription names (육군자탕⊂가미육군자탕 etc.), so a retrieved variant chunk fires a false recall hit. Magnitude unmeasured (Chroma DB absent).
- **수정**: Score recall via exact chunk/graph 처방명 metadata match (as check_clinical_recall.py already does with $eq) instead of a raw `in` on ctx text.

### 15. [LOW] Split disjointness is unenforced: evaluate reads FULL, prepare_data --with-rationale reads EXPLAINED; divergence would leak eval into SFT train
- **파일**: `training/prepare_data.py:42`  ·  **분류**: leakage-fragility
- **영향**: NO current leakage (verified byte-identical splits, 0 overlap in both modes, and no ft variant is in any reported result), and add_explanations structurally re-aligns EXPLAINED to FULL. But the guarantee rides on that alignment with no assertion, and the split recipe is copy-pasted in 3 files. Latent risk for any future fine-tune.
- **수정**: Compute the split once from FULL, persist the val key set (source,교시,번호) or a hash, and have every consumer load and assert membership; or assert EXPLAINED is index-aligned with FULL before splitting.

### 16. [LOW] --with-rationale fine-tune would be unmeasurable: eval caps non-reasoning cloud output at 16 tokens while the target is a full 해설 before '정답: N', under a contradictory 'do not explain' instruction
- **파일**: `training/config.py:167`  ·  **분류**: eval-train-mismatch
- **영향**: Latent only — the OpenAI fine-tune path is never executed (empty *_FINETUNED_MODEL, no ft result files, no data/ dir, HANDOVER states FT deliberately abandoned). Corrupts no reported number, but would zero-out any future rationale FT and trains under a self-contradictory prompt.
- **수정**: In --with-rationale use COT_ANSWER_INSTRUCTION for the user turn, and give ft/CoT cloud models a larger eval budget (e.g. EVAL_MAX_TOKENS_COT=640).

### 17. [LOW] One law subject is spelled two ways ('보건의약 관계 법규' ×30 vs '보건의약관계법규' ×20), splitting per-subject accuracy
- **파일**: `dataset/한의학_문제_전체.jsonl:1`  ·  **분류**: per-subject-metric
- **영향**: Overall accuracy unaffected; only fragments the per-subject breakdown table (true n=50 reported as two half-buckets) in every track. Consistent across tracks and already manually merged (README.md notes the 30+20→50 merge), so it is a display/noise issue, not a result error.
- **수정**: Normalize 과목 labels (strip spaces / canonical map) before all by-subject grouping.

### 18. [LOW] save_results overwrite guard uses errored < n, so a mostly-errored partial-failure run still clobbers saved results
- **파일**: `training/evaluate.py:391`  ·  **분류**: robustness
- **영향**: Edge-triggered (daemon crash mid-run or sustained rate-limit storm); the all-errored case it targets is handled, accuracy math is correct, and a warning is printed. A near-fully-failed run can still overwrite with a deflated accuracy and no backup.
- **수정**: Require a minimum valid-response fraction (e.g. errored/n < 0.2) before allowing overwrite; write to a temp file + os.replace and keep a backup.

### 19. [LOW] wilson() CI is defined but never called, yet MULTIMODEL_REPORT.md header claims 'Wilson 95% CI'
- **파일**: `graphrag/aggregate_models.py:16`  ·  **분류**: reporting
- **영향**: Cosmetic research-integrity nit: the report advertises uncertainty quantification it does not display; accuracy numbers are unaffected (computed solely by acc()).
- **수정**: Either compute and print the Wilson interval in the tables or delete the dead function and the 'Wilson 95% CI' claim.

### 20. [LOW] OpenAI() constructed at module import in finetune.py and add_explanations.py
- **파일**: `training/finetune.py:31`  ·  **분류**: portability
- **영향**: Minor: with the repo's load_dotenv + non-empty .env placeholder key, `--help`/import do not actually crash in normal usage; only a fresh pre-setup clone with no key set is affected. No CI/tests exist to trip it. Also ignores OPENAI_BASE_URL.
- **수정**: Defer client construction into main()/worker using the lazy config.make_client pattern (which is OPENAI_BASE_URL-aware).

## 우선순위 (권장 작업)

- **[M] Fix the term-RAG answer leak and re-measure the RAG uplift** — The reported global-model RAG improvement is entirely produced by injecting the gold option's own KIOM definition on definitional MCQs; corrected, term-RAG is net-negative. This is the project's central claim and must be re-run with option-term injection suppressed (and definitional items reported separately).
- **[M] Unify answer parsing into one shared, robust module across all three tracks** — run_eval_local's circled-blind first-digit parser scores the headline multi-model report and penalizes verbose graph/hae modes; three divergent parsers make cross-track numbers non-comparable. Reuse training's parse_answer (with the tier fixes below) everywhere, add retry-on-unparseable, and log failure types separately.
- **[S] Harden parse_answer's regex ladder** — Tier 1 takes the first match (contradicting siblings and the 'prefer conclusion' intent); tier 2 lacks the digit-boundary and 번째 guards tier 4 already has; tier 4's unit blacklist omits g/mg/kg/ml/%/주/단계/가지. These silently misscore CoT and explanatory outputs. Ship with an adversarial-Korean regression test corpus as a gate.
- **[S] Externalize gold/subjects/questions into the repo (drop D:\ paths)** — Every graphrag/bigse0u1 accuracy number depends on ungit-versioned Windows-path files; the scripts don't run and the metrics can't be audited or regenerated from the checkout. Commit the ground-truth files (or derive them deterministically) and use repo-relative paths.
- **[S] Merge multi-edition prescription evidence in build_evidence.rx_by_name** — Last-wins drops indications for ~31% of prescriptions and feeds incomplete option evidence to 92% of option-matched graded questions. Union 주치/구성/계통 across all chunks per name, as sym_to_rx already does.
- **[M] Make the eval reproducible and fix comparability confounds in bigse0u1/graphrag** — bigse0u1 벡터 vs GraphRAG read different Chroma collections for 83% of questions (corpus swap confound); collection choice and seed-extraction differ from graphrag. Standardize the vector collection within a track and treat bigse0u1 as non-comparable/archival.
- **[S] Enforce train/val split disjointness via a persisted manifest** — Split safety is currently an unasserted coincidence between FULL and EXPLAINED with the recipe copy-pasted in 3 files; a future fine-tune could silently train on eval. Compute the split once, persist the val key set, and assert membership in every consumer.
- **[S] Normalize 과목 labels and tighten the save_results overwrite guard** — The two spellings of the law subject fragment per-subject reporting; and a mostly-errored run can clobber saved results with a deflated accuracy. Canonicalize subject strings before grouping, and require a minimum valid-response fraction plus atomic temp-file writes.

## 총평
The codebase is competent research plumbing with sound fundamentals — consistent 517/587/86 question sets, correct figure filtering, deterministic single-shot scoring, a verified-disjoint train/val split, and genuinely leakage-free graph/selective/해설 RAG — but two confirmed high-severity defects undercut its headline claims: the training term-RAG 'uplift' is entirely an answer-leak artifact (net-negative once the gold option's own definition is stopped from being injected), and the multi-model GraphRAG report is scored by a circled-numeral-blind parser whose failures cluster in the very RAG modes being evaluated. Most other findings are real but low-impact or latent (regex-ladder edge cases, non-optional Cypher matches on prescriptions that are never gold, an unexecuted fine-tune path, dead code). The biggest structural weakness is that three tracks reimplement parsing/prompting independently and the graphrag/bigse0u1 ground truth lives on hardcoded Windows paths absent from the repo, so those numbers are neither cross-comparable nor reproducible as shipped. Net: within-track base/CoT/vote results and the 처방형 graph benefit are trustworthy; the term-RAG uplift and any cross-track absolute comparison are not, and both need a re-run after fixing the leak and unifying the parser.
