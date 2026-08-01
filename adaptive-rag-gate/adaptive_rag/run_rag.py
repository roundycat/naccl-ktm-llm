# -*- coding: utf-8 -*-
"""어댑티브 RAG 벤치마크 러너.

팀원의 tkm_pipeline.py(5단계 프롬프트 벤치마크)를 그대로 재사용하되,
build_prompt를 감싸(wrap) 각 문항에 '우리 분류기가 관련하다고 판정한 지식'을 주입한다.
→ 팀원 파이프라인 파일은 수정하지 않음(모듈 함수만 런타임에 교체).

같은 스크립트로 두 조건을 돌려 공정 비교:
  --rag 없이 : baseline (팀원 파이프라인과 동일, 주입 없음)
  --rag      : + 어댑티브 RAG (BGE-m3 검색 → 우리 BERT 게이트 → 관련 지식 주입)

실행 예 (박스, vLLM 서버가 :8000에 exaone/qwen 서빙 중일 때):
  # baseline
  python run_rag.py --model openai/qwen2.5-7b --api-base http://localhost:8000/v1 \
      --api-key sk-dummy --data ../baseline/KTM_data/2025.json --stage 4 \
      --output out_qwen_2025_stage4_base.json
  # + 어댑티브 RAG
  python run_rag.py --model openai/qwen2.5-7b --api-base http://localhost:8000/v1 \
      --api-key sk-dummy --data ../baseline/KTM_data/2025.json --stage 4 \
      --rag --corpus corpus.jsonl --classifier /workspace/bert_clf/best \
      --output out_qwen_2025_stage4_rag.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 팀원 벤치마크(tkm_pipeline.py) 경로 — 기본값은 adaptive-rag-gate/의 형제 디렉터리
# (저장소 최상위에 baseline/ 와 adaptive-rag-gate/ 가 나란히 있는 실제 구조),
# 환경변수로 재정의 가능.
BENCH_DIR = os.environ.get(
    "TKM_BENCH_DIR",
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "baseline"))
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, HERE)

import tkm_pipeline as tk          # noqa: E402  팀원 파이프라인
from adaptive_rag import AdaptiveRAG  # noqa: E402

_RAG: AdaptiveRAG | None = None
_INJECTIONS: dict | None = None      # 사전계산 주입 {질문id: 주입블록} (vLLM과 분류기 분리용)
_orig_build_prompt = tk.build_prompt
_inject_log: list[dict] = []

# 답변 디코딩을 greedy(temperature=0)로 강제 → 결정론적.
# 팀원 파이프라인 기본값은 temp=1.0(랜덤 샘플링)이라 단일 시행이 ±3%p 흔들려
# base vs RAG의 Δ가 노이즈에 묻힌다. temp 0으로 고정하면 두 조건이 '주입 여부'로만
# 달라져 Δ가 순수 분류기 효과가 된다(번역은 원래 temp 0이라 영향 없음).
_orig_call_llm = tk.call_llm


def _call_llm_greedy(model, system, user, temperature=0.0, api_base=None, api_key=None):
    return _orig_call_llm(model, system, user, temperature=0.0, api_base=api_base, api_key=api_key)


# 적용은 main()에서 조건부: stage<5는 greedy(temp0), stage 5는 self-consistency라 원래 temp 유지.


def build_prompt_with_rag(q, stage, model_for_translation, api_base=None, api_key=None):
    """원래 build_prompt 결과에 '참고 지식'을 앞에 덧붙인다(검색은 한국어 원문으로).

    두 모드: (a) _INJECTIONS 사전계산 사전에서 조회(분류기 미로드), (b) _RAG 실시간 게이트.
    """
    system, user = _orig_build_prompt(q, stage, model_for_translation,
                                      api_base=api_base, api_key=api_key)
    inj = ""
    if _INJECTIONS is not None:
        inj = _INJECTIONS.get(q.id, "")
        _inject_log.append({"id": q.id, "n_injected": 1 if inj else 0})
    elif _RAG is not None:
        query = q.question_kr + " " + " ".join(q.choices_kr)
        inj, docs = _RAG.augment(query)
        _inject_log.append({"id": q.id, "n_injected": len(docs),
                            "docs": [{"id": d["id"], "type": d["type"],
                                     "rel": round(d["relevance"], 3)} for d in docs]})
    if inj:
        user = inj + "\n\n" + user
    return system, user


def _ensure_translations(questions, cache_path, model, api_base, api_key, workers):
    """stage>=3 영어번역을 파일 캐시로 재사용. 있으면 로드(번역 스킵), 없으면 만들어 저장."""
    from concurrent.futures import ThreadPoolExecutor
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        hit = 0
        for q in questions:
            c = cache.get(q.id)
            if c:
                q._question_en, q._choices_en = c["q_en"], c["choices_en"]
                hit += 1
        print(f"[번역캐시] {hit}/{len(questions)} 로드 → 번역 스킵", flush=True)
        return
    print(f"[번역캐시] 없음 → {len(questions)}문항 번역 생성(1회) …", flush=True)

    def _tr(q):
        tk.get_translated_question(q, model, api_base=api_base, api_key=api_key)
        return q
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(_tr, questions))
    cache = {q.id: {"q_en": q._question_en, "choices_en": q._choices_en} for q in questions}
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[번역캐시] 저장 → {cache_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="어댑티브 RAG 벤치마크(팀원 tkm_pipeline 위에 게이트 주입)")
    # tkm_pipeline과 동일한 인자
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--stage", type=int, default=4, choices=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--n-trials", type=int, default=7)
    ap.add_argument("--translation-model", default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="평가 문항 수 제한(스모크 테스트용)")
    ap.add_argument("--translation-cache", default=None,
                    help="stage>=3 영어번역을 이 파일에 캐시(없으면 만들고 저장, 있으면 로드→번역 스킵). "
                         "같은 모델의 baseline/RAG가 번역을 재사용해 시간 절약.")
    ap.add_argument("--output", default=None)
    # 어댑티브 RAG 인자
    ap.add_argument("--rag", action="store_true", help="어댑티브 RAG 주입 켜기(없으면 baseline)")
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus.jsonl"))
    ap.add_argument("--classifier", default="/workspace/bert_clf/best",
                    help="관련성 분류기 경로(BERT 크로스인코더)")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--rag-top-k", type=int, default=20, help="1차 리트리버 후보 수")
    ap.add_argument("--rag-threshold", type=float, default=0.5, help="게이트 통과 확률 임계값")
    ap.add_argument("--rag-max-inject", type=int, default=5, help="주입 문서 최대 개수")
    ap.add_argument("--injections-file", default=None,
                    help="사전계산된 주입 사전 JSON({질문id:주입블록}). 주면 분류기/BGE 미로드 → "
                         "vLLM이 GPU 전체 사용 가능(도메인 모델용).")
    args = ap.parse_args()

    if args.stage < 5:
        tk.call_llm = _call_llm_greedy   # stage<5: greedy(temp0). stage 5: self-consistency라 원래 temp 유지.

    global _RAG, _INJECTIONS
    if args.rag and args.injections_file:
        _INJECTIONS = json.load(open(args.injections_file, encoding="utf-8"))
        n = sum(1 for v in _INJECTIONS.values() if v)
        print(f"[RAG] 사전계산 주입 로드 {args.injections_file}: {n}/{len(_INJECTIONS)} 주입 "
              f"(분류기 미로드)", flush=True)
        tk.build_prompt = build_prompt_with_rag
    elif args.rag:
        print(f"[RAG] 분류기={args.classifier} | top_k={args.rag_top_k} "
              f"thr={args.rag_threshold} max={args.rag_max_inject}", flush=True)
        _RAG = AdaptiveRAG(args.corpus, args.classifier, embed_model=args.embed_model,
                           top_k=args.rag_top_k, threshold=args.rag_threshold,
                           max_inject=args.rag_max_inject)
        tk.build_prompt = build_prompt_with_rag   # 런타임 교체(파일 수정 X)

    questions = tk.load_questions(args.data)
    if args.limit:
        questions = questions[: args.limit]

    # stage>=3 영어번역 캐시: 같은 모델의 baseline/RAG가 번역을 공유해 재번역 낭비 제거.
    if args.stage >= 3 and args.translation_cache:
        _ensure_translations(questions, args.translation_cache,
                             args.translation_model or args.model,
                             args.api_base, args.api_key, args.max_workers)

    summary = tk.evaluate(
        questions, model=args.model, stage=args.stage, n_trials=args.n_trials,
        translation_model=args.translation_model, api_base=args.api_base,
        api_key=args.api_key, max_workers=args.max_workers,
    )
    summary["rag"] = bool(args.rag)
    if args.rag:
        n_inj = sum(1 for e in _inject_log if e["n_injected"] > 0)
        summary["rag_stats"] = {
            "classifier": args.classifier, "top_k": args.rag_top_k,
            "threshold": args.rag_threshold, "max_inject": args.rag_max_inject,
            "questions_with_injection": n_inj, "total_questions": len(questions),
            "avg_injected": round(sum(e["n_injected"] for e in _inject_log) / max(1, len(_inject_log)), 2),
            "inject_log": _inject_log,
        }

    print("\n=== SUMMARY ===")
    print(f"Model    : {summary['model']}  | RAG: {'ON' if args.rag else 'OFF(baseline)'}")
    print(f"Stage    : {summary['stage']}  | 문항 {summary['n_questions']}")
    print(f"Accuracy : {summary['accuracy']*100:.2f}%")
    if args.rag:
        rs = summary["rag_stats"]
        print(f"주입된 문항: {rs['questions_with_injection']}/{rs['total_questions']} "
              f"(평균 {rs['avg_injected']}개 주입)")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
