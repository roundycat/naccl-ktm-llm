# -*- coding: utf-8 -*-
"""persubj 지반 파일 생성(코드리뷰 #7) — 하드코딩 D:\\tmp\\persubj\\ 를 대체.

레포에 동봉된 dataset/한의학_문제.jsonl(그림 제외 517)에서 결정론적으로 유도한다:
  graphrag/data/persubj/gold.json            idx → 정답(1~5)
  graphrag/data/persubj/subjects.json        과목 → [idx]
  graphrag/data/persubj/questions_noanswer.jsonl   idx·question·options·과목·has_figure (정답 제외)

보기 순서는 데이터셋 자연순(셔플 없음) → 완전 재현 가능. 과목 라벨 공백 정규화(#17).

⚠️ 이 파일들을 새로 만들면 보기 순서가 원본(미동봉 셔플)과 달라진다. 따라서
   기존 eval_out/grageval_*.json(원본 순서 예측)은 무효 → derive 후 반드시
   build_evidence.py → build_hae_evidence.py → run_eval_local.py(전 모델) → aggregate_models.py
   순으로 재실행해야 정합적이다.

사용:  python graphrag/derive_persubj.py   (PERSUBJ_DIR / KTM_DATASET 로 경로 재정의 가능)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
# 전체(587, 과목·has_figure 메타 포함)에서 그림문항을 제외해 텍스트 517을 얻는다.
SRC = os.environ.get("KTM_DATASET", os.path.join(HERE, "..", "dataset", "한의학_문제_전체.jsonl"))
OUT = os.environ.get("PERSUBJ_DIR", os.path.join(HERE, "data", "persubj"))


def norm_subj(s: str) -> str:
    return "".join((s or "미상").split())   # '보건의약 관계 법규' ↔ '보건의약관계법규' 병합(#17)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not r.get("has_figure", False)]   # 그림 제외 → 텍스트 517
    gold, subjects, qn = {}, {}, []
    for idx, r in enumerate(rows):
        gold[str(idx)] = r["answer"]
        subj = norm_subj(r.get("과목"))
        subjects.setdefault(subj, []).append(idx)
        qn.append({"idx": idx, "question": r["question"], "options": r["options"],
                   "과목": subj, "has_figure": bool(r.get("has_figure", False))})

    json.dump(gold, open(os.path.join(OUT, "gold.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(subjects, open(os.path.join(OUT, "subjects.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    with open(os.path.join(OUT, "questions_noanswer.jsonl"), "w", encoding="utf-8") as f:
        for q in qn:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"생성 완료: {OUT}")
    print(f"  gold {len(gold)} · subjects {len(subjects)} · questions {len(qn)}")
    print("  다음: build_evidence.py → build_hae_evidence.py → run_eval_local.py(전 모델) → aggregate_models.py")


if __name__ == "__main__":
    main()
