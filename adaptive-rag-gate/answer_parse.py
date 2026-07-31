# -*- coding: utf-8 -*-
"""공유 정답 파서 — 세 트랙(training/·graphrag/·bigse0u1/)이 동일 로직으로 채점하도록 단일화.

기존엔 트랙마다 파서가 달라(training 4단, graphrag/run_eval_local 원문자-불능, step4 first-digit)
같은 모델 출력이 다르게 채점되고 원문자(①~⑤) 답이 오답 처리됐다(코드리뷰 #2/#3/#4).
이 모듈이 유일한 정답 추출기다. 무거운 의존성 없음(re 만) → 어느 트랙에서든 import 가능.

추출 우선순위(설명형·CoT 출력 대응):
  1) "정답/답 … N" 결론 표기 — 마지막 유효 매치(자기수정 대응), 부정/추측 인접 표현은 배제
  2) "N번" — 숫자경계 + 서수('번째') 배제
  3) 원문자 ①~⑤ — 마지막 등장(단, ①..⑤ 오름차순 나열=보기 에코는 무시)
  4) 독립 숫자 1~5 — 다자리/범위(1~5)/단위·카운터(회·세·mg·번째 등) 배제

parse_answer 는 실패 시 None(기권/미응답)을 돌려준다. -1 규약을 쓰는 호출부는 parse_choice 사용.
"""
from __future__ import annotations
import re

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}


def _to_num(ch: str) -> int:
    return CIRCLED[ch] if ch in CIRCLED else int(ch)


# 정답 앵커 뒤에 오면 그 숫자를 '결론'으로 인정하지 않는 표현(부정·추측·오답 지목).
_ANCHOR_SKIP = re.compile(
    r"^\s*(?:[은는이가]\s*)?(?:아니|아닌|틀|오답)"   # 3이 아니라 / 3은 아님 / 틀린
    r"|^\s*번?\s*(?:처럼|같|보이|보임)"              # 3번처럼 / 3 같아 보임
)

# 독립 숫자 뒤에 오면 정답이 아닌 단위·카운터·서수.
_UNIT_AHEAD = (r"(?:[0-9~\-]"
               r"|\s*(?:회|번째|세|개|명|년|월|일|시간|시|분|초|차|도|가지|단계|층|군|형|위|과목|"
               r"g|mg|kg|ml|cc|주|퍼센트|%|배|톤|cm|mm|mL))")


def parse_answer(text: str | None) -> int | None:
    """모델 출력에서 정답 번호(1~5)를 견고하게 추출. 실패 시 None."""
    if not text:
        return None

    # 1) "정답/답 … N" 결론 표기. 조사 다양체 허용, 부정·추측 인접 표현 배제, 마지막 유효 매치.
    chosen = None
    for m in re.finditer(r"(?:정답|답)\s*[은는이가을를로과와도의\s]*[:：]?\s*([1-5①-⑤])", text):
        after = text[m.end():m.end() + 8]
        if _ANCHOR_SKIP.match(after):
            continue
        chosen = _to_num(m.group(1))
    if chosen is not None:
        return chosen

    # 2) "N번" — 앞 숫자경계, 뒤 '째'(서수) 배제. 마지막 우선.
    bm = re.findall(r"(?<![0-9])([1-5])\s*번(?!째)", text)
    if bm:
        return int(bm[-1])

    # 3) 원문자 ①~⑤. 단 ①②③④⑤가 오름차순으로 죽 나열되면 '보기 에코'로 보고 무시.
    present = [(text.find(ch), num) for ch, num in CIRCLED.items() if ch in text]
    if present:
        asc = sorted(present)
        is_option_echo = len(asc) >= 4 and [n for _, n in asc] == sorted(n for _, n in asc)
        if not is_option_echo:
            last = [(text.rfind(ch), num) for ch, num in CIRCLED.items() if ch in text]
            return max(last)[1]

    # 4) 독립 숫자 1~5(마지막). 다자리·범위·단위·서수 배제.
    nums = re.findall(r"(?<![0-9~\-])([1-5])(?!" + _UNIT_AHEAD + r")", text)
    return int(nums[-1]) if nums else None


def parse_choice(text: str | None) -> int:
    """-1 규약 호출부용 래퍼(파싱 실패=-1)."""
    v = parse_answer(text)
    return v if v is not None else -1
