# 절삭 대상 수정: 프롬프트 꼬리(=문제·보기)가 아니라 [참고 자료] 블록만 줄인다.
# build_prompt 구조:  "[참고 자료]\n{근거}\n\n[문제]\n{지문}\n\n{보기}"
import sys

P = "/workspace/bench/ktm-llm-benchmark/tkm_pipeline.py"
src = open(P, encoding="utf-8").read()

BAD = '''            _keep = int(len(_u) * 0.8)
            if _keep < 200:
                raise
            _u = _u[:_keep] + "\\n...(근거 일부 생략)\\n"'''

GOOD = '''            _m = re.match(r"(?s)^\\[참고 자료\\]\\n(.*?)\\n\\n(\\[문제\\].*)$", _u)
            if not _m:
                raise                                   # RAG 프롬프트가 아니면 줄일 근거가 없음
            _ev, _rest = _m.group(1), _m.group(2)
            _keep = int(len(_ev) * 0.75)
            if _keep < 100:
                raise                                   # 근거를 다 없애도 안 들어가면 포기(기록됨)
            _ev = _ev[:_keep]
            _u = "[참고 자료]\\n" + _ev + "\\n...(근거 일부 생략)\\n\\n" + _rest'''

if BAD not in src:
    print("!! 수정 대상 블록을 찾지 못함 — 파일 상태 확인 필요"); sys.exit(1)

src = src.replace(BAD, GOOD)
open(P, "w", encoding="utf-8").write(src)
import ast
ast.parse(open(P, encoding="utf-8").read())
print("[fix] 근거 블록만 절삭하도록 수정 + 문법 검증 통과")

# 자기검증: 실제 프롬프트 형태로 절삭이 문제/보기를 보존하는지 확인
import re
u = "[참고 자료]\n" + ("근거내용 " * 200) + "\n\n[문제]\n이 환자의 진단은?\n\n1. 가\n2. 나\n3. 다\n4. 라"
m = re.match(r"(?s)^\[참고 자료\]\n(.*?)\n\n(\[문제\].*)$", u)
assert m, "정규식 매칭 실패"
ev, rest = m.group(1), m.group(2)
out = "[참고 자료]\n" + ev[: int(len(ev) * 0.75)] + "\n...(근거 일부 생략)\n\n" + rest
assert "이 환자의 진단은?" in out and "4. 라" in out, "문제/보기가 보존되지 않음"
assert len(out) < len(u), "축소되지 않음"
print(f"[fix] 자기검증 통과 — 문제·보기 보존, 길이 {len(u)} -> {len(out)}")
