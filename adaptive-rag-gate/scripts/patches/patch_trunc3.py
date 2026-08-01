# 절삭 수렴 강화: 반복 12→24, 축소율 0.75→0.55, 하한 100→30자.
# SOLAR/MedLLaMA2 는 한국어 토큰 팽창 때문에 25% 축소로는 12회 안에 4096 밑으로 못 내려간다.
import sys
P = "/workspace/bench/ktm-llm-benchmark/tkm_pipeline.py"
s = open(P, encoding="utf-8").read()
subs = [("for _i in range(12):", "for _i in range(24):"),
        ("_keep = int(len(_ev) * 0.75)", "_keep = int(len(_ev) * 0.55)"),
        ("if _keep < 100:", "if _keep < 30:"),
        ('raise RuntimeError("컨텍스트 폴백 12회 실패")',
         'raise RuntimeError("컨텍스트 폴백 24회 실패(근거 최소화 후에도 초과)")')]
for a, b in subs:
    if a not in s:
        print(f"!! 대상 없음: {a}"); sys.exit(1)
    s = s.replace(a, b)
open(P, "w", encoding="utf-8").write(s)
import ast; ast.parse(open(P, encoding="utf-8").read())
print("[patch3] 수렴 강화 적용 + 문법 검증 통과")
# 수렴 확인: 근거 4000자가 24회 안에 30자 밑으로 내려가는지
n, L = 0, 4000
while L >= 30 and n < 24:
    L = int(L * 0.55); n += 1
print(f"[patch3] 시뮬레이션: 4000자 -> {L}자 ({n}회) — 24회 내 수렴 {'OK' if L < 30 else '실패'}")
