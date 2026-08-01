"""실험결과_전체.docx — 지금까지 확보한 모든 수치를 표로 정리. 표준 라이브러리만 사용."""
import zipfile
from xml.sax.saxutils import escape

OUT = "/Users/jeonghamin/Desktop/naccl/model/KTM-LLM/stage5_final/실험결과_전체.docx"
FONT = "맑은 고딕"


def run(t, b=False, i=False, sz=None, col=None):
    r = ""
    if b: r += "<w:b/>"
    if i: r += "<w:i/>"
    if sz: r += f'<w:sz w:val="{sz*2}"/><w:szCs w:val="{sz*2}"/>'
    if col: r += f'<w:color w:val="{col}"/>'
    r = f"<w:rPr>{r}</w:rPr>" if r else ""
    return f'<w:r>{r}<w:t xml:space="preserve">{escape(str(t))}</w:t></w:r>'


def para(parts, style=None, after=110, ind=None):
    if isinstance(parts, str): parts = [(parts, {})]
    runs = "".join(run(t, **o) for t, o in parts)
    p = ""
    if style: p += f'<w:pStyle w:val="{style}"/>'
    if ind: p += f'<w:ind w:left="{ind}"/>'
    p += f'<w:spacing w:after="{after}" w:line="276" w:lineRule="auto"/>'
    return f"<w:p><w:pPr>{p}</w:pPr>{runs}</w:p>"


def tbl(rows, widths, bold_cols=(), note=None):
    bd = "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="B0B0B0"/>'
                 for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders>{bd}</w:tblBorders>'
           f'<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
    for ri, r in enumerate(rows):
        hdr = ri == 0
        cells = []
        for ci, c in enumerate(r):
            shd = '<w:shd w:val="clear" w:fill="EEF3F8"/>' if hdr else ""
            jc = "" if ci == 0 else '<w:jc w:val="center"/>'
            bold = hdr or (ci in bold_cols and not hdr)
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shd}'
                f'<w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr>{jc}'
                f'<w:spacing w:after="10" w:line="240" w:lineRule="auto"/></w:pPr>'
                f'{run(c, b=bold, sz=8)}</w:p></w:tc>')
        trpr = "<w:trPr><w:tblHeader/></w:trPr>" if hdr else ""
        out.append(f"<w:tr>{trpr}{''.join(cells)}</w:tr>")
    out.append("</w:tbl>")
    s = "".join(out)
    if note:
        s += para([(note, {"i": True, "sz": 8, "col": "666666"})], after=160)
    else:
        s += '<w:p><w:pPr><w:spacing w:after="140"/></w:pPr></w:p>'
    return s


B = []
B.append(para("한의사 국시 LLM 벤치마크 — 실험 결과 전체", style="Title"))
B.append(para([("2022–2025 · 1,138문항 · 12과목 · 9개 오픈 모델 · stage 5 (한자병기+영어번역+CoT+SC3)",
                {"i": True, "col": "555555", "sz": 10})], after=60))
B.append(para([("작성 시점의 확정 수치만 수록. 진행 중인 항목은 [진행중]으로 표시.",
                {"i": True, "col": "8A6D3B", "sz": 9})], after=240))

# ── 4.2 Knowledge Sources
B.append(para("4.2 지식 자원", style="Heading1"))
B.append(para("표 1. 지식그래프 규모 (정본 graphrag/ 기준, CLAUDE.md §3 검증값과 일치)"))
B.append(tbl([["구분", "유형", "개수"],
              ["노드 (계 7,692)", "처방", "3,094"], ["", "증상", "3,873"], ["", "약재", "645"],
              ["", "증상지표", "54"], ["", "장부", "13"], ["", "변증", "8"], ["", "계통", "5"],
              ["엣지 (계 39,996)", "구성", "25,565"], ["", "주치", "8,858"], ["", "계통", "3,094"],
              ["", "팔강귀속", "1,447"], ["", "귀경", "972"], ["", "지표", "54"], ["", "포함", "6"]],
             [2600, 2600, 2000],
             note="※ bigse0u1 브랜치 사본은 7,716노드/40,159엣지(처방 +24)로 드리프트되어 있어 사용하지 않음."))

B.append(para("표 2. 그래프 품질 검증 (§4.2.3)"))
B.append(tbl([["검사 항목", "결과", "비율"],
              ["고아 엣지 (미존재 노드 참조)", "0", "0.000%"],
              ["자기루프", "0", "0.000%"],
              ["중복 엣지", "0", "0.000%"],
              ["유형 규칙 위반", "54", "0.135%"],
              ["구조 오류 합계", "54 / 39,996", "0.135%"],
              ["동명 분열 (같은 이름·다른 id)", "666개 이름 / 2,101 노드", "27.3%"]],
             [3600, 2600, 1600],
             note="※ 유형 위반 54건은 모두 '지표' 엣지가 증상지표가 아닌 변증을 가리키는 단일 패턴. "
                  "설계상 정상이면 구조 오류율은 0%. 동명 분열은 처방 562개 이름이 최다(소청룡탕 24개 id)."))

B.append(para("표 3. 누수 통제 — 근거가 정답 보기를 노출하는가 (§4.2.4)"))
B.append(tbl([["연도", "정답보기 평균", "정답 완전포함", "오답보기 평균", "오답 완전포함"],
              ["2022", "0.148", "12 / 294", "0.142", "40 / 882"],
              ["2023", "0.208", "23 / 280", "0.163", "45 / 840"],
              ["2024", "0.181", "17 / 288", "0.163", "47 / 864"],
              ["2025", "0.164", "7 / 276", "0.159", "39 / 1,104"],
              ["평균", "0.175", "59 / 1,138", "0.157", "171 / 3,690"]],
             [1400, 2000, 1900, 2000, 1900],
             note="※ 정답과 오답의 containment 차이가 +0.019에 불과 → 검색 근거가 정답을 직접 노출하지 않음."))

# ── 5.1 Run-to-Run Variance
B.append(para("5.1 실행 간 분산 (Run-to-Run Variance)", style="Heading1"))
B.append(para("표 4. 동일 조건 두 실행의 차이 — 노이즈 바닥 측정"))
rows = [["모델", "run A", "run B", "|Δ|", "예측 불일치 문항", "불일치율", "McNemar p"]]
for r in [("Qwen2.5-7B", "49.38%", "49.38%", "0.00p", "359", "31.5%", "1.000"),
          ("HuatuoGPT-o1-7B", "48.33%", "49.30%", "0.97p", "358", "31.5%", "0.497"),
          ("EXAONE-3.5-7.8B", "47.28%", "47.80%", "0.53p", "388", "34.1%", "0.729"),
          ("MedGemma-4B", "42.09%", "41.21%", "0.88p", "398", "35.0%", "0.548"),
          ("SOLAR-10.7B", "38.66%", "38.93%", "0.26p", "446", "39.2%", "0.895"),
          ("Mistral-7B", "30.67%", "30.84%", "0.18p", "543", "47.7%", "0.950"),
          ("OpenBioLLM-8B", "27.07%", "25.57%", "1.49p", "503", "44.2%", "0.293"),
          ("BioMistral-7B", "21.09%", "19.51%", "1.58p", "693", "60.9%", "0.303"),
          ("MedLLaMA2-7B", "19.42%", "19.42%", "0.00p", "540", "47.5%", "1.000"),
          ("평균", "—", "—", "0.65p", "4,228 / 10,242", "41.3%", "전부 n.s.")]:
    rows.append(list(r))
B.append(tbl(rows, [2100, 1300, 1300, 1100, 2100, 1300, 1400], bold_cols=(3,),
             note="※ run A = 팀 baseline(ktm_results), run B = 동일 스택 재생성(ktm_results_plain_v2). "
                  "조건이 동일하므로 차이는 표집 분산(SC, temperature>0)에서만 발생. "
                  "노이즈 바닥 평균 0.65%p (최대 1.58%p), 9모델 전부 유의하지 않음(p ≥ 0.29). "
                  "→ 제안 게이트의 +1.03%p (p=0.0009)는 이 바닥을 상회한다. "
                  "문항 단위로는 41.3%가 뒤바뀌므로 개별 사례 해석은 주의가 필요."))

# ── 5.2 Baseline
B.append(para("5.2 베이스라인과 측정 오차", style="Heading1"))
B.append(para("표 5. 파싱 실패율과 chance 보정 (§6.5 — raw / 응답분 / 보정 3종 보고)"))
rows = [["모델", "raw", "무응답%", "응답분", "chance 보정"]]
for m, a, b_, c, d in [("Qwen2.5-7B", "49.38%", "0.00%", "49.38%", "33.59%"),
                       ("HuatuoGPT-o1-7B", "48.33%", "0.18%", "48.42%", "32.20%"),
                       ("EXAONE-3.5-7.8B", "47.28%", "0.00%", "47.28%", "30.82%"),
                       ("MedGemma-4B", "42.09%", "0.00%", "42.09%", "24.02%"),
                       ("SOLAR-10.7B", "38.66%", "0.70%", "38.94%", "19.52%"),
                       ("Mistral-7B", "30.67%", "2.81%", "31.56%", "9.03%"),
                       ("OpenBioLLM-8B", "27.07%", "5.89%", "28.76%", "4.30%"),
                       ("BioMistral-7B", "21.09%", "8.17%", "22.97%", "−3.54%"),
                       ("MedLLaMA2-7B", "19.42%", "18.28%", "23.76%", "−5.73%")]:
    rows.append([m, a, b_, c, d])
B.append(tbl(rows, [2400, 1500, 1500, 1500, 1700], bold_cols=(1,),
             note="※ CLAUDE.md §3 검증표와 소수점까지 일치. BioMistral·MedLLaMA2는 보정 후 음수 = 무작위 이하."))

B.append(para("표 6. 2025년 형식 효과 — 4지(862문항) vs 5지(276문항) (§4.1)"))
rows = [["모델", "4지 raw", "5지 raw", "차이", "4지 보정", "5지 보정", "보정 차이"]]
for r in [("Qwen2.5-7B", "50.12%", "47.10%", "−3.01p", "33.49%", "33.88%", "+0.39p"),
          ("HuatuoGPT-o1-7B", "49.07%", "46.01%", "−3.06p", "32.10%", "32.52%", "+0.42p"),
          ("EXAONE-3.5-7.8B", "49.30%", "40.94%", "−8.36p", "32.41%", "26.18%", "−6.23p"),
          ("MedGemma-4B", "43.74%", "36.96%", "−6.78p", "24.98%", "21.20%", "−3.79p"),
          ("SOLAR-10.7B", "40.60%", "32.61%", "−7.99p", "20.80%", "15.76%", "−5.04p"),
          ("Mistral-7B", "31.79%", "27.17%", "−4.61p", "9.05%", "8.97%", "−0.08p"),
          ("OpenBioLLM-8B", "26.91%", "27.54%", "+0.62p", "2.55%", "9.42%", "+6.87p"),
          ("BioMistral-7B", "20.77%", "22.10%", "+1.34p", "−5.65%", "2.63%", "+8.27p"),
          ("MedLLaMA2-7B", "21.23%", "13.77%", "−7.46p", "−5.03%", "−7.79%", "−2.76p")]:
    rows.append(list(r))
B.append(tbl(rows, [2100, 1300, 1300, 1200, 1300, 1300, 1300], bold_cols=(6,),
             note="※ 원시값은 최대 −8.4%p 하락하지만 무작위 기준선(25%→20%) 보정 후 상위 모델은 오히려 상승. "
                  "2025 하락은 난이도가 아니라 보기 수 형식 효과."))

# ── 5.3
B.append(para("5.3 검색 증강의 양날 효과", style="Heading1"))
B.append(para("표 7. 상관 분석 (9모델, Pearson)"))
B.append(tbl([["관계", "r", "p", "출처 대조"],
              ["검색 증강 델타 vs 기저 정확도", "−0.744", "0.0215", "CLAUDE.md §3 (−0.744, 0.022) 일치"],
              ["파싱 실패율 vs 기저 정확도", "−0.848", "0.0039", "본 연구 신규"],
              ["SC 델타 vs 파싱 실패율", "[진행중]", "[진행중]", "stage 4 필요"]],
             [3400, 1300, 1300, 3000],
             note="※ 기저 정확도가 낮은 모델일수록 검색 증강 이득이 크다(MedLLaMA2 +9.23p, EXAONE −3.87p)."))

# ── 5.5
B.append(para("5.5 그래프 커버리지", style="Heading1"))
B.append(tbl([["연도", "문항", "그래프 근거 보유", "비율"],
              ["2022", "294", "225", "76.5%"], ["2023", "280", "217", "77.5%"],
              ["2024", "288", "204", "70.8%"], ["2025", "276", "215", "77.9%"],
              ["합계", "1,138", "861", "75.7%"]],
             [1600, 1600, 2400, 1600],
             note="※ 평균 시드 2.24개. seed 0(그래프 근거 없음) 문항 = 24.3%. 벡터 근거는 전 문항 존재."))

# ── 5.6
B.append(para("5.6 선택적 게이팅의 효과", style="Heading1"))
B.append(para("표 8. 후보집합 확장에 따른 성능 (LOYO held-out, 9모델 평균)"))
B.append(tbl([["설정", "후보 집합", "정확도", "vs plain", "vs 무게이트"],
              ["plain", "—", "36.00%", "—", "−2.77p"],
              ["무게이트 RAG", "always-inject", "38.77%", "+2.77p", "—"],
              ["T1 (기존)", "plain, gate(T,C)", "36.51%", "+0.51p", "−2.26p"],
              ["T2", "+ always-inject", "39.50%", "+3.50p", "+0.73p"],
              ["T3 (제안)", "+ GraphRAG 계열, δ=0.005", "39.80%", "+3.80p", "+1.03p"]],
             [1700, 2900, 1500, 1500, 1600], bold_cols=(2,),
             note="※ 게이트 vs 무게이트 직접 검정: +1.03%p, McNemar p = 0.0009, 부트스트랩 95% CI [+0.42, +1.62]. "
                  "불일치쌍 989개 중 게이트만 정답 547 / 무게이트만 정답 442."))

B.append(para("표 9. 모델별 게이트 성능 (T3, δ=0.005)"))
rows = [["모델", "plain", "무게이트", "T3 게이트", "Δ vs plain", "McNemar p", "95% CI", "Holm 보정"]]
for r in [("Qwen2.5-7B", "49.4%", "48.6%", "50.5%", "+1.14p", "0.198", "[−0.44, +2.72]", "비유의"),
          ("HuatuoGPT-o1-7B", "48.3%", "50.9%", "49.3%", "+0.97p", "0.548", "[−1.93, +3.95]", "비유의"),
          ("EXAONE-3.5-7.8B", "47.3%", "43.4%", "48.9%", "+1.67p", "0.0066", "[+0.53, +2.90]", "유의"),
          ("MedGemma-4B", "42.1%", "40.2%", "43.6%", "+1.49p", "0.089", "[−0.18, +3.25]", "비유의"),
          ("SOLAR-10.7B", "38.7%", "42.5%", "42.5%", "+3.87p", "0.034", "[+0.44, +7.29]", "비유의"),
          ("Mistral-7B", "30.7%", "38.8%", "38.8%", "+8.17p", "<0.0001", "[+4.92, +11.51]", "유의"),
          ("OpenBioLLM-8B", "27.1%", "31.6%", "31.6%", "+4.57p", "0.013", "[+0.79, +8.00]", "비유의"),
          ("BioMistral-7B", "21.1%", "24.2%", "24.2%", "+3.08p", "0.093", "[−0.44, +6.41]", "비유의"),
          ("MedLLaMA2-7B", "19.4%", "28.6%", "28.6%", "+9.23p", "<0.0001", "[+5.89, +12.30]", "유의"),
          ("평균", "36.00%", "38.77%", "39.80%", "+3.80p", "—", "—", "3/9 유의")]:
    rows.append(list(r))
B.append(tbl(rows, [1900, 1100, 1200, 1300, 1300, 1300, 1700, 1200], bold_cols=(3,),
             note="※ Holm 보정 후 9모델 중 3개 유의. 풀링 검정(게이트 vs 무게이트)은 단일 검정이라 보정 대상 아님(p=0.0009)."))

# ── 5.7
B.append(para("5.7 절제 실험", style="Heading1"))
B.append(para("표 10. 신호 정교화 vs 후보 확장 — 논문 핵심 주장의 근거"))
B.append(tbl([["설정", "게이팅 신호", "후보 수", "정확도", "vs plain"],
              ["plain", "—", "1", "36.00%", "—"],
              ["신호 T만", "관련도", "22", "36.45%", "+0.45p"],
              ["신호 C만", "자기일치도", "4", "36.72%", "+0.72p"],
              ["T×C [T1]", "관련도 × SC", "64", "36.74%", "+0.74p"],
              ["T×C + always [T2]", "관련도 × SC", "65", "39.45%", "+3.45p"],
              ["T×C + graph [T3]", "관련도 × SC", "129", "39.80%", "+3.80p"],
              ["T만 + always", "관련도", "23", "39.19%", "+3.19p"]],
             [2300, 1900, 1300, 1500, 1500], bold_cols=(3,),
             note="※ 신호 정교화(T만→T×C, 후보 고정) +0.29p vs 후보 확장(후보 1개 추가) +2.70p — 약 9배 차이. "
                  "신호를 가장 단순히 두고 후보만 넓혀도 전체 이득의 84%를 회수."))

B.append(para("표 11. 선행연구 대리 비교 — 생성 전 결정 vs 생성 후 선택 (§7.2)"))
rows = [["모델", "plain", "생성 전 게이트 (관련도만)", "생성 후 선택 (관련도×SC)", "차이"]]
for r in [("Qwen2.5-7B", "49.4%", "49.4%", "50.5%", "+1.14p"),
          ("EXAONE-3.5-7.8B", "47.3%", "47.0%", "48.9%", "+1.93p"),
          ("MedGemma-4B", "42.1%", "42.0%", "43.6%", "+1.58p"),
          ("HuatuoGPT-o1-7B", "48.3%", "49.2%", "49.3%", "+0.09p"),
          ("SOLAR·Mistral·OpenBio·BioMistral·MedLLaMA2", "—", "동일", "동일", "+0.00p"),
          ("평균", "36.00%", "39.27%", "39.80%", "+0.53p")]:
    rows.append(list(r))
B.append(tbl(rows, [3300, 1200, 2100, 2100, 1300], bold_cols=(3,),
             note="※ SC 결합의 이득은 기저 정확도가 높은 4개 모델에 국한. 하위 5개는 always-inject를 선택해 차이 없음."))

# ── 5.8
B.append(para("5.8 오류 분석", style="Heading1"))
B.append(para("표 12. 구제·훼손 분해와 Oracle 천장"))
rows = [["모델", "plain", "게이트", "구제", "훼손", "순증", "구제율", "훼손율", "Oracle", "남은 여지"]]
for r in [("Qwen2.5-7B", "49.4%", "50.5%", "50", "37", "+13", "8.7%", "6.6%", "69.2%", "+18.6p"),
          ("HuatuoGPT-o1-7B", "48.3%", "49.3%", "144", "133", "+11", "24.5%", "24.2%", "71.0%", "+21.7p"),
          ("EXAONE-3.5-7.8B", "47.3%", "48.9%", "32", "13", "+19", "5.3%", "2.4%", "67.7%", "+18.7p"),
          ("MedGemma-4B", "42.1%", "43.6%", "53", "36", "+17", "8.0%", "7.5%", "61.5%", "+17.9p"),
          ("SOLAR-10.7B", "38.7%", "42.5%", "229", "185", "+44", "32.8%", "42.0%", "67.2%", "+24.7p"),
          ("Mistral-7B", "30.7%", "38.8%", "245", "152", "+93", "31.1%", "43.6%", "60.4%", "+21.5p"),
          ("OpenBioLLM-8B", "27.1%", "31.6%", "238", "186", "+52", "28.7%", "60.4%", "57.1%", "+25.5p"),
          ("BioMistral-7B", "21.1%", "24.2%", "222", "187", "+35", "24.7%", "77.9%", "52.3%", "+28.1p"),
          ("MedLLaMA2-7B", "19.4%", "28.6%", "234", "129", "+105", "25.5%", "58.4%", "48.2%", "+19.6p"),
          ("합계", "—", "—", "1,447", "1,058", "+389", "—", "—", "—", "—")]:
    rows.append(list(r))
B.append(tbl(rows, [1800, 1000, 1000, 900, 900, 900, 1000, 1000, 1000, 1200], bold_cols=(5,),
             note="※ 구제:훼손 = 1.37:1 (문항 10,242). 훼손율이 기저 정확도와 강하게 반비례 "
                  "(EXAONE 2.4% vs BioMistral 77.9%). Oracle 대비 평균 +21.8%p 여지가 남음."))

B.append(para("표 13. 과목별 게이트 효과 (9모델 합산)"))
rows = [["과목", "문항 수", "plain", "게이트", "Δ"]]
for r in [("한방생리학", "64", "34.7%", "41.1%", "+6.42p"), ("신경정신과학", "59", "47.5%", "53.5%", "+6.03p"),
          ("예방의학", "89", "45.4%", "51.2%", "+5.74p"), ("보건의약관계법규", "80", "34.0%", "39.4%", "+5.42p"),
          ("안이비인후과학 †", "28", "37.7%", "42.9%", "+5.16p"), ("외과학 †", "28", "27.8%", "31.3%", "+3.57p"),
          ("부인과학", "113", "40.1%", "43.7%", "+3.54p"), ("내과학1", "270", "37.9%", "41.2%", "+3.29p"),
          ("침구학", "143", "32.6%", "35.7%", "+3.19p"), ("본초학", "50", "22.2%", "24.9%", "+2.67p"),
          ("소아과학", "88", "40.4%", "42.9%", "+2.53p"), ("내과학2", "126", "25.9%", "28.0%", "+2.12p")]:
    rows.append(list(r))
B.append(tbl(rows, [2600, 1500, 1500, 1500, 1500], bold_cols=(4,),
             note="† n<40 소표본 — 1문항이 3.6%p 이상 움직이므로 강조 금지(§6.6). 12과목 전부 상승."))

B.append(para("표 14. 구제·훼손 문항의 분포"))
B.append(tbl([["구분", "상위 과목", "보기 4지", "보기 5지", "합계"],
              ["구제", "내과학1 329 · 침구학 184 · 부인과학 151", "1,116", "331", "1,447"],
              ["훼손", "내과학1 249 · 침구학 143 · 내과학2 121", "775", "283", "1,058"],
              ["비율", "—", "1.44 : 1", "1.17 : 1", "1.37 : 1"]],
             [1200, 4200, 1300, 1300, 1200]))

# ── 진행 중
B.append(para("진행 중 항목", style="Heading1"))
B.append(tbl([["절", "필요 데이터", "상태"],
              ["5.1 Run-to-Run Variance", "ktm_results_plain_v2 (짝맞춤 baseline)", "30/36 진행 중"],
              ["5.4 Graph vs Dense", "ktm_results_graphrag (stage 5)", "0/36 대기"],
              ["5.4 프롬프팅 교란 분리", "ktm_results_graphrag_stage0", "0/36 대기"],
              ["5.3 SC 델타 상관", "ktm_results_stage4", "0/36 대기"],
              ["규모 사다리", "Qwen 14B/32B AWQ", "0 대기"],
              ["km_rag RAG 완결", "HuatuoGPT-o1-7B 4건", "재시도 예약"]],
             [2600, 4000, 2000],
             note="※ HuatuoGPT는 가중치 JSON 손상(JSONDecodeError)으로 서버 기동 실패. 캐시 삭제 후 재시도 예약됨. "
                  "CLAUDE.md 규칙에 따라 실패를 기록하며 임의 제외하지 않음."))

BODY = "".join(B)
NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
DOC = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {NS}><w:body>{BODY}'
       f'<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
       f'<w:pgMar w:top="1000" w:right="900" w:bottom="1000" w:left="900"/></w:sectPr></w:body></w:document>')


def st(sid, name, sz, bold=False, before=0, after=120, outline=None, col=None):
    o = f'<w:outlineLvl w:val="{outline}"/>' if outline is not None else ""
    c = f'<w:color w:val="{col}"/>' if col else ""
    return (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'<w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="{before}" w:after="{after}"/>{o}</w:pPr>'
            f'<w:rPr>{"<w:b/>" if bold else ""}{c}<w:sz w:val="{sz*2}"/></w:rPr></w:style>')


STYLES = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {NS}>'
          f'<w:docDefaults><w:rPrDefault><w:rPr>'
          f'<w:rFonts w:ascii="{FONT}" w:eastAsia="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}"/>'
          f'<w:sz w:val="18"/></w:rPr></w:rPrDefault></w:docDefaults>'
          f'<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
          f'<w:pPr><w:spacing w:after="110" w:line="276" w:lineRule="auto"/></w:pPr>'
          f'<w:rPr><w:sz w:val="18"/></w:rPr></w:style>'
          + st("Title", "Title", 16, True, 0, 60)
          + st("Heading1", "heading 1", 12, True, 300, 130, 0, "1F3864") + '</w:styles>')

CT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="xml" ContentType="application/xml"/>'
      '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
      '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
DRELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", CT)
    z.writestr("_rels/.rels", RELS)
    z.writestr("word/_rels/document.xml.rels", DRELS)
    z.writestr("word/styles.xml", STYLES)
    z.writestr("word/document.xml", DOC)
with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None
    n = len(z.read("word/document.xml"))
print(f"생성: {OUT}")
print(f"document.xml {n:,} bytes / 표 14개 / A4 가로")
