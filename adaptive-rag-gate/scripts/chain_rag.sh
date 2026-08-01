#!/bin/bash
# ktm_results_rag 생성 체인 — 의존성 설치 → 레포 클론 → FAISS 인덱스 → 9모델 RAG 벤치마크.
# nohup 으로 띄우므로 맥이 절전에 들어가도 계속 진행된다.
set -uo pipefail
echo "===== CHAIN_RAG START $(date '+%m-%d %H:%M:%S') ====="

VENV=/root/venv
BENCH=/workspace/bench/ktm-llm-benchmark

# ---------------------------------------------------------------- 1. 의존성
echo "[1/5] 의존성 설치 $(date '+%H:%M:%S')"
python3 -m venv "$VENV" 2>&1 | tail -2
"$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -2
"$VENV/bin/pip" install -q vllm litellm sentence-transformers faiss-cpu 2>&1 | tail -4
for m in vllm litellm sentence_transformers faiss; do
  printf "   %-22s %s\n" "$m" "$("$VENV/bin/python" -c "import $m; print('OK')" 2>&1 | tail -1)"
done
export PATH="$VENV/bin:$PATH"

# ---------------------------------------------------------------- 2. 레포
echo "[2/5] 레포 클론 $(date '+%H:%M:%S')"
rm -rf /workspace/bench
git clone -q https://github.com/roundycat/naccl-ktm-llm.git /workspace/bench 2>&1 | tail -2
cd "$BENCH" || { echo "!! 벤치마크 디렉토리 없음"; exit 1; }
echo "   KTM_data: $(ls KTM_data | wc -l)개 / km_rag: $(ls km_rag/*.jsonl 2>/dev/null | wc -l)개 청크파일"

# ---------------------------------------------------------------- 3. 논문 9모델로 제한
echo "[3/5] 모델 목록을 논문 9모델로 제한"
printf 'qwen2.5-7b\tQwen/Qwen2.5-7B-Instruct\n' > models/general.txt
printf 'exaone3.5-7.8b\tLGAI-EXAONE/EXAONE-3.5-7.8B-Instruct\t--trust-remote-code\n' >> models/general.txt
printf 'solar-10.7b\tupstage/SOLAR-10.7B-Instruct-v1.0\n' >> models/general.txt
printf 'mistral-7b\tmistralai/Mistral-7B-Instruct-v0.3\n' >> models/general.txt
printf 'huatuogpt-o1-7b\tFreedomIntelligence/HuatuoGPT-o1-7B\n' > models/domain/tcm.txt
printf 'medgemma-4b\tgoogle/medgemma-4b-it\n' > models/domain/western_med.txt
printf 'biomistral-7b\tBioMistral/BioMistral-7B\t--chat-template chat_templates/generic.jinja\n' >> models/domain/western_med.txt
printf 'medllama2-7b\tllSourcell/medllama2_7b\t--chat-template chat_templates/generic.jinja --max-model-len 4096\n' >> models/domain/western_med.txt
printf 'openbiollm-8b\taaditya/Llama3-OpenBioLLM-8B\t--chat-template chat_templates/generic.jinja\n' >> models/domain/western_med.txt
echo "   general 4 / tcm 1 / western_med 4 = 9모델"

# 가중치 캐시 유지(워크스페이스 여유 396T) → 재실행 시 다운로드 생략
sed -i 's/^CLEANUP_HF_CACHE=true/CLEANUP_HF_CACHE=false/' run_all_models_rag.sh
grep -m1 '^CLEANUP_HF_CACHE' run_all_models_rag.sh | sed 's/^/   /'

# ---------------------------------------------------------------- 3b. 컨텍스트 초과 폴백 패치
# medllama2(컨텍스트 4096)는 RAG 근거가 붙으면 입력+출력이 4096을 넘겨 죽는다.
# 초과 에러일 때만 출력 예산을 줄여 재시도하도록 call_llm 에 폴백을 넣는다.
python3 - <<'PYEOF'
import re
P = "/workspace/bench/ktm-llm-benchmark/tkm_pipeline.py"
src = open(P, encoding="utf-8").read()
if "컨텍스트 초과" in src:
    print("   [patch] 이미 적용됨"); raise SystemExit
m = re.search(r"\n(\s*)resp = completion\(\s*\n(.*?)\n\s*\)\n\s*return resp\[", src, re.S)
if not m:
    print("   [patch] !! 대상 코드 못 찾음 — 수동 확인 필요"); raise SystemExit(1)
old = m.group(0)
new = '''
    _msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    def _call(_mt):
        return completion(model=model, messages=_msgs, temperature=temperature,
                          max_tokens=_mt, api_base=api_base, api_key=api_key)

    _mt, resp = 1500, None
    for _ in range(8):
        try:
            resp = _call(_mt); break
        except Exception as _e:
            if "maximum context length" not in str(_e):
                raise
            _s = str(_e)
            _ctx = re.search(r"maximum context length is (\\d+)", _s)
            _inp = re.search(r"at least (\\d+) input tokens", _s)
            _new = (int(_ctx.group(1)) - int(_inp.group(1)) - 128) if (_ctx and _inp) else None
            if _new is None or _new >= _mt:
                _new = _mt // 2
            _mt = max(64, _new)
            print(f"[call_llm] 컨텍스트 초과 -> max_tokens={_mt} 재시도", flush=True)
    if resp is None:
        raise RuntimeError("컨텍스트 축소 재시도 8회 실패")
    return resp['''
open(P, "w", encoding="utf-8").write(src.replace(old, new))
import ast; ast.parse(open(P, encoding="utf-8").read())
print("   [patch] 컨텍스트 폴백 적용 + 문법 검증 통과")
PYEOF

# ---------------------------------------------------------------- 4. FAISS 인덱스
export HF_HOME=/workspace/dhf
export HF_TOKEN="$(cat /workspace/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0
echo "[4/5] FAISS 인덱스 생성 $(date '+%H:%M:%S')"
"$VENV/bin/python" rag_index.py 2>&1 | tail -12
if [[ ! -f km_rag/index/index.faiss ]]; then
  echo "!! 인덱스 생성 실패 — 중단"; exit 1
fi
echo "   인덱스: $(ls -la km_rag/index/index.faiss | awk '{print $5}') bytes"

# ---------------------------------------------------------------- 5. RAG 벤치마크
echo "[5/5] RAG 벤치마크 시작 $(date '+%H:%M:%S')  (9모델 x 4연도, 수 시간 예상)"
chmod +x run_all_models_rag.sh
./run_all_models_rag.sh
echo "===== CHAIN_RAG DONE $(date '+%m-%d %H:%M:%S') ====="
echo "결과: $(find ktm_results_rag -name '*.json' 2>/dev/null | wc -l)개 / 기대 36개"
