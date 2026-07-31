#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# KTM-LLM — RunPod/컨테이너-Pod 환경용 셋업 (docker compose 없이).
#   RunPod Pod 은 Docker-in-Docker 를 막으므로 compose 를 못 쓴다. 대신 Pod 안에
#   Ollama 바이너리 + 파이썬 venv 를 직접 깔아 training/ · graphrag/ 트랙을 돌린다.
#   (Neo4j 가 필요한 bigse0u1/ 원본 파이프라인은 Vast Linux VM + docker compose 권장 — REMOTE_SETUP.md)
#
#   Pod 안에선 Ollama 가 localhost:11434 에 뜨므로 코드 기본값 그대로 동작(엔드포인트 오버라이드 불필요).
#
#   사용법(Pod 터미널에서, /workspace 에 코드를 둔 상태):
#     bash runpod_setup.sh                 # Ollama + 기본 2모델 + venv(기본 의존성)
#     bash runpod_setup.sh --all           # 6모델 전부(~30GB)
#     bash runpod_setup.sh --graphrag      # + torch/chromadb 등 GraphRAG 무거운 의존성
#     bash runpod_setup.sh --all --graphrag
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
info(){ echo "${GREEN}▶ $*${NC}"; }
warn(){ echo "${YELLOW}⚠ $*${NC}"; }

WITH_GRAPHRAG=0; MODELS="exaone3.5:7.8b qwen2.5:7b"
ALL="exaone3.5:7.8b qwen2.5:7b gemma2:9b llama3.1:8b mistral:7b solar:10.7b"
for a in "$@"; do
  case "$a" in
    --all) MODELS="$ALL" ;;
    --graphrag) WITH_GRAPHRAG=1 ;;
    *) echo "알 수 없는 인자: $a"; exit 1 ;;
  esac
done

# 영구 볼륨(RunPod 은 /workspace 가 퍼시스턴트). 모델·HF캐시를 여기 둬 재기동 시 재다운로드 방지.
VOL="${WORKSPACE_DIR:-/workspace}"
export OLLAMA_MODELS="$VOL/ollama-models"
export HF_HOME="$VOL/hf-cache"
mkdir -p "$OLLAMA_MODELS" "$HF_HOME"

# ── 파이썬 버전 확인(3.10+ 필요: `int | None` 타입힌트) ──────────────────────
PYBIN="$(command -v python3 || true)"
[[ -z "$PYBIN" ]] && { echo "python3 가 없습니다. PyTorch/CUDA 템플릿 Pod 을 쓰세요."; exit 1; }
PYV=$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
info "python3 = $PYV"
"$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)' \
  || { warn "Python 3.10+ 필요(현재 $PYV). 3.10+ 템플릿 Pod 을 쓰세요."; exit 1; }

# ── Ollama 설치 + 기동 ───────────────────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
  info "Ollama 설치"; curl -fsSL https://ollama.com/install.sh | sh
fi
if ! pgrep -x ollama >/dev/null 2>&1; then
  info "Ollama 데몬 기동 (OLLAMA_MODELS=$OLLAMA_MODELS)"
  OLLAMA_MODELS="$OLLAMA_MODELS" nohup ollama serve >"$VOL/ollama.log" 2>&1 &
fi
for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 2; done
info "Ollama OK"

info "모델 pull: $MODELS"
for m in $MODELS; do info "  pull $m"; ollama pull "$m"; done

# ── 파이썬 venv + 의존성 ─────────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
  info "venv 생성"
  if command -v uv >/dev/null 2>&1; then uv venv --python "$PYV" .venv; else "$PYBIN" -m venv .venv; fi
fi
PIP=".venv/bin/pip"; command -v uv >/dev/null 2>&1 && PIP="uv pip install --python .venv"
info "기본 의존성 설치"
$PIP install -r requirements.txt >/dev/null 2>&1 || .venv/bin/pip install -r requirements.txt
if [[ "$WITH_GRAPHRAG" == "1" ]]; then
  warn "GraphRAG 무거운 의존성 설치(torch/chromadb/sentence-transformers) — 수 분"
  ($PIP install -r graphrag/requirements_graphrag.txt numpy) || \
    .venv/bin/pip install -r graphrag/requirements_graphrag.txt numpy
fi

[[ -f .env ]] || { info ".env 생성"; cp .env.example .env; }

# ── 스모크 테스트: 실제 코드 경로로 3문항 추론(results/ 미변경) ──────────────
info "스모크 테스트 (3문항 실제 추론, results/ 는 건드리지 않음)"
.venv/bin/python - <<'PY' || warn "스모크 실패 — 'cat $VOL/ollama.log' 및 'ollama list' 확인"
import sys; sys.path.insert(0, "training")
import config, evaluate
from evaluate import ask
rows = evaluate.load_rows()[:3]
ok = errs = 0
for r in rows:
    try:
        p = ask(config.LOCAL_MODEL, config.PROVIDER_LOCAL, r)
        ok += int(p == r["answer"])
        print(f"  {r.get('과목','?')} {r.get('번호','?')}: pred={p} gold={r['answer']} {'OK' if p==r['answer'] else '.'}")
    except Exception as e:
        errs += 1; print("  ERROR:", e)
if errs == len(rows):
    print("전부 오류 — Ollama/모델 점검 필요"); sys.exit(1)
print(f"스모크 완료: 정답 {ok}/{len(rows)} · 파이프라인 정상")
PY

echo
info "완료. Pod 안 localhost 에 Ollama 가 떠 있어 코드 기본값 그대로 동작합니다:"
cat <<'EOF'

  .venv/bin/python training/evaluate.py --limit 5        # 스모크
  .venv/bin/python training/evaluate.py --all            # 메인(517)
  .venv/bin/python training/evaluate.py --all --cot --sc 5
  .venv/bin/python training/report.py                    # → results/report.md
  .venv/bin/python graphrag/run_all_models.py            # 6모델(그래프RAG 선택적, Neo4j 불요)

  # ⚠️ Neo4j 가 필요한 bigse0u1/ 원본 파이프라인은 Vast Linux VM + docker compose 권장(REMOTE_SETUP.md)
  # 산출물(results/·dataset/·*.jsonl)은 /workspace 볼륨에 쌓임 → git commit 또는 rsync 로 회수
EOF
