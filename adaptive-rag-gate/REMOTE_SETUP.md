# 원격 GPU 서버 실험 환경 (용량·연산 걱정 없이)

> 노트북 디스크(99%)에서 벗어나 **원격 GPU 박스**에서 실험을 돌리기 위한 가이드.
> 용량·연산은 서버가 감당하고, git 멈춤·느린 import·모델 못 받음이 **뿌리 원인(디스크)째로** 사라진다.
> 전체 실험·결과·로드맵은 [CONTINUE_HERE.md](CONTINUE_HERE.md), 배경은 [HANDOVER.md](HANDOVER.md)·[README.md](README.md).

---

## ⚠️ 먼저 알아둘 것 — 배포 모드 2가지

RunPod/Vast 의 "**Pod**"는 그 자체가 컨테이너라 **`docker compose` 를 못 돌린다**(Docker-in-Docker 금지).
그래서 두 갈래가 있고, **하려는 실험에 따라 고르면 된다:**

| 모드 | 어디서 | 무엇을 돌리나 | 준비 |
|---|---|---|---|
| **B. Pod-native (권장 시작점)** | **RunPod Pod** (가장 싸고 즉시) | `training/`(용어RAG) + `graphrag/`(선택적RAG, **Neo4j 불요**) 전부 | `bash runpod_setup.sh` (compose 없이 Ollama+venv 직접) |
| **A. 풀 compose 스택** | **Vast.ai Linux VM** (compose 지원) | 위 + **`bigse0u1/` 원본 GraphRAG(Neo4j+Chroma)** 까지 | `bash docker/bootstrap.sh` (Ollama+Neo4j+app 3서비스) |

> 핵심: **Neo4j 가 필요한 건 `bigse0u1/` 원본 파이프라인 하나뿐**이다. 논문의 메인 결과(로컬 vs 글로벌,
> 용어RAG, CoT/SC, 선택적 GraphRAG)는 전부 **Ollama만으로** 돌아가므로 → **모드 B(RunPod)로 시작**하고,
> bigse0u1 Neo4j 재현이 필요할 때만 모드 A(Vast VM)로 가면 된다.

GPU 요구: 7~9B 모델 Q4 기준 모델당 VRAM ~5~6GB, 순차 로드라 **단일 16~24GB(RTX 4090/3090)면 6모델 전부 OK**.
비용은 실험 시간만큼: **전체 재현 $3~10, 개발 한 주 $10~20** (자세히 아래 §4). 연세대 랩 GPU가 있으면 $0.

---

## 모드 B — RunPod Pod (권장: 가장 싸고 간단, compose 없이)

### 1) 가입·크레딧
1. [runpod.io](https://www.runpod.io) 가입 → **Billing 에서 크레딧 $10** 충전(시작엔 충분).

### 2) Pod 생성
1. Console → **Pods → Deploy**.
2. **GPU**: `RTX 4090` (또는 3090) 선택 — 커뮤니티 클라우드가 저렴(~$0.34/hr).
3. **Template**: `RunPod PyTorch 2.x` (Ubuntu + CUDA + Python 3.10+, Docker/NVIDIA 불필요 — 우린 compose 안 씀).
4. **Disk**: Container Disk 20GB + **Volume Disk 60~80GB**(→ `/workspace` 에 마운트, **영구 보존**). 모델 30GB+산출물 고려.
5. **Deploy On-Demand** (초 단위 과금).

### 3) 접속 → 코드 → 셋업(한 줄)
```bash
# Pod 의 Web Terminal 또는 SSH 접속 후:
cd /workspace
git clone https://github.com/roundycat/KTM-LLM.git && cd KTM-LLM && git checkout main_1
bash runpod_setup.sh            # Ollama 설치+모델pull+venv (기본 2모델)
#   6모델 전부:        bash runpod_setup.sh --all
#   +GraphRAG 무거운 스택: bash runpod_setup.sh --graphrag
```
> `runpod_setup.sh` 는 모델·HF캐시를 `/workspace`(영구 볼륨)에 두므로 Pod 을 껐다 켜도 재다운로드 안 함.

### 4) 실험 실행
Pod 안 `localhost:11434` 에 Ollama 가 떠 있어 **코드 기본값 그대로** 동작(엔드포인트 설정 불필요):
```bash
.venv/bin/python training/evaluate.py --limit 5        # 스모크
.venv/bin/python training/evaluate.py --all            # 메인(517)
.venv/bin/python training/evaluate.py --all --cot --sc 5
.venv/bin/python training/report.py                    # → results/report.md
.venv/bin/python graphrag/run_all_models.py            # 선택적 GraphRAG 6모델(Neo4j 불요)
```
산출물은 `/workspace/KTM-LLM/results/` 등에 쌓임 → `git commit` 또는 `rsync` 로 노트북에 회수.

### 5) 비용 관리 (중요)
- 안 쓸 땐 Console 에서 Pod **Stop** → GPU 과금 정지, **`/workspace` 볼륨은 유지**(월 소액 저장소 요금).
- 완전 종료: **Terminate** → 볼륨까지 삭제($0). 다음엔 모델만 다시 받으면 됨.

---

## 모드 A — Vast.ai Linux VM (풀 compose 스택, bigse0u1 Neo4j 포함)

Vast 는 **Linux VM 인스턴스**에서 docker compose 를 지원한다(일반 Docker 인스턴스는 불가 — 반드시 **VM** 선택).

### 1) 가입·크레딧
1. [vast.ai](https://vast.ai) 가입 → 크레딧 충전.

### 2) VM 인스턴스 렌트
1. Console → 검색 필터에서 **"Linux VM"** 유형 선택(중요 — compose 지원은 VM에서만).
2. **GPU** `RTX 4090`(~$0.3/hr, 인터럽터블이면 더 저렴), **Disk 60~80GB**.
3. Rent → SSH 로 접속.

### 3) Docker 확인 → 코드 → 스택 기동(한 줄)
```bash
# VM 접속 후 (대부분 Docker+NVIDIA toolkit 포함. 없으면 아래로 설치)
docker compose version || (curl -fsSL https://get.docker.com | sh)
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # GPU 확인
#   NVIDIA runtime 미설정 시: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

git clone https://github.com/roundycat/KTM-LLM.git && cd KTM-LLM && git checkout main_1
bash docker/bootstrap.sh        # ollama+neo4j+app 빌드·기동 + 모델pull + 헬스체크
#   6모델 전부: bash docker/bootstrap.sh --all
```

### 4) 실험 실행 (app 컨테이너 안에서, 엔드포인트 자동 연결)
```bash
docker compose exec app python training/evaluate.py --all
docker compose exec app python training/report.py
# 원본 GraphRAG(Neo4j+Chroma):
docker compose exec -w /workspace/bigse0u1 app python step2_build_vectordb.py   # BGE-m3 → chroma
docker compose exec -w /workspace/bigse0u1 app python step1_load_neo4j.py       # 그래프 적재
docker compose exec -w /workspace/bigse0u1 app python step4_eval.py --rx-only --k 20
docker compose exec app bash    # 셸
```

### 5) 정지/삭제
```bash
docker compose down       # 컨테이너 정지(모델·그래프 볼륨 보존)
docker compose down -v    # 볼륨까지 삭제
```
그리고 Vast Console 에서 인스턴스 Stop(볼륨 유지) 또는 Destroy(완전 삭제).

---

## 4. 비용 요약

| 무엇 | 대략 GPU-시간 | @$0.35/hr |
|---|---|---|
| 메인 평가 1회(`--all`) + 리포트 | ~0.5h | ~$0.2 |
| 전체 재현(6모델 + CoT/SC + vote + GraphRAG) | ~10~20h | **$3~10** |
| 개발 한 주(끄고 켜며 ~30~40h) | 30~40h | **$10~20** |

- 시간 잡아먹는 건 **CoT+self-consistency**(517×5샘플)와 6모델 벤치마크. 나머진 몇 분~수십 분.
- **⚠️ 저장소 요금**: GPU 를 꺼도 볼륨은 과금(RunPod ~$0.07~0.10/GB·월, Vast ~$0.10~0.20/GB·월).
  80GB 유지 시 월 $5~10. 실험 끝나고 볼륨 삭제하면 $0.
- 아끼기: 인터럽터블/스팟(Vast, 절반값), 필요한 모델만(`--models`/기본 2모델), 안 쓸 때 Stop.

---

## 5. 트러블슈팅

| 증상 | 대응 |
|---|---|
| RunPod 에서 `docker compose` 안 됨 | **정상**(Pod=컨테이너, DinD 금지). 모드 B(`runpod_setup.sh`) 사용. compose 는 Vast **VM**에서만 |
| `nvidia`/GPU 예약 실패 | nvidia-container-toolkit 미설치 → §모드A 3). GPU 없이 CPU로만 돌리려면 compose 의 `ollama` `deploy:` 블록 삭제 |
| Neo4j 인증 실패 | `.env` 의 `NEO4J_PW`(기본 `ktmpassword`)와 일치해야. 'neo4j' 는 Neo4j 가 거부 |
| `run_all_models.py` 전부 skip | `graphrag/eval_out/*.json` 캐시 존재 → 재계산하려면 삭제(단 `aggregate_models.py` 는 `D:\` 하드코딩 이슈, CONTINUE_HERE.md §8-2) |
| 디스크 참 | 모델 골라 받기, `docker volume prune`, 볼륨 크기 ↑ |

---

## 부록: Docker 불가 HPC/공용 클러스터

컨테이너를 못 쓰는 학교 공용 클러스터면 모드 B 와 동일하게 **Ollama 바이너리 + venv 직접**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_MODELS=/scratch/$USER/ollama-models   # 큰 볼륨으로
ollama serve & 
bash runpod_setup.sh          # 같은 스크립트가 그대로 동작(compose 미사용)
```
Neo4j 도 무설치로 쓰려면 Apptainer/Singularity(`apptainer run docker://neo4j:5-community`) 활용.
