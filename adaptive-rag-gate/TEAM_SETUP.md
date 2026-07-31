# 팀 협업 가이드 (NACCL 버전 · 공유 박스)

> 이 저장소는 한의학 국시 LLM 연구의 **NACCL 팀 정본**입니다. 코드리뷰 수정(누수·파서·경로)이
> 반영된 버전이에요. 팀은 **공유 GPU 박스 1대 + GitHub**로 함께 작업합니다.
> 실험 셋업은 [REMOTE_SETUP.md](REMOTE_SETUP.md), 코드 상태는 [CODE_REVIEW.md](CODE_REVIEW.md),
> 전체 배경은 [CONTINUE_HERE.md](CONTINUE_HERE.md)·[HANDOVER.md](HANDOVER.md) 참고.

---

## 구조 한눈에

```
GitHub (private, roundycat/naccl-ktm-llm)   ← 코드 정본. 각자 브랜치로 작업
        │ clone / pull / push (branch)
        ▼
공유 GPU 박스 1대 (RunPod Pod + Network Volume 80GB)
  /workspace/naccl-ktm-llm   ← 팀 공용 작업 트리
  /workspace/ollama-models   ← 모델 6개(한 번만 다운로드, 공유)
  /workspace/hf-cache        ← BGE-m3 등
  Ollama(:11434) · (선택) Neo4j — 박스에서 공유
```

- **GPU는 1개** → 무거운 평가(CoT+SC, 6모델)는 **동시에 돌리지 말고 순번**. `ollama ps`로 사용 중인지 확인.
- **결과 충돌 방지**: 각자 브랜치에 커밋하거나 `results/<이름>/`에 저장. `evaluate.py` 저장가드가 실패런 클로버링은 막지만, 정상 병행 실행은 사람이 조율.

---

## 팀원: 접속 방법 (2가지만)

### 1) GitHub 접근
1. owner(roundycat)에게 **GitHub 아이디** 전달 → collaborator 초대 수락.
2. clone (private repo라 인증 필요):
   ```bash
   gh repo clone roundycat/naccl-ktm-llm      # gh 로그인돼 있으면 가장 쉬움
   # 또는 SSH:  git clone git@github.com:roundycat/naccl-ktm-llm.git
   ```

### 2) 공유 박스 SSH 접근
1. 본인 SSH 공개키 생성(없으면):
   ```bash
   ssh-keygen -t ed25519 -C "your-name"      # ~/.ssh/id_ed25519.pub 생성
   cat ~/.ssh/id_ed25519.pub                 # 이 한 줄을 owner에게 전달
   ```
2. owner가 박스에 등록하면, 받은 접속 명령으로 들어갑니다:
   ```bash
   ssh root@<박스IP> -p <PORT>                # owner가 알려줌
   cd /workspace/naccl-ktm-llm
   ```

---

## 팀원: 작업 흐름

```bash
cd /workspace/naccl-ktm-llm
git pull                                   # 최신 코드
git switch -c yourname/실험이름             # 개인 브랜치 (main 직접 커밋 X)

# 실험 (박스 안 localhost에 Ollama가 떠 있어 코드 기본값 그대로)
.venv/bin/python training/evaluate.py --limit 5           # 스모크
.venv/bin/python training/evaluate.py --all              # 메인(GPU 순번 확인 후)
.venv/bin/python tests/test_answer_parse.py              # 파서 회귀테스트

git add -A && git commit -m "..." && git push -u origin yourname/실험이름
# GitHub에서 PR → 리뷰 후 main 병합
```

**규칙**
- `main`은 보호 — 직접 커밋 말고 **브랜치 + PR**.
- 무거운 평가 전 **박스에 누가 GPU 쓰는지 확인**(`nvidia-smi`, `ollama ps`).
- `.env`는 공유 안 함(박스에 owner가 1회 설정). 개인 API키가 필요하면 개인 브랜치/셸에서만.
- 재측정 주의: 코드리뷰 수정으로 **기존 수치는 무효** → 새로 돌린 결과만 신뢰([CODE_REVIEW.md](CODE_REVIEW.md)).

---

## Owner(roundycat) 세팅 체크리스트

**GitHub**
- [ ] Settings → Collaborators → 팀원 GitHub 아이디 초대
- [ ] (권장) main 브랜치 보호 규칙(PR 필수)

**공유 박스**
- [ ] RunPod Pod 생성: RTX 4090 + **Network Volume 80GB(/workspace)** + SSH over exposed TCP
- [ ] `git clone` + `bash runpod_setup.sh --all` (Ollama·6모델·venv·스모크)
- [ ] 팀원 공개키를 박스에 추가:
      ```bash
      echo "ssh-ed25519 AAAA...팀원키" >> ~/.ssh/authorized_keys
      ```
      (여러 명이면 각자 키를 한 줄씩 추가. 또는 RunPod 계정 SSH Keys에 팀 키 모두 등록)
- [ ] 팀에 **박스 접속 주소(root@IP -p PORT)** 공유
- [ ] `.env` 1회 설정(필요 시 API키)

**비용 관리**: 안 쓸 때 Pod Stop(볼륨 유지, 모델 보존). Network Volume은 GPU 꺼도 월정액(~80GB $6~8/월).
