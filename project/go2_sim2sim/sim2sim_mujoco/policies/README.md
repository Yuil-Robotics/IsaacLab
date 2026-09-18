# Yuil Dog Sim2Sim Policy Management

`sim2sim_mujoco/policies/` 디렉토리는 MuJoCo Sim2Sim 검증에 사용할 정책 모델(`.pt`, TorchScript)을 관리하는 전용 폴더입니다.

## 1. 정책 파일 구조

```text
sim2sim_mujoco/policies/
├── README.md                     # [본 문서] 정책 관리 안내
├── policy.pt                     # [기본 실행 정책] 인자 없이 실행할 때 로드됨
├── yuil_dog_flat_3500_latest.pt  # 최신 학습 모델 (3,500 iters, 보폭 확장 & 체고 0.33m)
└── yuil_dog_flat_3000_baseline.pt # 기존 기준 모델 (3,000 iters, 전달용 baseline)
```

## 2. 사용 방법

### 기본 정책 실행
별도 옵션 없이 실행하면 `policy.pt`가 자동으로 로드되며 `Ctrl+C`를 누를 때까지 계속 실행됩니다:
```bash
./sim2sim_mujoco/run.sh
```

실행 시간을 제한하려면 `--duration 20`처럼 초 단위 시간을 지정합니다.

### 특정 정책 실행 (파일명만 입력 가능)
`--policy` 인자에 전체 경로 대신 파일 이름만 넘겨도 이 폴더에서 자동으로 찾습니다:
```bash
# 베이스라인 3000 iter 모델 실행
./sim2sim_mujoco/run.sh --policy yuil_dog_flat_3000_baseline --duration 20

# 최신 3500 iter 모델 실행
./sim2sim_mujoco/run.sh --policy yuil_dog_flat_3500_latest
```
(.pt 확장자 생략도 지원: `--policy yuil_dog_flat_3000_baseline`)

### 보유 정책 목록 확인
```bash
./sim2sim_mujoco/run.sh --list
```

## 3. 새로운 정책 추가 및 교체 방법

새로 학습한 모델(예: `logs/rsl_rl/.../exported/policy.pt`)을 테스트하려면:
1. 이 폴더에 원하는 이름(예: `my_new_policy.pt`)으로 복사합니다:
   ```bash
   cp <학습경로>/exported/policy.pt sim2sim_mujoco/policies/my_new_policy.pt
   ```
2. 기본 정책으로 만들고 싶다면 `policy.pt`로 덮어씌웁니다:
   ```bash
   cp <학습경로>/exported/policy.pt sim2sim_mujoco/policies/policy.pt
   ```
