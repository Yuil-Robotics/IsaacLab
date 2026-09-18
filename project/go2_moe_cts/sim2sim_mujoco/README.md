# Yuil Dog MoE-CTS MuJoCo 실행

이 폴더는 기존 `go2_sim2sim/sim2sim_mujoco`의 Rust GUI와 Python 정책 클라이언트를 복사한 독립 실행 구성입니다. 형제 프로젝트의 코드, 모델, 정책, 빌드 결과를 참조하지 않습니다. Rust 도구와 상위 IsaacLab Python 환경은 사용합니다.

프로젝트 루트에서 실행합니다.

```bash
./sim2sim_mujoco/run.sh
```

MuJoCo GUI가 열리고 `policies/policy.pt`의 student가 50 Hz로 동작합니다. 기본 명령은 `(0, 0, 0)`이며 GUI의 vx/vy/wz 슬라이더로 움직입니다. 범위는 각각 ±1 m/s, ±0.5 m/s, ±1 rad/s입니다. Ctrl+C로 정책과 이 명령이 시작한 서버를 종료합니다. 기본 포트는 `127.0.0.1:7001`이며 이미 사용 중이면 다른 서버에 연결하지 않고 종료합니다.

기본 정책은 로컬 학습 run `2026-09-18_12-24-32_rough_gui/model_1500.pt`에서 export했습니다. 정확한 출처와 iteration은 `policies/policy.json`에 기록됩니다. `model_*.pt` 학습 체크포인트는 직접 실행하지 않고 먼저 변환합니다.

```bash
# 체크포인트 경로는 프로젝트 루트 기준; 절대 경로도 가능
./sim2sim_mujoco/export_policy.sh \
  --checkpoint logs/rsl_rl/yuil_dog_rough_moe_cts/2026-09-18_12-24-32_rough_gui/model_1500.pt \
  --name student_1500

./sim2sim_mujoco/run.sh --policy student_1500
./sim2sim_mujoco/run.sh --list
./sim2sim_mujoco/run.sh --policy student_1500 --duration 20
```

`--name`을 생략하면 기본 `policy.pt`와 `policy.json`을 갱신합니다. Export는 시뮬레이터를 열지 않으며, 체크포인트에 저장된 네트워크 설정/가중치와 현재 프로젝트의 Yuil Dog 배포 설정을 사용합니다. 로봇·관측 설정을 바꾼 과거 체크포인트는 해당 학습 설정과의 일치를 별도로 확인해야 합니다.

기존 IsaacLab play에서 만든 `exported/policy.pt`도 `--policy`에 절대 경로로 지정할 수 있습니다. 같은 폴더의 `policy_contract.json`이 필요합니다. 관측 스케일이 다른 평지 PPO는 이 CTS 서버용 정책으로 사용할 수 없습니다.

서버와 클라이언트를 따로 실행할 수도 있습니다.

```bash
# 터미널 1: 기존 GUI 기능과 CLI 유지
./sim2sim_mujoco/run_server_gui.sh --command 0.5 0 0
# 터미널 2
./sim2sim_mujoco/run_policy.sh --policy student_1500
```

결과 JSON 기본 위치는 **프로젝트의 `logs/sim2sim_mujoco/<실행시각>/policy_result.json`**입니다. `--output`으로 변경할 수 있습니다. 이 JSON은 실행 시간과 action 통계이며 험지 성능 평가 보고서는 아닙니다.

## 유지한 GUI 기능

- 로봇 위치·몸체 높이·실제 속도·action RMS·episode·낙상 reset 횟수
- 명령 슬라이더, 로봇 reset, 시간 제한 없는 실행
- 발 접촉·높이·보행 주기·접촉/착지 충격량, 관절 토크 그래프
- 발 궤적과 표시 길이/좌표 모드
- 외력 방향 버튼, 크기와 지속 시간 조절
- 카메라 추종, 자유 시점, 방향별 preset, 차체 고정 시점 및 회전 추종

## 폴더 역할

| 경로 | 역할 |
|---|---|
| `src/` | Rust 물리 제어, TCP 서버, GUI와 상태/그래프 |
| `scripts/` | Python 정책 클라이언트, 체크포인트 export, URDF 변환 |
| `model/urdf`, `model/MJCF`, `model/meshes` | 로컬 로봇 모델과 STL |
| `policies/` | TorchScript student 및 정책별 JSON 계약/출처 |
| `crates/mujoco-rs/` | 기존 GUI 기능을 포함한 수정된 Rust 의존성 및 라이선스 |
| `.mujoco/` | 로컬 MuJoCo 라이브러리; 삭제 시 Cargo가 다시 다운로드 |
| `target/` | 이 프로젝트에서 독립적으로 생성한 빌드 캐시 |

관측은 45차원 × 10프레임, term-major/과거→현재의 450차원입니다. 각속도는 0.25, 관절속도는 0.05를 곱하며 원시 관측을 ±100으로 먼저 clip합니다. Action은 ±100 clip 후 `기본 관절각 + 0.25 × action`, 관절별 목표각 제한과 기존 PD/모터 토크 제한을 적용합니다. 물리 dt 0.005 s, decimation 4입니다. Teacher와 critic은 실행에 필요 없습니다.

MuJoCo 지형은 기존 구성대로 평지입니다. 낙상 판정도 기존 몸체 하단 높이 0.18 m 기준이므로 IsaacLab의 base contact 종료·험지 커리큘럼 생존율과 직접 비교하지 않습니다. 이 구성은 시뮬레이터 간 정책 동작 확인용입니다.

```bash
cd sim2sim_mujoco
cargo test --locked
cargo clippy --locked -- -D warnings
# 모델 재생성에는 Python mujoco 패키지가 필요합니다.
../../../isaaclab.sh -p scripts/convert_urdf_to_mjcf.py
```

검증 기록: 프로젝트 `logs/sim2sim_mujoco/validation_2026-09-18/`. 정지/전진 0.5 m/s 명령으로 각각 20초, 1,000 step 실행에서 낙상 reset 0회였습니다. 전진 시 x 이동량은 약 10.31 m였습니다. 이 짧은 평지 확인만으로 험지 성능을 보장하지 않습니다.
