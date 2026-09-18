# Yuil Dog Rust MuJoCo Sim2Sim

`Isaac-Velocity-Flat-Yuil-Dog-RobotLab-Eval-v0`에서 학습한 4족 보행 정책을
Isaac Lab과 독립된 MuJoCo 태스크에서 검증한다. 실행기, GUI, 물리 루프, 정책 계약은
`robotis_hand_cube/sim2sim-mujoco`와 같은 Rust 구조다. Python은 TorchScript 추론과
URDF 자산 변환에만 사용한다.

기본 정책은 `policies/policy.pt` (최신 3,500 iters 모델)이며, `policies/` 폴더에 모델을 넣어두고 파일명만으로 간편하게 교체 실행할 수 있다.

## 구조

```text
sim2sim_mujoco/
├── Cargo.toml
├── Cargo.lock
├── build.rs
├── src/
│   ├── main.rs                 # Rust GUI, CLI, 물리 스레드
│   ├── server.rs               # 12-D/450-D lock-step TCP 서버
│   ├── simulation.rs           # 보행 태스크, 관측, DC 모터 모델
│   └── state.rs                # 스레드 공유 상태
├── model/
│   ├── urdf/yuil_dog.urdf
│   ├── meshes/*.STL            # GUI가 직접 표시하는 원본 상세 visual mesh
│   └── MJCF/yuil_dog.xml
├── policies/                   # 정책 파일 (.pt) 전용 관리 폴더
│   ├── policy.pt               # [기본] 인자 없이 실행 시 로드됨
│   ├── yuil_dog_flat_3500_latest.pt
│   └── yuil_dog_flat_3000_baseline.pt
├── scripts/
│   ├── convert_urdf_to_mjcf.py
│   └── run_policy_client.py
├── run.sh                       # 서버, GUI, 정책 통합 실행
├── run_server.sh
└── run_policy.sh
```

변환 스크립트는 원본 STL을 `model/meshes`에 재현 가능하게 staging하고 Rust GUI가
이를 `model/stl`로 직접 표시한다. 접촉 계산은 URDF의 단순화된
box/cylinder/sphere collision만 사용하며, 상세 외형을 가리지 않도록 투명하게 둔다.

## 실행

최초 한 번 또는 URDF 변경 후 모델을 생성한다.

```bash
cd ~/workspace/IsaacLab/project/go2_sim2sim
../../isaaclab.sh -p sim2sim_mujoco/scripts/convert_urdf_to_mjcf.py
```

통합 스크립트 하나로 Rust 서버, GUI, 정책을 함께 실행한다. 서버의 초기 속도 명령은
항상 `(vx, vy, wz) = (0, 0, 0)`이며 GUI 슬라이더로 변경할 수 있다.

```bash
./sim2sim_mujoco/run.sh
```

GUI에는 STL 기반 외형, 차체 위치/속도, action RMS, episode/fall 횟수와
`vx`, `vy`, `wz` 명령 슬라이더가 표시된다. 최근 5초의 발 궤적은 월드 좌표계에서
FL=빨강, FR=초록, RL=파랑, RR=노랑 선으로 표시되며 에피소드 리셋 시 초기화된다.
`run_server_gui.sh`도 같은 명령의 호환 alias다.

**카메라 조작 (기본적으로 로봇을 항상 따라다님)**:
- **자동 추적**: 실행 즉시 카메라가 로봇 몸체(`base_link`)를 자동으로 따라다닙니다.
- **방향 변경**: 마우스 왼쪽 버튼 드래그로 로봇 중심 360° 궤도 회전(방향 및 앙각 조절).
- **줌 조절**: 마우스 오른쪽 버튼 드래그 또는 휠 스크롤로 확대/축소.
- **뷰 프리셋**: GUI 내 `3/4 View`, `Side View`, `Rear View`, `Top View` 버튼으로 원하는 방향 원클릭 설정.
- **단축키**: `F` 키(로봇 재추적), `Esc` 키(자유 시점 전환).

기본값은 `policies/policy.pt`이며 `--policy`로 `policies/` 안의 다른 파일을 선택할 수 있다.
확장자 `.pt`는 생략 가능하다.

```bash
./sim2sim_mujoco/run.sh --policy policy_v5
```

정책은 기본적으로 `Ctrl+C`를 누를 때까지 계속 실행되고, 종료 시 서버도 함께 종료된다.
제한된 시간만 실행하려면 `--duration 20`처럼 초 단위 실행 시간을 지정한다.

`policies/` 폴더 내의 특정 정책 파일명만으로 실행하거나 목록을 확인할 수 있다.

```bash
# 사용 가능한 정책 목록 확인
./sim2sim_mujoco/run.sh --list

# 베이스라인 정책 실행 (파일명만 입력)
./sim2sim_mujoco/run.sh --policy yuil_dog_flat_3000_baseline --duration 20

# 최신 3500 iter 정책 실행
./sim2sim_mujoco/run.sh --policy yuil_dog_flat_3500_latest
```

`run_server.sh`와 `run_policy.sh`는 서버와 정책을 별도 터미널에서 실행해야 할 때 사용할 수 있다.

## 정책 계약

- 입력: 45-D 한 스텝을 10개 보관한 450-D term-major history
- 한 스텝: body angular velocity 3, projected gravity 3, command 3,
  relative joint position 12, joint velocity 12, previous action 12
- 출력: `FL → FR → RL → RR`, 각 다리 `hip_roll → hip_pitch → knee_pitch`의 12-D action
- action: `[-3.5, 3.5]` clip 후 `q_default + 0.25 * action`, 안전 joint limit 적용
- 제어: RS03/RS04 PD gain과 DC motor torque-speed clipping
- 주기: MuJoCo 5 ms × decimation 4 = 정책 50 Hz
- 전송: little-endian `u32 payload byte length` + little-endian `f64[]`

모든 MuJoCo joint/qpos/dof 연결은 배열 번호가 아니라 이름으로 찾는다. 현재
`policy_to_mujoco_sign`은 학습과 동일 URDF 좌표를 사용하므로 전부 `+1`이다. 실제
모터 ID와 encoder 방향은 이 코드가 확정하지 않으며, 실기체 저속 단일 관절 시험 후
별도의 hardware adapter에서 명시해야 한다.

## 검사

```bash
cd ~/workspace/IsaacLab/project/go2_sim2sim/sim2sim_mujoco
cargo test
cargo clippy --all-targets -- -D warnings
```

이 재구축은 보상 재현이 아니라 배포 경계 검증용이다. MuJoCo와 PhysX/Newton의 접촉
결과가 완전히 같음을 전제하지 않고, 관측 인덱스·관절 순서·방향·action scaling·모터
제한·리셋 계약이 다른 엔진에서도 일관적인지를 확인한다.
