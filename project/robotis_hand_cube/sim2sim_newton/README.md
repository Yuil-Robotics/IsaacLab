<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# Robotis HX5 benchmark training and sim2sim_newton

`Robotis_left_benchmark_2026-07-30_16-01-04`의 PhysX/RSL-RL 학습 설정을
재현하고, 학습 산출물의 `149 observation / 20 action` TorchScript 정책을
IsaacLab Newton MJWarp에서 실행하는 환경이다.
기본 로봇은 더 이상 임시 MJCF 변환본이 아니라 아래 원본 USD를 합성한다.

V7 정책은 기존 benchmark 환경을 변경하지 않는 별도 task로 추가되어 있다. 실행법과
정확한 계약은 [`../V7_SIM2SIM.md`](../V7_SIM2SIM.md)를 참고한다.

```text
robotis_hand/robotis_hand_description/urdf/hx5_d20_left_2/
  hx5_d20_left_2/hx5_d20_left_2.usd
```

## 최초 준비

IsaacLab 루트에서 다음 명령을 한 번 실행한다.

```bash
./isaaclab.sh -p \
  project/robotis_hand_cube/sim2sim_newton/scripts/prepare_original_robot_usd.py
```

이 스크립트는 원본을 수정하지 않고 다음 얇은 래퍼를 만든다.

- `assets/hx5_d20_left_original_newton/hx5_d20_left.usda`: 원본 손 USD를
  reference하고 Newton이 읽을 수 있는 위치로 21개 collision API를 옮긴다.
- `assets/dex_cube_original_newton/dex_cube.usda`: 학습에 사용한 Isaac 5.1
  DexCube 외형을 유지하면서 최종 크기, 질량, 관성을 명시한다.
- DexCube의 작은 원본 USD 두 레이어와 텍스처는 로컬에 받아 이후 실행 시
  네트워크와 Isaac asset 버전에 의존하지 않는다.

손 래퍼는 `robotis_hand/` 아래의 sublayer와 mesh를 계속 참조하므로 그 폴더의
구조를 유지해야 한다.

## PhysX benchmark 학습

프로젝트 루트에서 다음 명령을 실행한다. 기본 task는 domain randomization과
action/observation noise가 없는 benchmark 재현 환경이다.

```bash
./run_robust_train.sh --baseline --headless --num_envs 8192
```

이 명령은 RSL-RL과 아래 기준값을 사용한다.

- physics 120 Hz, policy 30 Hz (`decimation=4`)
- 149 observation / 20 action
- actor/critic `1024, 512, 256, 128`, ELU
- rollout 16, PPO epochs 5, mini-batches 4
- learning rate `3e-4`, entropy coefficient `0.001`
- Isaac 5.1 DexCube 로컬 사본, scale `0.8`, 실측 질량 `0.077 kg`
- PhysX actuator stiffness/damping `1.0/0.1`, effort `1.03 N·m`, armature `0`

원본 benchmark의 action 순서와 `wxyz` quaternion 관측은
`RobotisHandBenchmarkEnv`가 현재 IsaacLab 내부의 `xyzw` 표현과 분리해 복원한다.
Sim2Real domain randomization은 별도 task에서만 활성화한다. 이 task의 actuator
armature nominal은 `0.0009 kg·m²`이고 reset 시 `0.5~2.0`배 log-uniform으로
샘플링된다. Robust task는 회전 오차가 tolerance 안에서 10 policy step(30 Hz 기준
약 0.333초) 연속 유지되어야 성공으로 인정하고, 큐브 낙하 시 `-50` reward를 준다.
원본 benchmark 재현 task는 기존 1-step 성공 및 낙하 비용 0을 유지한다.

```bash
./run_robust_train.sh --robust --headless --num_envs 8192
```

### A: Single-goal 학습

기존 Robust task는 B 방식으로 유지된다. 새 A task는 별도 Gym ID
`Isaac-Repose-Cube-Robotis-Single-Goal-Train-v0`를 사용하므로 기존 task의 설정과
checkpoint를 변경하지 않는다.

| 구분 | B: Continuous-goal | A: Single-goal |
|---|---|---|
| 성공 조건 | 0.2 rad 이내 10 policy step 유지 | 회전오차 0.2 rad·각속도 2.0 rad/s 이하를 5 step 유지 |
| 성공 보너스 | 목표마다 +250 | episode당 1회 +250 |
| 성공 직후 | 새 목표 orientation 생성 | 즉시 episode 종료/reset |
| 낙하 비용 | 기존 task 설정 유지 | -50 |
| 최대 episode | 10초 동안 여러 목표 | 성공·낙하 또는 10초 timeout까지 한 목표 |

A는 Robust task의 물리 domain randomization, action/observation noise, 149 observation,
20 action 및 30 Hz 제어 주기를 그대로 상속한다. 다만 목표 주변에서 시간을 끌며
proximity reward를 누적하지 못하도록 A의 reward만 다음과 같이 변경한다.

```text
reward =
    20.0 * max(best_rotation_error - rotation_error, 0)
    - 0.5 * rotation_error
    - 100.0 * max(in_hand_position_error - 0.06, 0)^2
    - 0.001 * sum((action - previous_action)^2)
    - 0.005 * sum(max(abs(action) - 0.95, 0)^2)
    - 0.0005 * sum(joint_velocity^2)
    + 1.0   (stable hold step)
    + 250.0 (success)
    - 50.0  (fall)
    - 50.0  (timeout without success)
    - 0.01  (each policy step)
```

현재 초기 curriculum의 Stable hold는 `rotation_error <= 0.2 rad`와
`cube_angular_speed <= 2.0 rad/s`를 동시에 5 policy step(약 0.167초) 만족해야 한다.
실제 6D pose의 translation 오차와 위치 미분 노이즈가
성공 판정에 영향을 주지 않도록 cube position, in-hand distance 및 linear velocity는
성공/Hold 조건에서 사용하지 않는다. 성공 판정 스텝에는 `+250`을 계산한 뒤 terminal
transition으로 저장하며 다음 목표를 만들지 않는다. 기존 B는 reciprocal proximity
reward와 연속 목표 동작을 그대로 유지한다.

Progress는 episode에서 기록한 최저 회전오차를 새로 갱신할 때만 지급하므로 왕복으로
반복 수확할 수 없다. 일시적으로 목표에서 멀어지는 탐색에는 별도 progress penalty를
주지 않는다. 대신 매 step `-0.5 * rotation_error`와 미성공 timeout `-50`을 적용해
목표를 시도하지 않고 버티는 전략이 이득이 되지 않게 한다. 거리 항목은 0.06 m 이내에서
0인 dead zone이므로 목표 위치에 정확히 정렬하는 별도 position task가 되지 않는다.
Action-rate와 별도로 0.95를 넘는 action에만 약한 soft saturation penalty를 적용한다.
성공률이 충분히 올라간 뒤 최종 단계에서 각속도 1.0 rad/s 및 10-step Hold로 강화한다.

TensorBoard에는 `reward_rotation_progress`, `reward_distance`, `reward_action_rate`,
`reward_orientation_error`, `reward_action_saturation`, `reward_joint_velocity`, `reward_hold`,
`reward_success`, `reward_fall`, `reward_timeout`, `reward_time`, `in_hand_distance`, `object_linear_speed`,
`object_angular_speed`, `action_saturation_rate`가 각각 기록된다. 위 scale과 안정 조건은
첫 학습용 초기값이므로 실제 분포를 확인한 뒤 한 항목이 total reward를 지배하지 않는
범위에서 조정한다.

A PhysX task의 GPU rigid patch capacity는 8192개 환경의 접촉 peak를 감당하도록
`163840`에서 `262144`로 올렸다. 기존 baseline과 B task의 값은 변경하지 않는다.
A 학습 checkpoint는 100 iteration마다 저장하며, 기존 baseline과 B task는 250
iteration 간격을 유지한다.

처음부터 A를 학습하려면 다음을 실행한다.

```bash
./run_robust_train.sh --single_goal --headless --num_envs 8192 \
  --max_iterations 10000 --run_name single_goal
```

처음부터 학습할 때는 2단계 curriculum을 권장한다. Stage 1은 Sim2Real 물성·모터
랜덤화, cube pose noise/sample-hold, joint offset/limit 랜덤화 및 action delay/hold를
끄지만 초기 손/큐브 자세와 목표 방향 샘플링은 유지한다. Stage 1이 끝나면 같은
actor, critic, observation normalizer와 optimizer를 checkpoint에서 읽어 전체
랜덤화를 적용한 Stage 2를 자동으로 시작한다.

```bash
NOMINAL_ITERATIONS=40000 ROBUST_ITERATIONS=10000 \
  ./run_single_goal_curriculum_train.sh --headless --num_envs 8192
```

성공률을 직접 확인한 뒤 원하는 시점에 Stage 2를 시작하려면 Stage 1만 먼저
실행하고, 생성된 run 이름을 `--from-run`에 지정한다.

```bash
./run_robust_train.sh --single_goal_nominal --headless --num_envs 8192 \
  --max_iterations 40000 --run_name single_goal_nominal_stage1

ROBUST_ITERATIONS=10000 ./run_single_goal_curriculum_train.sh \
  --from-run 2026-09-01_12-00-00_single_goal_nominal_stage1 \
  --headless --num_envs 8192
```

Stage 1의 task ID는
`Isaac-Repose-Cube-Robotis-Single-Goal-Nominal-Train-v0`, 전체 랜덤화를 다시
적용하는 Stage 2의 task ID는 기존
`Isaac-Repose-Cube-Robotis-Single-Goal-Train-v0`이다. 두 task의 observation/action
및 reward contract는 같아서 checkpoint를 그대로 이어받을 수 있다.

현재 최신 checkpoint를 초기 가중치로 사용해 A fine-tuning을 시작하려면 다음을
실행한다. `--resume`은 같은 experiment 아래에서 수정 시간이 가장 최근인 checkpoint를
고르므로, 출력되는 checkpoint 경로를 시작 전에 확인해야 한다.

```bash
./run_robust_train.sh --single_goal --headless --num_envs 8192 \
  --resume --max_iterations 10000 --run_name single_goal_from_latest
```

기존 benchmark checkpoint에서 Robust 학습을 이어갈 때는 run 폴더와 checkpoint를
지정한다. 두 task는 observation/action 및 RSL-RL network contract가 같으므로 actor,
critic, optimizer 상태를 그대로 이어받는다.

같은 학습 설정으로 가장 최근에 저장된 `model_*.pt`부터 재개할 때는 `--resume`만
지정하면 된다. 스크립트가 전체 experiment 로그에서 수정 시간이 가장 최근인
checkpoint와 해당 run을 자동으로 선택한다.

```bash
./run_robust_train.sh --baseline --headless --num_envs 8192 --resume
```

```bash
./run_robust_train.sh --robust --headless --num_envs 8192 \
  --resume \
  --load_run 2026-08-10_08-23-39 \
  --checkpoint 'model_.*.pt' \
  --max_iterations 10000 \
  --run_name sim2real_robust
```

일반 학습이 끝난 뒤 최신 checkpoint를 자동으로 찾아 Robust 학습까지 연속 실행하려면
다음을 사용한다.

```bash
BASE_ITERATIONS=50000 ROBUST_ITERATIONS=10000 \
  ./run_two_stage_train.sh --headless --num_envs 8192
```

이미 진행 중인 일반 학습에서 Robust 단계만 자동으로 이어가려면 다음과 같이 실행한다.

```bash
ROBUST_ITERATIONS=10000 ./run_two_stage_train.sh \
  --from-run 2026-08-10_08-23-39 \
  --headless --num_envs 8192
```

### Sim2Real randomization 범위

| 항목 | Nominal | Robust 범위 | 비고 |
|---|---:|---:|---|
| joint armature | 0.0009 kg·m² | 0.00045~0.0018 kg·m² | 0.5~2.0배 log-uniform |
| cube side / mass | 0.048 m / 0.077 kg | 0.047~0.049 m / 0.072287~0.081913 kg | 696.2529 kg/m³ 고정, 질량·관성 결합 계산 |
| static friction | 1.0 | 0.7~1.3 | robot/object 각각 샘플링 |
| dynamic friction | 1.0 | min(1.0, static) | `dynamic <= static` 강제 |
| restitution | 0.0 | 0.0 | 고정 |
| actuator stiffness | 1.0 | 0.9~1.1 | uniform scale |
| actuator damping | 0.1 | 0.08~0.12 | uniform scale |
| action white noise std | - | 0.05 | 매 policy step |
| action persistent bias std | - | 0.015 | reset 시 재샘플링 |
| observation white noise std | - | 0.002 | 전체 149-vector 임시 모델 |
| observation bias std | - | 0.0001 | reset 시 재샘플링 |

현재 observation noise는 좌표별 실측 단위를 반영하지 않은 임시 공통 noise다. 실기
6D pose의 10~15 Hz 갱신, timestamp latency, dropout, quaternion 각도 오차와 관절
encoder noise는 측정값을 확보한 뒤 component별 sensor model로 추가해야 한다. CoM,
effort limit, 통신 지연도 실측 근거가 없어 아직 randomize하지 않는다.

기준 benchmark는 이전 iteration 11500에서 이어진 기록이므로 완전히 같은 학습
곡선을 재현하려면 해당 선행 checkpoint가 필요하다. 현재 폴더의 `model_14500.pt`는
평가 또는 그 이후 재개에 사용할 수 있지만, 최초 0 iteration부터 동일한 가중치
경로를 보장하지는 않는다.

설정·정책·USD 구성을 시뮬레이터 실행 전에 검사하려면 다음을 사용한다.

```bash
./isaaclab.sh -p \
  project/robotis_hand_cube/sim2sim_newton/scripts/validate_setup.py
```

## 실행

이 디렉터리에서 바로 실행할 수 있다.

```bash
./run_sim2sim.sh
```

프로젝트 루트에서는 가장 최근에 수정된 RSL-RL checkpoint를 자동으로 선택하는
스크립트를 사용할 수 있다. Isaac PLAY는 선택한 checkpoint를 같은 run의
`exported/policy.pt`로 내보낸다.

```bash
./run_latest_isaac_play.sh
```

A 방식으로 성공 즉시 reset되는 PhysX PLAY는 다음과 같다.

```bash
./run_latest_isaac_play.sh --single_goal
```

A 방식 학습과 PLAY는 각 관절의 물리 한계 양 끝에서 5%씩 여유를 둔다. 정책의
정규화 action `[-1, 1]`은 축소된 90% command 범위 전체로 다시 매핑되며, 관절
위치 관측은 기존 checkpoint 호환을 위해 원래 물리 한계 기준을 유지한다.
관절 목표각 변화량은 강제로 제한하지 않으며, single-goal 관절 속도 reward
scale `-0.001`과 action-rate penalty로 정책이 느리고 부드럽게 움직이도록 유도한다.

동일한 A 방식에서 물성·모터·관측·지연 랜덤화를 끈 PhysX nominal PLAY는
다음과 같다. 초기 손/큐브 자세와 목표 방향 샘플링은 비교 난이도를 유지하기
위해 그대로 적용된다.

```bash
NUM_ENVS=16 ./run_latest_isaac_play.sh --single_goal_nominal
```

Newton-MuJoCo PLAY는 최신 checkpoint와 같은 run의 export를 사용하며, export가
없거나 checkpoint보다 오래되었으면 잘못된 정책을 실행하지 않고 Isaac PLAY를 먼저
실행하라는 오류를 출력한다.

```bash
./run_latest_newton_play.sh
```

Newton에서도 A와 동일한 episode 정의와 통계를 사용하려면 다음과 같이 실행한다.

```bash
./run_latest_newton_play.sh --single_goal --num_envs 64 --headless --stats_every 30
```

두 latest PLAY 스크립트는 기본적으로 학습 제어 주기인 30 Hz에 맞춰 실시간으로
실행되며 step 제한이 없다. 사용자가 `Ctrl+C`를 누르거나 시각화 창을 닫을 때까지
계속 실행된다.

Newton-MuJoCo에서 여러 환경을 병렬로 실행하면서 누적 통계를 확인할 수 있다.

```bash
./run_latest_newton_play.sh --num_envs 64 --headless --stats_every 30
```

상태 줄은 완료 episode 성공률, 낙하율, timeout 비율, episode당 목표 성공 횟수,
평균 episode 시간, 연속 성공, action 포화율 및 평균 관절 속도를 출력한다.

기본 Isaac task는 Robust task, 환경 수는 16이다. 환경 변수로 nominal task나 환경
수를 선택할 수 있다.

```bash
TASK=Isaac-Repose-Cube-Robotis-Direct-v0 NUM_ENVS=1 ./run_latest_isaac_play.sh
```

스크립트 위치를 기준으로 경로를 계산하므로 어느 작업 디렉터리에서 호출해도
된다. 기본은 학습 제어 주기와 같은 30 Hz real-time pacing이다. 각 step의 계산이
33.33 ms보다 빠르면 남은 시간만큼 기다리고, 계산 자체가 더 느리면 실제 시간보다
느려질 수 있다.

```bash
# GUI, 30 Hz real time
./run_sim2sim.sh

# 가능한 가장 빠르게 3000 policy step 실행
./run_sim2sim.sh --steps 3000 --no-real_time

# headless 검증 실행
./run_sim2sim.sh --headless --steps 300 --no-real_time
```

`--stats_every 30`으로 출력 간격을 바꿀 수 있다. `goal_dist`와 `cube_z`는 상태
줄에서 출력하지 않는다.

다른 호환 래퍼를 시험할 때만 `--robot_usd PATH` 또는
`ROBOTIS_HAND_USD=/absolute/path`를 사용한다. 받은 raw USD를 직접 지정하면
Newton이 잘못 배치된 collision API를 무시하므로 기본 래퍼 사용을 권장한다.

## 원본 학습 환경과 대조한 값

| 항목 | 원본 PhysX 학습 | 현재 Newton 환경 |
|---|---:|---:|
| random seed | 42 | 42 |
| physics step / decimation | 1/120 s / 4 | 1/120 s / 4 |
| policy rate | 30 Hz | 30 Hz |
| observation / action | 149 / 20 | 149 / 20 |
| 손 root 위치 | (0, -0.3, 0.5) m | 동일 |
| 큐브 초기 위치 | (0, -0.42, 0.6) m | 동일 |
| 큐브 한 변 | 0.06 x 0.8 = 0.048 m | 0.048 m |
| 실제 큐브 질량 | 0.077 kg | 0.077 kg |
| 큐브 관성 대각 | 2.9568e-05 kg·m² | 2.9568e-05 kg·m² |
| 마찰계수 | static/dynamic 1.0/1.0 | 1.0/1.0 |
| actuator stiffness / damping | 1.0 / 0.1 | 1.0 / 0.1 |
| effort / velocity limit | 1.03 N·m / 4.8 rad/s | 동일 |
| reset position / joint noise | 0.01 m / 0.2 | 동일 |
| single-goal success / fall threshold | 0.4 rad / 0.24 m | 동일 |

현재 명목 설정은 실측한 48 mm, 77 g으로 계산한 `696.2529 kg/m³`를
명시한다. A 태스크는 유효 충돌 한 변을 47~49 mm로 샘플링하고 이 밀도를
유지하도록 질량 `m=rho*L^3`과 정육면체 관성 `I=m*L^2/6`을 함께 계산한다.
Newton 명목 검증도 48 mm, 0.077 kg과 대응 관성을 사용한다.

원본 손은 20 revolute joint, 21 rigid body, 21 collider이며 고정 base이다. root를
고정하는 constraint는 policy joint가 아니므로 runtime `joints=20`이 정상이다.
22개로 보이던 결과는 이전 임시 변환 자산의 topology였으며 현재 기본 자산에서는
사용하지 않는다. 원본 손 전체 질량 `0.92023569 kg`과 각 링크의 authored inertia도
래퍼를 통해 그대로 합성된다.

원본 actuator의 `armature: null`과 USD를 PhysX에서 직접 읽은 실제 값은 정확히
`0.0 kg·m²`이다. Benchmark PhysX task는 이 값을 보존한다. Sim2Real PhysX task와
Newton의 nominal `0.0009 kg·m²`는 엔진 안정화와 reflected-inertia uncertainty를
다루기 위한 임시값이며 실기 식별값이라는 뜻은 아니다.

## 정책 호환 처리

받은 `policy.pt` 내부에는 149차원 empirical normalizer와 actor가 함께 들어 있어
별도의 RSL-RL runner가 필요 없다. 학습 당시 순서를 그대로 유지한다.

- action/joint: index, middle, ring, pinky, thumb 순서의 20개
- fingertip: link8, link12, link16, link20, link4 순서
- observation: joint state, object, goal, fingertip, 이전 action 순서

현재 IsaacLab/Newton tensor는 quaternion을 `xyzw`로 제공하지만 정책은 예전
IsaacLab의 `wxyz` 값을 학습했다. object, goal, relative, fingertip quaternion을
정책 직전에 `wxyz`로 되돌리고, 고정된 fingertip quaternion의 `q/-q` 부호도
학습 normalizer 평균과 같은 쪽으로 맞춘다.

## Newton에서 완전히 같게 만들 수 없는 부분

물성·pose·정책 contract는 위와 같이 맞췄지만 solver 자체는 PhysX TGS가 아니라
Newton MJWarp이다. 현재 Newton 안정화 설정은 `implicitfast`, 4 substeps,
100 iterations, pyramidal cone, `impratio=1`이며 `clone_in_fabric=False`는 Newton의
제약이다. PhysX 전용 articulation iteration/sleep/stabilization, gyroscopic flag,
torsional patch friction은 Newton에 일대일 대응하지 않는다. 따라서 같은 정책이라도
contact trajectory와 최종 성공률이 완전히 같을 수는 없다.

실행 시작 시 아래 값을 직접 출력하므로 잘못된 USD나 설정이 들어갔는지 즉시 확인할
수 있다.

```text
[physics] fixed_base=True | bodies=21 | joints=20 | hand_mass=0.920236 kg | ...
[physics] cube_mass=0.077 kg | cube_inertia_diag=[2.956800068e-05, ...]
```

큐브 중심의 목표 holding 높이는 `in_hand_pos.z=0.56 m`이다. 받은 학습 로그의
후반 평균 object height는 약 `0.569 m`였으므로 `mean_cube_z` 하나만 보기보다는
`mean_goal_dist`와 회전 성공 횟수를 함께 보는 것이 정확하다.
