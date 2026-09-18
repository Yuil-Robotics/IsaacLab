<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# Robotis HX5 V7 sim2sim profile

`robotis_hand_other_sim/exported/policy.pt`용 V7 실행 환경이다. 기존 benchmark
환경은 삭제하거나 기본값을 변경하지 않고 별도 task/profile로 보존한다.

| Backend | 기존 환경 | V7 환경 |
|---|---|---|
| Newton | `Isaac-Repose-Cube-Robotis-HX5-Newton-Play-v0` | `Isaac-Repose-Cube-Robotis-HX5-Newton-V7-Play-v0` |
| MuJoCo | 기본 `SIM_PROFILE=benchmark` | `SIM_PROFILE=v7` |
| Policy | `Robotis_left_benchmark_.../policy.pt` | `robotis_hand_other_sim/exported/policy.pt` |

## V7 contract

- physics 120 Hz, policy 30 Hz, observation/action `149/20`
- cube 48 mm, 0.0365 kg, nominal static/dynamic friction `1.0/1.0`
- cube reset `(0, -0.44, 0.60) m`, target `(0, -0.44, 0.56) m`
- actuator `kp=3.0`, `kd=0.05`, joint friction `0.01`, effort limit `1.03 N·m`
- raw action penalty와 별개로 control action을 `[-1, 1]`로 clip
- physical joint limit에서 양쪽 3%를 제외한 command range, EMA `0.8`, 최대 target 변화 `0.08 rad`
- command마다 0~2 physics-step delay, startup motor time constant 5~15 ms
- cube pose 1~2 policy-step delay, position noise 2 mm, orientation noise 2 deg
- delayed/noisy pose의 finite difference velocity, low-pass alpha `0.35`
- success는 rotation error `<=0.2 rad` 및 position error `<=0.045 m`를 동시에 만족할 때 발생
- 성공 직후 새 orientation goal을 만드는 continuous reorientation task

학습 DR의 cube mass/size/friction, actuator gain 및 hand-base ±1 mm 범위는 정책의
robustness 범위다. 두 sim2sim backend는 비교 가능한 재현성을 위해 그 범위의 nominal
물성(베이스 오프셋 0)을 사용하고, V7에 핵심적인 observation/control latency와 estimator
noise는 실행 중 샘플링한다. Newton joint armature는 V7 PhysX nominal인 `0.0`, MuJoCo는
원본 모델의 solver-stabilization 값 `0.004`를 유지한다.

## Newton

```bash
cd project/robotis_hand_cube/sim2sim_newton
./run_sim2sim_v7.sh

# headless smoke/evaluation
./run_sim2sim_v7.sh --headless --steps 300 --no-real_time --stats_every 30

# 64개 환경 병렬 평가: 두 명령은 동일
./run_sim2sim_v7.sh --num_envs 64 --headless --steps 3000 --no-real_time --stats_every 100
V7_NUM_ENVS=64 ./run_sim2sim_v7.sh --headless --steps 3000 --no-real_time --stats_every 100
```

병렬 실행에서 `step`은 모든 환경을 한 번씩 전진시킨 policy-loop 횟수이고,
`env_steps`는 `step × envs`로 계산한 총 policy transition 수다. `--steps`는 `step`을
기준으로 종료하므로 환경 수를 늘리면 같은 `--steps`에서 더 많은 episode 표본을 얻는다.

기존 `./run_sim2sim.sh`는 benchmark 백업 환경을 그대로 실행한다.

## MuJoCo

두 터미널을 사용한다.

```bash
# terminal 1
cd project/robotis_hand_cube/sim2sim-mujoco
./run_server_v7.sh

# terminal 2
cd project/robotis_hand_cube/sim2sim-mujoco
./run_policy_v7.sh
```

서버와 클라이언트는 반드시 둘 다 V7으로 실행해야 한다. 기존 `cargo run --release`와
`scripts/run_policy_client.py` 기본 실행은 benchmark profile을 유지한다.
V7 TCP 응답은 149차원 policy observation 뒤에 episode generation 1개를 덧붙인다.
클라이언트는 이 값을 policy 입력에서 제거하고, generation이 바뀔 때 pose-delay와
velocity-filter history를 초기화한다. benchmark 응답은 기존 149차원 그대로다.

## 검증

```bash
./isaaclab.sh -p -m pytest project/robotis_hand_cube/sim2sim_newton/tests -q
./isaaclab.sh -p -m pytest \
  project/robotis_hand_cube/sim2sim-mujoco/tests/test_v7_observation_adapter.py -q
cd project/robotis_hand_cube/sim2sim-mujoco && cargo test
```

Newton MJWarp와 MuJoCo는 PhysX와 contact solver, friction cone, collision 처리 방식이
다르므로 trajectory와 성공률이 동일하다는 보장은 없다. 이 프로필은 정책 I/O,
명목 물성, reset, control/observation pipeline 및 성공 정의를 일치시키는 데 초점을 둔다.
