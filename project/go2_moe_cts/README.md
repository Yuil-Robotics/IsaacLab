# Yuil Dog rough-terrain MoE-CTS

제작 로봇 **Yuil Dog**의 모델·모터·센서 코드를 포함한 독립 프로젝트입니다.
[`go2_rl_robotlab`](https://github.com/wertyuilife2/go2_rl_robotlab)의 험지 MoE-CTS를 적용한 프로젝트입니다.
기준 커밋: `28b4516d22617b11aeaf8ead63cc00b0c0bcd1bd`.

## 실행

**`go2_sim2sim` 폴더나 패키지는 필요하지 않습니다.** 로봇 USD와 모든 참조 layer가
`src/go2_moe_cts/robots/yuil_dog/usd/`에 포함되어 있으며 wheel에도 함께 패키징됩니다.
기존 IsaacLab 실행 환경은 필요합니다. 아래 shell 명령은 `IsaacLab/project/go2_moe_cts` 배치를 기준으로 합니다.
IsaacLab 환경에는 기존 `torch`, `tensordict`, `rsl-rl-lib`, `onnx`, `tensorboard`를 사용하며
원본의 구버전 RSL-RL을 덮어 설치하지 않습니다.

```bash
# IsaacLab 루트에서 패키지 설치 (최초 1회)
./isaaclab.sh -p -m pip install -e project/go2_moe_cts

cd project/go2_moe_cts

# 짧은 파이프라인 검증: 8 env, 2개 terrain row, 2 iterations
bash scripts/train_yuil_moe_cts.sh --smoke_test --num_envs 8

# 본 학습: 원본의 전체 10×20 험지 구성, 기본 headless
bash scripts/train_yuil_moe_cts.sh --num_envs 4096 --run_name rough_cts

# 학습 재개: 아래 런/체크포인트는 실제 파일명으로 지정
bash scripts/train_yuil_moe_cts.sh --num_envs 4096 --resume \
  --load_run '2026-09-18_09-00-00_rough_cts' --checkpoint 'model_500.pt'

# 최신 CTS 체크포인트로 student만 시각화하고 정책 내보내기
bash scripts/play_yuil_moe_cts.sh

# GUI 없이 특정 모델을 500스텝 평가 (경로는 현재 프로젝트 기준)
bash scripts/play_yuil_moe_cts.sh --viz none --num_envs 16 --num_steps 500 \
  --checkpoint logs/rsl_rl/yuil_dog_rough_moe_cts/<run>/model_500.pt
```

로그 경로는 **현재 프로젝트 `go2_moe_cts` 내부의** `logs/rsl_rl/yuil_dog_rough_moe_cts/<run>/`입니다.
train 저장·resume 검색·play 자동 검색은 모두 이 경로를 사용합니다.
play와 물리 검증의 상대 checkpoint 경로도 프로젝트 기준이며, 절대 경로도 사용할 수 있습니다.
기존 IsaacLab 루트의 CTS 로그는 프로젝트 로그 폴더에 복사해 두었습니다.

기본 학습 길이는 원본과 같은 300,000 iterations이며 `--max_iterations`로 변경할 수 있습니다.
재개 시 `--max_iterations`는 추가 학습 횟수입니다. 학습에는 최소 2개 환경이 필요하며,
teacher/student 각각의 rollout 크기가 mini-batch 수(기본 4)로 나누어져야 합니다.
평가에는 1개 환경도 사용할 수 있습니다.

`--smoke_test` 결과는 코드 동작 확인용이며 보행 성능을 검증한 정책이 아닙니다.
학습 checkpoint, 두 optimizer, iteration, 난수 상태는 복원하지만 물리 시뮬레이터 상태를 복원하는 것은 아닙니다.
기존 flat PPO 모델은 네트워크 구조가 달라 MoE-CTS로 직접 `--resume`할 수 없습니다.

## 태스크

| 태스크 ID | 용도 |
|---|---|
| `Isaac-Velocity-Rough-Yuil-Dog-MoECTS-v0` | 험지 CTS 학습 |
| `Isaac-Velocity-Rough-Yuil-Dog-MoECTS-Play-v0` | 노이즈·push 없이 student 재생 |
| `Isaac-Velocity-Rough-Yuil-Dog-MoECTS-Eval-v0` | 고정 seed, nominal 동역학, reset randomization 제거 |

Play는 학습의 질량·마찰 등 randomization을 유지합니다. Eval도 여러 지형과 속도 명령을
사용하므로, 동일한 seed/환경 수/설정으로 비교해야 합니다.

기존 `Isaac-Velocity-Rough-Go2-MoECTS[-Play/-Eval]-v0` 및
`train_go2.sh` / `play_go2.sh`는 **Phase 1 일반 PPO**로 유지됩니다.
새 CTS 학습은 위의 **Yuil-Dog-MoECTS** 태스크와 스크립트를 사용합니다.

## 구현 구조

```text
src/go2_moe_cts/
  robots/yuil_dog/
    asset_cfg.py            # USD 경로와 Articulation 조립
    actuators_cfg.py        # RS03/RS04 모터 gain·토크·속도·마찰
    joints_cfg.py           # 관절 순서·목표각 제한·초기 자세/높이
    usd/                    # 자체 포함된 로봇 모델과 mesh/layer
  sensors/
    hierarchical_contact_sensor.py
  tasks/yuil_dog/
    env_cfg.py              # 환경 조립, 센서·action, Train/Play/Eval
    observations_cfg.py     # 사용 관측·노이즈·스케일
    rewards_cfg.py          # 사용 보상·가중치·파라미터
    events_cfg.py           # randomization·reset·push 설정
    curriculum_cfg.py       # 지형·보상 curriculum 설정
  mdp/                      # 관측·보상·action·event·명령·지형의 계산 함수
  agents/                   # PPO/MoE-CTS 네트워크 크기·학습 하이퍼파라미터
  learning/                 # MoE 모델·알고리즘·storage·runner·export
scripts/
  training/train.py         # 학습 실행
  evaluation/play.py        # 평가·정책 export
  evaluation/validate_cts.py # 물리·관측·reset·export 검증
tests/                      # 단위 테스트 및 독립성 회귀 검사
```

세부 파일 지도는 [docs/STRUCTURE.md](docs/STRUCTURE.md)에 정리되어 있습니다.
`yuil_dog_env_cfg.py`, `mdp/cts.py`와 기존 `scripts/*.py`는 호환 경로입니다.
구현은 위 역할별 폴더에서 관리하며, 오래된 Python 모듈 경로는 deprecation 후 유지합니다.

| 항목 | 적용 내용 |
|---|---|
| CTS 학습 | Teacher 환경 75%, Student 환경 25%; 공유 actor/critic |
| 인코더 | Teacher MLP `[512,256]`, Student MoE `[512,256,256]`, 8 experts, L2-normalized latent 32 |
| 학습 손실 | Teacher/Student PPO surrogate 각각 평균 후 합산; student 환경에서 teacher latent MSE + gate load balancing |
| Gradient | Student encoder는 PPO에서 분리; 별도 optimizer로 latent 학습; critic의 latent도 detach |
| Actor 입력 | 센서 history 450 → student latent 32 + 현재 센서 프레임 45 → action 12 |
| Privileged 입력 | 현재 관측·선속도·관절 가속도/토크·발 접촉력·187-ray 높이맵, 총 263 |
| 지형 비율 | wave 5%, 상/하 경사 각 10%, rough slope 5%, 상/하 계단 25%/10%, 장애물 20%, 평지 15% |
| 지형 수준 | 10 rows × 20 columns, 에피소드 평균 명령 추종률 <50% 강등 / ≥70% 승급 |
| 보상 | 추종 2.0/1.0, std 0.5; 원본의 관절·접촉·발 움직임·action smoothness 항목 |
| 보상 curriculum | 수직속도 -2→0 (1500 iter), 지면 기준 몸체 높이 -1→-10 (5000 iter) |
| Domain randomization | 마찰·반발·질량·CoM, joint reset, gain ±10%, encoder zero ±0.035 rad, 4초 간격 push |

## 제작 로봇에 맞춘 차이

- Unitree Go2 대신 이 프로젝트의 `robots/yuil_dog/asset_cfg.py`와 검증된 낮은 초기 자세를 사용합니다.
  RS03/RS04의 관절별 nominal gain·토크/속도 한계·마찰·armature 및
  `FL/FR/RL/RR × hip_roll/hip_pitch/knee_pitch` 순서를 재사용합니다.
- Yuil의 관절 목표각 clip과 action scale 0.25를 유지합니다. 목표 root 높이는 0.325 m로 맞춥니다.
- 원본 Unitree 전용 모터 모델과 0–4 step 지연을 Yuil 모터에 그대로 적용하지 않습니다.
  nominal Yuil DCMotor를 사용하며, 기존 RS03/RS04 검증 조건을 기반으로 합니다.
- 물리 200 Hz / 정책 50 Hz. 환경 수 기본값은 이 장비에 맞춰 4096이며 원본의 16384와 다릅니다.
- action 표준편차는 양수 보장을 위해 log parameterization을 사용합니다(초기 std 1.0).
- `single_obs`는 noisy history의 최신 프레임을 재사용해 중복 noise 샘플링에 따른 불일치를 방지합니다.
- 학습은 단일 프로세스/장치를 지원합니다. RoboGauge 서버 연동, RND, symmetry, 분산 학습은 포함하지 않습니다.

## 정책 export와 외부 시뮬레이터 연결

학습 종료 또는 Play 시 `<run>/exported/`에 다음 파일이 생성됩니다.

- `policy.pt`: TorchScript student
- `policy.onnx`: 같은 인터페이스의 ONNX student
- `policy_contract.json`: 입력 순서·스케일·관절 순서·기본 자세·action scale/clip·제어 주기

입력은 `[batch, 450]`, 출력은 `[batch, 12]`입니다. **Teacher나 critic 관측을 입력하지 않습니다.**
History는 term-major이며 각 term 안에서 oldest→newest 순서입니다.
순서는 각속도(3), 중력(3), 명령(3), 관절 위치(12), 관절 속도(12), 이전 action(12)이고
스케일은 각각 `0.25, 1, 1, 1, 0.05, 1`입니다.
에피소드 reset 시 호출 측이 history를 초기화하고 첫 관측 프레임을 채워야 합니다.

기존 PPO 검증 클라이언트에 연결할 때 450차원이라는 사실만으로 관측 의미가 같다고 가정하지 말고
**이 정책의 `policy_contract.json`과 새 env 설정**으로 구성하세요.
이 폴더의 오래된 `deploy/deploy_go2.py`는 Unitree Go2용이며 Yuil Dog 배포 엔트리포인트가 아닙니다.
Yuil Dog 모델과 PhysX 검증 도구는 이 프로젝트에 포함되어 있습니다.
외부 Newton/MuJoCo/실기 클라이언트와 연결할 때에는 export된 계약을 맞춰야 합니다.

## 검증

```bash
# ROS pytest 플러그인의 환경 외 의존성 자동 로딩을 끄고 프로젝트 테스트 실행
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../isaaclab.sh -p -m pytest tests -q
```

테스트는 teacher/student 분리, rollout 순서·timeout bootstrap, gradient 격리,
두 encoder의 실제 업데이트, checkpoint/optimizer 복원, JIT/ONNX 수치 일치,
privileged 관측 비의존성 및 제작 로봇/태스크 설정을 확인합니다.
검증 결과와 실행 제한은 [VALIDATION.md](VALIDATION.md)를 참고하세요.

원본 구현의 출처와 라이선스는 [REFERENCE.md](REFERENCE.md)를 참고하세요.

실제 센서·history reset·student export를 PhysX에서 재검증하려면:

```bash
../../isaaclab.sh -p scripts/evaluation/validate_cts.py --viz none \
  --checkpoint logs/rsl_rl/yuil_dog_rough_moe_cts/<run>/model_500.pt
```

## 학습 로그 읽기

터미널과 `metrics.jsonl`/TensorBoard에 다음 지표를 기록합니다.

- `Train/mean_step_reward`: rollout의 환경·스텝 평균 보상. 에피소드 누적 보상과 다릅니다.
- `Episode/*_return`, `length_steps`, `length_s`: 최근 최대 100개 완료된 전체 에피소드의 누적 보상과 길이.
  all/teacher/student를 각각 집계합니다.
- `Episode/*_timeout_survival_pct`: 완료된 전체 에피소드 중 실패 없이 timeout에 도달한 비율 [%].
  timeout과 실패가 동시에 발생하면 실패로 처리합니다. 실기 성공률이나 모든 지형의 통과율은 아닙니다.
  시작·재개 직후 첫 부분 에피소드는 제외하며, 완료 표본이 없으면 `pending`으로 표시합니다.
- `Tracking/*`: pre-step 속도 명령과 실제 body-frame 속도의 오차. 선속도 XY 벡터 오차 [m/s],
  yaw 각속도 절대 오차 [rad/s]이며 작을수록 잘 추종합니다. teacher/student별 값도 기록합니다.
- `RewardRate/*`: 각 보상의 가중치 적용 후 시뮬레이션 초당 기여도. rollout 평균입니다.
  모두 더한 뒤 `step_dt`를 곱하면 평균 스텝 보상이 됩니다. 가중치는 curriculum에 따라 변합니다.
- `TerminationCount/*`: 현재 rollout의 종료 원인별 횟수. 동시 종료 원인은 중복 집계됩니다.
- `Loss/*`: PPO value/surrogate/entropy, CTS latent/load-balance 손실.
- `MoE/expert_*_weight`: student distillation mini-batch 전체에서 평균한 soft gating weight.
  hard expert 선택 빈도가 아니며, 합은 1입니다.
- `Perf/*`: 처리 속도, rollout/update 시간, 예상 잔여 시간.

에피소드 manager 지표는 reset이 발생한 모든 스텝에서 종료 환경 수로 가중 평균합니다.
reset이 없는 스텝에 남아 있는 이전 `extras`는 다시 집계하지 않습니다.

```bash
# 현재 프로젝트 폴더에서
../../isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl --port 6006
```

브라우저: `http://localhost:6006`. 터미널 전체 stdout은 자동 저장하지 않으므로
필요하면 shell의 `tee` 또는 `> 파일명 2>&1`을 사용합니다.

## GUI 명령 화살표

`go2_sim2sim`의 표시 방식을 로컬 코드로 복사·이식했습니다.

- 초록: 목표 `vx·vy·wz`, root 위 0.65 m.
- 파랑: 실제 `vx·vy·wz`, root 위 0.48 m.
- 이동 중: `atan2(vy + 0.25*wz, vx)` 방향으로 표시하고, 길이에 이동·회전 크기를 함께 반영합니다.
- 제자리 회전: 수평속도 < 0.05 m/s, |wz| > 0.05 rad/s이면 왼쪽/오른쪽으로 표시합니다.
- 기본 mesh scale `(0.5, 1.0, 1.0)`, 색·발광·roughness 모두 복사한 설정과 동일합니다.
- 곡선이 아닌 직선 화살표 두 개이며, 로봇의 전체 자세 회전을 따릅니다.
  이동과 회전을 시각적으로 합친 표시이므로 실제 XY 이동 벡터와는 다를 수 있습니다.

Kit GUI에서 자동 활성화됩니다. `--no_command_debug_vis`로 끌 수 있습니다.
새 실행부터 적용되며, 외부 폴더를 import하거나 참조하지 않습니다.
구현: `src/go2_moe_cts/visualization/velocity.py`의 `tracking_arrow`, `tracking_arrow_cfg`.

## 고정 명령 범위

사용자 지정에 따라 모든 학습 단계와 지형에서 `vx ±1.0 m/s`, `vy ±0.5 m/s`,
`wz ±1.0 rad/s`로 고정했습니다. 원본의 20,000/50,000 iteration 범위 확대는 해제했습니다.
`commands_cts.py`의 `Ranges`, `terrain_max_command_ranges`, `command_limits`에 설정하며,
샘플링 후와 command 갱신 시 최종 clamp도 적용합니다. 범위는 실제 속도가 아닌 목표 명령입니다.
새 실행·재개부터 적용되며, 이미 실행 중인 프로세스에는 자동 반영되지 않습니다.

## 명령 추종 기반 지형 커리큘럼

거리 4 m 기준을 사용하지 않습니다. 에피소드 reset 시 평균 명령 추종 점수로 판단합니다.

- 50% 미만: 한 단계 강등.
- 50% 이상 70% 미만: 유지.
- 70% 이상: 한 단계 승급.
- 기록한 스텝이 없는 최초 reset: 유지.

매 정책 스텝의 점수는 다음과 같습니다.

```text
limits = (1.0 m/s, 0.5 m/s, 1.0 rad/s)
u = (vx_command, vy_command, wz_command) / limits
error = norm((actual_velocity - command) / limits)
score = clamp(1 - error / max(norm(u), 0.1), 0, 1)
episode_tracking_pct = mean(score over episode) * 100
```

축별 단위를 명령 상한으로 정규화한 뒤 하나의 벡터 오차로 계산합니다.
명령 반대 방향·과속·명령하지 않은 축의 움직임도 감점됩니다.
매 스텝에서 오차를 계산하므로 서로 반대인 명령이 누적되어 상쇄되지 않습니다.
정지·아주 작은 명령에서는 정규화된 분모 0.1을 사용합니다.
예를 들어 완전 정지 명령에 vx만 0.01 m/s이면 90%, 0.1 m/s이면 0%입니다.

목표 vx=0.1 m/s, vy=wz=0을 유지하고 실제 다른 축 오차가 없을 때,
실제 vx=0.07이면 70%, 0.05이면 50%, 0이면 0%입니다.
부동소수점 경계 비교에 1e-6의 허용치를 적용합니다.

정상 스텝은 command 변경 전에 측정하고, 마지막 스텝은 물리 계산 후 reset 전에 포함합니다.
초기화·reset 직후 상태는 중복 집계하지 않습니다. 첫 부분 에피소드도 관측된 스텝을 기준으로 판단합니다.
생존·이동 거리의 추가 조건은 없습니다. 정지·회전 명령으로도 승급할 수 있고,
일찍 종료되어도 그때까지의 평균이 70% 이상이면 승급할 수 있습니다.
이는 지형 통과 인증이 아니라 사용자가 지정한 명령 추종 기반 난이도 조정입니다.
최저 0단계 유지·최고 단계 초과 시 무작위 재배치·지형 종류 유지 규칙은 같습니다.

임계값: `tasks/yuil_dog/curriculum_cfg.py`의 `promote_threshold`, `demote_threshold`.
점수 계산: `mdp/tracking.py`. 과거 거리 기반 함수는 호환성을 위해 남겨두었지만 새 task에서는 사용하지 않습니다.
`Curriculum/terrain_levels/` 지표를 JSONL/TensorBoard 및 reset이 있는 iteration의 터미널에 기록합니다.
로그의 count들은 기존 logger의 reset batch 집계 방식에 따라 가중 평균된 요약값입니다.
새 train/resume부터 적용하며, 실행 중인 프로세스에는 자동 반영되지 않습니다.

### MuJoCo에서 student 확인

```bash
./sim2sim_mujoco/run.sh
```

로컬 student 정책으로 기존 MuJoCo GUI 기능을 사용할 수 있습니다.
체크포인트 변환과 정책 선택 방법은 [MuJoCo 실행 안내](sim2sim_mujoco/README.md)를 참고하세요.
