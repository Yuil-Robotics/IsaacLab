# 구성 파일 지도

이 프로젝트는 `go2_sim2sim` 없이 동작합니다. IsaacLab 및 기존 RL 실행 환경은 필요합니다.
로봇 모델은 symlink가 아닌 실제 USD 파일로 포함되어 있고 모든 하위 참조가 로컬에서 해결됩니다.

## 로봇과 센서

| 변경하려는 내용 | 파일 |
|---|---|
| 로봇 모델 경로·Articulation | [asset_cfg.py](../src/go2_moe_cts/robots/yuil_dog/asset_cfg.py) |
| 관절별 motor gain·토크·속도 한계·마찰 | [actuators_cfg.py](../src/go2_moe_cts/robots/yuil_dog/actuators_cfg.py) |
| 관절 순서·관절 목표각 clip·초기 자세·높이 | [joints_cfg.py](../src/go2_moe_cts/robots/yuil_dog/joints_cfg.py) |
| 실제 USD 모델 | [yuil_dog.usda](../src/go2_moe_cts/robots/yuil_dog/usd/yuil_dog.usda) |
| 중첩된 body 구조의 PhysX 접촉 처리 | [hierarchical_contact_sensor.py](../src/go2_moe_cts/sensors/hierarchical_contact_sensor.py) |

기본 USD는 패키지 내부 파일입니다. 다른 모델을 의도적으로 사용할 때만
`YUIL_DOG_USD_PATH` 환경 변수로 경로를 override합니다.
이 override가 외부 파일을 가리키면 그 파일은 별도로 제공해야 합니다.

## 환경 cfg와 계산 함수

**cfg는 사용 항목·가중치·파라미터를 정하고, mdp는 계산식을 구현합니다.**

| 역할 | cfg | 계산 함수 |
|---|---|---|
| 환경 조립·센서·action·평가 모드 | [env_cfg.py](../src/go2_moe_cts/tasks/yuil_dog/env_cfg.py) | [actions.py](../src/go2_moe_cts/mdp/actions.py) |
| 관측 구성 | [observations_cfg.py](../src/go2_moe_cts/tasks/yuil_dog/observations_cfg.py) | [observations_cts.py](../src/go2_moe_cts/mdp/observations_cts.py) |
| 보상 선택·가중치 | [rewards_cfg.py](../src/go2_moe_cts/tasks/yuil_dog/rewards_cfg.py) | [rewards.py](../src/go2_moe_cts/mdp/rewards.py), [rewards_cts.py](../src/go2_moe_cts/mdp/rewards_cts.py) |
| randomization·reset·push | [events_cfg.py](../src/go2_moe_cts/tasks/yuil_dog/events_cfg.py) | [events.py](../src/go2_moe_cts/mdp/events.py), IsaacLab 기본 MDP |
| curriculum | [curriculum_cfg.py](../src/go2_moe_cts/tasks/yuil_dog/curriculum_cfg.py) | [curriculums_cts.py](../src/go2_moe_cts/mdp/curriculums_cts.py) |
| 명령·지형별 속도 제한 | [commands_cts.py](../src/go2_moe_cts/mdp/commands_cts.py) | 같은 파일 |
| 지형 종류·비율·형상 | [terrains.py](../src/go2_moe_cts/mdp/terrains.py) | 같은 파일 |

보상 가중치를 조정할 때 `rewards_cfg.py`와 `curriculum_cfg.py`를 함께 확인하세요.
Curriculum이 설정된 가중치는 학습 진행에 따라 바뀝니다.
관측 `policy`는 기존 `env_cfg.py`의 PolicyCfg를 상속하고, Yuil 관절 선택은 새 task에서 적용합니다.

## 학습과 실행

- [agents/moe_cts_cfg.py](../src/go2_moe_cts/agents/moe_cts_cfg.py): expert 수, 네트워크 크기, teacher 비율, 학습률.
- [learning/metrics.py](../src/go2_moe_cts/learning/metrics.py): 보상·추종·생존율·종료 원인 집계와 터미널 출력.
- `logs/rsl_rl/`: 현재 프로젝트의 학습 기록 및 resume/play 검색 위치.
- `learning/`: model, PPO/CTS update, storage, 저장·재개, export 구현.
- [scripts/training/train.py](../scripts/training/train.py): 학습 엔트리포인트.
- [scripts/evaluation/play.py](../scripts/evaluation/play.py): 평가·export 엔트리포인트.
- [scripts/evaluation/validate_cts.py](../scripts/evaluation/validate_cts.py): 실제 물리 실행 검증.
- `scripts/train_yuil_moe_cts.sh`, `scripts/play_yuil_moe_cts.sh`: 기존과 같은 편의 실행 명령.

기존 `go2_moe_cts.yuil_dog_env_cfg`와 `go2_moe_cts.mdp.cts`는 이전 설정 파일을 위한
deprecated import 호환 모듈입니다. 새 코드에서는 위의 역할별 경로를 사용합니다.
기존 `scripts/train.py`, `play.py`, `validate_cts.py`도 새 위치로 실행을 전달합니다.
네트워크 구조와 checkpoint format은 변경하지 않았으므로 기존 CTS checkpoint를 사용할 수 있습니다.

## 패키지 이동

```bash
# IsaacLab 루트에서
./isaaclab.sh -p -m pip wheel project/go2_moe_cts --no-deps --no-build-isolation -w /tmp/cts_wheel
```

wheel에는 Python 코드와 로봇 USD/mesh layer가 포함됩니다.
학습 스크립트까지 함께 옮기려면 프로젝트 폴더를 복사하세요.
Shell launcher는 IsaacLab/project 아래 배치를 기준으로 하며, 다른 위치에서는
해당 IsaacLab의 `isaaclab.sh -p /경로/scripts/training/train.py ...`로 실행할 수 있습니다.

## MuJoCo sim2sim

`sim2sim_mujoco/`는 로컬 Rust GUI/물리 서버, Python 정책 클라이언트, 모델/메시,
student 정책을 포함합니다. 기존 형제 프로젝트와 실행 의존성을 공유하지 않습니다.
실행/정책 변환은 [MuJoCo 안내](../sim2sim_mujoco/README.md)를 참고하세요.
결과는 프로젝트 `logs/sim2sim_mujoco/`에 저장됩니다.
