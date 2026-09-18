# MoE-CTS implementation validation — 2026-09-18

검증 환경: IsaacLab `6.1.14`, RSL-RL `5.0.1`, PyTorch `2.11.0+cu128`, RTX 5070 Ti.
아래 최초 CTS 검증 당시에는 sibling에서 로봇을 가져왔습니다.
현재 독립 구성에 대한 검증은 문서 하단의 추가 기록을 참고하세요.

## 자동 테스트

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../isaaclab.sh -p -m pytest tests -q`

**12 tests passed.**

- Teacher/student partition 및 arbitrary ratio.
- Teacher/student rollout 재정렬, timeout bootstrap, 실제 encoder 업데이트.
- Student PPO gradient 차단, critic→encoder gradient 차단.
- 양쪽 optimizer와 모델/iteration의 checkpoint 복원.
- Student JIT 및 ONNX(설치된 ONNX reference evaluator) 수치 비교.
- Privileged observation 비의존성, 1-env inference.
- Yuil 로봇 계약, 원본 지형/보상/관측, Eval randomization 제거.

Teacher ratio 0.6 회귀 테스트는 이식한 원본의 환경 분할 코드에서 먼저 실패함을 확인했고,
분할 수정 이후 통과했습니다. 기존 0.75 분할은 유지됩니다.

## 실제 PhysX 실행

1. `train_yuil_moe_cts.sh --smoke_test --num_envs 8 --max_iterations 3 --run_name cts_final_check`
   - Teacher 6 / Student 2.
   - 관측 shape: policy `(8,450)`, critic `(8,263)`, single_obs `(8,45)`.
   - 3 iterations 완료, 모든 loss 유한.
   - 마지막 value loss `0.03203`, latent loss `0.003189`, load-balance loss `0.01860`.
   - 모델 및 TorchScript/ONNX/contract 내보내기 완료.
2. 동일 checkpoint `model_3.pt`를 불러와 1 iteration 추가 실행.
   - `iteration=4`로 이어짐, 두 optimizer 복원, export 완료.
3. `play_yuil_moe_cts.sh --viz none --num_envs 1 --num_steps 100` 실행 완료.
   - 단일 환경 student 평가 및 export 확인.
4. 전체 **10 rows × 20 columns** 지형에서 8-env, 200-step 검증.
   - 모든 관측/보상 유한, 모든 ray hit 유효.
   - height scanner와 robot root의 XY 위치 오차 최대 `0.0 m`.
   - 실제 접촉 force 확인: 최대 `344.69 N`.
   - 실제 student와 exported TorchScript action의 최대 차이 `0.0`.
   - 부분 reset 후 해당 환경의 action history 초기화 및 첫 관측 반복 채움 확인.
   - 강제 timeout에 따른 자동 episode reset 확인.

재실행:

```bash
# 현재 프로젝트 폴더 기준; checkpoint는 실제 경로로 지정
../../isaaclab.sh -p scripts/evaluation/validate_cts.py --viz none \
  --checkpoint ../../logs/rsl_rl/yuil_dog_rough_moe_cts/<run>/model_3.pt
```

검증 로그/모델은 IsaacLab 루트에 보관했습니다.

- `logs/rsl_rl/yuil_dog_rough_moe_cts/2026-09-18_09-44-51_cts_final_check/`
- `logs/rsl_rl/yuil_dog_rough_moe_cts/2026-09-18_09-45-31_cts_resume_check/`

## 코드 검사와 범위

`./isaaclab.sh -f` 통과. 프로젝트가 Git 미추적 상태이므로 실제 프로젝트 파일을
명시한 `pre_commit run --files ...`도 별도로 실행해 통과했습니다.
기존 `source/isaaclab_tasks_experimental/.../__init__.py`의 사용자 변경은 수정하지 않았습니다.

Kit 초기화의 inotify watch 부족 메시지와 Yuil USD 경로를 탐색하는 PhysX 경고가 발생했습니다.
위 센서 위치·접촉력·유효 ray 검사를 실제 실행해 정상 동작을 확인했습니다.

이 검증은 **학습 및 배포 파이프라인의 동작 확인**입니다.
300,000-iteration 본 학습, 수렴/험지 주행 성능, 새로운 CTS 정책의 Newton/MuJoCo/실기 성능은
아직 검증하지 않았습니다. 생성된 짧은 학습 모델을 실기용 정책으로 취급하지 마세요.

## 독립 패키지 및 역할별 구조 검증 (2026-09-18)

- `go2_sim2sim` import를 차단한 전체 테스트: **16 passed**.
  동일 독립성 테스트가 변경 전 코드에서는 `ModuleNotFoundError`로 실패하는 것을 확인했습니다.
- 로봇 USD를 임시 폴더에 복사해 하위 layer 10개가 모두 복사 위치 안에서 해결되고,
  미해결 참조와 symlink가 없음을 확인했습니다.
- wheel을 빌드한 뒤 별도 위치에 풀어 해당 wheel의 Python 코드와 로봇 파일만으로
  환경 cfg를 생성하고 USD layer 10개를 로딩했습니다.
- Python import 및 파일 접근 guard로 `go2_sim2sim` 접근을 차단한 상태에서
  8-env, teacher 6 / student 2, 2-iteration 학습과 export를 완료했습니다.
  결과: `logs/rsl_rl/yuil_dog_rough_moe_cts/2026-09-18_10-29-01_standalone_layout_check/`.
- 같은 guard 아래 기존 `cts_final_check/model_3.pt`를 로딩해 8-env, 200-step 물리 검증을 완료했습니다.
  전체 10 × 20 지형, 유효 ray 비율 1.0, scanner XY 오차 0.0 m,
  최대 접촉력 344.69 N, TorchScript action 오차 0.0,
  부분 reset 및 timeout reset 정상 동작을 확인했습니다.

원래 `go2_sim2sim` 폴더는 변경하지 않았습니다. Python 접근 차단과 USD 의존성 검사를
조합한 검증이며, 다른 컴퓨터에서의 설치 검증은 아닙니다. IsaacLab 실행 환경은 필요합니다.
새 파일 위치와 이전 경로 호환성은 [구성 파일 지도](docs/STRUCTURE.md)에 정리했습니다.

## 프로젝트 로그 경로와 PPO형 진단 출력 (2026-09-18)

현재 train 저장·resume 검색·play 자동 검색 경로는 프로젝트 내부 `logs/rsl_rl/`입니다.
기존 IsaacLab 루트의 `yuil_dog_rough_moe_cts`, `go2_moe_cts_45x10`,
`go2_moe_cts_phase1` 기록을 원본 삭제 없이 복사했습니다.
이후 실행은 이전 로그 위치를 자동 참조하지 않습니다.

- 전체 테스트 18개 통과. 첫 부분 에피소드 제외, timeout과 실패가 동시에 발생한 경우,
  teacher/student별 생존율, reset별 가중 평균과 stale extras 제외를 확인했습니다.
- 프로젝트 로그 위치에서 8-env, 3-iteration 학습·체크포인트·export 완료.
- 해당 `model_3.pt`로 60 iterations 추가 학습, `iteration=63`까지 복원·저장 완료.
  결과: `logs/rsl_rl/yuil_dog_rough_moe_cts/2026-09-18_10-45-57_logging_resume_check/`.
- 60개 JSONL 행 전체에서 보상 항목 합 × 0.02 s가 평균 스텝 보상과 일치하고,
  expert gating weight 합이 1임을 확인했습니다. 마지막 로그에 전체 에피소드 14개가 집계됐습니다.
- 전체 저장 지표는 JSONL 및 TensorBoard에도 기록됩니다. 지표 정의는 README의
  '학습 로그 읽기'를 참고하세요. 짧은 테스트 정책의 생존율은 본 학습 성능을 뜻하지 않습니다.
- `/tmp`에서 play launcher를 실행해 프로젝트 내부 최신 `model_63.pt`의 자동 검색,
  1-env 20-step 재생과 export를 확인했습니다.

## GUI 속도 화살표 (2026-09-18)

- 전체 테스트 20개 통과: body heading 변환, 기울기와 무관한 수평 표시,
  yaw 부호와 0 명령 처리 포함.
- 실제 PhysX 2-env 실행에서 XY/yaw 목표·실제 마커 각 2개 갱신과 표시 토글 확인.
- wheel에 독립 화살표 USD 포함 확인. 프로젝트 코드 검사 통과.
- 별도 Kit GUI 검증 실행은 앱 초기화 중 진행이 멈춰 종료했습니다.
  화면 렌더링 결과는 확인하지 못했으며, 위 런타임 검증은 headless에서 마커 callback을 직접 호출했습니다.

## 통합 명령 화살표 (2026-09-18)

- 이전 네 색상 비교 표시를 로봇당 초록색 곡선 화살표 하나로 변경했습니다.
  목표 body twist를 0.6초 유지한 곡선이며, 순수 회전은 상징적 원호입니다.
- 전체 테스트 21개 통과: 직진·옆 이동·좌우 회전 결합·순수 회전·정지 포함.
- 실제 PhysX 2-env에서 총 2개 화살촉과 32개 shaft segment의 갱신 및 표시 토글 확인.
  렌더링 창의 시각 검증은 수행하지 않았습니다.

## 명령 범위 고정 (2026-09-18)

사용자 요청으로 vx ±1.0 m/s, vy ±0.5 m/s, wz ±1.0 rad/s로 고정했습니다.
초기 범위·지형별 상한을 통일하고 명령 범위 확대 curriculum을 해제했습니다.
샘플링 후와 명령 갱신 시 최종 clamp를 적용합니다.
전체 테스트 21개 통과. PhysX에서 0/20,000/50,000/100,000 iteration에 해당하는
단계와 dynamic/일반·끝값 샘플링 조합 320회 및 범위 초과 입력 clamp를 검증했습니다.

## sim2sim 표시 방식 일치 (2026-09-18)

사용자 요청에 따라 단일 곡선 표시 대신 목표(초록, root +0.65 m)와 실제 속도(파랑,
root +0.48 m)를 비교하는 직선 화살표로 변경했습니다. 각 화살표는 vx/vy/wz를 함께 반영합니다.
`go2_sim2sim`의 현재 표시 수식과 크기·재질·mesh를 로컬에 이식했고 런타임 의존성은 없습니다.
100개 입력(순수 회전·정지·기울어진 자세 포함)에서 원본과 scale/quaternion이 정확히 일치했습니다.
PhysX 2-env에서 두 종류의 마커 갱신·표시 토글을 확인했고 전체 테스트 21개가 통과했습니다.
GUI 화면 자체의 렌더링 검증은 별도로 수행하지 않았습니다. 명령 범위 제한은 유지했습니다.

## 명령 추종 기반 승급·강등 (2026-09-18)

거리 기반 조건을 에피소드 평균 명령 추종 점수 기반으로 교체했습니다.
50% 미만 강등, 50% 이상 70% 미만 유지, 70% 이상 승급이며 무표본 reset은 유지합니다.
기존 거리 기반 함수는 보존하되 현재 Yuil task에서 사용하지 않습니다.

- 전체 테스트 24개 통과: 저속 70% 추종, 정지·역방향·과속, 임계값,
  무표본 상태, 마지막 스텝 중복 집계 방지 포함.
- 실제 PhysX 8-env 3-iteration 학습 및 export 완료.
- PhysX 2-env에서 한 환경만 timeout 처리하여 마지막 스텝 반영, 해당 환경만 누적 초기화,
  reset 직후 관측 제외, 다음 스텝 정상 집계를 확인했습니다.
- 새 규칙은 명령 추종률만 판단하며 생존·지형 통과 조건을 추가하지 않았습니다.
  정규화와 정지 명령의 분모 하한은 README에 기록했습니다.
