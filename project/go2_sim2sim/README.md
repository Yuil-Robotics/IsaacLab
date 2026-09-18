# Go2 flat & rough ground PhysX–Newton Sim2Sim

This project trains a Unitree Go2 (and custom quadruped robots) velocity-tracking policy in
PhysX for both **flat-ground** and **rough-terrain (험지)** locomotion, replays the frozen policy in PhysX or Newton/MJWarp, and produces a
deterministic quantitative comparison from the two backends.

The project reuses the Isaac Lab Go2 asset and official locomotion frameworks while providing modular URDF-to-USD conversion tooling to swap in custom robot models. The training logs and task IDs are project-owned so they do not mix with Isaac Lab's built-in runs.

## 0. Download the Go2 model

The official Isaac Lab Go2 and ground-plane USD files are hosted remotely.
Mirror the root USD files and all referenced meshes and materials into this
project before the first run:

```bash
project/go2_sim2sim/scripts/setup_assets.sh
```

All training and evaluation scripts require the local asset. This prevents a
network failure or a remote asset update from changing one side of a Sim2Sim
comparison.

## Tasks

### Flat-Ground Locomotion Tasks
```text
Isaac-Velocity-Flat-Go2-Sim2Sim-v0
Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0
Isaac-Velocity-Flat-Go2-Sim2Sim-Eval-v0
```

### Rough-Terrain (험지) Locomotion Tasks
```text
Isaac-Velocity-Rough-Go2-Sim2Sim-v0
Isaac-Velocity-Rough-Go2-Sim2Sim-Play-v0
Isaac-Velocity-Rough-Go2-Sim2Sim-Eval-v0
```

### Yuil Dog Rough-Terrain Tasks

```text
Isaac-Velocity-Rough-Yuil-Dog-v0
Isaac-Velocity-Rough-Yuil-Dog-Play-v0
Isaac-Velocity-Rough-Yuil-Dog-Eval-v0
```

These task IDs and their `yuil_dog_rough_leg_major` checkpoint directory are
separate from Go2. They reuse the project Go2 rough-terrain reward functions,
weights, terrain, and command distribution. Only the robot asset, joint/body
selectors, and actuator contract are replaced for Yuil Dog. The Yuil task uses
the nominal RS03/RS04 actuator configuration: action/motor delay, motor-strength
scaling, actuator-gain randomization, and joint-friction randomization are not
applied. General environment randomization such as contact material, base mass,
base COM, and training pushes remains inherited from the Go2 training task.

- `Train`: parallel RL training across all sub-terrains and difficulty levels simultaneously (without a level curriculum), contact domain randomization, actuator gain scaling, and external disturbance pushes.
- `Play`: policy visualization across various sub-terrains with observation noise and pushes disabled.
- `Eval`: deterministic reset, nominal mass, clean observations, fixed command suite, and no external pushes.

### Yuil Dog Flat-Ground Tasks

```text
Isaac-Velocity-Flat-Yuil-Dog-v0
Isaac-Velocity-Flat-Yuil-Dog-Play-v0
Isaac-Velocity-Flat-Yuil-Dog-Eval-v0
```

The flat-ground configuration lives in `src/go2_sim2sim/yuil_dog_flat_cfg.py`
and writes checkpoints under `logs/rsl_rl/yuil_dog_flat_leg_major`. The
original rough-terrain configuration and its checkpoints remain available as
the backup version. `scripts/train_yuil_dog.sh` and
`scripts/play_yuil_dog.sh` now select the flat-ground task by default.

The policy rate is 50 Hz (`0.02 s`) and the physics rate is 200 Hz (`0.005 s`).
The Yuil Dog action and joint-observation contract uses leg-major ordering:
`FL(hip_roll, hip_pitch, knee_pitch)`, `FR(...)`, `RL(...)`, then `RR(...)`.
The corresponding zero-based leg index groups are `FL=[0,1,2]`, `FR=[3,4,5]`,
`RL=[6,7,8]`, and `RR=[9,10,11]`.
Rough-terrain training randomizes the policy-command delay over `0-1` policy
steps (`0-20 ms`) and the motor-command delay over `0-3` physics steps
(`0-15 ms`) independently for every environment and episode. Available motor
torque is sampled from `80-100%` of nominal torque per episode. Joint static,
dynamic, and viscous friction are sampled at startup in the `0.0-0.1` range.
Play and evaluation retain nominal actions and actuators without these training
perturbations.

### Yuil Dog RobotLab Recovery Training

The RobotLab flat task keeps the 450-dimensional actor observation and
12-dimensional action contracts used by existing checkpoints while fine-tuning
for disturbance recovery. As in the saved `2026-09-16_11-08-38` environment,
training applies only XY velocity pushes in `[-0.5, 0.5] m/s` every `3-8 s`.
The external force/torque event remains configured at zero over a `10-15 s`
interval, and no angular-velocity impulse is applied. Thirty percent of
environments receive a zero-velocity command so stationary bracing and
recovery stepping are sampled frequently.

The reward suite and curriculum are anchored to the saved
`2026-09-16_11-08-38` environment configuration. To encourage softer foot
placement, the touchdown-speed penalty starts at `0.25 m/s` with twice the
reference weight, while the contact-force limit starts at `100 N` with twice
the reference weight. Swing clearance tracks an `8 cm` target with a wider
`5 cm` reward width and a `0.7` weight so low-clearance policies receive a
useful lift signal without relaxing the touchdown penalties.

Recovery fine-tuning defaults to the verified stable
`2026-09-16_11-08-38/model_24995.pt` checkpoint rather than the saturated
`2026-09-16_17-03-34` run. Override the anchor with
`YUIL_DOG_RECOVERY_LOAD_RUN` and `YUIL_DOG_RECOVERY_CHECKPOINT` if needed.
Train and run the disturbed cross-backend evaluation with:

```bash
project/go2_sim2sim/scripts/train_yuil_dog_recovery.sh \
  --num_envs 4096 --max_iterations 5000 --headless
project/go2_sim2sim/scripts/run_sim2sim_yuil_dog_robotlab_recovery.sh
```

The standard `RobotLab-Eval-v0` task remains disturbance-free. The dedicated
`RobotLab-Recovery-Eval-v0` task uses deterministic resets, nominal actuators,
clean observations, and the reference-run XY velocity pushes.

### Yuil Dog Low-Profile Flat-Ground Tasks

```text
Isaac-Velocity-Flat-Yuil-Dog-Low-Profile-v0
Isaac-Velocity-Flat-Yuil-Dog-Low-Profile-Play-v0
Isaac-Velocity-Flat-Yuil-Dog-Low-Profile-Eval-v0
```

This task inherits the robot, plane, observations, commands, actuator model,
and domain randomization from `yuil_dog_flat_robotlab_cfg.py`, but replaces the
entire reward suite. Base-bottom clearance above `0.35 m` receives a sharply
normalized continuous penalty without immediately terminating the episode.
Cadence is not part of the reward. Completed stance times are instead shaped
toward minimums of `0.30 s` for sagittal motion, `0.24 s` for lateral motion,
and `0.22 s` for turning. The command-aligned stride target increases from
`0.18 m` up to `0.52 m` without using a frequency conversion. Cadence remains
available only as an aggregate and per-command diagnostic metric. Swing-foot
clearance is measured relative to the most recent support height and has a
zero-penalty band from `0.025 m` to `0.08 m`; this prevents both scuffing and
excessive leg lifting without rewarding a particular height. Contact slip and
long swing phases are penalized separately. Hip-roll position and
velocity are strongly regulated during standing and forward/backward motion,
then smoothly released for lateral and yaw commands. Checkpoints are isolated under
`logs/rsl_rl/yuil_dog_flat_low_profile`.

```bash
project/go2_sim2sim/scripts/train_yuil_dog_low_profile.sh --num_envs 4096 --max_iterations 20000 --headless
project/go2_sim2sim/scripts/play_yuil_dog_low_profile.sh --gui
project/go2_sim2sim/scripts/teleop_yuil_dog_low_profile.sh --gui
project/go2_sim2sim/scripts/run_sim2sim_yuil_dog_low_profile.sh
```

### Zero-Command Stationary Balance Sampling

The RobotLab recovery task samples `rel_standing_envs = 0.30`; other flat and
rough tasks retain their task-specific standing ratios. When commanded with
`[0.0, 0.0, 0.0]`:

- The RobotLab policy keeps a weak nominal-pose prior while stable, then reduces
  it to 10-15% strength during a detected recovery so the feet can reposition.
- The other flat policy uses a 5× joint-position regularizer, while the rough
  policy uses a less restrictive 3× scale so it can widen its stance on uneven
  support.
- The rough feet air-time reward transitions smoothly between stance support and rhythmic stepping instead of switching on for every non-zero command.
- Base motion and slip penalties ensure the robot maintains a rock-solid, drift-free stance even on slopes and uneven footholds.

The rough task additionally assigns 25% of all environments pure forward/backward commands
`[vx, 0.0, 0.0]`. These environments use a mild straight-motion hip-velocity penalty to reduce
lateral foot swing without constraining turning or lateral locomotion.

### Rough-Terrain Reward Suite & Sub-Terrains
The rough terrain task trains on a procedural grid of diverse sub-terrains:
- **Random Grid Boxes**: Discrete obstacle heights $0.025 \sim 0.10\,\text{m}$, width $0.45\,\text{m}$.
- **Random Rough Heightfield**: Noise amplitude $0.01 \sim 0.06\,\text{m}$.
- **Pyramid Slopes**: Incline angles $0^\circ \sim 20^\circ$ ($0.0 \sim 0.35\,\text{rad}$).

The reward function enforces:
- **Linear & Angular Velocity Tracking**: Squared exponential tracking keeps accurate command following distinct from merely surviving.
- **Air Time & Cadence**: Smoothly blends rhythmic stepping while moving with continuous stance support while stopped.
- **Terrain-Relative Foot Clearance**: Measures swing height from each foot's most recent support height rather than absolute world height.
- **Command-Gated Trot Coordination**: Prefers a diagonal trot at meaningful planar speeds while allowing adaptive low-speed, turning, and rough-terrain contacts.
- **Four-Foot Flight Suppression**: Allows brief contact transitions, then removes gait, air-time, and clearance incentives and directly penalizes sustained all-foot flight.
- **Foot Slip Penalty**: Penalizes horizontal sliding on slopes and step edges.
- **Undesired Contact Penalty**: Heavily penalizes collision of the base torso, hips, thighs, or calves with step edges.
- **Slope-Compliant Stability**: Uses squared vertical-motion, roll/pitch-rate, and orientation costs so normal terrain adaptation is weakly penalized while unstable motion grows expensive.
- **Action & Motor Regularization**: Suppresses jerky commands, joint acceleration, and motor torque spikes while directly penalizing soft joint-limit violations.

---

## 1. Custom Robot URDF -> USD Conversion & Swapping

For the Yuil Dog description in this repository, run the dedicated converter:

```bash
project/go2_sim2sim/scripts/convert_yuil_dog_urdf.sh
```

The converter validates the URDF and referenced meshes, imports detailed mesh collisions with convex decomposition by
default, and writes the stable training entrypoint to `assets/yuil_dog/yuil_dog.usda`. This preserves concave link
geometry more closely than a single convex hull. Joint drives are intentionally disabled in the USD; define the
training actuator gains in the Yuil Dog `ArticulationCfg` instead. Fixed joints are preserved by default so each
`*_foot` remains an independent rigid body required by the Go2 foot-contact, clearance, and slip rewards.

The project pairs this USD with `YUIL_DOG_CFG` in `src/go2_sim2sim/asset_cfg.py`. Its three nominal actuator groups
model the RS03 hip-roll motors and the RS04 hip-pitch and knee motors. The asset configuration intentionally contains
no command delay or motor-strength randomization; task-specific training configuration can add those later.

Use `--strict` once the URDF is production-ready to reject placeholder joint limits. Detailed visual meshes reused as
collision sources are accepted with convex decomposition and reported as notes because they increase first-load cooking
time and asset size.

When your own robot's URDF description is ready, convert it into an IsaacLab-ready layered USD asset:

```bash
project/go2_sim2sim/scripts/convert_urdf_to_usd.sh \
  --urdf /path/to/my_robot.urdf \
  --output_dir project/go2_sim2sim/assets/custom_robot \
  --usd_filename robot.usd \
  --robot_type Quadruped \
  --stiffness 25.0 \
  --damping 0.5 \
  --collision_type "Convex Hull"
```

### Activating the Custom Robot
To train or evaluate policies with your custom robot instead of Unitree Go2, set the environment variables:

```bash
export ROBOT_TYPE=custom
export CUSTOM_ROBOT_USD_PATH="project/go2_sim2sim/assets/custom_robot/robot.usd"

# Optional: if your URDF uses custom joint/foot names
export CUSTOM_ROBOT_JOINT_NAMES="FL_hip_joint,FR_hip_joint,RL_hip_joint,RR_hip_joint,FL_thigh_joint,FR_thigh_joint,RL_thigh_joint,RR_thigh_joint,FL_calf_joint,FR_calf_joint,RL_calf_joint,RR_calf_joint"
export CUSTOM_ROBOT_FOOT_NAMES="FL_foot,FR_foot,RL_foot,RR_foot"
```

All flat, rough, and rear-standing tasks will automatically instantiate and simulate your custom robot asset.

---

## 2. Train Locomotion Policies in PhysX

### Flat-Ground Training:
```bash
project/go2_sim2sim/scripts/train_physx.sh \
  --num_envs 4096 \
  --headless
```

### Rough-Terrain (험지) Training:
```bash
project/go2_sim2sim/scripts/train_rough.sh \
  --num_envs 4096 \
  --headless
```

### Yuil Dog Rough-Terrain Training:

```bash
project/go2_sim2sim/scripts/train_yuil_dog.sh \
  --num_envs 4096 \
  --headless
```

Play the latest Yuil Dog checkpoint with:

```bash
project/go2_sim2sim/scripts/play_yuil_dog.sh --gui
```

Teleoperate the Yuil Dog RL policy with LiDAR and vision point clouds disabled by
default. Either perception path can be enabled without changing the policy
observation contract:

```bash
project/go2_sim2sim/scripts/teleop_policy.sh
project/go2_sim2sim/scripts/teleop_policy.sh --lidar-vis --camera-vis
project/go2_sim2sim/scripts/teleop_mapping.sh
project/go2_sim2sim/scripts/teleop_mapping.sh --lidar-vis --camera-vis
```

For a short GUI smoke test:
```bash
project/go2_sim2sim/scripts/train_rough.sh \
  --num_envs 64 \
  --max_iterations 2 \
  --gui
```

Resume from the latest checkpoint:
```bash
project/go2_sim2sim/scripts/train_rough.sh \
  --num_envs 4096 \
  --resume \
  --headless
```

### Rough-Terrain History Training

The history task keeps the rough-task rewards unchanged. Its actor receives
five flattened 45-value observations (225 values total) without base linear
velocity. During training, the critic additionally receives the current
three-axis base linear velocity as privileged state (228 values total).

```text
Isaac-Velocity-Rough-Go2-History-v0
Isaac-Velocity-Rough-Go2-History-Play-v0
Isaac-Velocity-Rough-Go2-History-Eval-v0
```

Start or resume its isolated training run:

```bash
project/go2_sim2sim/scripts/train_rough_history.sh --num_envs 4096 --headless
project/go2_sim2sim/scripts/train_rough_history.sh --num_envs 4096 --resume --headless
```

Visualize the latest history-policy checkpoint with either backend:

```bash
project/go2_sim2sim/scripts/play_rough_history.sh --physics physx --gui
project/go2_sim2sim/scripts/play_rough_history.sh --physics newton_mjwarp --gui
```

Generate a PhysX--Newton/MJWarp comparison for the latest history checkpoint.
The output directory includes the selected training-run timestamp, checkpoint
iteration, and a unique evaluation timestamp so repeated evaluations never
overwrite an open report:

```bash
project/go2_sim2sim/scripts/run_sim2sim_history.sh
```

This command defaults to the latest checkpoint under
`logs/rsl_rl/go2_rough_history_5step_sim2real/` and uses the 225-dimensional
history evaluation task.

Checkpoints are saved under:

```text
logs/rsl_rl/go2_rough_history_5step_sim2real/
```

Checkpoints are saved under:
```text
Flat:  logs/rsl_rl/go2_flat_spot_rewards_sim2sim_dr/
Rough: logs/rsl_rl/go2_rough_spot_rewards_sim2sim_dr/
```

---

## 3. Visual Policy Checks

### Play Flat-Ground Policy:
```bash
project/go2_sim2sim/scripts/play_policy.sh --physics physx
```

### Play Rough-Terrain Policy on Diverse Terrains:
```bash
project/go2_sim2sim/scripts/play_rough.sh --physics physx --num_envs 16 --gui
```

Check resolved checkpoint:
```bash
project/go2_sim2sim/scripts/play_rough.sh --print-checkpoint
```

---

## 4. Rear-foot standing task

The rear-standing experiment is isolated from the velocity policy and its
checkpoints:

```text
Isaac-Rear-Stand-Go2-Sim2Sim-v0
Isaac-Rear-Stand-Go2-Sim2Sim-Play-v0
logs/rsl_rl/go2_rear_bent_balance/
```

Train and visualize the task with:

```bash
project/go2_sim2sim/scripts/train_rear_stand.sh --num_envs 4096 --headless
project/go2_sim2sim/scripts/play_rear_stand.sh
```

---

## 5. Quantitative policy Sim2Sim

The default Sim2Sim task is the flat-ground Yuil Dog RobotLab evaluation task,
matching checkpoints trained with `yuil_dog_flat_robotlab_cfg.py`. Run both
backends and generate the comparison report with:

```bash
project/go2_sim2sim/scripts/run_sim2sim.sh
```

Defaults can be changed with environment variables:

```bash
GO2_EVAL_NUM_ENVS=512 \
GO2_EVAL_DURATION=15 \
GO2_EVAL_SEED=42 \
project/go2_sim2sim/scripts/run_sim2sim.sh
```

Each environment is assigned one of these fixed commands:

```text
stand
forward 0.5 m/s
forward 1.0 m/s
forward 2.0 m/s
backward 0.5 m/s
lateral 0.5 m/s
yaw 0.5 rad/s
combined: vx=0.5 m/s, vy=0.2 m/s, wz=0.4 rad/s
```

Results are written below a checkpoint-specific directory:

```text
logs/sim2sim/<training-run>_<model>_yuil_dog_flat_robotlab/<evaluation-run>/
  physx.json
  newton_mjwarp.json
  comparison.json
  report.md
```

The action-saturation section reports total, positive, and negative saturation
rates for every joint, both overall and per command. Directional rates use the
raw policy output before the simulator wrapper clamps it.

### Yuil Dog RobotLab PhysX--Newton/MJWarp comparison

The RobotLab flat-ground configuration uses the same multi-physics preset as
the other project tasks. Play a checkpoint with NVIDIA Newton's MuJoCo Warp
solver using:

```bash
project/go2_sim2sim/scripts/play_yuil_dog_robotlab.sh \
  --physics newton_mjwarp \
  --checkpoint /path/to/model.pt
```

Run the same frozen RobotLab policy in PhysX and Newton/MJWarp, then feed both
JSON files into `scripts/compare_policy_metrics.py` automatically:

```bash
project/go2_sim2sim/scripts/run_sim2sim_yuil_dog_robotlab.sh

# Optional explicit checkpoint and shorter smoke comparison
project/go2_sim2sim/scripts/run_sim2sim_yuil_dog_robotlab.sh \
  --checkpoint /path/to/model.pt \
  --num_envs 8 \
  --duration 1.0
```

The evaluator derives the joint contract from
`YuilDogFlatRobotLabEvalEnvCfg.actions.joint_pos`, so action saturation, joint
velocity, torque, power, and model-contract arrays follow the Yuil Dog
leg-major order instead of the Go2 order. Results are written as:

```text
logs/sim2sim/<training-run>_<model>_yuil_dog_flat_robotlab/<evaluation-run>/
  physx.json
  newton_mjwarp.json
  comparison.json
  report.md
```
