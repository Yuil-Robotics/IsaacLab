# Yuil 3 kg Robot Arm Reinforcement Learning

This project owns the Isaac Lab training configuration for the company-provided
`YBL-A603_250924.SLDASM` six-axis robot. It uses the converted local USD and
leaves the supplied URDF, USD layers, and STL meshes unchanged.

Registered tasks:

```text
Isaac-Reach-Yuil-3kg-v0
Isaac-Reach-Yuil-3kg-Play-v0
```

## Current control contract

| Item | Value |
|---|---|
| Controlled joints | `j1_link_rev` through `j6_link_rev` |
| Base frame | `base_link` |
| End effector | `j6_adapter_endplate` |
| Observation | `[joint_pos(6), joint_vel(6), target_pos(3), target_quat(4)]` |
| Observation dimension | 19 |
| Action | Six relative joint-position commands |
| Action mapping | `q_target = q_measured + 0.05 * action` |
| Physics rate | 120 Hz |
| Policy rate | 60 Hz |
| Rollout | 64 steps per environment |
| Episode | 12 seconds / 720 policy steps |
| Default environments | 4096 |
| PPO iterations | 1500 |

The environment inherits the Isaac Lab Deploy Reach reward and PPO architecture.
The company robot asset, joint names, frame transformer, workspace, visual
markers, action scale, and output directory are owned by this project.

## Important provisional settings

The provided URDF defines joint limits, mass, inertia, effort limits, and velocity
limits. The converted USD also contains position-drive gains, but their angular
unit conversion produces runtime stiffness and damping values about 57.3 times
larger than the intended values. Restoring only the pre-conversion values was
useful for the gravity-off diagnostic, but those gains could not support the arm
after gravity was enabled. The current gravity-on baseline uses explicit
per-joint stiffness and critical damping together with the URDF effort and
velocity limits. Gain and joint-friction randomization remain disabled until
the baseline and real actuator data are validated.

The following values must be validated with the company before real deployment:

- real home/zero joint position;
- controller position-loop gains and command interface;
- whether the real controller performs gravity compensation;
- motor-side versus joint-side encoder convention and gear ratios;
- safe Cartesian workspace;
- physical TCP offset from `j6_adapter_endplate`;
- continuous-joint wrapping convention;
- payload mass, center of mass, and inertia;
- collision geometry and permitted self-collision pairs.
- final joint position-drive stiffness and damping.

Gravity is enabled at `-9.81 m/s^2`; self-collision remains disabled. The
current controller does not apply gravity feed-forward, so the joint PD drives
must support the arm within the policy's relative command range. Whether the
real controller performs gravity compensation remains unverified.

The inherited remote table and ground assets are disabled. The initial Reach
task is fixed-base, contact-free, and uses only company-local USD dependencies.

## Inspect the supplied robot

This command does not launch Isaac Sim:

```bash
./isaaclab.sh -p project/yuil_3kg_robot_arm/scripts/inspect_urdf.py
```

Inspect one instantiated Isaac environment:

```bash
./isaaclab.sh -p project/yuil_3kg_robot_arm/scripts/inspect_isaac_env.py \
  --device cpu \
  --viz none
```

## Smoke-test training

Start with one environment and one iteration before using thousands of
environments:

```bash
project/yuil_3kg_robot_arm/scripts/train_isaac.sh \
  --num_envs 1 \
  --max_iterations 1 \
  --headless
```

Then inspect the robot and coordinate frames in the GUI:

```bash
project/yuil_3kg_robot_arm/scripts/train_isaac.sh \
  --num_envs 1 \
  --max_iterations 2 \
  --gui
```

## USD-based PhysX-to-Newton sim2sim

The Yuil task exposes `physx` and `newton_mjwarp` physics presets. Both load the
same company USD; no MJCF model file is used as an input.

Run the deterministic two-second joint-step probe on both backends:

```bash
project/yuil_3kg_robot_arm/scripts/run_sim2sim.sh
```

The probe disables reset-time joint-friction randomization and compares the
runtime mass, center of mass, inertia, joint limits, drive gains, friction, and
joint response. Both backends use the same `-9.81 m/s^2` scene gravity. Results
are written to:

```text
logs/sim2sim/physx.json
logs/sim2sim/newton_mjwarp.json
```

This probe is the contact-free articulation baseline. Newton currently warns
that the converted USD has `CollisionAPI` on unsupported intermediate
`UsdGeomGPrim` nodes. Validate or rebuild the collision layer before using this
asset for contact-rich sim2sim.

Select the Newton/MJWarp backend for an ordinary project command with:

```bash
project/yuil_3kg_robot_arm/scripts/train_isaac.sh \
  --num_envs 64 \
  --max_iterations 1 \
  --headless \
  physics=newton_mjwarp
```

The Newton preset uses MJWarp's `implicitfast` integrator and its own solver
iterations. PhysX articulation solver-iteration settings are not reused by
MJWarp.

The GUI markers are:

```text
/Visuals/Yuil3kg/EndEffectorFrame
/Visuals/Yuil3kg/TargetFrame
```

## Full training

```bash
project/yuil_3kg_robot_arm/scripts/train_isaac.sh
```

Override the defaults:

```bash
project/yuil_3kg_robot_arm/scripts/train_isaac.sh \
  --num_envs 512 \
  --max_iterations 1500 \
  --headless
```

Resume from the highest numbered checkpoint in the gravity-on experiment:

```bash
project/yuil_3kg_robot_arm/scripts/train_isaac.sh \
  --num_envs 2048 \
  --gui \
  --resume
```

Inspect the checkpoint that `--resume` will load without launching training:

```bash
project/yuil_3kg_robot_arm/scripts/train_isaac.sh \
  --print-resume-checkpoint
```

Equivalent environment variables:

```text
YUIL_NUM_ENVS
YUIL_MAX_ITERATIONS
YUIL_VISUALIZER
```

All training outputs are project-local:

```text
project/yuil_3kg_robot_arm/logs/rsl_rl/reach_yuil_3kg_gravity/
```

## Play and export

Evaluate the latest checkpoint:

```bash
project/yuil_3kg_robot_arm/scripts/play_isaac.sh
```

Automatic selection uses the highest numeric `model_<iteration>.pt` under this
project's `reach_yuil_3kg_gravity` logs. Gravity-off checkpoints from the old
`reach_yuil_3kg` experiment are deliberately excluded. It does not select the
newest run directory, which may contain only a smoke-test `model_0.pt`. Check
the selected file without launching the simulator:

```bash
project/yuil_3kg_robot_arm/scripts/play_isaac.sh --print-checkpoint
```

Evaluate a specific checkpoint:

```bash
project/yuil_3kg_robot_arm/scripts/play_isaac.sh \
  --checkpoint project/yuil_3kg_robot_arm/logs/rsl_rl/reach_yuil_3kg_gravity/<run>/model_1499.pt
```

Use that same automatically selected policy with Newton/MJWarp:

```bash
project/yuil_3kg_robot_arm/scripts/play_isaac.sh \
  --headless \
  physics=newton_mjwarp
```

Export TorchScript and ONNX policies:

```bash
project/yuil_3kg_robot_arm/scripts/export_policy.sh \
  project/yuil_3kg_robot_arm/logs/rsl_rl/reach_yuil_3kg_gravity/<run>/model_1499.pt
```

The exported policies are written under the selected run:

```text
<run>/exported/policy.pt
<run>/exported/policy.onnx
```

## Project layout

```text
src/yuil_3kg_robot_arm/asset_cfg.py      Local USD and actuator configuration
src/yuil_3kg_robot_arm/isaaclab_cfg.py   Reach environment, PPO, Gym tasks
scripts/inspect_urdf.py                  URDF contract check
scripts/inspect_isaac_env.py             Runtime Isaac environment check
scripts/sim2sim_probe.py                 Deterministic single-backend probe
scripts/compare_sim2sim.py               PhysX/Newton result comparison
scripts/run_sim2sim.sh                    Two-backend USD sim2sim runner
scripts/train_isaac.sh                   Project-local training
scripts/play_isaac.sh                    Policy evaluation
scripts/export_policy.sh                 TorchScript/ONNX export
tests/test_project_cfg.py                Static configuration tests
```
