# UR10e Isaac Lab to MuJoCo Sim2Sim

This project inherits `Isaac-Deploy-Reach-UR10e-v0` and registers project-owned
training tasks:

```text
Isaac-Reach-UR10e-Sim2Sim-v0
Isaac-Reach-UR10e-Sim2Sim-Play-v0
```

The project task transfers its trained policy from Isaac Lab to MuJoCo without
modifying the upstream Isaac Lab configurations.

The transfer contract is:

| Item | Value |
|---|---|
| Policy observation | `[joint_pos(6), joint_vel(6), target_pos(3), target_quat_xyzw(4)]` |
| Policy action | Relative joint-position command, 6 values |
| Action mapping | `q_target = q + 0.0625 * action` |
| Physics rate | 120 Hz |
| Policy rate | 60 Hz |
| Target frame | UR controller `base` frame |
| End effector | `wrist_3_link` |

Project-owned Isaac environment and PPO overrides are located in:

```text
src/ur10_sim2sim/isaaclab_cfg.py
```

The PPO configuration inherits the upstream deployment configuration and sets
`num_steps_per_env = 64`.

The MuJoCo model is intentionally mesh-free. Its kinematic dimensions, masses,
and inertias come from the UR10e URDF bundled with Isaac Sim, while primitive
geometries keep this project portable.

## 1. Train in Isaac Lab

From the IsaacLab repository root:

```bash
project/UR10_sim2sim/scripts/train_isaac.sh
```

For a smoke test, override the number of environments and iterations:

```bash
project/UR10_sim2sim/scripts/train_isaac.sh \
  --num_envs 32 \
  --max_iterations 2
```

The explicit project defaults are:

```text
parallel environments = 4096
PPO iterations        = 1500
rollout steps/env      = 64
visualizer            = none (headless)
```

Run training with the Kit GUI:

```bash
project/UR10_sim2sim/scripts/train_isaac.sh \
  --num_envs 64 \
  --gui
```

Both project tasks visualize two coordinate frames when a GUI is active:

```text
/Visuals/UR10Sim2Sim/EndEffectorFrame  current wrist_3_link frame
/Visuals/UR10Sim2Sim/TargetFrame       commanded target frame
```

The target frame includes the 180-degree transform from ROS `base_link` to the
UR controller `base` frame. Headless training disables these markers
automatically.

Explicitly select headless mode:

```bash
project/UR10_sim2sim/scripts/train_isaac.sh \
  --num_envs 4096 \
  --headless
```

The defaults can also be changed with environment variables:

```bash
UR10_NUM_ENVS=512 \
UR10_MAX_ITERATIONS=1000 \
UR10_VISUALIZER=none \
  project/UR10_sim2sim/scripts/train_isaac.sh
```

Use `--help` or `--project-help` to display the wrapper-specific options.

Training outputs are always written inside this project:

```text
project/UR10_sim2sim/logs/rsl_rl/reach_ur10_sim2sim/
```

The wrapper changes its working directory to the project root before launching
Isaac Lab, so this is independent of the directory from which the wrapper is
called.

## 2. Evaluate and export

Evaluate the latest checkpoint:

```bash
project/UR10_sim2sim/scripts/play_isaac.sh
```

Play opens the Kit GUI by default. It can also run without visualization:

```bash
project/UR10_sim2sim/scripts/play_isaac.sh --headless
```

Export a specific checkpoint:

```bash
project/UR10_sim2sim/scripts/export_policy.sh \
  project/UR10_sim2sim/logs/rsl_rl/reach_ur10_sim2sim/<run>/model_1500.pt
```

Isaac Lab writes both files next to the checkpoint:

```text
<run>/exported/policy.pt
<run>/exported/policy.onnx
```

The MuJoCo runner uses `policy.pt`, which includes the trained observation
normalizer.

## 3. Validate the MuJoCo model

Inspect the Isaac source environment:

```bash
./isaaclab.sh -p project/UR10_sim2sim/scripts/inspect_isaac_env.py \
  --device cpu \
  --viz none
```

Inspect the MuJoCo target environment:

```bash
./isaaclab.sh -p project/UR10_sim2sim/scripts/inspect_mujoco_model.py
```

This checks the joint order, actuator order, timestep, initial state, and
end-effector forward kinematics.

At the nominal joint configuration, the implemented model was verified against
the official Isaac UR10e USD:

```text
Isaac wrist_3_link:  [0.6914002, 0.17415038, 0.67685] m
MuJoCo wrist_3_link: [0.6914000, 0.17415000, 0.67685] m
```

## 4. Run the policy in MuJoCo

With the interactive viewer:

```bash
./isaaclab.sh -p project/UR10_sim2sim/scripts/run_mujoco.py \
  --policy project/UR10_sim2sim/logs/rsl_rl/reach_ur10_sim2sim/<run>/exported/policy.pt
```

Continuously cycle through five targets and slow the GUI to quarter speed:

```bash
./isaaclab.sh -p project/UR10_sim2sim/scripts/run_mujoco.py \
  --policy project/UR10_sim2sim/logs/rsl_rl/reach_ur10_sim2sim/<run>/exported/policy.pt \
  --cycle-targets \
  --duration 0 \
  --playback-speed 0.25 \
  --target-hold 1.0
```

Each waypoint changes both position and orientation within the Isaac training
ranges. The target changes after both the position error is below 3 cm and the
orientation error is below 0.15 rad for one simulation second. A duration of zero
keeps a GUI rollout active until the viewer is closed.

The viewer displays the target frame with long translucent axes and the current
`wrist_3_link` frame with shorter opaque axes. Both use the standard axis colors:
X red, Y green, and Z blue. The final console summary reports both position and
orientation error.

Headless rollout with an `.npz` record:

```bash
./isaaclab.sh -p project/UR10_sim2sim/scripts/run_mujoco.py \
  --policy project/UR10_sim2sim/logs/rsl_rl/reach_ur10_sim2sim/<run>/exported/policy.pt \
  --headless \
  --duration 12 \
  --record project/UR10_sim2sim/artifacts/mujoco_rollout.npz
```

The default target is the center of the Isaac Lab training distribution:

```text
position = [0.8875, -0.225, 0.2] m
RPY      = [pi, 0, -pi/2] rad
```

It can be overridden:

```bash
./isaaclab.sh -p project/UR10_sim2sim/scripts/run_mujoco.py \
  --policy <policy.pt> \
  --target-pos 0.8 -0.2 0.25 \
  --target-rpy 3.1415926536 0.0 -1.5707963268
```

## Important fidelity notes

- Isaac's policy action is not a persistent desired position. At every 60 Hz
  policy step, it is added to the measured joint position.
- Isaac's UR10e has gravity disabled. The MuJoCo model therefore also disables
  gravity.
- The policy target quaternion uses Isaac Lab's `x, y, z, w` order. It is
  reordered to MuJoCo's `w, x, y, z` only for the target visualization body.
- The policy was trained relative to the UR controller `base` frame. The MJCF
  kinematic chain is rooted in that frame, avoiding an extra 180-degree
  transformation in the observation.
- MuJoCo and PhysX contact behavior is irrelevant for this reach-only task,
  but actuator dynamics can still differ. Start by comparing joint and
  end-effector trajectories before tuning gains.

## Project layout

```text
assets/ur10e.xml             MuJoCo robot and scene
scripts/train_isaac.sh       Isaac Lab training
scripts/play_isaac.sh        Isaac Lab evaluation
scripts/export_policy.sh     TorchScript/ONNX export
scripts/run_mujoco.py        MuJoCo policy rollout
scripts/inspect_isaac_env.py Isaac source contract inspection
scripts/inspect_mujoco_model.py
src/ur10_sim2sim/            Shared policy and simulation code
tests/                       Simulator-independent and MuJoCo smoke tests
```
