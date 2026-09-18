#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate a trained Yuil Dog RL policy with keyboard velocity commands."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go2_sim2sim.asset_cfg import (  # noqa: E402
    GO2_FOOT_NAMES,
    GO2_JOINT_NAMES,
    YUIL_DOG_BASE_PRIM_PATH,
    YUIL_DOG_FOOT_NAMES,
    YUIL_DOG_JOINT_NAMES,
)
from go2_sim2sim.isaaclab_cfg import register_tasks  # noqa: E402
from go2_sim2sim.robotlab_rewards import get_base_bottom_height_from_reward  # noqa: E402
from go2_sim2sim.rough_mdp import (  # noqa: E402
    get_ground_relative_base_height,
    get_ground_relative_foot_heights,
)
from go2_sim2sim.sensor_patterns import create_d435i_camera_cfg, create_lidar_cfg  # noqa: E402
from go2_sim2sim.teleop_input import LongitudinalHoldCommand, UnifiedKeyboardTeleop  # noqa: E402
from go2_sim2sim.teleop_recorder import (  # noqa: E402
    RollingFootContactFrequency,
    TeleopDataRecorder,
    extract_foot_contact_forces,
    extract_foot_contact_states,
)
from go2_sim2sim.yuil_dog_cfg import YUIL_DOG_ROUGH_PLAY_TASK_ID  # noqa: E402

register_tasks()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=YUIL_DOG_ROUGH_PLAY_TASK_ID, help="Gymnasium task ID.")
parser.add_argument("--checkpoint", type=Path, required=True, help="Policy checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulation environments.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument("--vx", type=float, default=0.5, help="Linear velocity magnitude along X [m/s].")
parser.add_argument(
    "--vx_fast",
    "--vx-fast",
    type=float,
    default=1.0,
    help="Linear X speed after holding W/S [m/s].",
)
parser.add_argument(
    "--vx_hold_time",
    "--vx-hold-time",
    type=float,
    default=1.0,
    help="W/S hold duration before accelerating [s].",
)
parser.add_argument("--vy", type=float, default=0.5, help="Linear velocity magnitude along Y [m/s].")
parser.add_argument("--wz", type=float, default=1.0, help="Angular yaw velocity magnitude [rad/s].")
parser.add_argument("--lidar_vis", "--lidar-vis", action="store_true", help="Enable LiDAR point cloud visualization.")
parser.add_argument(
    "--camera_vis",
    "--camera-vis",
    "--cam-vis",
    action="store_true",
    default=False,
    help="Enable D435i depth point-cloud visualization.",
)
parser.add_argument(
    "--real-time", "--real_time", action="store_true", default=False, help="Run in real-time step rate."
)
parser.add_argument("--gui", action="store_true", default=True, help="Enable Kit GUI visualizer.")
parser.add_argument(
    "--record_dir",
    "--record-dir",
    type=Path,
    default=PROJECT_ROOT / "logs" / "teleop_recordings",
    help="Directory to save teleoperation CSV recordings.",
)
parser.add_argument(
    "--record_duration",
    "--record-duration",
    type=float,
    default=10.0,
    help="Recording duration in seconds of simulation time (default: 10.0s).",
)
parser.add_argument(
    "--push",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable automatic periodic push disturbances (default: True).",
)
parser.add_argument(
    "--push_vel",
    "--push-vel",
    type=float,
    default=0.5,
    help="Push velocity magnitude [m/s] (default: 0.5 m/s matching train).",
)
parser.add_argument(
    "--push_interval_min",
    "--push-interval-min",
    type=float,
    default=10.0,
    help="Minimum interval between automatic pushes [s] (default: 10.0s matching train).",
)
parser.add_argument(
    "--push_interval_max",
    "--push-interval-max",
    type=float,
    default=15.0,
    help="Maximum interval between automatic pushes [s] (default: 15.0s matching train).",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# Enable cameras subsystem when camera_vis is requested
if getattr(args_cli, "camera_vis", False):
    args_cli.enable_cameras = True

# Ensure Kit GUI visualizer is default unless headless is specified
if getattr(args_cli, "headless", False):
    args_cli.visualizer = ["none"]
elif not getattr(args_cli, "visualizer", None):
    args_cli.visualizer = ["kit"]


def _resolve_sensor_prim_path(task_id: str) -> str:
    """Return the robot base prim used to mount optional teleoperation sensors."""
    if "Yuil-Dog" in task_id:
        return YUIL_DOG_BASE_PRIM_PATH
    return "{ENV_REGEX_NS}/Robot/base"


def main() -> None:
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Disable heading controller so direct angular velocity (wz) teleop is used
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.ranges.heading = None

    # Disable episode time_out so the robot never resets automatically
    env_cfg.episode_length_s = 1.0e9
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    # Configure push / force disturbances during teleoperation
    if hasattr(env_cfg, "events"):
        if not args_cli.push:
            if hasattr(env_cfg.events, "push_robot"):
                env_cfg.events.push_robot = None
            if hasattr(env_cfg.events, "base_external_force_torque"):
                env_cfg.events.base_external_force_torque = None
        else:
            from isaaclab.envs import mdp
            from isaaclab.managers import EventTermCfg, SceneEntityCfg

            env_cfg.events.push_robot = EventTermCfg(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(args_cli.push_interval_min, args_cli.push_interval_max),
                params={
                    "velocity_range": {
                        "x": (-args_cli.push_vel, args_cli.push_vel),
                        "y": (-args_cli.push_vel, args_cli.push_vel),
                    }
                },
            )
            # Physical external force/torque wrench matching train configuration
            base_asset_cfg = SceneEntityCfg("robot", body_names="base_link")
            env_cfg.events.base_external_force_torque = EventTermCfg(
                func=mdp.apply_external_force_torque,
                mode="interval",
                interval_range_s=(args_cli.push_interval_min, args_cli.push_interval_max),
                params={
                    "asset_cfg": base_asset_cfg,
                    "force_range": (0.0, 0.0),
                    "torque_range": (0.0, 0.0),
                },
            )

    # Keep optional perception sensors out of the policy observation and disabled by default.
    env_cfg.scene.lidar = None
    env_cfg.scene.camera = None
    sensor_prim_path = _resolve_sensor_prim_path(args_cli.task)

    # Ensure height_scanner is present on base prim for ground-relative height measurement on rough terrain
    terrain_type = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_type", None)
    if env_cfg.scene.height_scanner is None and terrain_type not in (None, "plane"):
        from isaaclab.sensors.ray_caster import RayCasterCfg
        from isaaclab.sensors.ray_caster.patterns import grid_pattern
        from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg

        env_cfg.scene.height_scanner = RayCasterCfg(
            prim_path=sensor_prim_path,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=GridPatternCfg(func=grid_pattern, resolution=0.1, size=[0.1, 0.1]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
    # Policy remains strictly blind (proprioceptive only)
    env_cfg.observations.policy.height_scan = None

    # Apply LiDAR point-cloud visualization if requested.
    if args_cli.lidar_vis:
        env_cfg.scene.lidar = create_lidar_cfg(
            prim_path=sensor_prim_path,
            debug_vis=True,
        )

    # Apply D435i depth point-cloud visualization if requested.
    if getattr(args_cli, "camera_vis", False):
        env_cfg.scene.camera = create_d435i_camera_cfg(
            prim_path=sensor_prim_path,
            debug_vis=True,
        )

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = str(checkpoint.parent)
        agent_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            agent_cfg.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        configure_seed(args_cli.seed, True)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        unwrapped = vec_env.unwrapped
        command_term = unwrapped.command_manager.get_term("base_velocity")

        vx = args_cli.vx
        vx_fast = args_cli.vx_fast
        vy = args_cli.vy
        wz = args_cli.wz
        if hasattr(command_term, "cfg") and hasattr(command_term.cfg, "ranges"):
            ranges = command_term.cfg.ranges
            if hasattr(ranges, "lin_vel_x") and ranges.lin_vel_x is not None:
                max_x = max(abs(ranges.lin_vel_x[0]), abs(ranges.lin_vel_x[1]))
                if vx_fast > max_x:
                    vx_fast = max_x
                if vx > max_x:
                    vx = max_x * 0.5
            if hasattr(ranges, "lin_vel_y") and ranges.lin_vel_y is not None:
                max_y = max(abs(ranges.lin_vel_y[0]), abs(ranges.lin_vel_y[1]))
                if vy > max_y:
                    vy = max_y
            if hasattr(ranges, "ang_vel_z") and ranges.ang_vel_z is not None:
                max_z = max(abs(ranges.ang_vel_z[0]), abs(ranges.ang_vel_z[1]))
                if wz > max_z:
                    wz = max_z

        teleop = UnifiedKeyboardTeleop(
            vx=vx,
            vx_fast=vx_fast,
            vx_hold_time=args_cli.vx_hold_time,
            vy=vy,
            wz=wz,
            device=vec_env.unwrapped.device,
        )

        foot_names = YUIL_DOG_FOOT_NAMES if "Yuil-Dog" in args_cli.task else GO2_FOOT_NAMES
        recorder = TeleopDataRecorder(
            output_dir=args_cli.record_dir,
            joint_names=YUIL_DOG_JOINT_NAMES if "Yuil-Dog" in args_cli.task else GO2_JOINT_NAMES,
            foot_names=foot_names,
            record_duration_s=args_cli.record_duration,
            checkpoint=checkpoint,
        )

        obs = vec_env.get_observations()
        sim_dt = unwrapped.step_dt
        foot_frequency_tracker = RollingFootContactFrequency(foot_names, step_dt=sim_dt, window_s=1.0)
        foot_frequencies_hz = tuple(0.0 for _ in foot_names)
        try:
            checkpoint_display = str(checkpoint.relative_to(PROJECT_ROOT / "logs" / "rsl_rl"))
        except ValueError:
            try:
                checkpoint_display = str(checkpoint.relative_to(PROJECT_ROOT))
            except ValueError:
                checkpoint_display = f"{checkpoint.parent.name}/{checkpoint.name}"

        robot = unwrapped.scene["robot"]
        ordered_joint_ids = None
        if hasattr(robot.data, "joint_names"):
            name_to_id = {name: i for i, name in enumerate(robot.data.joint_names)}
            for candidate_names in (YUIL_DOG_JOINT_NAMES, GO2_JOINT_NAMES):
                if all(name in name_to_id for name in candidate_names):
                    ordered_joint_ids = [name_to_id[name] for name in candidate_names]
                    break
        if ordered_joint_ids is None and hasattr(robot, "find_joints"):
            for candidate_names in (YUIL_DOG_JOINT_NAMES, GO2_JOINT_NAMES):
                matched_ids, _ = robot.find_joints(list(candidate_names), preserve_order=True)
                if len(matched_ids) == 12:
                    ordered_joint_ids = matched_ids
                    break
        if ordered_joint_ids is None:
            ordered_joint_ids = list(range(min(12, getattr(robot, "num_joints", 12))))

        last_print_cmd = None
        last_status_time = 0.0
        printed_lines = 0

        try:
            while True:
                step_start_time = time.time()
                cmd_tensor = teleop.get_command()

                if teleop.check_record_trigger():
                    if not recorder.is_recording:
                        recorder.start(task_name=args_cli.task, step_dt=sim_dt, checkpoint=checkpoint)
                        ckpt_info_str = f" | {recorder.checkpoint_display}" if recorder.checkpoint_display else ""
                        print(
                            f"\n[REC] Started {recorder.record_duration_s:.1f}s recording "
                            f"({recorder._target_steps} steps @ {1.0 / sim_dt:.1f}Hz){ckpt_info_str}..."
                        )

                if teleop.check_push_trigger():
                    angle = float(np.random.uniform(0, 2 * np.pi))
                    push_mag = args_cli.push_vel
                    push_vx = float(np.cos(angle) * push_mag)
                    push_vy = float(np.sin(angle) * push_mag)
                    if hasattr(robot.data, "root_vel_w"):
                        vel_w = (
                            robot.data.root_vel_w.torch.clone()
                            if hasattr(robot.data.root_vel_w, "torch")
                            else robot.data.root_vel_w.clone()
                        )
                        vel_w[:, 0] += push_vx
                        vel_w[:, 1] += push_vy
                        robot.write_root_velocity_to_sim(root_velocity=vel_w)
                        printed_lines = 0
                        print(
                            f"\n[PUSH] Applied manual external force impulse: "
                            f"vx={push_vx:+.2f} m/s, vy={push_vy:+.2f} m/s (|v|={push_mag:.2f} m/s)"
                        )

                # Broadcast command to all environments
                command_term.vel_command_b[:, :3] = cmd_tensor
                command_term.command[:, :3] = cmd_tensor
                command_term.is_heading_env.zero_()
                command_term.is_standing_env.zero_()
                if hasattr(command_term, "is_straight_env"):
                    command_term.is_straight_env.zero_()
                if hasattr(command_term, "is_turning_env"):
                    command_term.is_turning_env.zero_()
                if hasattr(command_term, "is_lateral_env"):
                    command_term.is_lateral_env.zero_()

                cmd_tuple = (
                    round(float(cmd_tensor[0]), 2),
                    round(float(cmd_tensor[1]), 2),
                    round(float(cmd_tensor[2]), 2),
                )
                command_changed = cmd_tuple != last_print_cmd

                with torch.inference_mode():
                    actions = policy(obs)
                    obs_next, _, dones, _ = vec_env.step(actions)

                done_env_0 = bool(dones[0].item()) if isinstance(dones, torch.Tensor) else bool(dones[0])
                if done_env_0:
                    foot_frequency_tracker.reset()
                foot_contact_states = extract_foot_contact_states(unwrapped, foot_names=foot_names)
                foot_frequencies_hz = foot_frequency_tracker.update(foot_contact_states)

                # Extract joint positions and action targets
                action_term = unwrapped.action_manager.get_term("joint_pos")
                target_joint_pos_rad = (
                    action_term.processed_actions[0]
                    if hasattr(action_term, "processed_actions")
                    else torch.zeros(12, device=unwrapped.device)
                )

                if hasattr(robot.data, "joint_pos"):
                    pos_tensor = (
                        robot.data.joint_pos.torch
                        if hasattr(robot.data.joint_pos, "torch")
                        else robot.data.joint_pos
                    )
                    jp = [pos_tensor[0, j_id].item() for j_id in ordered_joint_ids]
                else:
                    jp = [0.0] * 12
                while len(jp) < 12:
                    jp.append(0.0)

                current_base_height = None
                if recorder.is_recording:
                    reward_aligned_heights = get_base_bottom_height_from_reward(unwrapped)
                    if reward_aligned_heights is None:
                        current_base_height = get_ground_relative_base_height(unwrapped.scene)
                    else:
                        current_base_height = reward_aligned_heights[0].item()

                    is_done = recorder.record_step(
                        unwrapped=unwrapped,
                        obs=obs,
                        actions=actions,
                        target_joint_pos_rad=target_joint_pos_rad,
                        actual_joint_pos_rad=jp,
                        base_height=current_base_height,
                    )
                    if is_done:
                        printed_lines = 0
                        print(
                            f"\n[REC] Recording complete! Saved {recorder._last_saved_steps} rows to: "
                            f"{recorder.last_saved_file}"
                        )
                        recorder.print_summary_metrics()

                obs = obs_next

                now = time.monotonic()
                if command_changed or now - last_status_time > 0.15:
                    if current_base_height is not None:
                        base_height = current_base_height
                    else:
                        reward_aligned_heights = get_base_bottom_height_from_reward(unwrapped)
                        if reward_aligned_heights is None:
                            base_height = get_ground_relative_base_height(unwrapped.scene)
                        else:
                            base_height = reward_aligned_heights[0].item()
                    feet_heights = get_ground_relative_foot_heights(unwrapped.scene)
                    foot_contact_forces = extract_foot_contact_forces(unwrapped, foot_names=foot_names)
                    foot_force_norms = tuple(force[3] for force in foot_contact_forces)

                    # Extract actual base linear and angular velocities in robot body frame
                    if hasattr(robot.data, "root_lin_vel_b"):
                        lin_vel_tensor = (
                            robot.data.root_lin_vel_b.torch
                            if hasattr(robot.data.root_lin_vel_b, "torch")
                            else robot.data.root_lin_vel_b
                        )
                        act_vx = lin_vel_tensor[0, 0].item()
                        act_vy = lin_vel_tensor[0, 1].item()
                    else:
                        act_vx, act_vy = 0.0, 0.0

                    if hasattr(robot.data, "root_ang_vel_b"):
                        ang_vel_tensor = (
                            robot.data.root_ang_vel_b.torch
                            if hasattr(robot.data.root_ang_vel_b, "torch")
                            else robot.data.root_ang_vel_b
                        )
                        act_wz = ang_vel_tensor[0, 2].item()
                    else:
                        act_wz = 0.0

                    status_lines = [
                        "[Teleop Status]",
                        f"  Command     : vx={cmd_tuple[0]:+5.2f} m/s | vy={cmd_tuple[1]:+5.2f} m/s | wz={cmd_tuple[2]:+5.2f} rad/s",
                        f"  Actual Vel  : vx={act_vx:+5.2f} m/s | vy={act_vy:+5.2f} m/s | wz={act_wz:+5.2f} rad/s",
                        f"  Base Height : {base_height:.3f} m",
                        f"  Foot Height : FL: {feet_heights[0]:.3f} m | FR: {feet_heights[1]:.3f} m | RL: {feet_heights[2]:.3f} m | RR: {feet_heights[3]:.3f} m",
                        f"  Foot Hz (last 1.0s): FL: {foot_frequencies_hz[0]:4.1f} | FR: {foot_frequencies_hz[1]:4.1f} | RL: {foot_frequencies_hz[2]:4.1f} | RR: {foot_frequencies_hz[3]:4.1f}",
                        f"  Foot Force |F| [N] : FL: {foot_force_norms[0]:6.1f} | FR: {foot_force_norms[1]:6.1f} | RL: {foot_force_norms[2]:6.1f} | RR: {foot_force_norms[3]:6.1f}",
                        "  Joints [rad] (roll, pitch, knee):",
                        f"    FL: [{jp[0]:+6.3f}, {jp[1]:+6.3f}, {jp[2]:+6.3f}] | FR: [{jp[3]:+6.3f}, {jp[4]:+6.3f}, {jp[5]:+6.3f}]",
                        f"    RL: [{jp[6]:+6.3f}, {jp[7]:+6.3f}, {jp[8]:+6.3f}] | RR: [{jp[9]:+6.3f}, {jp[10]:+6.3f}, {jp[11]:+6.3f}]",
                        f"  Recording   : {recorder.get_status_str()}",
                        f"  Checkpoint  : {checkpoint_display}",
                    ]
                    if printed_lines > 0:
                        sys.stdout.write(f"\033[{printed_lines}A")
                    for line in status_lines:
                        sys.stdout.write(f"\r\033[K{line}\n")
                    sys.stdout.flush()
                    printed_lines = len(status_lines)
                    last_print_cmd = cmd_tuple
                    last_status_time = now

                if args_cli.real_time:
                    sleep_time = sim_dt - (time.time() - step_start_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received.")
        finally:
            if recorder.is_recording:
                partial_file = recorder.save()
                if partial_file:
                    print(
                        f"\n[REC] Interrupted! Saved partial recording "
                        f"({recorder._last_saved_steps} steps) to: {partial_file}"
                    )
                    recorder.print_summary_metrics()
            print("\n[INFO] Exiting teleoperation.")
            teleop.close()
            vec_env.close()


if __name__ == "__main__":
    main()
