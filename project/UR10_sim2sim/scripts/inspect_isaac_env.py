#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the Isaac Lab UR10e environment used as the transfer source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

from ur10_sim2sim.isaaclab_cfg import PLAY_TASK_ID, register_tasks

register_tasks()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=PLAY_TASK_ID)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    """Create one environment and print the policy-transfer contract."""
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env.reset()
            unwrapped = env.unwrapped
            robot = unwrapped.scene["robot"]
            with torch.inference_mode():
                joint_pos = robot.data.default_joint_pos.torch.clone()
                joint_vel = torch.zeros_like(joint_pos)
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                robot.set_joint_position_target_index(target=joint_pos)
                robot.write_data_to_sim()
                unwrapped.sim.step()
                unwrapped.scene.update(unwrapped.physics_dt)
                observation = unwrapped.observation_manager.compute()

            ee_frame = unwrapped.scene["ee_frame_wrt_base_frame"]
            command_term = unwrapped.command_manager.get_term("ee_pose")
            command = command_term.command
            policy_obs = observation["policy"]

            print(f"Task: {args_cli.task}")
            print(f"Joint order: {tuple(robot.joint_names)}")
            print(f"Joint position [rad]: {robot.data.joint_pos.torch[0].cpu().numpy()}")
            print(f"Joint velocity [rad/s]: {robot.data.joint_vel.torch[0].cpu().numpy()}")
            print(f"Policy observation shape: {tuple(policy_obs.shape)}")
            print(f"Pose command class: {type(command_term).__name__}")
            print(f"Pose debug visualization: {command_term.cfg.debug_vis}")
            print(f"Current frame marker: {command_term.cfg.current_pose_visualizer_cfg.prim_path}")
            print(f"Target frame marker: {command_term.cfg.goal_pose_visualizer_cfg.prim_path}")
            print(f"Pose command in base [pos, quat_xyzw]: {command[0].cpu().numpy()}")
            print(f"Pose command in world [pos, quat_xyzw]: {command_term.pose_command_w[0].cpu().numpy()}")
            print(
                "wrist_3_link in base [pos]: "
                f"{ee_frame.data.target_pos_source.torch[0, 0].cpu().numpy()}"
            )
            print(
                "wrist_3_link in base [quat_xyzw]: "
                f"{ee_frame.data.target_quat_source.torch[0, 0].cpu().numpy()}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
