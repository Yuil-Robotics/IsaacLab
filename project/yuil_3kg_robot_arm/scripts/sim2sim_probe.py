#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure a deterministic joint-step response for one Yuil physics backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

from yuil_3kg_robot_arm.asset_cfg import ROBOT_USD_PATH
from yuil_3kg_robot_arm.isaaclab_cfg import PLAY_TASK_ID, register_tasks

register_tasks()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=PLAY_TASK_ID)
parser.add_argument("--duration", type=float, default=2.0, help="Joint-step duration [s].")
parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _tensor_list(value: torch.Tensor) -> list:
    """Copy a tensor to a JSON-compatible nested list."""
    return value.detach().cpu().tolist()


def main() -> None:
    """Run the probe and write the loaded model contract and response metrics."""
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        env_cfg.seed = 0
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.commands.ee_pose.debug_vis = False
        env_cfg.events.joint_friction = None
        env_cfg.events.reset_robot_joints = None

        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env.reset()
            unwrapped = env.unwrapped
            robot = unwrapped.scene["robot"]
            num_steps = max(1, round(args_cli.duration / unwrapped.physics_dt))

            with torch.inference_mode():
                initial_pos = torch.zeros_like(robot.data.joint_pos.torch)
                initial_vel = torch.zeros_like(robot.data.joint_vel.torch)
                direction = torch.tensor(
                    [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]],
                    device=initial_pos.device,
                    dtype=initial_pos.dtype,
                )
                target_pos = initial_pos + 0.1 * direction

                robot.write_joint_position_to_sim_index(position=initial_pos)
                robot.write_joint_velocity_to_sim_index(velocity=initial_vel)
                robot.set_joint_position_target_index(target=initial_pos)
                robot.write_data_to_sim()
                unwrapped.sim.step()
                unwrapped.scene.update(unwrapped.physics_dt)

                positions = []
                velocities = []
                for _ in range(num_steps):
                    robot.set_joint_position_target_index(target=target_pos)
                    robot.write_data_to_sim()
                    unwrapped.sim.step()
                    unwrapped.scene.update(unwrapped.physics_dt)
                    positions.append(robot.data.joint_pos.torch[0].clone())
                    velocities.append(robot.data.joint_vel.torch[0].clone())

                position_history = torch.stack(positions)
                velocity_history = torch.stack(velocities)
                target = target_pos[0]
                signed_error = (position_history - target) * direction[0]
                overshoot = torch.clamp(signed_error.max(dim=0).values, min=0.0)
                final_error = position_history[-1] - target
                rms_velocity_tail = torch.sqrt(torch.mean(velocity_history[num_steps // 2 :] ** 2, dim=0))

            physics_cfg = unwrapped.cfg.sim.physics
            result = {
                "backend": type(physics_cfg).__name__,
                "usd_path": str(ROBOT_USD_PATH),
                "physics_dt_s": unwrapped.physics_dt,
                "gravity_m_s2": list(unwrapped.cfg.sim.gravity),
                "duration_s": args_cli.duration,
                "num_steps": num_steps,
                "joint_names": list(robot.joint_names),
                "body_names": list(robot.body_names),
                "body_mass_kg": _tensor_list(robot.data.body_mass.torch[0]),
                "body_com_m": _tensor_list(robot.data.body_com_pos_b.torch[0]),
                "body_inertia_kg_m2": _tensor_list(robot.data.body_inertia.torch[0]),
                "joint_limits_rad": _tensor_list(robot.data.joint_pos_limits.torch[0]),
                "joint_stiffness": _tensor_list(robot.data.joint_stiffness.torch[0]),
                "joint_damping": _tensor_list(robot.data.joint_damping.torch[0]),
                "joint_friction": _tensor_list(robot.data.joint_friction_coeff.torch[0]),
                "target_position_rad": _tensor_list(target),
                "final_position_rad": _tensor_list(position_history[-1]),
                "final_velocity_rad_s": _tensor_list(velocity_history[-1]),
                "final_position_error_rad": _tensor_list(final_error),
                "overshoot_rad": _tensor_list(overshoot),
                "tail_rms_velocity_rad_s": _tensor_list(rms_velocity_tail),
            }
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
