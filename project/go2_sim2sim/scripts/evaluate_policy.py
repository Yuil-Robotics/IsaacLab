#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one frozen quadruped policy with deterministic commands and write JSON metrics."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go2_sim2sim.checkpoint import resolve_checkpoint  # noqa: E402
from go2_sim2sim.isaaclab_cfg import EVAL_TASK_ID, register_tasks  # noqa: E402
from go2_sim2sim.rough_mdp import get_ground_relative_base_height_tensor  # noqa: E402

COMMAND_PROFILES = (
    ("stand", 0.0, 0.0, 0.0),
    ("forward_0p5", 0.5, 0.0, 0.0),
    ("forward_1p0", 1.0, 0.0, 0.0),
    # ("forward_2p0", 2.0, 0.0, 0.0),  # Held: training vx range is [-1.0, 1.0]
    ("backward_0p5", -0.5, 0.0, 0.0),
    ("lateral_0p5", 0.0, 0.5, 0.0),
    ("yaw_0p5", 0.0, 0.0, 0.5),
    ("combined", 0.5, 0.2, 0.4),
)
CONTACT_FORCE_THRESHOLDS_N = (1.0, 5.0, 10.0, 20.0)
SUSTAINED_CONTACT_FRAMES = (2, 3)

register_tasks()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=EVAL_TASK_ID)
parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
    help="Checkpoint file, run directory, or experiment directory containing timestamped runs.",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--duration", type=float, default=10.0, help="Evaluation duration [s].")
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _tensor_list(value: torch.Tensor) -> list:
    """Copy a tensor to a JSON-compatible nested list."""
    return value.detach().cpu().tolist()


def _mean_for_mask(value: torch.Tensor, mask: torch.Tensor) -> float:
    """Return the scalar mean for the selected environments."""
    return value[mask].mean().item()


def _build_command_table(device: str, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic per-environment commands and scenario indices."""
    scenario_ids = torch.arange(num_envs, device=device, dtype=torch.long) % len(COMMAND_PROFILES)
    profile_values = torch.tensor(
        [profile[1:] for profile in COMMAND_PROFILES],
        device=device,
        dtype=torch.float32,
    )
    return profile_values[scenario_ids], scenario_ids


def _resolve_foot_indices(robot, contact_sensor) -> tuple[list[int], list[int], list[str]]:
    """Resolve corresponding quadruped foot indices in the articulation and contact sensor."""
    foot_body_ids, foot_names = robot.find_bodies(".*_foot", preserve_order=True)
    sensor_names = getattr(contact_sensor, "sensor_names", None)
    if sensor_names is None:
        sensor_names = contact_sensor.body_names
    sensor_name_to_id = {name: index for index, name in enumerate(sensor_names)}
    sensor_ids = [sensor_name_to_id[name] for name in foot_names]
    return foot_body_ids, sensor_ids, foot_names


def _resolve_action_joint_contract(env, robot) -> tuple[list[int], list[str]]:
    """Resolve joint indices in the exact order used by the policy action term."""
    action_cfg = env.cfg.actions.joint_pos
    joint_ids, joint_names = robot.find_joints(
        action_cfg.joint_names,
        preserve_order=action_cfg.preserve_order,
    )
    action_dim = env.action_manager.total_action_dim
    if len(joint_names) != action_dim:
        raise ValueError(
            "The evaluator requires one joint-position action per resolved joint, "
            f"but resolved {len(joint_names)} joints for an action dimension of {action_dim}."
        )
    return joint_ids, joint_names


def _summarize(
    accumulators: dict[str, torch.Tensor],
    scenario_ids: torch.Tensor,
    num_steps: int,
    action_delta_steps: int,
    num_feet: int,
    policy_dt_s: float,
    joint_names: list[str],
) -> tuple[dict, dict]:
    """Aggregate per-environment accumulators overall and by command profile."""

    def summarize_mask(mask: torch.Tensor) -> dict:
        env_count = int(mask.sum().item())
        xy_rmse_env = torch.sqrt(accumulators["xy_error_sq"][mask] / num_steps)
        yaw_rmse_env = torch.sqrt(accumulators["yaw_error_sq"][mask] / num_steps)
        fall_free = accumulators["fall_count"][mask] == 0
        successful = fall_free & (xy_rmse_env < 0.5) & (yaw_rmse_env < 0.4)
        height_mean = _mean_for_mask(accumulators["height_sum"] / num_steps, mask)
        height_second_moment = _mean_for_mask(accumulators["height_sq_sum"] / num_steps, mask)
        contact_samples = accumulators["contact_count"][mask].sum().item()
        slip_sum = accumulators["foot_slip_sum"][mask].sum().item()
        threshold_sweep = {}
        for index, threshold in enumerate(CONTACT_FORCE_THRESHOLDS_N):
            threshold_contact_samples = accumulators["contact_count_by_threshold"][mask, index].sum().item()
            threshold_slip_sum = accumulators["foot_slip_sum_by_threshold"][mask, index].sum().item()
            threshold_sweep[f"{threshold:g}"] = {
                "mean_contact_foot_slip_m_s": threshold_slip_sum / max(threshold_contact_samples, 1.0),
                "foot_contact_ratio": threshold_contact_samples / max(env_count * num_steps * num_feet, 1),
            }
        sustained_contact_sweep = {}
        for index, minimum_frames in enumerate(SUSTAINED_CONTACT_FRAMES):
            sustained_samples = accumulators["sustained_contact_count"][mask, index].sum().item()
            horizontal_displacement = accumulators["sustained_contact_displacement_sum"][mask, index].sum().item()
            sustained_contact_sweep[str(minimum_frames)] = {
                "mean_foot_horizontal_displacement_m_per_policy_step": horizontal_displacement
                / max(sustained_samples, 1.0),
                "mean_foot_horizontal_speed_m_s": horizontal_displacement
                / max(sustained_samples * policy_dt_s, policy_dt_s),
                "eligible_foot_sample_ratio": sustained_samples / max(env_count * num_steps * num_feet, 1),
            }
        joint_saturation_count = accumulators["action_saturation_count"][mask].sum(dim=0)
        positive_joint_saturation_count = accumulators["positive_action_saturation_count"][mask].sum(dim=0)
        negative_joint_saturation_count = accumulators["negative_action_saturation_count"][mask].sum(dim=0)
        joint_saturation_rate = joint_saturation_count / max(env_count * num_steps, 1)
        positive_joint_saturation_rate = positive_joint_saturation_count / max(env_count * num_steps, 1)
        negative_joint_saturation_rate = negative_joint_saturation_count / max(env_count * num_steps, 1)
        return {
            "num_envs": env_count,
            "success_rate": successful.float().mean().item(),
            "fall_rate": (~fall_free).float().mean().item(),
            "falls_per_env": _mean_for_mask(accumulators["fall_count"], mask),
            "lin_vel_xy_rmse_m_s": torch.sqrt(accumulators["xy_error_sq"][mask].sum() / (env_count * num_steps)).item(),
            "yaw_rate_rmse_rad_s": torch.sqrt(
                accumulators["yaw_error_sq"][mask].sum() / (env_count * num_steps)
            ).item(),
            "mean_reward_per_step": _mean_for_mask(accumulators["reward_sum"] / num_steps, mask),
            "mean_base_height_m": height_mean,
            "base_height_std_m": max(height_second_moment - height_mean**2, 0.0) ** 0.5,
            "base_tilt_rms": torch.sqrt(accumulators["tilt_sq_sum"][mask].sum() / (env_count * num_steps)).item(),
            "action_rms": torch.sqrt(accumulators["action_sq_sum"][mask].sum() / (env_count * num_steps)).item(),
            "action_saturation_rate": (
                joint_saturation_count.sum() / max(env_count * num_steps * len(joint_names), 1)
            ).item(),
            "action_any_saturation_rate": _mean_for_mask(
                accumulators["action_any_saturation_count"] / num_steps,
                mask,
            ),
            "action_saturation_rate_by_joint": {
                joint_name: joint_saturation_rate[index].item() for index, joint_name in enumerate(joint_names)
            },
            "directional_action_saturation_rate_by_joint": {
                joint_name: {
                    "positive": positive_joint_saturation_rate[index].item(),
                    "negative": negative_joint_saturation_rate[index].item(),
                }
                for index, joint_name in enumerate(joint_names)
            },
            "action_delta_rms_per_policy_step": torch.sqrt(
                accumulators["action_delta_sq_sum"][mask].sum() / (env_count * action_delta_steps)
            ).item(),
            "joint_velocity_rms_rad_s": torch.sqrt(
                accumulators["joint_velocity_sq_sum"][mask].sum() / (env_count * num_steps)
            ).item(),
            "joint_torque_rms_nm": torch.sqrt(
                accumulators["joint_torque_sq_sum"][mask].sum() / (env_count * num_steps)
            ).item(),
            "mean_abs_mechanical_power_w": _mean_for_mask(accumulators["power_sum"] / num_steps, mask),
            "mean_contact_foot_slip_m_s": slip_sum / max(contact_samples, 1.0),
            "foot_contact_ratio": contact_samples / max(env_count * num_steps * num_feet, 1),
            "contact_force_threshold_sweep": threshold_sweep,
            "sustained_contact_sweep": sustained_contact_sweep,
        }

    overall_mask = torch.ones_like(scenario_ids, dtype=torch.bool)
    overall = summarize_mask(overall_mask)
    by_scenario = {}
    for scenario_index, profile in enumerate(COMMAND_PROFILES):
        by_scenario[profile[0]] = {
            "command": list(profile[1:]),
            **summarize_mask(scenario_ids == scenario_index),
        }
    return overall, by_scenario


def main() -> None:
    """Run deterministic policy evaluation."""
    if args_cli.num_envs < len(COMMAND_PROFILES):
        raise ValueError(f"--num_envs must be at least {len(COMMAND_PROFILES)}.")
    if args_cli.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    checkpoint = resolve_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint.expanduser().resolve() != checkpoint:
        print(f"[INFO] Auto-selected latest checkpoint: {checkpoint}")

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.episode_length_s = max(2.0 * args_cli.duration, 20.0)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = str(checkpoint.parent)
        agent_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            agent_cfg.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = None
        try:
            vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            configure_seed(args_cli.seed, True)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

            unwrapped = vec_env.unwrapped
            robot = unwrapped.scene["robot"]
            joint_ids, joint_names = _resolve_action_joint_contract(unwrapped, robot)
            action_clip_limit = agent_cfg.clip_actions
            if action_clip_limit is None or action_clip_limit <= 0.0:
                raise ValueError("Sim2sim action saturation metrics require a positive agent clip_actions value.")
            command_term = unwrapped.command_manager.get_term("base_velocity")
            contact_sensor = unwrapped.scene.sensors["contact_forces"]
            foot_body_ids, foot_sensor_ids, foot_names = _resolve_foot_indices(robot, contact_sensor)
            commands, scenario_ids = _build_command_table(unwrapped.device, unwrapped.num_envs)
            num_steps = max(1, round(args_cli.duration / unwrapped.step_dt))

            accumulators = {
                name: torch.zeros(unwrapped.num_envs, device=unwrapped.device)
                for name in (
                    "xy_error_sq",
                    "yaw_error_sq",
                    "reward_sum",
                    "height_sum",
                    "height_sq_sum",
                    "tilt_sq_sum",
                    "action_sq_sum",
                    "action_delta_sq_sum",
                    "joint_velocity_sq_sum",
                    "joint_torque_sq_sum",
                    "power_sum",
                    "foot_slip_sum",
                    "contact_count",
                    "fall_count",
                    "action_any_saturation_count",
                )
            }
            accumulators["action_saturation_count"] = torch.zeros(
                (unwrapped.num_envs, len(joint_names)),
                device=unwrapped.device,
            )
            accumulators["positive_action_saturation_count"] = torch.zeros(
                (unwrapped.num_envs, len(joint_names)),
                device=unwrapped.device,
            )
            accumulators["negative_action_saturation_count"] = torch.zeros(
                (unwrapped.num_envs, len(joint_names)),
                device=unwrapped.device,
            )
            threshold_accumulator_shape = (unwrapped.num_envs, len(CONTACT_FORCE_THRESHOLDS_N))
            accumulators["foot_slip_sum_by_threshold"] = torch.zeros(
                threshold_accumulator_shape,
                device=unwrapped.device,
            )
            accumulators["contact_count_by_threshold"] = torch.zeros(
                threshold_accumulator_shape,
                device=unwrapped.device,
            )
            sustained_accumulator_shape = (unwrapped.num_envs, len(SUSTAINED_CONTACT_FRAMES))
            accumulators["sustained_contact_displacement_sum"] = torch.zeros(
                sustained_accumulator_shape,
                device=unwrapped.device,
            )
            accumulators["sustained_contact_count"] = torch.zeros(
                sustained_accumulator_shape,
                device=unwrapped.device,
            )
            contact_force_thresholds = torch.tensor(
                CONTACT_FORCE_THRESHOLDS_N,
                device=unwrapped.device,
                dtype=torch.float32,
            )

            previous_actions = None
            previous_foot_position_xy = None
            contact_streak = torch.zeros(
                (unwrapped.num_envs, len(foot_names)),
                device=unwrapped.device,
                dtype=torch.long,
            )
            action_delta_steps = 0
            with torch.inference_mode():
                for _ in range(num_steps):
                    command_term.vel_command_b.copy_(commands)
                    command_term.is_heading_env.zero_()
                    command_term.is_standing_env.zero_()
                    if hasattr(command_term, "is_straight_env"):
                        command_term.is_straight_env.zero_()
                    if hasattr(command_term, "is_turning_env"):
                        command_term.is_turning_env.zero_()
                    if hasattr(command_term, "is_lateral_env"):
                        command_term.is_lateral_env.zero_()
                    obs = vec_env.get_observations()
                    actions = policy(obs)
                    _, reward, dones, _ = vec_env.step(actions)

                    lin_error = commands[:, :2] - robot.data.root_lin_vel_b.torch[:, :2]
                    yaw_error = commands[:, 2] - robot.data.root_ang_vel_b.torch[:, 2]
                    base_height = get_ground_relative_base_height_tensor(unwrapped.scene)
                    base_tilt = torch.linalg.norm(robot.data.projected_gravity_b.torch[:, :2], dim=1)
                    joint_velocity = robot.data.joint_vel.torch[:, joint_ids]
                    joint_torque = robot.data.applied_torque.torch[:, joint_ids]

                    contact_force = torch.linalg.norm(
                        contact_sensor.data.net_forces_w.torch[:, foot_sensor_ids],
                        dim=-1,
                    )
                    in_contact = contact_force > 1.0
                    in_contact_by_threshold = contact_force.unsqueeze(-1) > contact_force_thresholds
                    foot_speed = torch.linalg.norm(
                        robot.data.body_lin_vel_w.torch[:, foot_body_ids, :2],
                        dim=-1,
                    )
                    foot_position_xy = robot.data.body_pos_w.torch[:, foot_body_ids, :2]
                    contact_streak = torch.where(in_contact, contact_streak + 1, 0)

                    accumulators["xy_error_sq"] += torch.sum(lin_error**2, dim=1)
                    accumulators["yaw_error_sq"] += yaw_error**2
                    accumulators["reward_sum"] += reward
                    accumulators["height_sum"] += base_height
                    accumulators["height_sq_sum"] += base_height**2
                    accumulators["tilt_sq_sum"] += base_tilt**2
                    accumulators["action_sq_sum"] += torch.mean(actions**2, dim=1)
                    positive_action_saturated = actions >= action_clip_limit
                    negative_action_saturated = actions <= -action_clip_limit
                    action_saturated = positive_action_saturated | negative_action_saturated
                    accumulators["action_saturation_count"] += action_saturated
                    accumulators["positive_action_saturation_count"] += positive_action_saturated
                    accumulators["negative_action_saturation_count"] += negative_action_saturated
                    accumulators["action_any_saturation_count"] += torch.any(action_saturated, dim=1)
                    accumulators["joint_velocity_sq_sum"] += torch.mean(joint_velocity**2, dim=1)
                    accumulators["joint_torque_sq_sum"] += torch.mean(joint_torque**2, dim=1)
                    accumulators["power_sum"] += torch.sum(torch.abs(joint_torque * joint_velocity), dim=1)
                    accumulators["foot_slip_sum"] += torch.sum(foot_speed * in_contact, dim=1)
                    accumulators["contact_count"] += torch.sum(in_contact, dim=1)
                    accumulators["foot_slip_sum_by_threshold"] += torch.sum(
                        foot_speed.unsqueeze(-1) * in_contact_by_threshold,
                        dim=1,
                    )
                    accumulators["contact_count_by_threshold"] += torch.sum(in_contact_by_threshold, dim=1)
                    if previous_foot_position_xy is not None:
                        horizontal_displacement = torch.linalg.norm(
                            foot_position_xy - previous_foot_position_xy,
                            dim=-1,
                        )
                        valid_transition = ~dones.unsqueeze(1)
                        for index, minimum_frames in enumerate(SUSTAINED_CONTACT_FRAMES):
                            sustained_contact = (contact_streak >= minimum_frames) & valid_transition
                            accumulators["sustained_contact_displacement_sum"][:, index] += torch.sum(
                                horizontal_displacement * sustained_contact,
                                dim=1,
                            )
                            accumulators["sustained_contact_count"][:, index] += torch.sum(
                                sustained_contact,
                                dim=1,
                            )
                    accumulators["fall_count"] += dones.float()
                    if previous_actions is not None:
                        accumulators["action_delta_sq_sum"] += torch.mean((actions - previous_actions) ** 2, dim=1)
                        action_delta_steps += 1
                    previous_actions = actions.clone()
                    previous_foot_position_xy = foot_position_xy.clone()
                    contact_streak[dones] = 0
                    if hasattr(policy, "reset"):
                        policy.reset(dones)

            overall, by_scenario = _summarize(
                accumulators,
                scenario_ids,
                num_steps,
                max(action_delta_steps, 1),
                len(foot_names),
                unwrapped.step_dt,
                joint_names,
            )
            physics_cfg = unwrapped.cfg.sim.physics
            terrain_material = unwrapped.cfg.scene.terrain.physics_material
            body_names = sorted(robot.body_names)
            body_ids = [robot.body_names.index(name) for name in body_names]
            result = {
                "schema_version": 5,
                "backend": type(physics_cfg).__name__,
                "checkpoint": str(checkpoint),
                "task": args_cli.task,
                "seed": args_cli.seed,
                "num_envs": unwrapped.num_envs,
                "duration_s": args_cli.duration,
                "num_policy_steps": num_steps,
                "physics_dt_s": unwrapped.physics_dt,
                "policy_dt_s": unwrapped.step_dt,
                "action_clip_limit": action_clip_limit,
                "foot_names": foot_names,
                "contact_force_thresholds_n": list(CONTACT_FORCE_THRESHOLDS_N),
                "sustained_contact_frames": list(SUSTAINED_CONTACT_FRAMES),
                "commands": [
                    {
                        "name": profile[0],
                        "lin_vel_x_m_s": profile[1],
                        "lin_vel_y_m_s": profile[2],
                        "ang_vel_z_rad_s": profile[3],
                    }
                    for profile in COMMAND_PROFILES
                ],
                "model_contract": {
                    "joint_names": joint_names,
                    "body_names": body_names,
                    "body_mass_kg": _tensor_list(robot.data.body_mass.torch[0, body_ids]),
                    "body_com_m": _tensor_list(robot.data.body_com_pos_b.torch[0, body_ids]),
                    "body_inertia_kg_m2": _tensor_list(robot.data.body_inertia.torch[0, body_ids]),
                    "joint_armature_kg_m2": _tensor_list(robot.data.joint_armature.torch[0, joint_ids]),
                    "joint_stiffness_nm_rad": _tensor_list(robot.data.joint_stiffness.torch[0, joint_ids]),
                    "joint_damping_nm_s_rad": _tensor_list(robot.data.joint_damping.torch[0, joint_ids]),
                    "joint_friction": _tensor_list(robot.data.joint_friction_coeff.torch[0, joint_ids]),
                    "ground_static_friction": terrain_material.static_friction,
                    "ground_dynamic_friction": terrain_material.dynamic_friction,
                    "ground_restitution": terrain_material.restitution,
                },
                "metrics": {
                    "overall": overall,
                    "by_command": by_scenario,
                },
            }
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result["metrics"], indent=2))
            print(f"[Policy evaluation] wrote {args_cli.output.resolve()}")
        finally:
            if vec_env is not None:
                vec_env.close()
            else:
                env.close()


if __name__ == "__main__":
    main()
