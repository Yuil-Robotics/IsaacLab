# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play the exported Robotis HX5 policy in the Newton environment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROBOTIS_PROJECT_DIR = PROJECT_DIR.parent
SOURCE_DIR = PROJECT_DIR / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import sim2sim_newton  # noqa: E402, F401
from sim2sim_newton.tasks.hx5_cube.hx5_cube_newton_env_cfg import (  # noqa: E402
    GOAL_MARKER_POSITION,
    NEWTON_CUBE_USD,
    OBJECT_INITIAL_POSITION,
    POLICY_ACTION_DIM,
    POLICY_OBSERVATION_DIM,
    ROBOT_ROOT_POSITION,
    SINGLE_GOAL_OBJECT_INITIAL_POSITION,
    Hx5CubeNewtonEnvCfg,
    Hx5CubeNewtonSingleGoalEnvCfg,
    default_robot_usd_path,
    make_robot_cfg,
)
from sim2sim_newton.tasks.hx5_cube.hx5_cube_v7_newton_env_cfg import (  # noqa: E402
    V7_CUBE_USD,
    V7_OBJECT_INITIAL_POSITION,
    Hx5CubeV7NewtonEnvCfg,
    make_v7_robot_cfg,
)

from isaaclab_tasks.utils import add_launcher_args, launch_simulation  # noqa: E402

CONTINUOUS_TASK_ID = "Isaac-Repose-Cube-Robotis-HX5-Newton-Play-v0"
SINGLE_GOAL_TASK_ID = "Isaac-Repose-Cube-Robotis-HX5-Newton-Single-Goal-Play-v0"
V7_TASK_ID = "Isaac-Repose-Cube-Robotis-HX5-Newton-V7-Play-v0"
DEFAULT_POLICY_PATH = ROBOTIS_PROJECT_DIR / "Robotis_left_benchmark_2026-07-30_16-01-04" / "exported" / "policy.pt"
V7_POLICY_PATH = ROBOTIS_PROJECT_DIR / "robotis_hand_other_sim" / "exported" / "policy.pt"


def format_vector(values: torch.Tensor) -> str:
    """Format a short tensor as a compact numeric vector."""
    return "[" + ", ".join(f"{float(value):+.4f}" for value in values.detach().cpu()) + "]"


def format_range(values: torch.Tensor) -> str:
    """Format a tensor's minimum and maximum values."""
    return f"{float(values.min()):.6g}..{float(values.max()):.6g}"


def format_scientific_vector(values: torch.Tensor) -> str:
    """Format a short tensor using scientific notation."""
    return "[" + ", ".join(f"{float(value):.9e}" for value in values.detach().cpu()) + "]"


def print_physics_diagnostics(env) -> None:
    """Print the effective hand properties written to the Newton solver."""
    unwrapped = env.unwrapped
    hand = unwrapped.hand
    hand_mass = hand.data.body_mass.torch[0]
    cube_mass = unwrapped.object.data.body_mass.torch[0, 0]
    cube_inertia = unwrapped.object.data.body_inertia.torch[0, 0]
    print(
        f"[physics] fixed_base={hand.is_fixed_base} | bodies={hand.num_bodies} | joints={hand.num_joints} | "
        f"hand_mass={float(hand_mass.sum()):.6g} kg | "
        f"armature={format_range(hand.data.joint_armature.torch[0])} kg*m^2 | "
        f"stiffness={format_range(hand.data.joint_stiffness.torch[0])} N*m/rad | "
        f"damping={format_range(hand.data.joint_damping.torch[0])} N*m*s/rad | "
        f"effort_limit={format_range(hand.data.joint_effort_limits.torch[0])} N*m | "
        f"velocity_limit={format_range(hand.data.joint_vel_limits.torch[0])} rad/s"
    )
    print(
        f"[physics] cube_mass={float(cube_mass):.6g} kg | "
        f"cube_inertia_diag={format_scientific_vector(cube_inertia[[0, 4, 8]])} kg*m^2"
    )


def print_initial_pose_diagnostics(env, object_initial_position: tuple[float, float, float]) -> None:
    """Print restored task poses and fail on a large root-placement error."""
    unwrapped = env.unwrapped
    hand_root_pose = unwrapped.hand.data.root_link_pose_w.torch[0]
    cube_position = unwrapped.object_pos[0]
    env_origin = unwrapped.scene.env_origins[0]
    hand_position_local = hand_root_pose[:3] - env_origin
    expected_hand_position = torch.tensor(ROBOT_ROOT_POSITION, device=hand_root_pose.device)
    expected_cube_position = torch.tensor(object_initial_position, device=cube_position.device)
    hand_position_error = torch.linalg.vector_norm(hand_position_local - expected_hand_position)
    cube_position_error = torch.linalg.vector_norm(cube_position - expected_cube_position)
    root_to_cube = torch.linalg.vector_norm(cube_position - hand_position_local)
    fingertip_distances = torch.linalg.vector_norm(unwrapped.fingertip_pos[0] - cube_position, dim=-1)

    print(
        f"[pose] env_origin={format_vector(env_origin)} | hand_root_local={format_vector(hand_position_local)} "
        f"hand_quat_xyzw={format_vector(hand_root_pose[3:7])} | "
        f"cube={format_vector(cube_position)} | in_hand={format_vector(unwrapped.in_hand_pos[0])} | "
        f"goal_marker={list(GOAL_MARKER_POSITION)} | root_to_cube={float(root_to_cube):.4f} m | "
        f"tip_to_cube={format_vector(fingertip_distances)} m"
    )
    if hand_position_error > 0.02 or cube_position_error > 0.02:
        raise RuntimeError(
            "Restored task pose mismatch: "
            f"hand error={float(hand_position_error):.4f} m, cube error={float(cube_position_error):.4f} m."
        )


def parse_args() -> argparse.Namespace:
    """Parse Newton play command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_usd", type=Path, default=default_robot_usd_path(), help="HX5 robot USD path.")
    parser.add_argument("--policy", type=Path, default=None, help="Exported TorchScript policy path.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of Newton environments.")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many policy steps; zero runs forever.")
    parser.add_argument("--stats_every", type=int, default=30, help="Print one status line every N policy steps.")
    parser.add_argument(
        "--spawn_drop_steps",
        type=int,
        default=15,
        help="Classify drops within this many policy steps after reset as spawn drops.",
    )
    parser.add_argument(
        "--single_goal",
        action="store_true",
        help="Terminate after one 15-step stable goal instead of generating another goal.",
    )
    parser.add_argument(
        "--v7",
        action="store_true",
        help="Use the isolated V7 Newton task and the V7 policy by default.",
    )
    parser.add_argument(
        "--real_time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace policy steps at the trained 30 Hz control rate.",
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton"])
    return parser.parse_args()


def validate_paths(robot_usd: Path, policy_path: Path, cube_usd: Path) -> tuple[Path, Path]:
    """Resolve required input paths and raise actionable errors when missing."""
    robot_usd = robot_usd.expanduser().resolve()
    policy_path = policy_path.expanduser().resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(
            f"Robot USD was not found: {robot_usd}\n"
            "Keep the original robotis_hand folder in the project and run "
            f"{PROJECT_DIR / 'scripts' / 'prepare_original_robot_usd.py'}, or pass --robot_usd PATH."
        )
    if not policy_path.is_file():
        raise FileNotFoundError(f"Exported TorchScript policy was not found: {policy_path}")
    if not cube_usd.is_file():
        raise FileNotFoundError(
            f"Newton DexCube wrapper was not found: {cube_usd}\n"
            f"Run {PROJECT_DIR / 'scripts' / 'prepare_original_robot_usd.py'} first."
        )
    return robot_usd, policy_path


def main() -> None:
    """Run policy inference and report compact success statistics."""
    args = parse_args()
    if args.v7 and args.single_goal:
        raise ValueError("--v7 is a continuous-goal task and cannot be combined with --single_goal.")
    if args.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args.num_envs}.")

    default_policy_path = V7_POLICY_PATH if args.v7 else DEFAULT_POLICY_PATH
    selected_policy_path = args.policy if args.policy is not None else default_policy_path
    cube_usd = V7_CUBE_USD if args.v7 else NEWTON_CUBE_USD
    if args.v7:
        object_initial_position = V7_OBJECT_INITIAL_POSITION
    elif args.single_goal:
        object_initial_position = SINGLE_GOAL_OBJECT_INITIAL_POSITION
    else:
        object_initial_position = OBJECT_INITIAL_POSITION
    robot_usd, policy_path = validate_paths(args.robot_usd, selected_policy_path, cube_usd)

    if args.v7:
        task_id = V7_TASK_ID
        cfg = Hx5CubeV7NewtonEnvCfg()
        cfg.robot_cfg = make_v7_robot_cfg(robot_usd)
    else:
        task_id = SINGLE_GOAL_TASK_ID if args.single_goal else CONTINUOUS_TASK_ID
        cfg = Hx5CubeNewtonSingleGoalEnvCfg() if args.single_goal else Hx5CubeNewtonEnvCfg()
        cfg.robot_cfg = make_robot_cfg(robot_usd)
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device

    print(f"[setup] task={task_id} | backend=Newton MJWarp | robot_usd={robot_usd}")
    print(f"[setup] policy={policy_path} | envs={args.num_envs} | control_rate=30 Hz | real_time={args.real_time}")

    with launch_simulation(cfg, args):
        env = gym.make(task_id, cfg=cfg)
        policy = torch.jit.load(str(policy_path), map_location=env.unwrapped.device).eval()

        observations, _ = env.reset()
        print_physics_diagnostics(env)
        print_initial_pose_diagnostics(env, object_initial_position)
        policy_observations = observations["policy"]
        if policy_observations.shape[-1] != POLICY_OBSERVATION_DIM:
            raise RuntimeError(
                f"Expected {POLICY_OBSERVATION_DIM} policy observations, got {policy_observations.shape}."
            )

        total_goals = 0
        completed_episodes = 0
        successful_episodes = 0
        dropped_episodes = 0
        spawn_dropped_episodes = 0
        policy_dropped_episodes = 0
        timed_out_episodes = 0
        timeout_never_reached_episodes = 0
        timeout_hold_failed_episodes = 0
        completed_episode_goals = 0
        completed_episode_steps = 0
        action_limit_hits = 0
        action_samples = 0
        joint_speed_sum = 0.0
        joint_tracking_error_sum = 0.0
        control_samples = 0
        drop_linear_speed_sum = 0.0
        drop_angular_speed_sum = 0.0
        timeout_min_rot_dist_sum = 0.0
        timeout_max_hold_sum = 0
        episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=env.unwrapped.device)
        current_hold_steps = torch.zeros(args.num_envs, dtype=torch.long, device=env.unwrapped.device)
        max_hold_steps = torch.zeros(args.num_envs, dtype=torch.long, device=env.unwrapped.device)
        min_rot_dist = torch.full((args.num_envs,), torch.inf, device=env.unwrapped.device)
        step = 0
        control_period = cfg.sim.dt * cfg.decimation
        next_step_time = time.perf_counter()
        sim = env.unwrapped.sim

        try:
            while args.steps <= 0 or step < args.steps:
                if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break

                previous_successes = env.unwrapped.successes.clone()
                with torch.inference_mode():
                    actions = policy(policy_observations)
                    if not isinstance(actions, torch.Tensor):
                        raise TypeError(f"The TorchScript policy returned {type(actions).__name__}, not a tensor.")
                    if actions.shape != (args.num_envs, POLICY_ACTION_DIM):
                        raise RuntimeError(
                            f"Expected policy actions shaped ({args.num_envs}, {POLICY_ACTION_DIM}), "
                            f"got {actions.shape}."
                        )
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("The policy produced NaN or infinite actions.")
                    observations, _, terminated, truncated, _ = env.step(actions)

                policy_observations = observations["policy"]
                episode_steps += 1
                rot_dist = env.unwrapped._last_step_rot_dist
                within_tolerance = rot_dist <= cfg.success_tolerance
                if args.v7:
                    within_tolerance &= env.unwrapped._last_step_pos_dist <= cfg.pos_success_tolerance
                min_rot_dist = torch.minimum(min_rot_dist, rot_dist)
                current_hold_steps = torch.where(
                    within_tolerance,
                    current_hold_steps + 1,
                    torch.zeros_like(current_hold_steps),
                )
                max_hold_steps = torch.maximum(max_hold_steps, current_hold_steps)
                action_limit_hits += int((torch.abs(actions) >= 0.99).sum().item())
                action_samples += actions.numel()
                joint_speed_sum += float(env.unwrapped._last_step_joint_speed.sum().item())
                joint_tracking_error_sum += float(env.unwrapped._last_step_joint_tracking_error.sum().item())
                control_samples += args.num_envs
                success_delta = env.unwrapped.successes - previous_successes
                total_goals += int(torch.clamp_min(success_delta, 0.0).sum().item())

                done = terminated | truncated
                done_count = int(done.sum().item())
                if done_count:
                    successful_done = env.unwrapped._last_episode_success[done]
                    completed_episodes += done_count
                    successful_episodes += int(successful_done.sum().item())
                    if args.single_goal:
                        successful_termination = torch.zeros_like(done)
                        successful_termination[done] = successful_done
                        dropped = terminated & ~successful_termination
                        timed_out = done & ~successful_termination & ~dropped
                    else:
                        dropped = terminated
                        timed_out = truncated & ~terminated
                    spawn_dropped = dropped & (episode_steps <= args.spawn_drop_steps)
                    policy_dropped = dropped & ~spawn_dropped
                    if args.v7:
                        timeout_never_reached = timed_out & ~env.unwrapped._last_episode_success
                        timeout_hold_failed = timed_out & env.unwrapped._last_episode_success
                    else:
                        timeout_never_reached = timed_out & (max_hold_steps == 0)
                        timeout_hold_failed = timed_out & ~timeout_never_reached
                    dropped_episodes += int(dropped.sum().item())
                    spawn_dropped_episodes += int(spawn_dropped.sum().item())
                    policy_dropped_episodes += int(policy_dropped.sum().item())
                    timed_out_episodes += int(timed_out.sum().item())
                    timeout_never_reached_episodes += int(timeout_never_reached.sum().item())
                    timeout_hold_failed_episodes += int(timeout_hold_failed.sum().item())
                    drop_linear_speed_sum += float(env.unwrapped._last_step_object_linear_speed[dropped].sum().item())
                    drop_angular_speed_sum += float(env.unwrapped._last_step_object_angular_speed[dropped].sum().item())
                    timeout_min_rot_dist_sum += float(min_rot_dist[timed_out].sum().item())
                    timeout_max_hold_sum += int(max_hold_steps[timed_out].sum().item())
                    completed_episode_goals += int(previous_successes[done].sum().item())
                    completed_episode_steps += int(episode_steps[done].sum().item())
                    episode_steps[done] = 0
                    current_hold_steps[done] = 0
                    max_hold_steps[done] = 0
                    min_rot_dist[done] = torch.inf
                    if args.single_goal:
                        total_goals = successful_episodes
                        completed_episode_goals = successful_episodes

                step += 1
                if args.stats_every > 0 and step % args.stats_every == 0:
                    success_rate = 100.0 * successful_episodes / completed_episodes if completed_episodes else 0.0
                    drop_rate = 100.0 * dropped_episodes / completed_episodes if completed_episodes else 0.0
                    spawn_drop_rate = 100.0 * spawn_dropped_episodes / completed_episodes if completed_episodes else 0.0
                    policy_drop_rate = (
                        100.0 * policy_dropped_episodes / completed_episodes if completed_episodes else 0.0
                    )
                    timeout_rate = 100.0 * timed_out_episodes / completed_episodes if completed_episodes else 0.0
                    timeout_never_rate = (
                        100.0 * timeout_never_reached_episodes / completed_episodes if completed_episodes else 0.0
                    )
                    timeout_hold_rate = (
                        100.0 * timeout_hold_failed_episodes / completed_episodes if completed_episodes else 0.0
                    )
                    goals_per_episode = completed_episode_goals / completed_episodes if completed_episodes else 0.0
                    mean_episode_time = (
                        completed_episode_steps * control_period / completed_episodes if completed_episodes else 0.0
                    )
                    action_limit_rate = 100.0 * action_limit_hits / action_samples if action_samples else 0.0
                    mean_joint_speed = joint_speed_sum / control_samples if control_samples else 0.0
                    mean_tracking_error = joint_tracking_error_sum / control_samples if control_samples else 0.0
                    mean_drop_linear_speed = drop_linear_speed_sum / dropped_episodes if dropped_episodes else 0.0
                    mean_drop_angular_speed = drop_angular_speed_sum / dropped_episodes if dropped_episodes else 0.0
                    mean_timeout_min_rot_dist = (
                        timeout_min_rot_dist_sum / timed_out_episodes if timed_out_episodes else 0.0
                    )
                    mean_timeout_max_hold = timeout_max_hold_sum / timed_out_episodes if timed_out_episodes else 0.0
                    mean_consecutive = float(env.unwrapped.consecutive_successes.mean().item())
                    timeout_detail = (
                        f"no_goal={timeout_never_rate:.2f}% | after_goal={timeout_hold_rate:.2f}%"
                        if args.v7
                        else f"never_reached={timeout_never_rate:.2f}% | hold_failed={timeout_hold_rate:.2f}%"
                    )
                    print(
                        f"step={step} | envs={args.num_envs} | env_steps={step * args.num_envs} | "
                        f"goals={total_goals} | "
                        f"episodes={completed_episodes} | success_rate={success_rate:.2f}% | "
                        f"drop_rate={drop_rate:.2f}% "
                        f"(spawn_drop_rate={spawn_drop_rate:.2f}% | policy_drop_rate={policy_drop_rate:.2f}%) | "
                        f"timeout_rate={timeout_rate:.2f}% ({timeout_detail}) | "
                        f"goals_per_episode={goals_per_episode:.3f} | "
                        f"mean_episode_time={mean_episode_time:.2f}s | "
                        f"mean_consecutive_successes={mean_consecutive:.3f} | "
                        f"action_limit_rate={action_limit_rate:.2f}% | "
                        f"mean_joint_speed={mean_joint_speed:.3f}rad/s | "
                        f"mean_joint_tracking_error={mean_tracking_error:.4f}rad"
                    )
                    timeout_progress = (
                        f"timeout_best_theta={mean_timeout_min_rot_dist:.3f}rad"
                        if args.v7
                        else (
                            f"timeout_min_theta={mean_timeout_min_rot_dist:.3f}rad "
                            f"| timeout_max_hold={mean_timeout_max_hold:.2f}steps"
                        )
                    )
                    print(
                        f"  failures | drop_linear_speed={mean_drop_linear_speed:.3f}m/s "
                        f"| drop_angular_speed={mean_drop_angular_speed:.3f}rad/s | {timeout_progress}"
                    )

                if args.real_time:
                    next_step_time += control_period
                    remaining = next_step_time - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)
        finally:
            env.close()


if __name__ == "__main__":
    main()
