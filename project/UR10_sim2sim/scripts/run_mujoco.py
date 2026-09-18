#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Run an Isaac Lab exported UR10e policy in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur10_sim2sim.mujoco_env import UR10eMuJoCoEnv  # noqa: E402
from ur10_sim2sim.policy import TorchScriptPolicy  # noqa: E402

CYCLING_TARGETS = np.array(
    [
        [0.8875, -0.225, 0.20],
        [0.70, -0.20, 0.15],
        [0.90, -0.32, 0.25],
        [1.05, -0.15, 0.20],
        [0.80, -0.28, 0.28],
    ],
    dtype=np.float64,
)
CYCLING_TARGET_RPYS = np.array(
    [
        [np.pi, 0.00, -np.pi / 2.0],
        [np.pi - 0.25, 0.15, -1.20],
        [np.pi, 0.00, -np.pi / 2.0],
        [np.pi - 0.15, -0.25, -0.80],
        [np.pi, 0.00, -np.pi / 2.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported TorchScript policy.pt.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "assets" / "ur10e.xml")
    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help="Rollout duration in simulation seconds; use 0 to run the GUI until it is closed.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--record", type=Path, help="Optional output .npz trajectory.")
    parser.add_argument("--target-pos", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-rpy", type=float, nargs=3, metavar=("R", "P", "Y"))
    parser.add_argument(
        "--cycle-targets",
        action="store_true",
        help="Cycle through deterministic targets inside the Isaac training workspace.",
    )
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=0.03,
        help="Position error [m] below which a cycling target is considered reached.",
    )
    parser.add_argument(
        "--orientation-threshold",
        type=float,
        default=0.15,
        help="Orientation error [rad] below which a cycling target is considered reached.",
    )
    parser.add_argument(
        "--target-hold",
        type=float,
        default=1.0,
        help="Simulation time [s] to hold a reached target before switching.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="GUI playback speed multiplier; use 0.25 for four-times-slower viewing.",
    )
    parser.add_argument(
        "--action-clip",
        type=float,
        default=None,
        help="Optional symmetric action clip. Leave unset for exact Isaac configuration parity.",
    )
    return parser.parse_args()


def quaternion_error_rad(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> float:
    """Return the shortest angular distance between two quaternions [rad]."""
    current = np.asarray(current_wxyz, dtype=np.float64)
    target = np.asarray(target_wxyz, dtype=np.float64)
    cosine = np.clip(abs(np.dot(current / np.linalg.norm(current), target / np.linalg.norm(target))), 0.0, 1.0)
    return float(2.0 * np.arccos(cosine))


def main() -> None:
    """Run the rollout."""
    args = parse_args()
    if args.duration < 0.0:
        raise ValueError("--duration must be non-negative.")
    if args.headless and args.duration == 0.0:
        raise ValueError("--duration 0 requires the GUI; headless runs must have a finite duration.")
    if args.reach_threshold <= 0.0 or args.orientation_threshold <= 0.0:
        raise ValueError("Reach and orientation thresholds must be positive.")
    if args.target_hold < 0.0:
        raise ValueError("--target-hold must be non-negative.")
    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive.")

    env = UR10eMuJoCoEnv(
        args.model,
        TorchScriptPolicy(args.policy),
        target_pos=np.array(args.target_pos) if args.target_pos else None,
        target_rpy=np.array(args.target_rpy) if args.target_rpy else None,
        action_clip=args.action_clip,
    )
    max_steps = None if args.duration == 0.0 else int(np.ceil(args.duration / env.config.policy_dt))
    records: list[dict[str, np.ndarray | float]] = []
    last_record: dict[str, np.ndarray | float] | None = None
    completed_steps = 0
    reached_steps = 0
    target_index = 0

    initial_rpy = np.array(args.target_rpy) if args.target_rpy else env.config.target_rpy.copy()
    targets = [(env.target_pos.copy(), initial_rpy)]
    if args.cycle_targets:
        targets.extend(
            (target_pos.copy(), target_rpy.copy())
            for target_pos, target_rpy in zip(CYCLING_TARGETS, CYCLING_TARGET_RPYS)
            if not (np.allclose(target_pos, env.target_pos) and np.allclose(target_rpy, initial_rpy))
        )
        print(f"Cycling through {len(targets)} target poses.")
        print(f"Initial position: {targets[0][0]}, RPY: {targets[0][1]}")

    def advance_policy() -> None:
        nonlocal completed_steps, last_record, reached_steps, target_index
        last_record = env.policy_step()
        completed_steps += 1
        if args.record:
            records.append(last_record)

        if not args.cycle_targets:
            return
        position_error = float(np.linalg.norm(np.asarray(last_record["ee_pos"]) - env.target_pos))
        orientation_error = quaternion_error_rad(np.asarray(last_record["ee_quat"]), env.target_quat_wxyz)
        if position_error <= args.reach_threshold and orientation_error <= args.orientation_threshold:
            reached_steps += 1
        else:
            reached_steps = 0
        hold_steps = int(np.ceil(args.target_hold / env.config.policy_dt))
        if reached_steps >= hold_steps:
            target_index = (target_index + 1) % len(targets)
            next_position, next_rpy = targets[target_index]
            env.set_target(next_position, next_rpy)
            reached_steps = 0
            print(
                f"[{float(last_record['time']):7.3f}s] Target reached "
                f"(position={position_error:.4f} m, orientation={orientation_error:.4f} rad); "
                f"next target {target_index + 1}/{len(targets)}: position={env.target_pos}, RPY={next_rpy}"
            )

    if args.headless:
        while max_steps is None or completed_steps < max_steps:
            advance_policy()
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while (max_steps is None or completed_steps < max_steps) and viewer.is_running():
                start = time.perf_counter()
                advance_policy()
                viewer.sync()
                wall_step = env.config.policy_dt / args.playback_speed
                remaining = wall_step - (time.perf_counter() - start)
                if remaining > 0.0:
                    time.sleep(remaining)

    if last_record is None:
        raise RuntimeError("The rollout ended before any policy steps were completed.")
    final = last_record
    error = np.linalg.norm(np.asarray(final["ee_pos"]) - env.target_pos)
    orientation_error = quaternion_error_rad(np.asarray(final["ee_quat"]), env.target_quat_wxyz)
    print(f"Completed {completed_steps} policy steps ({float(final['time']):.3f} s).")
    print(f"Final end-effector position: {np.asarray(final['ee_pos'])}")
    print(f"Target position:             {env.target_pos}")
    print(f"Final position error:        {error:.6f} m")
    print(f"Final orientation error:     {orientation_error:.6f} rad ({np.degrees(orientation_error):.3f} deg)")

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        keys = records[0].keys()
        arrays = {key: np.stack([np.asarray(record[key]) for record in records]) for key in keys}
        np.savez_compressed(args.record, **arrays)
        print(f"Saved trajectory: {args.record.resolve()}")


if __name__ == "__main__":
    main()
