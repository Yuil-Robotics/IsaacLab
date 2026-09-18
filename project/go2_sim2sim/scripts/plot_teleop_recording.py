#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot teleoperation recorded CSV data (commands, IMU, actions, joint tracking)."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go2_sim2sim.asset_cfg import YUIL_DOG_JOINT_NAMES  # noqa: E402

ARTIFACT_DIR = Path("/home/chung/.gemini/antigravity-ide/brain/32d5466d-7b6f-46c2-8921-1dbcfccb62f0")

LEG_GROUPS = {
    "Front Left (FL)": ["FL_hip_roll", "FL_hip_pitch", "FL_knee_pitch"],
    "Front Right (FR)": ["FR_hip_roll", "FR_hip_pitch", "FR_knee_pitch"],
    "Rear Left (RL)": ["RL_hip_roll", "RL_hip_pitch", "RL_knee_pitch"],
    "Rear Right (RR)": ["RR_hip_roll", "RR_hip_pitch", "RR_knee_pitch"],
}

COLOR_MAP = {
    "roll": "#2563EB",   # Vibrant Blue
    "pitch": "#059669",  # Emerald Green
    "knee": "#DC2626",   # Crimson Red
}


def find_latest_csv(directory: Path) -> Path | None:
    """Find the most recently modified CSV recording file."""
    if not directory.is_dir():
        return None
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


YUIL_DOG_DEFAULT_JOINT_POS = {
    "FL_hip_roll": 0.0, "FL_hip_pitch": -0.7, "FL_knee_pitch": 0.7,
    "FR_hip_roll": 0.0, "FR_hip_pitch": 0.7, "FR_knee_pitch": -0.7,
    "RL_hip_roll": 0.0, "RL_hip_pitch": -0.7, "RL_knee_pitch": 0.7,
    "RR_hip_roll": 0.0, "RR_hip_pitch": 0.7, "RR_knee_pitch": -0.7,
}


def load_teleop_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Parse teleoperation recording CSV into dictionary of numpy arrays."""
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    data_matrix = np.array([[float(val) for val in row] for row in raw_rows], dtype=np.float64)
    n_cols = data_matrix.shape[1]

    data: dict[str, np.ndarray] = {}
    if n_cols == len(headers):
        data = {header: data_matrix[:, col_idx] for col_idx, header in enumerate(headers)}
    elif n_cols == 72:
        # Columns in 72-col recording:
        # [0]: time_s, [1]: step
        # [2:47]: 45 input values
        # [47:59]: 12 out_action values
        # [59]: 1 action_joint_rad (FL_hip_roll)
        # [60:72]: 12 actual_joint_rad values
        data["time_s"] = data_matrix[:, 0]
        data["step"] = data_matrix[:, 1]
        for idx, h in enumerate(headers[2:47], start=2):
            data[h] = data_matrix[:, idx]
        for idx, name in enumerate(YUIL_DOG_JOINT_NAMES):
            out_act = data_matrix[:, 47 + idx]
            data[f"out_action_{name}"] = out_act
            # Reconstruct target joint angles: q_target = q_default + 0.25 * action
            def_pos = YUIL_DOG_DEFAULT_JOINT_POS.get(name, 0.0)
            data[f"action_joint_rad_{name}"] = def_pos + 0.25 * np.clip(out_act, -3.5, 3.5)
            data[f"actual_joint_rad_{name}"] = data_matrix[:, 60 + idx]
    else:
        # Map as many as available
        for col_idx in range(min(n_cols, len(headers))):
            data[headers[col_idx]] = data_matrix[:, col_idx]

    # Ensure delta_q_rad_* exists for all joints
    for name in YUIL_DOG_JOINT_NAMES:
        dq_key = f"delta_q_rad_{name}"
        if dq_key not in data and f"actual_joint_rad_{name}" in data:
            def_pos = YUIL_DOG_DEFAULT_JOINT_POS.get(name, 0.0)
            data[dq_key] = data[f"actual_joint_rad_{name}"] - def_pos

    return data


def compute_teleop_rms_metrics(data: dict[str, np.ndarray]) -> dict[str, float | dict[str, float]]:
    """Compute aggregate RMS(a) and Δq summary statistics from loaded teleop data.

    Returns:
        Dictionary with overall and per-group RMS actions, displacements, and per-leg breakdowns.
    """
    hip_joints = ["FL_hip_roll", "FR_hip_roll", "RL_hip_roll", "RR_hip_roll"]
    thigh_joints = ["FL_hip_pitch", "FR_hip_pitch", "RL_hip_pitch", "RR_hip_pitch"]
    calf_joints = ["FL_knee_pitch", "FR_knee_pitch", "RL_knee_pitch", "RR_knee_pitch"]

    # Actions
    act_hip = np.concatenate([data[f"out_action_{j}"] for j in hip_joints if f"out_action_{j}" in data]) if any(f"out_action_{j}" in data for j in hip_joints) else np.array([0.0])
    act_thigh = np.concatenate([data[f"out_action_{j}"] for j in thigh_joints if f"out_action_{j}" in data]) if any(f"out_action_{j}" in data for j in thigh_joints) else np.array([0.0])
    act_calf = np.concatenate([data[f"out_action_{j}"] for j in calf_joints if f"out_action_{j}" in data]) if any(f"out_action_{j}" in data for j in calf_joints) else np.array([0.0])

    rms_a_hip = float(np.sqrt(np.mean(act_hip ** 2)))
    rms_a_thigh = float(np.sqrt(np.mean(act_thigh ** 2)))
    rms_a_calf = float(np.sqrt(np.mean(act_calf ** 2)))

    # Joint Displacements Δq (rad & deg)
    dq_hip = np.concatenate([data[f"delta_q_rad_{j}"] for j in hip_joints if f"delta_q_rad_{j}" in data]) if any(f"delta_q_rad_{j}" in data for j in hip_joints) else np.array([0.0])
    dq_thigh = np.concatenate([data[f"delta_q_rad_{j}"] for j in thigh_joints if f"delta_q_rad_{j}" in data]) if any(f"delta_q_rad_{j}" in data for j in thigh_joints) else np.array([0.0])
    dq_calf = np.concatenate([data[f"delta_q_rad_{j}"] for j in calf_joints if f"delta_q_rad_{j}" in data]) if any(f"delta_q_rad_{j}" in data for j in calf_joints) else np.array([0.0])

    rms_dq_hip_rad = float(np.sqrt(np.mean(dq_hip ** 2)))
    rms_dq_thigh_rad = float(np.sqrt(np.mean(dq_thigh ** 2)))
    rms_dq_calf_rad = float(np.sqrt(np.mean(dq_calf ** 2)))

    rms_dq_hip_deg = float(np.degrees(rms_dq_hip_rad))
    rms_dq_thigh_deg = float(np.degrees(rms_dq_thigh_rad))
    rms_dq_calf_deg = float(np.degrees(rms_dq_calf_rad))

    max_dq_hip_deg = float(np.degrees(np.max(np.abs(dq_hip))))
    max_dq_thigh_deg = float(np.degrees(np.max(np.abs(dq_thigh))))
    max_dq_calf_deg = float(np.degrees(np.max(np.abs(dq_calf))))

    mean_dq_hip_deg = float(np.degrees(np.mean(np.abs(dq_hip))))
    mean_dq_thigh_deg = float(np.degrees(np.mean(np.abs(dq_thigh))))
    mean_dq_calf_deg = float(np.degrees(np.mean(np.abs(dq_calf))))

    per_leg = {}
    for leg in ["FL", "FR", "RL", "RR"]:
        hj, tj, cj = f"{leg}_hip_roll", f"{leg}_hip_pitch", f"{leg}_knee_pitch"
        if f"out_action_{hj}" in data:
            per_leg[f"{leg}_rms_a_hip"] = float(np.sqrt(np.mean(data[f"out_action_{hj}"] ** 2)))
        if f"out_action_{tj}" in data:
            per_leg[f"{leg}_rms_a_thigh"] = float(np.sqrt(np.mean(data[f"out_action_{tj}"] ** 2)))
        if f"out_action_{cj}" in data:
            per_leg[f"{leg}_rms_a_calf"] = float(np.sqrt(np.mean(data[f"out_action_{cj}"] ** 2)))

        if f"delta_q_rad_{hj}" in data:
            per_leg[f"{leg}_rms_dq_hip_deg"] = float(np.degrees(np.sqrt(np.mean(data[f"delta_q_rad_{hj}"] ** 2))))
        if f"delta_q_rad_{tj}" in data:
            per_leg[f"{leg}_rms_dq_thigh_deg"] = float(np.degrees(np.sqrt(np.mean(data[f"delta_q_rad_{tj}"] ** 2))))
        if f"delta_q_rad_{cj}" in data:
            per_leg[f"{leg}_rms_dq_calf_deg"] = float(np.degrees(np.sqrt(np.mean(data[f"delta_q_rad_{cj}"] ** 2))))

    return {
        "rms_a_hip": rms_a_hip,
        "rms_a_thigh": rms_a_thigh,
        "rms_a_calf": rms_a_calf,
        "rms_dq_hip_rad": rms_dq_hip_rad,
        "rms_dq_thigh_rad": rms_dq_thigh_rad,
        "rms_dq_calf_rad": rms_dq_calf_rad,
        "rms_dq_hip_deg": rms_dq_hip_deg,
        "rms_dq_thigh_deg": rms_dq_thigh_deg,
        "rms_dq_calf_deg": rms_dq_calf_deg,
        "max_abs_dq_hip_deg": max_dq_hip_deg,
        "max_abs_dq_thigh_deg": max_dq_thigh_deg,
        "max_abs_dq_calf_deg": max_dq_calf_deg,
        "mean_abs_dq_hip_deg": mean_dq_hip_deg,
        "mean_abs_dq_thigh_deg": mean_dq_thigh_deg,
        "mean_abs_dq_calf_deg": mean_dq_calf_deg,
        "max_dq_hip_deg": max_dq_hip_deg,
        "max_dq_thigh_deg": max_dq_thigh_deg,
        "max_dq_calf_deg": max_dq_calf_deg,
        "mean_dq_hip_deg": mean_dq_hip_deg,
        "mean_dq_thigh_deg": mean_dq_thigh_deg,
        "mean_dq_calf_deg": mean_dq_calf_deg,
        "per_leg": per_leg,
    }


def plot_base_dynamics(data: dict[str, np.ndarray], output_path: Path, title_suffix: str = ""):
    """Plot velocity commands, base angular velocities, and projected gravity tilt."""
    t = data["time_s"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, dpi=200)
    fig.suptitle(f"Base Dynamics & Teleoperation Commands {title_suffix}", fontsize=14, fontweight="bold")

    # 1. Velocity Commands
    ax = axes[0]
    ax.plot(t, data["in_cmd_vx"], label="Cmd Vx [m/s]", color="#2563EB", linewidth=1.8)
    ax.plot(t, data["in_cmd_vy"], label="Cmd Vy [m/s]", color="#10B981", linewidth=1.5, linestyle="--")
    ax.plot(t, data["in_cmd_wz"], label="Cmd Wz [rad/s]", color="#F59E0B", linewidth=1.5, linestyle="-.")
    ax.set_ylabel("Command", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Velocity Commands (Longitudinal, Lateral, Yaw)", fontsize=11)

    # 2. Base Angular Velocity (IMU Gyro)
    ax = axes[1]
    ax.plot(t, data["in_base_ang_vel_x"], label="Roll Rate Wx [rad/s]", color="#8B5CF6", linewidth=1.2, alpha=0.85)
    ax.plot(t, data["in_base_ang_vel_y"], label="Pitch Rate Wy [rad/s]", color="#EC4899", linewidth=1.2, alpha=0.85)
    ax.plot(t, data["in_base_ang_vel_z"], label="Yaw Rate Wz [rad/s]", color="#F59E0B", linewidth=1.5)
    ax.set_ylabel("Ang Vel [rad/s]", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Base Body Angular Velocity (Gyro)", fontsize=11)

    # 3. Projected Gravity & Estimated Pitch/Roll
    ax = axes[2]
    # Projected gravity components
    pitch_est_deg = np.degrees(np.arcsin(np.clip(data["in_proj_grav_x"], -1.0, 1.0)))
    roll_est_deg = np.degrees(np.arcsin(np.clip(-data["in_proj_grav_y"], -1.0, 1.0)))

    ax.plot(t, pitch_est_deg, label="Est Base Pitch [deg]", color="#059669", linewidth=1.6)
    ax.plot(t, roll_est_deg, label="Est Base Roll [deg]", color="#2563EB", linewidth=1.4)
    ax.plot(t, data["in_proj_grav_z"], label="Proj Gravity Gz", color="#6B7280", linewidth=1.2, linestyle=":")
    ax.set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tilt [deg] / Gz", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Body Tilt & Projected Gravity Vector", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_joint_tracking(data: dict[str, np.ndarray], output_path: Path, title_suffix: str = ""):
    """Plot target vs actual joint angles in radians for all 4 legs."""
    t = data["time_s"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, dpi=200)
    fig.suptitle(f"12-Joint Position Tracking: Target (Dashed) vs Actual (Solid) [rad] {title_suffix}",
                 fontsize=14, fontweight="bold")

    leg_axes = [
        ("Front Left (FL)", axes[0, 0]),
        ("Front Right (FR)", axes[0, 1]),
        ("Rear Left (RL)", axes[1, 0]),
        ("Rear Right (RR)", axes[1, 1]),
    ]

    for leg_name, ax in leg_axes:
        joints = LEG_GROUPS[leg_name]
        roll_j, pitch_j, knee_j = joints

        # Hip Roll
        ax.plot(t, data[f"action_joint_rad_{roll_j}"], label=f"{roll_j} Target",
                color=COLOR_MAP["roll"], linestyle="--", linewidth=1.5, alpha=0.85)
        ax.plot(t, data[f"actual_joint_rad_{roll_j}"], label=f"{roll_j} Actual",
                color=COLOR_MAP["roll"], linestyle="-", linewidth=1.8)

        # Hip Pitch
        ax.plot(t, data[f"action_joint_rad_{pitch_j}"], label=f"{pitch_j} Target",
                color=COLOR_MAP["pitch"], linestyle="--", linewidth=1.5, alpha=0.85)
        ax.plot(t, data[f"actual_joint_rad_{pitch_j}"], label=f"{pitch_j} Actual",
                color=COLOR_MAP["pitch"], linestyle="-", linewidth=1.8)

        # Knee Pitch
        ax.plot(t, data[f"action_joint_rad_{knee_j}"], label=f"{knee_j} Target",
                color=COLOR_MAP["knee"], linestyle="--", linewidth=1.5, alpha=0.85)
        ax.plot(t, data[f"actual_joint_rad_{knee_j}"], label=f"{knee_j} Actual",
                color=COLOR_MAP["knee"], linestyle="-", linewidth=1.8)

        ax.set_title(leg_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Joint Angle [rad]", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8, ncol=3, framealpha=0.85)

    axes[1, 0].set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Time [s]", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_tracking_error_and_velocities(data: dict[str, np.ndarray], output_path: Path, title_suffix: str = ""):
    """Plot joint tracking error (target - actual) and joint velocities."""
    t = data["time_s"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, dpi=200)
    fig.suptitle(f"Joint Tracking Errors & Angular Velocities {title_suffix}", fontsize=14, fontweight="bold")

    # 1. Hip Roll Errors
    ax_roll = axes[0]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_hip_roll"
        err = data[f"action_joint_rad_{name}"] - data[f"actual_joint_rad_{name}"]
        ax_roll.plot(t, np.degrees(err), label=f"{name} (MAE={np.degrees(np.mean(np.abs(err))):.2f}°)", linewidth=1.3)
    ax_roll.set_ylabel("Roll Error [deg]", fontsize=10, fontweight="bold")
    ax_roll.set_title("Hip Roll Tracking Error", fontsize=11)
    ax_roll.grid(True, linestyle=":", alpha=0.6)
    ax_roll.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    # 2. Hip Pitch & Knee Pitch Errors
    ax_pitch = axes[1]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_hip_pitch"
        err = data[f"action_joint_rad_{name}"] - data[f"actual_joint_rad_{name}"]
        ax_pitch.plot(t, np.degrees(err), label=f"{name} (MAE={np.degrees(np.mean(np.abs(err))):.2f}°)", linewidth=1.3)
    ax_pitch.set_ylabel("Pitch Error [deg]", fontsize=10, fontweight="bold")
    ax_pitch.set_title("Hip Pitch Tracking Error", fontsize=11)
    ax_pitch.grid(True, linestyle=":", alpha=0.6)
    ax_pitch.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    # 3. Knee Pitch Errors
    ax_knee = axes[2]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_knee_pitch"
        err = data[f"action_joint_rad_{name}"] - data[f"actual_joint_rad_{name}"]
        ax_knee.plot(t, np.degrees(err), label=f"{name} (MAE={np.degrees(np.mean(np.abs(err))):.2f}°)", linewidth=1.3)
    ax_knee.set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    ax_knee.set_ylabel("Knee Error [deg]", fontsize=10, fontweight="bold")
    ax_knee.set_title("Knee Pitch Tracking Error", fontsize=11)
    ax_knee.grid(True, linestyle=":", alpha=0.6)
    ax_knee.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_actions_and_saturation(data: dict[str, np.ndarray], output_path: Path, title_suffix: str = ""):
    """Plot raw policy output actions (dimensionless) and check saturation limits."""
    t = data["time_s"]
    m = compute_teleop_rms_metrics(data)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, dpi=200)
    fig.suptitle(
        f"Policy Action Outputs & Saturation Check [-3.5, +3.5] {title_suffix}\n"
        f"RMS(a_hip) = {m['rms_a_hip']:.4f}  |  RMS(a_thigh) = {m['rms_a_thigh']:.4f}  |  RMS(a_calf) = {m['rms_a_calf']:.4f}",
        fontsize=13, fontweight="bold",
    )

    # 1. Hip Roll Actions
    ax0 = axes[0]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_hip_roll"
        if f"out_action_{name}" in data:
            leg_rms = m.get("per_leg", {}).get(f"{leg}_rms_a_hip", 0.0)
            ax0.plot(t, data[f"out_action_{name}"], label=f"{name} (RMS={leg_rms:.3f})", linewidth=1.3)
    ax0.axhline(3.5, color="red", linestyle=":", alpha=0.7, label="Saturation Limit (±3.5)")
    ax0.axhline(-3.5, color="red", linestyle=":", alpha=0.7)
    ax0.set_ylabel("Action (Roll)", fontsize=10, fontweight="bold")
    ax0.set_title(f"Hip Roll Raw Actions — Group RMS(a_hip) = {m['rms_a_hip']:.4f}", fontsize=11, fontweight="bold")
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(loc="upper right", fontsize=8, ncol=5, framealpha=0.9)

    # 2. Hip Pitch Actions
    ax1 = axes[1]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_hip_pitch"
        if f"out_action_{name}" in data:
            leg_rms = m.get("per_leg", {}).get(f"{leg}_rms_a_thigh", 0.0)
            ax1.plot(t, data[f"out_action_{name}"], label=f"{name} (RMS={leg_rms:.3f})", linewidth=1.3)
    ax1.axhline(3.5, color="red", linestyle=":", alpha=0.7)
    ax1.axhline(-3.5, color="red", linestyle=":", alpha=0.7)
    ax1.set_ylabel("Action (Pitch)", fontsize=10, fontweight="bold")
    ax1.set_title(f"Hip Pitch (Thigh) Raw Actions — Group RMS(a_thigh) = {m['rms_a_thigh']:.4f}", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    # 3. Knee Pitch Actions
    ax2 = axes[2]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_knee_pitch"
        if f"out_action_{name}" in data:
            leg_rms = m.get("per_leg", {}).get(f"{leg}_rms_a_calf", 0.0)
            ax2.plot(t, data[f"out_action_{name}"], label=f"{name} (RMS={leg_rms:.3f})", linewidth=1.3)
    ax2.axhline(3.5, color="red", linestyle=":", alpha=0.7)
    ax2.axhline(-3.5, color="red", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Action (Knee)", fontsize=10, fontweight="bold")
    ax2.set_title(f"Knee Pitch (Calf) Raw Actions — Group RMS(a_calf) = {m['rms_a_calf']:.4f}", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_joint_displacements(data: dict[str, np.ndarray], output_path: Path, title_suffix: str = ""):
    """Plot joint displacements from default pose (Δq = q_actual - q_default) in degrees."""
    t = data["time_s"]
    m = compute_teleop_rms_metrics(data)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, dpi=200)
    fig.suptitle(
        f"Joint Displacements from Default Pose (Δq = q_actual - q_default) [deg] {title_suffix}\n"
        f"RMS(Δq_hip) = {m['rms_dq_hip_deg']:.2f}°  |  RMS(Δq_thigh) = {m['rms_dq_thigh_deg']:.2f}°  |  RMS(Δq_calf) = {m['rms_dq_calf_deg']:.2f}°",
        fontsize=13, fontweight="bold",
    )

    # 1. Hip Roll Displacements
    ax0 = axes[0]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_hip_roll"
        if f"delta_q_rad_{name}" in data:
            deg = np.degrees(data[f"delta_q_rad_{name}"])
            leg_rms = m.get("per_leg", {}).get(f"{leg}_rms_dq_hip_deg", 0.0)
            ax0.plot(t, deg, label=f"{name} (RMS={leg_rms:.2f}°)", linewidth=1.3)
    ax0.set_ylabel("Δq Hip [deg]", fontsize=10, fontweight="bold")
    ax0.set_title(f"Hip Roll Displacement Δq_hip — RMS: {m['rms_dq_hip_deg']:.2f}° | Max: {m['max_abs_dq_hip_deg']:.2f}°", fontsize=11, fontweight="bold")
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    # 2. Hip Pitch (Thigh) Displacements
    ax1 = axes[1]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_hip_pitch"
        if f"delta_q_rad_{name}" in data:
            deg = np.degrees(data[f"delta_q_rad_{name}"])
            leg_rms = m.get("per_leg", {}).get(f"{leg}_rms_dq_thigh_deg", 0.0)
            ax1.plot(t, deg, label=f"{name} (RMS={leg_rms:.2f}°)", linewidth=1.3)
    ax1.set_ylabel("Δq Thigh [deg]", fontsize=10, fontweight="bold")
    ax1.set_title(f"Hip Pitch (Thigh) Displacement Δq_thigh — RMS: {m['rms_dq_thigh_deg']:.2f}° | Max: {m['max_abs_dq_thigh_deg']:.2f}°", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    # 3. Knee Pitch (Calf) Displacements
    ax2 = axes[2]
    for leg in ["FL", "FR", "RL", "RR"]:
        name = f"{leg}_knee_pitch"
        if f"delta_q_rad_{name}" in data:
            deg = np.degrees(data[f"delta_q_rad_{name}"])
            leg_rms = m.get("per_leg", {}).get(f"{leg}_rms_dq_calf_deg", 0.0)
            ax2.plot(t, deg, label=f"{name} (RMS={leg_rms:.2f}°)", linewidth=1.3)
    ax2.set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Δq Calf [deg]", fontsize=10, fontweight="bold")
    ax2.set_title(f"Knee Pitch (Calf) Displacement Δq_calf — RMS: {m['rms_dq_calf_deg']:.2f}° | Max: {m['max_abs_dq_calf_deg']:.2f}°", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_summary_dashboard(data: dict[str, np.ndarray], output_path: Path, title_suffix: str = ""):
    """Create a single comprehensive executive dashboard plot."""
    t = data["time_s"]

    fig = plt.figure(figsize=(18, 12), dpi=200)
    fig.suptitle(f"Yuil Dog Teleoperation Performance Summary Dashboard {title_suffix}",
                 fontsize=15, fontweight="bold")

    gs = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.20)

    # 1. Top-Left: Velocity Command & Pitch/Roll
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, data["in_cmd_vx"], label="Cmd Vx [m/s]", color="#2563EB", linewidth=2.0)
    pitch_est_deg = np.degrees(np.arcsin(np.clip(data["in_proj_grav_x"], -1.0, 1.0)))
    roll_est_deg = np.degrees(np.arcsin(np.clip(-data["in_proj_grav_y"], -1.0, 1.0)))
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, pitch_est_deg, label="Pitch [deg]", color="#059669", linewidth=1.3, alpha=0.75)
    ax1_twin.plot(t, roll_est_deg, label="Roll [deg]", color="#EC4899", linewidth=1.3, alpha=0.75)
    ax1.set_ylabel("Command [m/s]", fontsize=10, fontweight="bold", color="#2563EB")
    ax1_twin.set_ylabel("Tilt [deg]", fontsize=10, fontweight="bold", color="#059669")
    ax1.set_title("(a) Velocity Command & Robot Tilt Dynamics", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # 2. Top-Right: Front Left (FL) Leg Target vs Actual
    ax2 = fig.add_subplot(gs[0, 1])
    for j_name, col in [("FL_hip_pitch", "#059669"), ("FL_knee_pitch", "#DC2626")]:
        ax2.plot(t, data[f"action_joint_rad_{j_name}"], label=f"{j_name} Tgt", color=col, linestyle="--", linewidth=1.4)
        ax2.plot(t, data[f"actual_joint_rad_{j_name}"], label=f"{j_name} Act", color=col, linestyle="-", linewidth=1.8)
    ax2.set_ylabel("Angle [rad]", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Front-Left (FL) Hip/Knee Pitch Tracking", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    # 3. Mid-Left: Front Right (FR) Leg Target vs Actual
    ax3 = fig.add_subplot(gs[1, 0])
    for j_name, col in [("FR_hip_pitch", "#059669"), ("FR_knee_pitch", "#DC2626")]:
        ax3.plot(t, data[f"action_joint_rad_{j_name}"], label=f"{j_name} Tgt", color=col, linestyle="--", linewidth=1.4)
        ax3.plot(t, data[f"actual_joint_rad_{j_name}"], label=f"{j_name} Act", color=col, linestyle="-", linewidth=1.8)
    ax3.set_ylabel("Angle [rad]", fontsize=10, fontweight="bold")
    ax3.set_title("(c) Front-Right (FR) Hip/Knee Pitch Tracking", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8, ncol=2)

    # 4. Mid-Right: Rear Legs Knee Pitch Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, data["action_joint_rad_RL_knee_pitch"], label="RL Knee Tgt", color="#2563EB", linestyle="--", linewidth=1.4)
    ax4.plot(t, data["actual_joint_rad_RL_knee_pitch"], label="RL Knee Act", color="#2563EB", linestyle="-", linewidth=1.8)
    ax4.plot(t, data["action_joint_rad_RR_knee_pitch"], label="RR Knee Tgt", color="#D97706", linestyle="--", linewidth=1.4)
    ax4.plot(t, data["actual_joint_rad_RR_knee_pitch"], label="RR Knee Act", color="#D97706", linestyle="-", linewidth=1.8)
    ax4.set_ylabel("Angle [rad]", fontsize=10, fontweight="bold")
    ax4.set_title("(d) Rear Left (RL) vs Rear Right (RR) Knee Pitch Tracking", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8, ncol=2)

    # 5. Bottom-Left: Knee Pitch Tracking Errors
    ax5 = fig.add_subplot(gs[2, 0])
    for leg, color in [("FL", "#2563EB"), ("FR", "#059669"), ("RL", "#D97706"), ("RR", "#DC2626")]:
        knee_err = data[f"action_joint_rad_{leg}_knee_pitch"] - data[f"actual_joint_rad_{leg}_knee_pitch"]
        ax5.plot(t, np.degrees(knee_err), label=f"{leg} Knee (MAE={np.degrees(np.mean(np.abs(knee_err))):.2f}°)",
                 color=color, linewidth=1.3)
    ax5.set_xlabel("Time [s]", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Error [deg]", fontsize=10, fontweight="bold")
    ax5.set_title("(e) Knee Tracking Error Across 4 Legs", fontsize=11, fontweight="bold")
    ax5.grid(True, linestyle=":", alpha=0.6)
    ax5.legend(loc="upper right", fontsize=8, ncol=2)

    # 6. Bottom-Right: Raw Output Actions (Knee Saturation)
    ax6 = fig.add_subplot(gs[2, 1])
    for leg, color in [("FL", "#2563EB"), ("FR", "#059669"), ("RL", "#D97706"), ("RR", "#DC2626")]:
        ax6.plot(t, data[f"out_action_{leg}_knee_pitch"], label=f"{leg}_knee", color=color, linewidth=1.3)
    ax6.axhline(3.5, color="red", linestyle=":", label="Clip Limit (±3.5)")
    ax6.axhline(-3.5, color="red", linestyle=":")
    ax6.set_xlabel("Time [s]", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Raw Action", fontsize=10, fontweight="bold")
    ax6.set_title("(f) Policy Raw Output Actions (Knee Saturation Check)", fontsize=11, fontweight="bold")
    ax6.grid(True, linestyle=":", alpha=0.6)
    ax6.legend(loc="upper right", fontsize=8, ncol=3)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_base_trajectory(data: dict[str, np.ndarray], output_path: Path, title_suffix: str = ""):
    """Plot 2D base XY trajectory, position over time, and base height."""
    t = data["time_s"]
    x = data.get("base_pos_x")
    y = data.get("base_pos_y")
    h = data.get("base_height")

    if x is None or y is None or h is None:
        return

    fig = plt.figure(figsize=(15, 10), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    fig.suptitle(f"Base Link Kinematics & Trajectory {title_suffix}", fontsize=14, fontweight="bold")

    # 1. Top Panel (spanning both columns): 2D XY Trajectory
    ax_xy = fig.add_subplot(gs[0, :])
    ax_xy.plot(x, y, color="#2563EB", linewidth=2.0, label="Base Path (X vs Y)")
    ax_xy.scatter([x[0]], [y[0]], color="#059669", s=100, zorder=5, label=f"Start ({x[0]:.2f}, {y[0]:.2f})")
    ax_xy.scatter([x[-1]], [y[-1]], color="#DC2626", s=100, zorder=5, label=f"End ({x[-1]:.2f}, {y[-1]:.2f})")

    # Calculate total path distance and net displacement
    dx = np.diff(x)
    dy = np.diff(y)
    total_dist = np.sum(np.sqrt(dx**2 + dy**2))
    net_disp = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

    ax_xy.set_xlabel("X Position (Forward) [m]", fontsize=11, fontweight="bold")
    ax_xy.set_ylabel("Y Position (Lateral) [m]", fontsize=11, fontweight="bold")
    ax_xy.set_title(
        f"Top-Down 2D Odometry Trajectory (Path Distance: {total_dist:.2f} m | Net Displacement: {net_disp:.2f} m)",
        fontsize=12,
        fontweight="bold",
    )
    ax_xy.grid(True, linestyle=":", alpha=0.6)
    ax_xy.legend(loc="best", framealpha=0.9)
    ax_xy.axis("equal")

    # 2. Bottom-Left Panel: Base X & Y vs Time
    ax_t = fig.add_subplot(gs[1, 0])
    ax_t.plot(t, x, label="Base X (Forward) [m]", color="#2563EB", linewidth=1.8)
    ax_t.plot(t, y, label="Base Y (Lateral) [m]", color="#F59E0B", linewidth=1.5, linestyle="--")
    ax_t.set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    ax_t.set_ylabel("Position [m]", fontsize=11, fontweight="bold")
    ax_t.set_title("Base Position vs Time", fontsize=12, fontweight="bold")
    ax_t.grid(True, linestyle=":", alpha=0.6)
    ax_t.legend(loc="best", framealpha=0.9)

    # 3. Bottom-Right Panel: Base Height vs Time
    ax_h = fig.add_subplot(gs[1, 1])
    mean_h = np.mean(h)
    std_h = np.std(h)
    ax_h.plot(t, h, label=f"Base Height (Mean={mean_h:.3f}m ± {std_h:.3f}m)", color="#059669", linewidth=1.8)
    ax_h.axhline(0.27, color="#DC2626", linestyle=":", linewidth=1.5, label="Target Height (0.27m)")
    ax_h.set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    ax_h.set_ylabel("Clearance [m]", fontsize=11, fontweight="bold")
    ax_h.set_title("Base Height (Ground-Relative Clearance)", fontsize=12, fontweight="bold")
    ax_h.grid(True, linestyle=":", alpha=0.6)
    ax_h.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, nargs="?", default=None, help="Path to teleop recording CSV file.")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None, help="Directory to save generated plots.")
    parser.add_argument("--artifact-dir", "--artifact_dir", type=Path, default=None, help="Artifact directory to copy plots to.")
    parser.add_argument("--copy-to-artifact", action="store_true", default=True, help="Copy plots to IDE artifact directory.")
    args = parser.parse_args()

    default_record_dir = PROJECT_ROOT / "logs" / "teleop_recordings"

    if args.csv_path is None:
        csv_file = find_latest_csv(default_record_dir)
        if csv_file is None:
            print(f"[ERROR] No CSV file found in {default_record_dir}. Please provide a path.")
            sys.exit(1)
        print(f"[INFO] Auto-selected latest recording: {csv_file.name}")
    else:
        csv_file = Path(args.csv_path).expanduser().resolve()
        if not csv_file.is_file():
            print(f"[ERROR] File does not exist: {csv_file}")
            sys.exit(1)

    # Determine output directories
    stem = csv_file.stem
    if args.output_dir is not None:
        plot_dir = Path(args.output_dir).expanduser().resolve()
    else:
        plot_dir = default_record_dir / "plots" / stem
    plot_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = None
    if args.artifact_dir is not None:
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    elif ARTIFACT_DIR.is_dir():
        artifact_dir = ARTIFACT_DIR

    print(f"[INFO] Loading {csv_file} ...")
    data = load_teleop_csv(csv_file)
    n_steps = len(data["time_s"])
    duration = data["time_s"][-1]
    print(f"[INFO] Loaded {n_steps} steps ({duration:.2f} seconds).")

    # Print summary metrics box
    m = compute_teleop_rms_metrics(data)
    pl = m.get("per_leg", {})
    print("\n" + "=" * 72)
    print(f" [REC] Teleoperation Recording Metric Summary ({duration:.2f}s, {n_steps} steps)")
    print("=" * 72)
    print("1. Policy Action Root-Mean-Square (RMS):")
    print(f"   * RMS(a_hip)   = {m['rms_a_hip']:.4f}")
    print(f"   * RMS(a_thigh) = {m['rms_a_thigh']:.4f}")
    print(f"   * RMS(a_calf)  = {m['rms_a_calf']:.4f}")
    print("   [Per-Leg Action RMS]")
    print(f"     FL: hip={pl.get('FL_rms_a_hip', 0.0):.3f}, thigh={pl.get('FL_rms_a_thigh', 0.0):.3f}, calf={pl.get('FL_rms_a_calf', 0.0):.3f}")
    print(f"     FR: hip={pl.get('FR_rms_a_hip', 0.0):.3f}, thigh={pl.get('FR_rms_a_thigh', 0.0):.3f}, calf={pl.get('FR_rms_a_calf', 0.0):.3f}")
    print(f"     RL: hip={pl.get('RL_rms_a_hip', 0.0):.3f}, thigh={pl.get('RL_rms_a_thigh', 0.0):.3f}, calf={pl.get('RL_rms_a_calf', 0.0):.3f}")
    print(f"     RR: hip={pl.get('RR_rms_a_hip', 0.0):.3f}, thigh={pl.get('RR_rms_a_thigh', 0.0):.3f}, calf={pl.get('RR_rms_a_calf', 0.0):.3f}")
    print("\n2. Joint Displacement from Default Pose (Δq = q_actual - q_default):")
    print(f"   * RMS(Δq_hip)   = {m['rms_dq_hip_rad']:.4f} rad ({m['rms_dq_hip_deg']:6.2f} deg)  | Max: {m['max_abs_dq_hip_deg']:5.2f} deg, Mean: {m['mean_abs_dq_hip_deg']:5.2f} deg")
    print(f"   * RMS(Δq_thigh) = {m['rms_dq_thigh_rad']:.4f} rad ({m['rms_dq_thigh_deg']:6.2f} deg)  | Max: {m['max_abs_dq_thigh_deg']:5.2f} deg, Mean: {m['mean_abs_dq_thigh_deg']:5.2f} deg")
    print(f"   * RMS(Δq_calf)  = {m['rms_dq_calf_rad']:.4f} rad ({m['rms_dq_calf_deg']:6.2f} deg)  | Max: {m['max_abs_dq_calf_deg']:5.2f} deg, Mean: {m['mean_abs_dq_calf_deg']:5.2f} deg")
    print("   [Per-Leg Δq RMS (deg)]")
    print(f"     FL: hip={pl.get('FL_rms_dq_hip_deg', 0.0):5.2f} deg, thigh={pl.get('FL_rms_dq_thigh_deg', 0.0):5.2f} deg, calf={pl.get('FL_rms_dq_calf_deg', 0.0):5.2f} deg")
    print(f"     FR: hip={pl.get('FR_rms_dq_hip_deg', 0.0):5.2f} deg, thigh={pl.get('FR_rms_dq_thigh_deg', 0.0):5.2f} deg, calf={pl.get('FR_rms_dq_calf_deg', 0.0):5.2f} deg")
    print(f"     RL: hip={pl.get('RL_rms_dq_hip_deg', 0.0):5.2f} deg, thigh={pl.get('RL_rms_dq_thigh_deg', 0.0):5.2f} deg, calf={pl.get('RL_rms_dq_calf_deg', 0.0):5.2f} deg")
    print(f"     RR: hip={pl.get('RR_rms_dq_hip_deg', 0.0):5.2f} deg, thigh={pl.get('RR_rms_dq_thigh_deg', 0.0):5.2f} deg, calf={pl.get('RR_rms_dq_calf_deg', 0.0):5.2f} deg")
    print("=" * 72 + "\n")

    # Generate figures
    plots = [
        ("teleop_base_dynamics.png", plot_base_dynamics),
        ("teleop_joint_tracking.png", plot_joint_tracking),
        ("teleop_tracking_error.png", plot_tracking_error_and_velocities),
        ("teleop_actions_and_limits.png", plot_actions_and_saturation),
        ("teleop_joint_displacements.png", plot_joint_displacements),
        ("teleop_summary_dashboard.png", plot_summary_dashboard),
    ]
    if "base_pos_x" in data and "base_height" in data:
        plots.append(("teleop_base_trajectory.png", plot_base_trajectory))

    generated_paths: list[Path] = []
    for filename, plot_func in plots:
        out_file = plot_dir / filename
        print(f"[INFO] Generating {filename} ...")
        plot_func(data, out_file, title_suffix=f"({stem})")
        generated_paths.append(out_file)

        # Copy to artifact directory if available
        if args.copy_to_artifact and artifact_dir is not None and artifact_dir.is_dir():
            artifact_file = artifact_dir / filename
            shutil.copy2(out_file, artifact_file)

    print("\n" + "=" * 65)
    print("Teleoperation Graphs Generated Successfully!")
    print("=" * 65)
    print(f"Output Directory: {plot_dir}")
    for p in generated_paths:
        print(f"  - {p.name}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
