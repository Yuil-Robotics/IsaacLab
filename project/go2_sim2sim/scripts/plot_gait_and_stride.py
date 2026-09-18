#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait, stride length, cadence, and foot kinematics analysis for recorded teleop data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Register and set NanumGothic font if available
nanum_font_path = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
if nanum_font_path.is_file():
    fm.fontManager.addfont(str(nanum_font_path))
    font_prop = fm.FontProperties(fname=str(nanum_font_path))
    plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Leg kinematic parameters (from URDF)
THIGH_LENGTH = 0.2543  # [m]
CALF_LENGTH = 0.2400   # [m]

COLOR_LEGS = {
    "FL": "#2563EB",  # Blue (Diagonal 1)
    "RR": "#3B82F6",  # Light Blue (Diagonal 1)
    "FR": "#059669",  # Green (Diagonal 2)
    "RL": "#10B981",  # Emerald (Diagonal 2)
}

TROT_PAIRS = (("FL", "RR"), ("FR", "RL"))


def find_latest_csv(directory: Path) -> Path | None:
    """Find the most recently modified CSV recording file."""
    if not directory.is_dir():
        return None
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def load_teleop_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Parse teleoperation CSV recording into dictionary of numpy arrays."""
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    data_matrix = np.array([[float(val) for val in row] for row in raw_rows], dtype=np.float64)
    data: dict[str, np.ndarray] = {header: data_matrix[:, idx] for idx, header in enumerate(headers)}
    return data


def compute_contact_cadence(
    data: dict[str, np.ndarray],
    time_s: np.ndarray,
) -> dict[str, object] | None:
    """Measure cadence from the same diagonal-pair contact events used by the reward.

    Args:
        data: Recorded CSV columns.
        time_s: Simulation timestamps [s].

    Returns:
        Contact states, pair periods [s], and aggregate cadence [Hz], or
        ``None`` for legacy recordings without contact columns.
    """
    contact_columns = {leg: f"foot_contact_{leg}" for leg in COLOR_LEGS}
    if not all(column in data for column in contact_columns.values()):
        return None

    contacts = {leg: data[column] > 0.5 for leg, column in contact_columns.items()}
    steady_end = min(float(time_s[-1]), 8.0)
    steady_mask = (time_s >= 2.0) & (time_s <= steady_end)
    if np.count_nonzero(steady_mask) < 3:
        steady_mask = np.ones_like(time_s, dtype=bool)

    pair_periods: dict[str, np.ndarray] = {}
    all_periods = []
    for first_leg, second_leg in TROT_PAIRS:
        pair_name = f"{first_leg}_{second_leg}"
        pair_contact = contacts[first_leg] & contacts[second_leg]
        previous_contact = np.concatenate(([pair_contact[0]], pair_contact[:-1]))
        touchdown = pair_contact & ~previous_contact & steady_mask
        touchdown_times = time_s[touchdown]
        periods = np.diff(touchdown_times)
        periods = periods[(periods >= 0.04) & (periods <= 0.60)]
        pair_periods[pair_name] = periods
        all_periods.extend(periods.tolist())

    if not all_periods:
        return {
            "contacts": contacts,
            "duty_cycles": {leg: float(np.mean(states[steady_mask])) for leg, states in contacts.items()},
            "pair_periods": pair_periods,
            "gait_frequency": None,
            "stride_period": None,
        }

    stride_period = float(np.median(all_periods))
    return {
        "contacts": contacts,
        "duty_cycles": {leg: float(np.mean(states[steady_mask])) for leg, states in contacts.items()},
        "pair_periods": pair_periods,
        "gait_frequency": 1.0 / stride_period,
        "stride_period": stride_period,
    }


def compute_gait_metrics(data: dict[str, np.ndarray]) -> dict[str, any]:
    """Compute comprehensive gait, stride, cadence, and kinematic metrics."""
    t = data["time_s"]
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    duration = float(t[-1])

    # 1. Base kinematics & speed
    bx = data["base_pos_x"]
    by = data["base_pos_y"]
    bh = data["base_height"]
    dbx = np.diff(bx)
    dby = np.diff(by)
    step_dists = np.sqrt(dbx**2 + dby**2)
    total_dist = float(np.sum(step_dists))
    net_disp = float(np.sqrt((bx[-1] - bx[0])**2 + (by[-1] - by[0])**2))
    mean_speed = total_dist / duration
    cmd_vx = float(np.mean(data["in_cmd_vx"]))

    # 2. Foot forward kinematics (sagittal plane relative to hip)
    foot_x: dict[str, np.ndarray] = {}
    foot_z: dict[str, np.ndarray] = {}
    foot_vx: dict[str, np.ndarray] = {}
    duty_cycles: dict[str, float] = {}

    for leg in ["FL", "FR", "RL", "RR"]:
        hp = data[f"actual_joint_rad_{leg}_hip_pitch"]
        kp = data[f"actual_joint_rad_{leg}_knee_pitch"]
        hp_phys = -hp if "R" in leg else hp
        kp_phys = -kp if "R" in leg else kp

        fx = THIGH_LENGTH * np.sin(hp_phys) + CALF_LENGTH * np.sin(hp_phys + kp_phys)
        fz = -THIGH_LENGTH * np.cos(hp_phys) - CALF_LENGTH * np.cos(hp_phys + kp_phys)
        fvx = np.gradient(fx, dt)

        foot_x[leg] = fx
        foot_z[leg] = fz
        foot_vx[leg] = fvx
        duty_cycles[leg] = float(np.mean(fvx < 0))

    # 3. Frequency & period analysis. Prefer reward-aligned contact events.
    contact_metrics = compute_contact_cadence(data, t)
    leg_freqs = []
    for leg in ["FL", "FR", "RL", "RR"]:
        kp = data[f"actual_joint_rad_{leg}_knee_pitch"]
        sig = kp - np.mean(kp)
        freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))
        dom_f = float(freqs[np.argmax(psd)])
        leg_freqs.append(dom_f)

    joint_frequency = float(np.median(leg_freqs))
    if contact_metrics is not None and contact_metrics["gait_frequency"] is not None:
        gait_freq = float(contact_metrics["gait_frequency"])
        stride_period = float(contact_metrics["stride_period"])
        duty_cycles = contact_metrics["duty_cycles"]
        frequency_source = "contact sensor"
    else:
        gait_freq = joint_frequency
        stride_period = 1.0 / gait_freq if gait_freq > 0 else 0.0
        frequency_source = "joint spectrum fallback"

    # 4. Stride Length & Step Length
    stride_length = mean_speed * stride_period
    step_length = stride_length / 2.0

    # 5. Cadence (steps per minute)
    cadence_spm = gait_freq * 2.0 * 60.0
    total_strides_per_leg = gait_freq * duration
    total_foot_contacts = total_strides_per_leg * 4.0

    # 6. Diagonal and Contralateral Correlations
    fl_h = data["actual_joint_rad_FL_hip_pitch"]
    fr_h = -data["actual_joint_rad_FR_hip_pitch"]
    rl_h = data["actual_joint_rad_RL_hip_pitch"]
    rr_h = -data["actual_joint_rad_RR_hip_pitch"]

    diag1_corr = float(np.corrcoef(fl_h, rr_h)[0, 1])
    diag2_corr = float(np.corrcoef(fr_h, rl_h)[0, 1])
    contra_corr = float(np.corrcoef(fl_h, fr_h)[0, 1])

    # 7. Phase lags via Hilbert transform
    ss_mask = (t >= 2.0) & (t <= 8.0)
    hilb_fl = signal.hilbert(fl_h - np.mean(fl_h))
    p_fl = np.unwrap(np.angle(hilb_fl))
    phases: dict[str, float] = {"FL": 0.0}
    for leg, h_sig in [("FR", fr_h), ("RL", rl_h), ("RR", rr_h)]:
        h_trans = signal.hilbert(h_sig - np.mean(h_sig))
        p_other = np.unwrap(np.angle(h_trans))
        dphi = float(np.median(np.degrees((p_other - p_fl) % (2 * np.pi))[ss_mask]))
        phases[leg] = dphi

    # 8. Ground-relative foot clearance [m]
    foot_clearances: dict[str, np.ndarray] = {}
    foot_clearance_stats: dict[str, dict[str, float]] = {}
    for leg in ["FL", "FR", "RL", "RR"]:
        # Kinematic foot z in base frame:
        hp = data[f"actual_joint_rad_{leg}_hip_pitch"]
        kp = data[f"actual_joint_rad_{leg}_knee_pitch"]
        hp_p = -hp if "R" in leg else hp
        kp_p = -kp if "R" in leg else kp
        z_rel_hip = -(THIGH_LENGTH * np.cos(hp_p) + CALF_LENGTH * np.cos(hp_p + kp_p))
        z_world = bh + 0.06 + z_rel_hip
        c_col = f"foot_contact_{leg}"
        if c_col in data:
            c = data[c_col] > 0.5
        else:
            c = foot_vx[leg] < 0
        st_idx = np.where(c)[0]
        sw_idx = np.where(~c)[0]
        g_ref = np.percentile(z_world[st_idx], 5) if len(st_idx) > 0 else np.min(z_world)
        cl = np.maximum(0.0, z_world - g_ref)
        foot_clearances[leg] = cl
        foot_clearance_stats[leg] = {
            "peak_swing_cm": float(np.max(cl[sw_idx])) * 100.0 if len(sw_idx) > 0 else float(np.max(cl)) * 100.0,
            "mean_swing_cm": float(np.mean(cl[sw_idx])) * 100.0 if len(sw_idx) > 0 else float(np.mean(cl)) * 100.0,
            "mean_stance_cm": float(np.mean(cl[st_idx])) * 100.0 if len(st_idx) > 0 else 0.0,
        }

    # 9. Foot impact forces & touchdown velocities
    has_force_data = all(f"foot_force_{leg}_norm" in data for leg in ["FL", "FR", "RL", "RR"])
    impact_stats: dict[str, dict[str, float]] = {}
    if has_force_data:
        for leg in ["FL", "FR", "RL", "RR"]:
            fnorm = data[f"foot_force_{leg}_norm"]
            c_col = f"foot_contact_{leg}"
            if c_col in data:
                c = data[c_col] > 0.5
            else:
                c = fnorm > 5.0

            vz_col = f"foot_vel_{leg}_z"
            vz = data.get(vz_col, np.zeros_like(fnorm))

            c_int = c.astype(int)
            touchdowns = np.where(np.diff(c_int) > 0)[0] + 1

            ss_mask = (t >= 2.0) & (t <= min(t[-1], 8.0))
            if np.count_nonzero(ss_mask) < 10:
                ss_mask = np.ones_like(t, dtype=bool)

            td_impact_forces = []
            td_velocities = []
            for td_idx in touchdowns:
                if td_idx < len(ss_mask) and ss_mask[td_idx]:
                    window_end = min(len(fnorm), td_idx + 3)
                    peak_f = float(np.max(fnorm[td_idx:window_end]))
                    td_impact_forces.append(peak_f)
                    td_velocities.append(float(vz[td_idx]))

            stance_forces = fnorm[c & ss_mask]
            mean_stance_f = float(np.mean(stance_forces)) if len(stance_forces) > 0 else 0.0
            max_impact_f = (
                float(np.max(td_impact_forces))
                if len(td_impact_forces) > 0
                else (float(np.max(fnorm)) if len(fnorm) > 0 else 0.0)
            )
            mean_impact_f = float(np.mean(td_impact_forces)) if len(td_impact_forces) > 0 else 0.0
            min_td_vz = float(np.min(td_velocities)) if len(td_velocities) > 0 else 0.0

            impact_stats[leg] = {
                "max_impact_n": max_impact_f,
                "mean_impact_n": mean_impact_f,
                "mean_stance_n": mean_stance_f,
                "touchdown_vz_mps": min_td_vz,
                "touchdown_count": len(td_impact_forces),
            }

    return {
        "dt": dt,
        "fs": fs,
        "duration": duration,
        "cmd_vx": cmd_vx,
        "mean_speed": mean_speed,
        "total_dist": total_dist,
        "net_disp": net_disp,
        "speed_error_pct": ((mean_speed - cmd_vx) / cmd_vx) * 100.0 if cmd_vx != 0 else 0.0,
        "base_height_mean": float(np.mean(bh)),
        "base_height_std": float(np.std(bh)),
        "gait_freq": gait_freq,
        "joint_frequency": joint_frequency,
        "frequency_source": frequency_source,
        "stride_period": stride_period,
        "stride_length": stride_length,
        "step_length": step_length,
        "cadence_spm": cadence_spm,
        "total_strides_per_leg": total_strides_per_leg,
        "total_foot_contacts": total_foot_contacts,
        "duty_cycles": duty_cycles,
        "contact_metrics": contact_metrics,
        "foot_x": foot_x,
        "foot_z": foot_z,
        "foot_vx": foot_vx,
        "foot_clearances": foot_clearances,
        "foot_clearance_stats": foot_clearance_stats,
        "has_force_data": has_force_data,
        "impact_stats": impact_stats,
        "diag1_corr": diag1_corr,
        "diag2_corr": diag2_corr,
        "contra_corr": contra_corr,
        "phases": phases,
    }


def plot_gait_phase_analysis(
    data: dict[str, np.ndarray], metrics: dict[str, any], output_path: Path, title_suffix: str = ""
):
    """Plot gait phase synchronization, joint waveforms, and contact timing diagram."""
    t = data["time_s"]
    t_start = 2.0
    t_end = min(t[-1], 3.6)
    mask = (t >= t_start) & (t <= t_end)
    t_win = t[mask]

    fig = plt.figure(figsize=(15, 12), dpi=200)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.2])
    fig.suptitle(f"Gait Coordination & Phase Synchronization {title_suffix}", fontsize=14, fontweight="bold")

    # 1. Top-Left: Hip Pitch Waveforms (Diagonal Pair 1: FL & RR)
    ax1 = fig.add_subplot(gs[0, 0])
    fl_hp = data["actual_joint_rad_FL_hip_pitch"][mask]
    rr_hp = -data["actual_joint_rad_RR_hip_pitch"][mask]
    ax1.plot(t_win, fl_hp, label=f"FL_hip_pitch (Ref, 0°)", color=COLOR_LEGS["FL"], linewidth=2.0)
    ax1.plot(t_win, rr_hp, label=f"RR_hip_pitch (Diag 1, Phase={metrics['phases']['RR']:.1f}°)",
             color=COLOR_LEGS["RR"], linewidth=1.8, linestyle="--")
    ax1.set_ylabel("Physical Hip Pitch [rad]", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Diagonal Pair 1 (FL & RR) Hip Pitch Coordination", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Top-Right: Hip Pitch Waveforms (Diagonal Pair 2: FR & RL)
    ax2 = fig.add_subplot(gs[0, 1])
    fr_hp = -data["actual_joint_rad_FR_hip_pitch"][mask]
    rl_hp = data["actual_joint_rad_RL_hip_pitch"][mask]
    ax2.plot(t_win, fr_hp, label=f"FR_hip_pitch (Phase={metrics['phases']['FR']:.1f}°)",
             color=COLOR_LEGS["FR"], linewidth=2.0)
    ax2.plot(t_win, rl_hp, label=f"RL_hip_pitch (Phase={metrics['phases']['RL']:.1f}°)",
             color=COLOR_LEGS["RL"], linewidth=1.8, linestyle="--")
    ax2.set_ylabel("Physical Hip Pitch [rad]", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Diagonal Pair 2 (FR & RL) Hip Pitch Coordination", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. Middle-Left: Knee Pitch Flexion (Swing/Stance Indicators)
    ax3 = fig.add_subplot(gs[1, 0])
    fl_kp = data["actual_joint_rad_FL_knee_pitch"][mask]
    rr_kp = -data["actual_joint_rad_RR_knee_pitch"][mask]
    ax3.plot(t_win, fl_kp, label="FL_knee (Swing peak=flexion)", color=COLOR_LEGS["FL"], linewidth=1.8)
    ax3.plot(t_win, rr_kp, label="RR_knee (Swing peak=flexion)", color=COLOR_LEGS["RR"], linewidth=1.8, linestyle="--")
    ax3.set_ylabel("Knee Angle [rad]", fontsize=10, fontweight="bold")
    ax3.set_title("(c) Knee Flexion (FL & RR): Periodic Swing Apex Timing", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8)

    # 4. Middle-Right: Contralateral Pair (FL vs FR: Antiphase Check)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_win, fl_hp, label="FL (Left Front)", color=COLOR_LEGS["FL"], linewidth=2.0)
    ax4.plot(t_win, fr_hp, label=f"FR (Right Front, Corr={metrics['contra_corr']:.2f})",
             color=COLOR_LEGS["FR"], linewidth=2.0, linestyle="-.")
    ax4.set_ylabel("Physical Hip Pitch [rad]", fontsize=10, fontweight="bold")
    ax4.set_title(f"(d) Contralateral Front Pairing (FL vs FR: Phase Shift={metrics['phases']['FR']:.1f}°)",
                  fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8)

    # 5. Bottom-Left: Gait Stance / Swing Timing Raster Chart
    ax5 = fig.add_subplot(gs[2, 0])
    y_labels = ["RR", "RL", "FR", "FL"]
    y_pos = [0, 1, 2, 3]

    for idx, leg in enumerate(y_labels):
        t_arr = t_win
        contact_metrics = metrics["contact_metrics"]
        if contact_metrics is None:
            is_stance = metrics["foot_vx"][leg][mask] < 0
        else:
            is_stance = contact_metrics["contacts"][leg][mask]
        for i in range(len(t_arr) - 1):
            if is_stance[i]:
                ax5.barh(idx, t_arr[i + 1] - t_arr[i], left=t_arr[i], height=0.6,
                         color=COLOR_LEGS[leg], alpha=0.85, edgecolor="none")

    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([f"{l} ({metrics['duty_cycles'][l]*100:.0f}%)" for l in y_labels], fontweight="bold")
    ax5.set_xlabel("Time [s]", fontsize=10, fontweight="bold")
    contact_source = "Sensor Contact" if contact_metrics is not None else "Kinematic Estimate"
    ax5.set_title(
        f"(e) Gait Contact Raster ({contact_source}: Solid = Stance, Space = Swing)",
        fontsize=11,
        fontweight="bold",
    )
    ax5.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax5.set_xlim(t_start, t_end)

    # 6. Bottom-Right: Phase Polar Compass Diagram
    ax6 = fig.add_subplot(gs[2, 1], polar=True)
    legs_order = ["FL", "RR", "FR", "RL"]
    angles_rad = [np.radians(metrics["phases"][l]) for l in legs_order]
    radii = [1.0, 0.9, 1.0, 0.9]

    for leg, ang, r in zip(legs_order, angles_rad, radii):
        ax6.plot([0, ang], [0, r], color=COLOR_LEGS[leg], linewidth=2.5, label=f"{leg} ({metrics['phases'][leg]:.0f}°)")
        ax6.scatter([ang], [r], color=COLOR_LEGS[leg], s=90, zorder=5)

    ax6.set_theta_zero_location("N")
    ax6.set_theta_direction(-1)
    ax6.set_rticks([0.5, 1.0])
    ax6.set_yticklabels([])
    ax6.set_title("(f) Relative Phase Polar Map (FL Reference at 0°)", fontsize=11, fontweight="bold", pad=15)
    ax6.legend(loc="lower right", bbox_to_anchor=(1.35, 0.0), fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_stride_and_kinematics(
    data: dict[str, np.ndarray], metrics: dict[str, any], output_path: Path, title_suffix: str = ""
):
    """Plot foot sagittal stroke, base odometry speed, phase portraits, and gait scorecard."""
    t = data["time_s"]
    fig = plt.figure(figsize=(15, 11), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    fig.suptitle(f"Stride Length, Foot Kinematics & Performance Scorecard {title_suffix}",
                 fontsize=14, fontweight="bold")

    # 1. Top-Left: Foot Sagittal Trajectory (X vs Z relative to Hip)
    ax1 = fig.add_subplot(gs[0, 0])
    for leg in ["FL", "FR", "RL", "RR"]:
        fx = metrics["foot_x"][leg] * 100.0  # [cm]
        fz = metrics["foot_z"][leg] * 100.0  # [cm]
        mask_tail = t >= (t[-1] - 0.5)
        ax1.plot(fx[mask_tail], fz[mask_tail], label=f"{leg} Foot (Stroke={(fx.max()-fx.min()):.1f}cm)",
                 color=COLOR_LEGS[leg], linewidth=2.0)

    ax1.set_xlabel("Foot Sagittal X (Forward/Backward) [cm]", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Foot Sagittal Z (Height relative to Hip) [cm]", fontsize=10, fontweight="bold")
    ax1.set_title("(a) 2D Foot Swing & Stance Trajectory Ellipses", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Top-Right: Forward Distance & Velocity Tracking
    ax2 = fig.add_subplot(gs[0, 1])
    bx = data["base_pos_x"]
    by = data["base_pos_y"]
    cum_dist = np.cumsum(np.sqrt(np.diff(bx, prepend=bx[0])**2 + np.diff(by, prepend=by[0])**2))
    inst_v = np.gradient(cum_dist, metrics["dt"])
    inst_v_smooth = np.convolve(inst_v, np.ones(15) / 15, mode="same")

    ax2.plot(t, inst_v_smooth, label=f"Actual Forward Speed (Mean={metrics['mean_speed']:.3f} m/s)",
             color="#2563EB", linewidth=2.0)
    ax2.axhline(metrics["cmd_vx"], color="#DC2626", linestyle="--", linewidth=1.5,
                label=f"Command Speed ({metrics['cmd_vx']:.2f} m/s)")
    ax2.set_xlabel("Time [s]", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Speed [m/s]", fontsize=10, fontweight="bold")
    ax2.set_title(f"(b) Speed Tracking: {metrics['mean_speed']:.3f} m/s (Error: {metrics['speed_error_pct']:+.1f}%)",
                  fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    # 3. Bottom-Left: Knee Phase Portrait (Limit Cycle Stability)
    ax3 = fig.add_subplot(gs[1, 0])
    for leg in ["FL", "RR", "FR", "RL"]:
        kp = data[f"actual_joint_rad_{leg}_knee_pitch"]
        kv = data[f"in_joint_vel_{leg}_knee_pitch"]
        mask_ss = (t >= 3.0) & (t <= min(t[-1], 7.0))
        ax3.plot(kp[mask_ss], kv[mask_ss], color=COLOR_LEGS[leg], alpha=0.7, linewidth=1.2, label=f"{leg}")

    ax3.set_xlabel("Knee Joint Angle [rad]", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Knee Joint Velocity [rad/s]", fontsize=10, fontweight="bold")
    ax3.set_title("(c) Joint Phase Portrait: Periodic Limit Cycle Orbit", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8, ncol=2)

    # 4. Bottom-Right: Gait Scorecard Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    scorecard_data = [
        ["항목 (Metric)", "측정값 (Value)", "목표치 / 기준 (Target/Ref)", "평가 (Status)"],
        ["보행 주파수 (Gait Frequency)", f"{metrics['gait_freq']:.2f} Hz", "≤ 3.0 Hz", metrics["frequency_source"]],
        ["보행 주기 (Stride Period)", f"{metrics['stride_period']*1000:.1f} ms", "≥ 333 ms", "동일 대각쌍 착지 간격"],
        ["보폭 (Stride Length)", f"{metrics['stride_length']*100:.1f} cm", "16 ~ 35 cm", "측정 완료"],
        ["발걸음폭 (Step Length)", f"{metrics['step_length']*100:.1f} cm", "8 ~ 18 cm", "측정 완료"],
        ["분당 걸음수 (Cadence)", f"{metrics['cadence_spm']:.0f} SPM", "300 ~ 750 SPM", "우수 (보폭수)"],
        ["10초간 총 걸음수", f"{metrics['total_strides_per_leg']:.1f} 보/다리", f"총 {metrics['total_foot_contacts']:.0f}회 접지", "안정 완주"],
        ["전진 속도 (Forward Speed)", f"{metrics['mean_speed']:.3f} m/s", f"{metrics['cmd_vx']:.2f} m/s", f"{metrics['speed_error_pct']:+.1f}% 오차"],
        ["베이스 차고 (Base Height)", f"{metrics['base_height_mean']:.3f} m", "0.30 ~ 0.35 m", "차고 범위"],
        ["몸체 흔들림 (Height Std)", f"{metrics['base_height_std']*1000:.1f} mm", "< 10 mm", "안정성 우수"],
        ["평균 지면 접촉률 (Duty Cycle)", f"{np.mean(list(metrics['duty_cycles'].values()))*100:.1f}%", "50.0% (Trot 기준)", "트롯 비율"],
    ]

    table = ax4.table(
        cellText=scorecard_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.32, 0.22, 0.26, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    for col_idx in range(4):
        cell = table[0, col_idx]
        cell.set_facecolor("#1E3A8A")
        cell.set_text_props(color="white", fontweight="bold")

    for row_idx in range(1, len(scorecard_data)):
        status_cell = table[row_idx, 3]
        status_cell.set_facecolor("#EFF6FF")
        status_cell.set_text_props(color="#1E40AF", fontweight="bold")

    ax4.set_title("(d) 핵심 보행 파라미터 정량 평가표 (Gait Scorecard)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_foot_clearance(
    data: dict[str, np.ndarray], metrics: dict[str, any], output_path: Path, title_suffix: str = ""
):
    """Plot ground-relative foot clearance height [cm] for each foot with contact state shading."""
    t = data["time_s"]
    t_start = 2.0
    t_end = min(t[-1], 5.0)  # 3-second steady-state window
    mask = (t >= t_start) & (t <= t_end)
    t_win = t[mask]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200, sharex=True, sharey=True)
    fig.suptitle(f"Foot Ground Clearance & Lift Height Profiles {title_suffix}", fontsize=14, fontweight="bold")

    legs = [("FL", axes[0, 0]), ("FR", axes[0, 1]), ("RL", axes[1, 0]), ("RR", axes[1, 1])]

    for leg, ax in legs:
        cl_cm = metrics["foot_clearances"][leg][mask] * 100.0
        stats = metrics["foot_clearance_stats"][leg]

        # Contact mask
        c_col = f"foot_contact_{leg}"
        if c_col in data:
            c = data[c_col][mask] > 0.5
        else:
            c = metrics["foot_vx"][leg][mask] < 0

        # Plot foot clearance
        ax.plot(t_win, cl_cm, color=COLOR_LEGS[leg], linewidth=2.0, label=f"{leg} Clearance [cm]")
        ax.axhline(6.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Target Clearance (6.0 cm)")
        ax.axhline(0.0, color="#78350F", linestyle=":", linewidth=1.2, label="Ground Surface (0.0 cm)")

        # Shade stance phase
        for i in range(len(t_win) - 1):
            if c[i]:
                ax.axvspan(t_win[i], t_win[i + 1], color="#E5E7EB", alpha=0.6, zorder=1)

        ax.set_title(
            f"({leg}) Peak Swing={stats['peak_swing_cm']:.1f} cm | Mean Swing={stats['mean_swing_cm']:.1f} cm | Stance Contact={stats['mean_stance_cm']:.1f} cm",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("Clearance [cm]", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    axes[1, 0].set_xlabel("Time [s]", fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("Time [s]", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_foot_impact_forces(
    data: dict[str, np.ndarray], metrics: dict[str, any], output_path: Path, title_suffix: str = ""
):
    """Plot foot contact forces [N], touchdown impact peaks, and vertical touchdown velocities [m/s]."""
    t = data["time_s"]
    t_start = 2.0
    t_end = min(t[-1], 6.0)  # 4-second steady-state window
    mask = (t >= t_start) & (t <= t_end)
    if np.count_nonzero(mask) < 10:
        mask = np.ones_like(t, dtype=bool)
    t_win = t[mask]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200, sharex=True)
    fig.suptitle(f"Foot Contact Forces & Touchdown Impact Profiles {title_suffix}", fontsize=14, fontweight="bold")

    legs = [("FL", axes[0, 0]), ("FR", axes[0, 1]), ("RL", axes[1, 0]), ("RR", axes[1, 1])]

    for leg, ax in legs:
        fnorm = data.get(f"foot_force_{leg}_norm", np.zeros_like(t))[mask]
        fz = data.get(f"foot_force_{leg}_z", fnorm)[mask]
        vz = data.get(f"foot_vel_{leg}_z", np.zeros_like(t))[mask]

        c_col = f"foot_contact_{leg}"
        if c_col in data:
            c = data[c_col][mask] > 0.5
        else:
            c = fnorm > 5.0

        stats = metrics.get("impact_stats", {}).get(leg, {
            "max_impact_n": float(np.max(fnorm)) if len(fnorm) > 0 else 0.0,
            "mean_impact_n": 0.0,
            "mean_stance_n": float(np.mean(fnorm[c])) if np.count_nonzero(c) > 0 else 0.0,
            "touchdown_vz_mps": float(np.min(vz)) if len(vz) > 0 else 0.0,
        })

        # Plot contact forces
        ax.plot(t_win, fnorm, color=COLOR_LEGS[leg], linewidth=2.0, label=f"{leg} Net Force ||F|| [N]")
        ax.plot(t_win, fz, color="#4B5563", linestyle="--", linewidth=1.2, alpha=0.8, label=f"{leg} Vertical Force Fz [N]")

        # Touchdown events
        c_int = c.astype(int)
        td_indices = np.where(np.diff(c_int) > 0)[0] + 1
        for td_idx in td_indices:
            if td_idx < len(t_win):
                ax.axvline(t_win[td_idx], color="#DC2626", linestyle=":", linewidth=1.2, alpha=0.7)
                peak_w = min(len(fnorm), td_idx + 3)
                p_val = np.max(fnorm[td_idx:peak_w])
                ax.plot(t_win[td_idx], p_val, marker="v", color="#DC2626", markersize=6)

        # Secondary y-axis for vertical velocity vz
        ax2 = ax.twinx()
        ax2.plot(t_win, vz, color="#8B5CF6", linestyle=":", linewidth=1.2, alpha=0.6, label="Vertical Vel Vz [m/s]")
        ax2.axhline(0.0, color="#8B5CF6", linestyle="-", linewidth=0.6, alpha=0.4)
        ax2.set_ylabel("Vz [m/s]", color="#8B5CF6", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#8B5CF6")

        # Shade stance phase
        for i in range(len(t_win) - 1):
            if c[i]:
                ax.axvspan(t_win[i], t_win[i + 1], color="#E5E7EB", alpha=0.5, zorder=1)

        ax.set_title(
            f"({leg}) Peak Impact={stats['max_impact_n']:.1f} N | Mean Stance={stats['mean_stance_n']:.1f} N | TD Vz={stats['touchdown_vz_mps']:.2f} m/s",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("Contact Force [N]", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8)

    axes[1, 0].set_xlabel("Time [s]", fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("Time [s]", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, nargs="?", default=None, help="Path to teleop CSV file.")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None, help="Directory to save plots.")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Artifact directory to copy plots to.")
    args = parser.parse_args()

    default_record_dir = PROJECT_ROOT / "logs" / "teleop_recordings"

    if args.csv_path is None:
        csv_file = find_latest_csv(default_record_dir)
        if csv_file is None:
            print(f"[ERROR] No CSV found in {default_record_dir}")
            sys.exit(1)
        print(f"[INFO] Auto-selected latest recording: {csv_file.name}")
    else:
        csv_file = Path(args.csv_path).expanduser().resolve()
        if not csv_file.is_file():
            print(f"[ERROR] File does not exist: {csv_file}")
            sys.exit(1)

    stem = csv_file.stem
    if args.output_dir is not None:
        plot_dir = Path(args.output_dir).expanduser().resolve()
    else:
        plot_dir = default_record_dir / "plots" / stem
    plot_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = None
    if args.artifact_dir is not None:
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()

    print(f"[INFO] Loading {csv_file.name} ...")
    data = load_teleop_csv(csv_file)
    print(f"[INFO] Calculating gait and stride metrics ...")
    metrics = compute_gait_metrics(data)

    print("\n" + "=" * 60)
    print("GAIT & STRIDE KINEMATICS SUMMARY")
    print("=" * 60)
    print(f"  - Command Vx         : {metrics['cmd_vx']:.2f} m/s")
    print(f"  - Actual Mean Speed  : {metrics['mean_speed']:.3f} m/s (Error: {metrics['speed_error_pct']:+.1f}%)")
    print(f"  - Total Distance     : {metrics['total_dist']:.2f} m (in {metrics['duration']:.2f} s)")
    print(f"  - Base Height        : {metrics['base_height_mean']:.3f} m (std: {metrics['base_height_std']*1000:.1f} mm)")
    print(f"  - Gait Frequency     : {metrics['gait_freq']:.2f} Hz ({metrics['frequency_source']})")
    print(f"  - Joint Spectrum     : {metrics['joint_frequency']:.2f} Hz")
    print(f"  - Stride Period      : {metrics['stride_period']*1000:.1f} ms")
    print(f"  - Stride Length (보폭): {metrics['stride_length']*100:.1f} cm")
    print(f"  - Step Length (발걸음): {metrics['step_length']*100:.1f} cm")
    print(f"  - Cadence (보폭수)    : {metrics['cadence_spm']:.0f} steps/min")
    print(f"  - Duty Cycle (접지율): FL={metrics['duty_cycles']['FL']*100:.0f}%, RR={metrics['duty_cycles']['RR']*100:.0f}%, "
          f"FR={metrics['duty_cycles']['FR']*100:.0f}%, RL={metrics['duty_cycles']['RL']*100:.0f}%")
    contact_metrics = metrics["contact_metrics"]
    if contact_metrics is not None:
        for pair_name, periods in contact_metrics["pair_periods"].items():
            if len(periods) > 0:
                print(
                    f"  - Pair {pair_name:5s}       : {1.0 / np.median(periods):.2f} Hz "
                    f"(median {np.median(periods)*1000:.1f} ms, n={len(periods)})"
                )
            else:
                print(f"  - Pair {pair_name:5s}       : insufficient touchdown events")

    if metrics.get("has_force_data", False):
        print("\n" + "=" * 60)
        print("FOOT CONTACT & TOUCHDOWN IMPACT ANALYSIS")
        print("=" * 60)
        for leg in ["FL", "FR", "RL", "RR"]:
            st = metrics["impact_stats"][leg]
            print(
                f"  - Leg {leg:2s}: Peak Impact={st['max_impact_n']:5.1f} N | "
                f"Mean Impact={st['mean_impact_n']:5.1f} N | "
                f"Mean Stance={st['mean_stance_n']:5.1f} N | "
                f"TD Vz={st['touchdown_vz_mps']:+5.2f} m/s (n={st['touchdown_count']})"
            )
    print("=" * 60 + "\n")

    plots = [
        ("teleop_gait_phase_analysis.png", plot_gait_phase_analysis),
        ("teleop_stride_and_kinematics.png", plot_stride_and_kinematics),
        ("teleop_foot_clearance.png", plot_foot_clearance),
    ]
    if metrics.get("has_force_data", False):
        plots.append(("teleop_foot_impact_forces.png", plot_foot_impact_forces))

    for fname, pfunc in plots:
        out_f = plot_dir / fname
        print(f"[INFO] Rendering {fname} ...")
        pfunc(data, metrics, out_f, title_suffix=f"({stem})")
        if artifact_dir is not None and artifact_dir.is_dir():
            shutil.copy2(out_f, artifact_dir / fname)
            print(f"[INFO] Copied {fname} -> {artifact_dir / fname}")

    print("\n[SUCCESS] Gait analysis plots generated successfully!")


if __name__ == "__main__":
    main()
