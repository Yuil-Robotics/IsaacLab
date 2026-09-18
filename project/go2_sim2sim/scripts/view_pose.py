#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the Yuil Dog robot in mid-air and interactively adjust joint angles.

Usage (run via isaaclab.sh):
  ./isaaclab.sh -p scripts/view_pose.py [--headless]

Keyboard shortcuts (terminal):
  q/a  : FL_hip_roll   +/-
  w/s  : FL_hip_pitch  +/-
  e/d  : FL_knee_pitch +/-
  r/f  : FR_hip_roll   +/-
  t/g  : FR_hip_pitch  +/-
  y/h  : FR_knee_pitch +/-
  u/j  : RL_hip_roll   +/-
  i/k  : RL_hip_pitch  +/-
  o/l  : RL_knee_pitch +/-
  p/;  : RR_hip_roll   +/-
  [/'  : RR_hip_pitch  +/-
  ]/\\ : RR_knee_pitch +/-
  0    : Reset all joints to default pose
  ESC  : Quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── CLI args BEFORE SimulationApp ──────────────────────────────────────────────
# AppLauncher must be imported first to register its own CLI flags
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Yuil Dog pose viewer – suspend robot and tune joint angles.")
parser.add_argument(
    "--spawn_height",
    type=float,
    default=1.2,
    help="Height above ground to suspend the robot [m].",
)
# Register all Isaac Sim / Kit launcher flags (--headless, --livestream, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force the Kit GUI window to open unless --headless was explicitly requested.
# Without this AppLauncher defaults to no-render mode and nothing appears on screen.
if getattr(args_cli, "headless", False):
    args_cli.visualizer = ["none"]
else:
    args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab / USD modules ─────────────────────────────────
import contextlib
import select
import termios
import tty

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

# ── Project path so we can import go2_sim2sim ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go2_sim2sim.asset_cfg import (  # noqa: E402
    YUIL_DOG_CFG,
    YUIL_DOG_JOINT_NAMES,
)

# ── Joint step size (radians per key press) ────────────────────────────────────
STEP_SIZE = 0.05  # rad


# ── Default joint angles — read directly from YUIL_DOG_CFG so that any change
# in asset_cfg.py is automatically reflected here.
def _resolve_default_joint_pos() -> np.ndarray:
    """Resolve YUIL_DOG_CFG.init_state.joint_pos patterns → per-joint array.

    The init_state uses regex-style glob keys (e.g. ``".*L_hip_pitch"``).
    We match each joint name against those keys in priority order (most specific
    first) and fall back to 0.0 for any unmatched joint.
    """
    import re

    init_joint_pos: dict[str, float] = YUIL_DOG_CFG.init_state.joint_pos
    joint_names_list = list(YUIL_DOG_JOINT_NAMES)
    result = np.zeros(len(joint_names_list), dtype=np.float64)

    for i, jname in enumerate(joint_names_list):
        for pattern, value in init_joint_pos.items():
            # Isaac Lab uses regex patterns; convert ".*" glob to proper regex
            if re.fullmatch(pattern, jname):
                result[i] = float(value)
                break  # first match wins

    return result


DEFAULT_JOINT_POS: np.ndarray = _resolve_default_joint_pos()

# ── Hardware-safe limits (rad) matching yuil_dog_cfg.py ───────────────────────
JOINT_LIMITS = {
    0: (-0.38, 0.46),  # FL_hip_roll
    1: (-1.80, 1.20),  # FL_hip_pitch
    2: (0.05, 1.57),  # FL_knee_pitch
    3: (-0.46, 0.38),  # FR_hip_roll
    4: (-1.20, 1.80),  # FR_hip_pitch
    5: (-1.57, -0.05),  # FR_knee_pitch
    6: (-0.46, 0.38),  # RL_hip_roll
    7: (-1.80, 1.20),  # RL_hip_pitch
    8: (0.05, 1.57),  # RL_knee_pitch
    9: (-0.38, 0.46),  # RR_hip_roll
    10: (-1.20, 1.80),  # RR_hip_pitch
    11: (-1.57, -0.05),  # RR_knee_pitch
}

# ── Key -> (joint_index, direction) map ───────────────────────────────────────
KEY_MAP: dict[str, tuple[int, float]] = {
    "q": (0, +1),
    "a": (0, -1),  # FL_hip_roll
    "w": (1, +1),
    "s": (1, -1),  # FL_hip_pitch
    "e": (2, +1),
    "d": (2, -1),  # FL_knee_pitch
    "r": (3, +1),
    "f": (3, -1),  # FR_hip_roll
    "t": (4, +1),
    "g": (4, -1),  # FR_hip_pitch
    "y": (5, +1),
    "h": (5, -1),  # FR_knee_pitch
    "u": (6, +1),
    "j": (6, -1),  # RL_hip_roll
    "i": (7, +1),
    "k": (7, -1),  # RL_hip_pitch
    "o": (8, +1),
    "l": (8, -1),  # RL_knee_pitch
    "p": (9, +1),
    ";": (9, -1),  # RR_hip_roll
    "[": (10, +1),
    "'": (10, -1),  # RR_hip_pitch
    "]": (11, +1),
    "\\": (11, -1),  # RR_knee_pitch
}


# ─────────────────────────────────────────────────────────────────────────────
# Terminal keyboard helpers
# ─────────────────────────────────────────────────────────────────────────────


def _setup_term() -> list | None:
    """Switch stdin to raw / no-echo mode. Returns saved settings or None."""
    if not sys.stdin.isatty():
        return None
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    return settings


def _restore_term(settings: list | None) -> None:
    if settings is not None:
        with contextlib.suppress(Exception):
            termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)
    with contextlib.suppress(Exception):
        import os

        os.system("stty sane 2>/dev/null")


def _read_key_nonblocking() -> str | None:
    """Return a single character from stdin if available, else None."""
    if not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        ch = sys.stdin.read(1)
        # Eat escape sequences (arrows etc.) without acting on them
        if ch == "\x1b":
            select.select([sys.stdin], [], [], 0.01)
            r2, _, _ = select.select([sys.stdin], [], [], 0)
            if r2:
                sys.stdin.read(2)
            return None
        return ch
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scene setup
# ─────────────────────────────────────────────────────────────────────────────


def _build_scene(height: float) -> Articulation:
    """Create the simulation scene and return the robot articulation."""

    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # Distant dome light
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 1.0))
    cfg_light.func("/World/DomeLight", cfg_light)

    # Robot – suspended in the air; gravity disabled on rigid body so it floats
    robot_cfg: ArticulationCfg = YUIL_DOG_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, height),
        joint_pos={
            ".*_hip_roll": 0.0,
            ".*L_hip_pitch": -0.4,
            ".*R_hip_pitch": 0.4,
            ".*L_knee_pitch": 0.8,
            ".*R_knee_pitch": -0.8,
        },
        joint_vel={".*": 0.0},
    )
    # Disable gravity + heavy damping so the body stays perfectly still
    robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        retain_accelerations=False,
        linear_damping=500.0,
        angular_damping=500.0,
        max_linear_velocity=1.0,
        max_angular_velocity=1.0,
        max_depenetration_velocity=1.0,
    )

    robot = Articulation(robot_cfg)
    return robot


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_controls() -> None:
    header = "\n" + "=" * 62
    print(header)
    print("  Yuil Dog Pose Viewer  --  Keyboard Controls")
    print("=" * 62)
    rows = [
        ("FL_hip_roll   [0]", "q", "a"),
        ("FL_hip_pitch  [4]", "w", "s"),
        ("FL_knee_pitch [8]", "e", "d"),
        ("FR_hip_roll   [1]", "r", "f"),
        ("FR_hip_pitch  [5]", "t", "g"),
        ("FR_knee_pitch [9]", "y", "h"),
        ("RL_hip_roll   [2]", "u", "j"),
        ("RL_hip_pitch  [6]", "i", "k"),
        ("RL_knee_pitch[10]", "o", "l"),
        ("RR_hip_roll   [3]", "p", ";"),
        ("RR_hip_pitch  [7]", "[", "'"),
        ("RR_knee_pitch[11]", "]", "\\"),
    ]
    print(f"  {'Joint':<22} {'Inc':>4}  {'Dec':>4}")
    print("-" * 40)
    for name, inc, dec in rows:
        print(f"  {name:<22} {inc:>4}  {dec:>4}")
    print("-" * 62)
    print("  0   -> Reset all joints to default pose")
    print(f"  Step size: {STEP_SIZE} rad per key press")
    print("  ESC -> Quit")
    print("=" * 62 + "\n")


def _print_pose(joint_pos: np.ndarray) -> None:
    """Print current joint angles with bar chart."""
    print("\n  Current joint angles (rad):")
    print("  " + "-" * 60)
    for i, name in enumerate(YUIL_DOG_JOINT_NAMES):
        lo, hi = JOINT_LIMITS[i]
        bar_len = 20
        ratio = (joint_pos[i] - lo) / (hi - lo) if hi > lo else 0.5
        ratio = max(0.0, min(1.0, ratio))
        filled = int(ratio * bar_len)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"  {name:<18} {joint_pos[i]:+7.4f} rad  [{bar}]  ({lo:+.2f}~{hi:+.2f})")
    print("  " + "-" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Set up sim, spawn robot, run interactive loop."""

    # Simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)

    # Isometric camera view
    sim.set_camera_view(eye=[2.5, 2.5, 1.5], target=[0.0, 0.0, 0.5])

    # Build scene
    robot = _build_scene(args_cli.spawn_height)

    # Reset sim (spawns everything)
    sim.reset()

    # Resolve joint ordering inside articulation
    joint_names_list = list(YUIL_DOG_JOINT_NAMES)
    joint_indices = [robot.find_joints(n)[0][0] for n in joint_names_list]

    device = sim.device
    num_joints = robot.num_joints

    # Working copy of joint target angles
    current_pos = DEFAULT_JOINT_POS.copy()

    def _apply_pose() -> None:
        """Write current_pos to robot position drive targets."""
        pos_tensor = torch.zeros((1, num_joints), device=device)
        for ctrl_i, art_i in enumerate(joint_indices):
            pos_tensor[0, art_i] = float(current_pos[ctrl_i])
        robot.set_joint_position_target(pos_tensor)
        # Teleport joints directly so they snap immediately
        robot.write_joint_state_to_sim(pos_tensor, torch.zeros_like(pos_tensor))

    def _freeze_root() -> None:
        """Keep the root body exactly at spawn position every step."""
        root_state = robot.data.root_state_w.clone()
        root_state[0, 0] = 0.0
        root_state[0, 1] = 0.0
        root_state[0, 2] = float(args_cli.spawn_height)
        root_state[0, 3] = 0.0  # qx
        root_state[0, 4] = 0.0  # qy
        root_state[0, 5] = 0.0  # qz
        root_state[0, 6] = 1.0  # qw
        root_state[0, 7:13] = 0.0  # linear + angular vel
        robot.write_root_state_to_sim(root_state)

    # Apply default pose before first step
    _apply_pose()

    # Terminal raw mode
    term_settings = _setup_term()
    _print_controls()
    _print_pose(current_pos)
    print("  [Sim running -- press keys in terminal to adjust joints]\n")

    try:
        while simulation_app.is_running():
            # ── Keyboard input ──────────────────────────────────────
            ch = _read_key_nonblocking()
            changed = False

            if ch is not None:
                if ch in ("\x1b", "\x03"):  # ESC or Ctrl-C
                    print("\n  Exiting pose viewer...")
                    break

                if ch == "0":
                    current_pos[:] = DEFAULT_JOINT_POS
                    changed = True
                    print("\n  -> Reset to default pose")

                elif ch in KEY_MAP:
                    j_idx, direction = KEY_MAP[ch]
                    new_val = current_pos[j_idx] + direction * STEP_SIZE
                    lo, hi = JOINT_LIMITS[j_idx]
                    new_val = float(np.clip(new_val, lo, hi))
                    if new_val != current_pos[j_idx]:
                        current_pos[j_idx] = new_val
                        changed = True
                        jname = joint_names_list[j_idx]
                        print(f"\r  {jname:<20} -> {new_val:+.4f} rad", end="", flush=True)

            if changed:
                _apply_pose()
                _print_pose(current_pos)

            # ── Write back & freeze root ────────────────────────────
            robot.write_data_to_sim()
            _freeze_root()

            # ── Step simulation ─────────────────────────────────────
            sim.step(render=not args_cli.headless)
            robot.update(sim.get_physics_dt())

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        _restore_term(term_settings)

    simulation_app.close()


if __name__ == "__main__":
    main()
