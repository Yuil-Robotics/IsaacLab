# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder for teleoperation policy interactions (inputs, outputs, and joint angles)."""

from __future__ import annotations

import csv
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .asset_cfg import YUIL_DOG_FOOT_NAMES, YUIL_DOG_JOINT_NAMES

DEFAULT_INPUT_NAMES_45 = (
    # 1. Base angular velocity (3)
    "in_base_ang_vel_x",
    "in_base_ang_vel_y",
    "in_base_ang_vel_z",
    # 2. Projected gravity (3)
    "in_proj_grav_x",
    "in_proj_grav_y",
    "in_proj_grav_z",
    # 3. Velocity commands (3)
    "in_cmd_vx",
    "in_cmd_vy",
    "in_cmd_wz",
    # 4. Joint positions relative to default (12)
    *(f"in_joint_pos_rel_{name}" for name in YUIL_DOG_JOINT_NAMES),
    # 5. Joint velocities (12)
    *(f"in_joint_vel_{name}" for name in YUIL_DOG_JOINT_NAMES),
    # 6. Previous step actions (12)
    *(f"in_last_action_{name}" for name in YUIL_DOG_JOINT_NAMES),
)


DEFAULT_BASE_STATE_NAMES = (
    "base_pos_x",
    "base_pos_y",
    "base_height",
)

YUIL_DOG_DEFAULT_JOINT_POS: dict[str, float] = {
    "FL_hip_roll": -0.1,
    "FL_hip_pitch": -0.7,
    "FL_knee_pitch": 0.7,
    "FR_hip_roll": 0.1,
    "FR_hip_pitch": 0.7,
    "FR_knee_pitch": -0.7,
    "RL_hip_roll": 0.1,
    "RL_hip_pitch": -0.7,
    "RL_knee_pitch": 0.7,
    "RR_hip_roll": -0.1,
    "RR_hip_pitch": 0.7,
    "RR_knee_pitch": -0.7,
}

DEFAULT_FOOT_POS_AXES = ("x", "y", "z")
DEFAULT_FOOT_FORCE_AXES = ("x", "y", "z")


class RollingFootContactFrequency:
    """Track per-foot touchdown frequency over a rolling simulation-time window."""

    def __init__(
        self,
        foot_names: Sequence[str],
        step_dt: float,
        window_s: float = 1.0,
    ) -> None:
        """Initialize the rolling touchdown tracker.

        Args:
            foot_names: Foot names in display order.
            step_dt: Simulation time between updates [s].
            window_s: Rolling touchdown-count window [s]. Defaults to 1.0.

        Raises:
            ValueError: If no feet are supplied or a time parameter is not positive.
        """
        if not foot_names:
            raise ValueError("Expected at least one foot name.")
        if step_dt <= 0.0:
            raise ValueError(f"Expected positive step_dt, got {step_dt}.")
        if window_s <= 0.0:
            raise ValueError(f"Expected positive window_s, got {window_s}.")

        self.foot_names = tuple(foot_names)
        self.step_dt = float(step_dt)
        self.window_s = float(window_s)
        self.reset()

    def reset(self) -> None:
        """Clear all touchdown history and restart elapsed simulation time."""
        self._elapsed_time_s = 0.0
        self._initialized = False
        self._previous_contact = tuple(False for _ in self.foot_names)
        self._last_touchdown_time_s: list[float | None] = [None for _ in self.foot_names]
        self._touchdown_intervals = tuple(deque() for _ in self.foot_names)

    def update(self, contact_states: Sequence[bool]) -> tuple[float, ...]:
        """Update contacts and return mean touchdown-cycle rates over the rolling window [Hz].

        Each rate is the reciprocal of the mean interval between consecutive
        touchdowns whose ending event occurred during the latest window. This
        produces fractional values such as 3.3 Hz instead of integer event
        counts per second.

        Args:
            contact_states: Current per-foot contact states in :attr:`foot_names` order.

        Returns:
            Per-foot touchdown frequencies [Hz] over :attr:`window_s`.

        Raises:
            ValueError: If the number of contact states does not match the foot count.
        """
        if len(contact_states) != len(self.foot_names):
            raise ValueError(
                f"Expected {len(self.foot_names)} contact states, got {len(contact_states)}."
            )

        self._elapsed_time_s += self.step_dt
        current_contact = tuple(bool(state) for state in contact_states)
        if self._initialized:
            for foot_index, (was_in_contact, is_in_contact) in enumerate(
                zip(self._previous_contact, current_contact)
            ):
                if is_in_contact and not was_in_contact:
                    last_touchdown_time_s = self._last_touchdown_time_s[foot_index]
                    if last_touchdown_time_s is not None:
                        interval_s = self._elapsed_time_s - last_touchdown_time_s
                        self._touchdown_intervals[foot_index].append((self._elapsed_time_s, interval_s))
                    self._last_touchdown_time_s[foot_index] = self._elapsed_time_s
        else:
            self._initialized = True

        cutoff_time_s = self._elapsed_time_s - self.window_s
        for touchdown_intervals in self._touchdown_intervals:
            while touchdown_intervals and touchdown_intervals[0][0] <= cutoff_time_s:
                touchdown_intervals.popleft()

        self._previous_contact = current_contact
        return tuple(
            len(touchdown_intervals) / sum(interval_s for _, interval_s in touchdown_intervals)
            if touchdown_intervals
            else 0.0
            for touchdown_intervals in self._touchdown_intervals
        )


def extract_default_joint_pos(unwrapped, joint_names: Sequence[str]) -> np.ndarray:
    """Extract default joint positions in radians matching the given joint sequence.

    Args:
        unwrapped: The unwrapped ManagerBasedRLEnv environment.
        joint_names: Sequence of joint names in desired order.

    Returns:
        1D numpy array of default joint angles [rad].
    """
    if unwrapped is not None and hasattr(unwrapped, "scene"):
        robot = None
        try:
            robot = unwrapped.scene["robot"]
        except Exception:
            robot = getattr(unwrapped.scene, "robot", None)
        if robot is not None and hasattr(robot, "data"):
            d_pos = getattr(robot.data, "default_joint_pos", None)
            if d_pos is not None:
                if hasattr(d_pos, "torch"):
                    d_pos = d_pos.torch
                if isinstance(d_pos, torch.Tensor):
                    arr = d_pos[0].detach().cpu().numpy().flatten()
                else:
                    arr = np.asarray(d_pos)[0].flatten()

                body_joint_names = getattr(robot.data, "joint_names", None)
                if body_joint_names is not None:
                    name_list = list(body_joint_names)
                    out = []
                    for name in joint_names:
                        if name in name_list:
                            out.append(float(arr[name_list.index(name)]))
                        else:
                            out.append(float(YUIL_DOG_DEFAULT_JOINT_POS.get(name, 0.0)))
                    return np.array(out, dtype=np.float32)
                elif len(arr) >= len(joint_names):
                    return arr[: len(joint_names)].astype(np.float32)

    return np.array([YUIL_DOG_DEFAULT_JOINT_POS.get(name, 0.0) for name in joint_names], dtype=np.float32)


def extract_foot_positions(
    unwrapped,
    foot_names: Sequence[str] = YUIL_DOG_FOOT_NAMES,
    env_idx: int = 0,
) -> tuple[tuple[float, float, float], ...]:
    """Extract 3D foot positions (x, y, z) [m] relative to the environment origin.

    Args:
        unwrapped: The unwrapped ManagerBasedRLEnv environment.
        foot_names: Foot body names in the desired CSV column order.
        env_idx: Index of the environment. Defaults to 0.

    Returns:
        Per-foot (x, y, z) coordinates in :paramref:`foot_names` order. Missing
        or invalid data produces (0.0, 0.0, 0.0) tuples.
    """
    no_positions = tuple((0.0, 0.0, 0.0) for _ in foot_names)
    if unwrapped is None:
        return no_positions

    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        return no_positions

    robot = None
    try:
        robot = scene["robot"]
    except (TypeError, KeyError, AttributeError):
        robot = getattr(scene, "robot", None)

    if robot is None or not hasattr(robot, "data"):
        return no_positions

    body_pos_w = getattr(robot.data, "body_pos_w", None)
    if body_pos_w is None:
        return no_positions

    if hasattr(body_pos_w, "torch"):
        pos_tensor = body_pos_w.torch
    else:
        pos_tensor = body_pos_w

    env_origins = getattr(scene, "env_origins", None)
    ox, oy, oz = 0.0, 0.0, 0.0
    if env_origins is not None:
        if hasattr(env_origins, "torch"):
            orig_t = env_origins.torch
        else:
            orig_t = env_origins
        if isinstance(orig_t, torch.Tensor):
            orig_arr = orig_t[env_idx].detach().cpu().numpy().flatten()
        else:
            orig_arr = np.asarray(orig_t)[env_idx].flatten()
        if len(orig_arr) >= 1:
            ox = float(orig_arr[0])
        if len(orig_arr) >= 2:
            oy = float(orig_arr[1])
        if len(orig_arr) >= 3:
            oz = float(orig_arr[2])

    body_names = getattr(robot.data, "body_names", None)
    name_to_id: dict[str, int] = {}
    if body_names is not None:
        name_to_id = {name: i for i, name in enumerate(body_names)}

    positions = []
    for foot_name in foot_names:
        body_id = name_to_id.get(foot_name)
        if body_id is None and hasattr(robot, "find_bodies"):
            try:
                res = robot.find_bodies([foot_name])
            except Exception:
                res = robot.find_bodies(foot_name)
            if isinstance(res, tuple) and len(res) >= 1 and len(res[0]) > 0:
                body_id = res[0][0]
            elif isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], int):
                body_id = res[0]

        if body_id is None:
            positions.append((0.0, 0.0, 0.0))
            continue

        try:
            if isinstance(pos_tensor, torch.Tensor):
                p = pos_tensor[env_idx, body_id].detach().cpu().numpy().flatten()
            else:
                p = np.asarray(pos_tensor)[env_idx, body_id].flatten()
            px = float(p[0] - ox) if len(p) >= 1 else 0.0
            py = float(p[1] - oy) if len(p) >= 2 else 0.0
            pz = float(p[2] - oz) if len(p) >= 3 else 0.0
            positions.append((px, py, pz))
        except (IndexError, TypeError, KeyError):
            positions.append((0.0, 0.0, 0.0))

    return tuple(positions)


def extract_foot_contact_states(
    unwrapped,
    foot_names: Sequence[str] = YUIL_DOG_FOOT_NAMES,
) -> tuple[bool, ...]:
    """Extract foot contact states from the task contact-time sensor.

    This uses the same ``current_contact_time > 0`` condition as the cadence
    reward, so recorded touchdown events match the reward's event definition.

    Args:
        unwrapped: The unwrapped ManagerBasedRLEnv environment.
        foot_names: Foot body names in the desired CSV column order.

    Returns:
        Per-foot contact states in :paramref:`foot_names` order. Missing sensor
        data produces all-false states.
    """
    no_contact = tuple(False for _ in foot_names)
    if unwrapped is None:
        return no_contact

    scene = getattr(unwrapped, "scene", None)
    sensors = getattr(scene, "sensors", None)
    if sensors is None or "contact_forces" not in sensors:
        return no_contact

    contact_sensor = sensors["contact_forces"]
    current_contact_time = getattr(getattr(contact_sensor, "data", None), "current_contact_time", None)
    if current_contact_time is None:
        return no_contact
    contact_time = current_contact_time.torch if hasattr(current_contact_time, "torch") else current_contact_time

    states = []
    for foot_name in foot_names:
        sensor_ids, _ = contact_sensor.find_sensors([foot_name])
        if not sensor_ids:
            states.append(False)
            continue
        states.append(bool(contact_time[0, sensor_ids[0]].item() > 0.0))
    return tuple(states)


def extract_foot_contact_forces(
    unwrapped,
    foot_names: Sequence[str] = YUIL_DOG_FOOT_NAMES,
    env_idx: int = 0,
) -> tuple[tuple[float, float, float, float], ...]:
    """Extract 3D net contact force (fx, fy, fz) [N] and magnitude (norm) [N] for each foot.

    Args:
        unwrapped: The unwrapped ManagerBasedRLEnv environment.
        foot_names: Foot body names in the desired CSV column order.
        env_idx: Index of the environment. Defaults to 0.

    Returns:
        Per-foot (fx, fy, fz, norm) contact forces [N] in :paramref:`foot_names` order.
        Missing or invalid data produces (0.0, 0.0, 0.0, 0.0) tuples.
    """
    no_forces = tuple((0.0, 0.0, 0.0, 0.0) for _ in foot_names)
    if unwrapped is None:
        return no_forces

    scene = getattr(unwrapped, "scene", None)
    sensors = getattr(scene, "sensors", None)
    if sensors is None or "contact_forces" not in sensors:
        return no_forces

    contact_sensor = sensors["contact_forces"]
    net_forces_w = getattr(getattr(contact_sensor, "data", None), "net_forces_w", None)
    if net_forces_w is None:
        return no_forces

    forces_t = net_forces_w.torch if hasattr(net_forces_w, "torch") else net_forces_w

    forces = []
    for foot_name in foot_names:
        sensor_ids, _ = contact_sensor.find_sensors([foot_name])
        if not sensor_ids:
            forces.append((0.0, 0.0, 0.0, 0.0))
            continue
        sid = sensor_ids[0]
        try:
            if isinstance(forces_t, torch.Tensor):
                f = forces_t[env_idx, sid].detach().cpu().numpy().flatten()
            else:
                f = np.asarray(forces_t)[env_idx, sid].flatten()
            fx = float(f[0]) if len(f) >= 1 else 0.0
            fy = float(f[1]) if len(f) >= 2 else 0.0
            fz = float(f[2]) if len(f) >= 3 else 0.0
            fnorm = float(np.sqrt(fx * fx + fy * fy + fz * fz))
            forces.append((fx, fy, fz, fnorm))
        except (IndexError, TypeError, KeyError):
            forces.append((0.0, 0.0, 0.0, 0.0))

    return tuple(forces)


def extract_foot_velocities(
    unwrapped,
    foot_names: Sequence[str] = YUIL_DOG_FOOT_NAMES,
    env_idx: int = 0,
) -> tuple[tuple[float, float, float], ...]:
    """Extract 3D foot velocity (vx, vy, vz) [m/s] in world frame for each foot.

    Args:
        unwrapped: The unwrapped ManagerBasedRLEnv environment.
        foot_names: Foot body names in the desired CSV column order.
        env_idx: Index of the environment. Defaults to 0.

    Returns:
        Per-foot (vx, vy, vz) linear velocities [m/s] in :paramref:`foot_names` order.
        Missing or invalid data produces (0.0, 0.0, 0.0) tuples.
    """
    no_vels = tuple((0.0, 0.0, 0.0) for _ in foot_names)
    if unwrapped is None:
        return no_vels

    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        return no_vels

    robot = None
    try:
        robot = scene["robot"]
    except (TypeError, KeyError, AttributeError):
        robot = getattr(scene, "robot", None)

    if robot is None or not hasattr(robot, "data"):
        return no_vels

    body_lin_vel_w = getattr(robot.data, "body_lin_vel_w", None)
    if body_lin_vel_w is None:
        return no_vels

    vel_tensor = body_lin_vel_w.torch if hasattr(body_lin_vel_w, "torch") else body_lin_vel_w

    body_names = getattr(robot.data, "body_names", None)
    name_to_id: dict[str, int] = {}
    if body_names is not None:
        name_to_id = {name: i for i, name in enumerate(body_names)}

    velocities = []
    for foot_name in foot_names:
        body_id = name_to_id.get(foot_name)
        if body_id is None and hasattr(robot, "find_bodies"):
            try:
                res = robot.find_bodies([foot_name])
            except Exception:
                res = robot.find_bodies(foot_name)
            if isinstance(res, tuple) and len(res) >= 1 and len(res[0]) > 0:
                body_id = res[0][0]
            elif isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], int):
                body_id = res[0]

        if body_id is None:
            velocities.append((0.0, 0.0, 0.0))
            continue

        try:
            if isinstance(vel_tensor, torch.Tensor):
                v = vel_tensor[env_idx, body_id].detach().cpu().numpy().flatten()
            else:
                v = np.asarray(vel_tensor)[env_idx, body_id].flatten()
            vx = float(v[0]) if len(v) >= 1 else 0.0
            vy = float(v[1]) if len(v) >= 2 else 0.0
            vz = float(v[2]) if len(v) >= 3 else 0.0
            velocities.append((vx, vy, vz))
        except (IndexError, TypeError, KeyError):
            velocities.append((0.0, 0.0, 0.0))

    return tuple(velocities)


def extract_current_policy_input(unwrapped, obs: torch.Tensor | np.ndarray | None = None) -> np.ndarray:
    """Extract single-step policy input from the environment's observation manager.

    Args:
        unwrapped: The unwrapped ManagerBasedRLEnv environment.
        obs: Optional raw policy observation tensor.

    Returns:
        1D numpy array containing the most recent single-step policy input terms (e.g. 45 values).
    """
    obs_mgr = getattr(unwrapped, "observation_manager", None)
    if obs_mgr is not None:
        history_buffers = getattr(obs_mgr, "_group_obs_term_history_buffer", {}).get("policy", {})
        term_names = getattr(obs_mgr, "_group_obs_term_names", {}).get("policy", [])
        if term_names and history_buffers:
            parts = []
            for term_name in term_names:
                if term_name in history_buffers:
                    cb = history_buffers[term_name]
                    if getattr(cb, "_buffer", None) is not None:
                        # cb.buffer shape: [batch, max_len, *term_dim]
                        val = cb.buffer[0, -1].detach().cpu().numpy().flatten()
                        parts.append(val)
            if parts:
                return np.concatenate(parts).astype(np.float32)

    # Fallback from obs tensor if history slicing can be deduced
    if obs is not None:
        if isinstance(obs, torch.Tensor):
            obs_arr = obs[0].detach().cpu().numpy().flatten()
        else:
            obs_arr = np.asarray(obs).flatten()

        if len(obs_arr) == 45:
            return obs_arr.astype(np.float32)
        elif len(obs_arr) == 450:
            # 10 steps of [3, 3, 3, 12, 12, 12]
            return np.concatenate([
                obs_arr[27:30],    # base_ang_vel
                obs_arr[57:60],    # projected_gravity
                obs_arr[87:90],    # velocity_commands
                obs_arr[198:210],  # joint_pos
                obs_arr[318:330],  # joint_vel
                obs_arr[438:450],  # actions
            ]).astype(np.float32)
        elif len(obs_arr) == 225:
            # 5 steps of [3, 3, 3, 12, 12, 12]
            return np.concatenate([
                obs_arr[12:15],
                obs_arr[27:30],
                obs_arr[42:45],
                obs_arr[96:108],
                obs_arr[156:168],
                obs_arr[213:225],
            ]).astype(np.float32)

    return np.zeros(45, dtype=np.float32)


def extract_base_link_state(
    unwrapped,
    base_pos: Sequence[float] | torch.Tensor | np.ndarray | None = None,
    base_height: float | torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Extract (base_pos_x, base_pos_y, base_height) in meters.

    Args:
        unwrapped: The unwrapped ManagerBasedRLEnv environment.
        base_pos: Optional explicit base position (x, y) or (x, y, z).
        base_height: Optional explicit base height above ground.

    Returns:
        Tuple of (base_pos_x, base_pos_y, base_height) as floats.
    """
    x, y, z = 0.0, 0.0, 0.0
    h: float | None = None

    if base_pos is not None:
        if isinstance(base_pos, torch.Tensor):
            arr = base_pos.detach().cpu().numpy().flatten()
        else:
            arr = np.asarray(base_pos).flatten()
        if len(arr) >= 2:
            x, y = float(arr[0]), float(arr[1])
        if len(arr) >= 3:
            z = float(arr[2])

    if base_height is not None:
        if isinstance(base_height, torch.Tensor):
            h = float(base_height.detach().cpu().item())
        else:
            h = float(base_height)

    # Extract base position from unwrapped if not explicitly provided
    if base_pos is None and unwrapped is not None:
        scene = getattr(unwrapped, "scene", None)
        if scene is not None:
            robot = None
            try:
                robot = scene["robot"]
            except (TypeError, KeyError, AttributeError):
                robot = getattr(scene, "robot", None)

            if robot is not None and hasattr(robot, "data"):
                root_pos_w = getattr(robot.data, "root_pos_w", None)
                if root_pos_w is not None:
                    if hasattr(root_pos_w, "torch"):
                        pos_t = root_pos_w.torch
                    else:
                        pos_t = root_pos_w
                    if isinstance(pos_t, torch.Tensor):
                        pos_arr = pos_t[0].detach().cpu().numpy().flatten()
                    else:
                        pos_arr = np.asarray(pos_t).flatten()

                    env_origins = getattr(scene, "env_origins", None)
                    if env_origins is not None:
                        if hasattr(env_origins, "torch"):
                            orig_t = env_origins.torch
                        else:
                            orig_t = env_origins
                        if isinstance(orig_t, torch.Tensor):
                            orig_arr = orig_t[0].detach().cpu().numpy().flatten()
                        else:
                            orig_arr = np.asarray(orig_t).flatten()
                        x = float(pos_arr[0] - orig_arr[0])
                        y = float(pos_arr[1] - orig_arr[1])
                    else:
                        x = float(pos_arr[0])
                        y = float(pos_arr[1])
                    if len(pos_arr) >= 3:
                        z = float(pos_arr[2])

    # Extract ground-relative base height if not explicitly provided
    if h is None:
        if unwrapped is not None:
            try:
                from .robotlab_rewards import get_base_bottom_height_from_reward

                reward_aligned_heights = get_base_bottom_height_from_reward(unwrapped)
                if reward_aligned_heights is not None:
                    h = float(reward_aligned_heights[0].detach().cpu().item())
            except Exception:
                pass

        if h is None and unwrapped is not None and hasattr(unwrapped, "scene"):
            try:
                from .rough_mdp import get_ground_relative_base_height

                h = float(get_ground_relative_base_height(unwrapped.scene))
            except Exception:
                pass

        if h is None:
            h = z

    return (x, y, h)


class TeleopDataRecorder:
    """Manages recording of teleoperation steps to CSV with RMS action and joint displacement metrics."""

    def __init__(
        self,
        output_dir: Path | str,
        joint_names: Sequence[str] | None = None,
        record_duration_s: float = 10.0,
        foot_names: Sequence[str] | None = None,
        checkpoint: str | Path | None = None,
    ) -> None:
        """Initialize the teleoperation recorder.

        Args:
            output_dir: Directory where CSV files will be saved.
            joint_names: Names of 12 joints in order. Defaults to YUIL_DOG_JOINT_NAMES.
            record_duration_s: Target recording duration in seconds of simulation time.
            foot_names: Names of four feet in recording order. Defaults to YUIL_DOG_FOOT_NAMES.
            checkpoint: Optional policy checkpoint path. Used for run metadata and file tagging.
        """
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.joint_names = tuple(joint_names) if joint_names is not None else YUIL_DOG_JOINT_NAMES
        self.foot_names = tuple(foot_names) if foot_names is not None else YUIL_DOG_FOOT_NAMES
        self.record_duration_s = float(record_duration_s)
        self._checkpoint: Path | None = Path(checkpoint) if checkpoint is not None else None

        self._is_recording = False
        self._task_name = "teleop"
        self._step_dt = 0.02
        self._target_steps = int(round(self.record_duration_s / self._step_dt))
        self._step_count = 0
        self._rows: list[list[float | int]] = []
        self._start_timestamp = ""
        self._last_saved_file: Path | None = None
        self._last_saved_steps = 0
        self._input_dim = 45
        self._default_joint_pos: np.ndarray | None = None
        self._last_summary_dict: dict = {}
        self._last_summary_text: str = ""

        # Precompute joint indices by anatomical group
        self._hip_idx, self._thigh_idx, self._calf_idx = self._resolve_joint_group_indices()

    @property
    def checkpoint(self) -> Path | None:
        """Policy checkpoint path if configured."""
        return self._checkpoint

    @property
    def checkpoint_run_name(self) -> str:
        """Run folder name of the checkpoint (e.g. '2026-09-16_11-08-38'), or empty string."""
        if self._checkpoint is None:
            return ""
        parent = self._checkpoint.parent.name
        return parent if parent and parent != "." else ""

    @property
    def checkpoint_model_name(self) -> str:
        """Checkpoint filename (e.g. 'model_24995.pt'), or empty string."""
        if self._checkpoint is None:
            return ""
        return self._checkpoint.name

    @property
    def checkpoint_stem(self) -> str:
        """Checkpoint stem (e.g. 'model_24995'), or empty string."""
        if self._checkpoint is None:
            return ""
        return self._checkpoint.stem

    @property
    def checkpoint_tag(self) -> str:
        """Short identifier combining run name and checkpoint stem (e.g. '2026-09-16_11-08-38_model_24995')."""
        run = self.checkpoint_run_name
        stem = self.checkpoint_stem
        if run and stem:
            return f"{run}_{stem}"
        return stem or run

    @property
    def checkpoint_display(self) -> str:
        """Readable display string for checkpoint (e.g. '2026-09-16_11-08-38/model_24995.pt')."""
        if self._checkpoint is None:
            return ""
        run = self.checkpoint_run_name
        name = self.checkpoint_model_name
        if run and name:
            return f"{run}/{name}"
        return name or str(self._checkpoint)

    def _resolve_joint_group_indices(self) -> tuple[list[int], list[int], list[int]]:
        """Resolve joint indices for hip, thigh, and calf groups."""
        hip_idx = [i for i, name in enumerate(self.joint_names) if "hip_roll" in name or "hip_joint" in name]
        thigh_idx = [i for i, name in enumerate(self.joint_names) if "hip_pitch" in name or "thigh" in name]
        calf_idx = [i for i, name in enumerate(self.joint_names) if "knee" in name or "calf" in name]

        # Fallback to leg-major 3-joints-per-leg indexing if names don't match
        if len(hip_idx) != 4 or len(thigh_idx) != 4 or len(calf_idx) != 4:
            hip_idx = [0, 3, 6, 9]
            thigh_idx = [1, 4, 7, 10]
            calf_idx = [2, 5, 8, 11]

        return hip_idx, thigh_idx, calf_idx

    @property
    def is_recording(self) -> bool:
        """Whether recording is currently active."""
        return self._is_recording

    @property
    def last_saved_file(self) -> Path | None:
        """Path of the most recently saved CSV recording."""
        return self._last_saved_file

    def get_headers(self, input_dim: int = 45) -> list[str]:
        """Construct descriptive column headers for the CSV file.

        Args:
            input_dim: Dimension of single-step policy input.

        Returns:
            List of column header names.
        """
        headers = ["time_s", "step"]

        # Input columns (45)
        if input_dim == 45:
            headers.extend(DEFAULT_INPUT_NAMES_45)
        else:
            headers.extend(f"in_{i}" for i in range(input_dim))

        # Output columns (12 raw policy actions)
        headers.extend(f"out_action_{name}" for name in self.joint_names)

        # Converted target joint positions (12 rad)
        headers.extend(f"action_joint_rad_{name}" for name in self.joint_names)

        # Actual joint positions (12 rad)
        headers.extend(f"actual_joint_rad_{name}" for name in self.joint_names)

        # Base link position and height (3)
        headers.extend(DEFAULT_BASE_STATE_NAMES)

        # Contact states use the cadence reward's current_contact_time definition (4)
        headers.extend(f"foot_contact_{name.removesuffix('_foot')}" for name in self.foot_names)

        # Foot positions relative to env origin (12)
        for name in self.foot_names:
            leg = name.removesuffix("_foot")
            headers.extend(f"foot_pos_{leg}_{axis}" for axis in DEFAULT_FOOT_POS_AXES)

        # Foot contact forces [N] (fx, fy, fz, norm for each foot = 16)
        for name in self.foot_names:
            leg = name.removesuffix("_foot")
            headers.extend(f"foot_force_{leg}_{axis}" for axis in DEFAULT_FOOT_FORCE_AXES)
            headers.append(f"foot_force_{leg}_norm")

        # Foot vertical impact velocity [m/s] (vz for each foot = 4)
        for name in self.foot_names:
            leg = name.removesuffix("_foot")
            headers.append(f"foot_vel_{leg}_z")

        # Joint displacement from default pose (Δq = q_actual - q_default) [rad] (12)
        headers.extend(f"delta_q_rad_{name}" for name in self.joint_names)

        # Step-wise RMS Action by group (3)
        headers.extend(["rms_action_hip", "rms_action_thigh", "rms_action_calf"])

        # Step-wise RMS Joint Displacement by group [rad] (3)
        headers.extend(["rms_delta_q_hip_rad", "rms_delta_q_thigh_rad", "rms_delta_q_calf_rad"])

        # Step-wise RMS Joint Displacement by group [deg] (3)
        headers.extend(["rms_delta_q_hip_deg", "rms_delta_q_thigh_deg", "rms_delta_q_calf_deg"])

        return headers

    def start(
        self,
        task_name: str,
        step_dt: float,
        checkpoint: str | Path | None = None,
    ) -> None:
        """Start a new recording session.

        Args:
            task_name: Task or robot identifier used for filename.
            step_dt: Simulation step time interval [s].
            checkpoint: Optional policy checkpoint path. Overrides instance checkpoint if given.
        """
        if checkpoint is not None:
            self._checkpoint = Path(checkpoint)
        self._task_name = task_name.replace(":", "_").replace("/", "_")
        self._step_dt = max(float(step_dt), 1e-4)
        self._target_steps = int(round(self.record_duration_s / self._step_dt))
        self._step_count = 0
        self._rows.clear()
        self._start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._is_recording = True
        self._default_joint_pos = None
        self._last_summary_dict = {}
        self._last_summary_text = ""

    def record_step(
        self,
        unwrapped,
        obs: torch.Tensor | np.ndarray | None,
        actions: torch.Tensor | np.ndarray,
        target_joint_pos_rad: torch.Tensor | np.ndarray | Sequence[float],
        actual_joint_pos_rad: torch.Tensor | np.ndarray | Sequence[float],
        base_pos: Sequence[float] | torch.Tensor | np.ndarray | None = None,
        base_height: float | torch.Tensor | None = None,
        foot_contacts: Sequence[bool | float] | torch.Tensor | np.ndarray | None = None,
        foot_positions: Sequence[Sequence[float]] | Sequence[float] | torch.Tensor | np.ndarray | None = None,
        foot_forces: Sequence[Sequence[float]] | Sequence[float] | torch.Tensor | np.ndarray | None = None,
        foot_velocities: Sequence[Sequence[float]] | Sequence[float] | torch.Tensor | np.ndarray | None = None,
    ) -> bool:
        """Record one control step.

        Args:
            unwrapped: The unwrapped environment.
            obs: The policy observation tensor.
            actions: The raw 12-dim action output from policy.
            target_joint_pos_rad: The 12-dim target joint position in radians (converted from action).
            actual_joint_pos_rad: The 12-dim actual physical joint position in radians.
            base_pos: Optional explicit base position (x, y) or (x, y, z) [m].
            base_height: Optional explicit base height above ground [m].
            foot_contacts: Optional per-foot contact states in :attr:`foot_names` order.
            foot_positions: Optional per-foot 3D positions (x, y, z) in :attr:`foot_names` order [m].
            foot_forces: Optional per-foot contact forces (fx, fy, fz, norm) or (fx, fy, fz) [N].
            foot_velocities: Optional per-foot velocities (vx, vy, vz) or vertical velocity vz [m/s].

        Returns:
            True if recording is complete and file was saved, False otherwise.
        """
        if not self._is_recording:
            return False

        elapsed_time = round(self._step_count * self._step_dt, 4)

        # 1. Input 45
        in_values = extract_current_policy_input(unwrapped, obs)
        self._input_dim = len(in_values)

        # 2. Output 12
        if isinstance(actions, torch.Tensor):
            out_act = actions.detach().cpu().numpy()
        else:
            out_act = np.asarray(actions)
        if out_act.ndim == 2:
            out_act = out_act[0]
        out_act = out_act.flatten()
        if len(out_act) > 12:
            out_act = out_act[:12]

        # 3. Target joint rad 12
        if isinstance(target_joint_pos_rad, torch.Tensor):
            tgt_rad = target_joint_pos_rad.detach().cpu().numpy()
        else:
            tgt_rad = np.asarray(target_joint_pos_rad)
        if tgt_rad.ndim == 2:
            tgt_rad = tgt_rad[0]
        tgt_rad = tgt_rad.flatten()
        if len(tgt_rad) > 12:
            tgt_rad = tgt_rad[:12]

        # 4. Actual joint rad 12
        if isinstance(actual_joint_pos_rad, torch.Tensor):
            act_rad = actual_joint_pos_rad.detach().cpu().numpy()
        else:
            act_rad = np.asarray(actual_joint_pos_rad)
        if act_rad.ndim == 2:
            act_rad = act_rad[0]
        act_rad = act_rad.flatten()
        if len(act_rad) > 12:
            act_rad = act_rad[:12]

        # 5. Base link x, y, and base_height
        base_x, base_y, base_h = extract_base_link_state(
            unwrapped=unwrapped,
            base_pos=base_pos,
            base_height=base_height,
        )

        if foot_contacts is None:
            contact_states = extract_foot_contact_states(unwrapped, self.foot_names)
        elif isinstance(foot_contacts, torch.Tensor):
            contact_states = foot_contacts.detach().cpu().numpy().flatten()
        else:
            contact_states = np.asarray(foot_contacts).flatten()
        if len(contact_states) != len(self.foot_names):
            raise ValueError(
                f"Expected {len(self.foot_names)} foot contact states, got {len(contact_states)}."
            )

        # 6. Foot positions (4 feet x 3 xyz = 12)
        if foot_positions is None:
            pos_tuples = extract_foot_positions(unwrapped, self.foot_names)
            flat_pos = [coord for pt in pos_tuples for coord in pt]
        elif isinstance(foot_positions, torch.Tensor):
            flat_pos = [float(v) for v in foot_positions.detach().cpu().numpy().flatten()]
        else:
            flat_pos = [float(v) for v in np.asarray(foot_positions).flatten()]

        expected_coords = len(self.foot_names) * len(DEFAULT_FOOT_POS_AXES)
        if len(flat_pos) != expected_coords:
            raise ValueError(
                f"Expected {expected_coords} foot position coordinates, got {len(flat_pos)}."
            )

        # 7. Foot contact forces (4 feet x 4 (fx, fy, fz, norm) = 16)
        if foot_forces is None:
            force_tuples = extract_foot_contact_forces(unwrapped, self.foot_names)
            flat_forces = [float(val) for quad in force_tuples for val in quad]
        elif isinstance(foot_forces, torch.Tensor):
            flat_forces = [float(v) for v in foot_forces.detach().cpu().numpy().flatten()]
        else:
            flat_forces = [float(v) for v in np.asarray(foot_forces).flatten()]

        expected_forces = len(self.foot_names) * (len(DEFAULT_FOOT_FORCE_AXES) + 1)
        if len(flat_forces) == len(self.foot_names) * len(DEFAULT_FOOT_FORCE_AXES):
            reconstructed_forces = []
            for i in range(len(self.foot_names)):
                fx, fy, fz = flat_forces[i * 3 : (i + 1) * 3]
                fnorm = float(np.sqrt(fx * fx + fy * fy + fz * fz))
                reconstructed_forces.extend([fx, fy, fz, fnorm])
            flat_forces = reconstructed_forces

        if len(flat_forces) != expected_forces:
            raise ValueError(
                f"Expected {expected_forces} foot force values, got {len(flat_forces)}."
            )

        # 8. Foot vertical impact velocities (4 feet x 1 (vz) = 4)
        if foot_velocities is None:
            vel_tuples = extract_foot_velocities(unwrapped, self.foot_names)
            flat_vels = [float(triple[2]) for triple in vel_tuples]
        elif isinstance(foot_velocities, torch.Tensor):
            arr_vel = foot_velocities.detach().cpu().numpy()
            if arr_vel.shape[-1] == 3 and arr_vel.size == len(self.foot_names) * 3:
                flat_vels = [float(arr_vel.reshape(-1, 3)[i, 2]) for i in range(len(self.foot_names))]
            else:
                flat_vels = [float(v) for v in arr_vel.flatten()]
        else:
            arr_vel = np.asarray(foot_velocities)
            if arr_vel.shape[-1] == 3 and arr_vel.size == len(self.foot_names) * 3:
                flat_vels = [float(arr_vel.reshape(-1, 3)[i, 2]) for i in range(len(self.foot_names))]
            else:
                flat_vels = [float(v) for v in arr_vel.flatten()]

        expected_vels = len(self.foot_names)
        if len(flat_vels) != expected_vels:
            raise ValueError(
                f"Expected {expected_vels} foot vertical velocity values, got {len(flat_vels)}."
            )

        # 9. Joint displacement metrics (Δq = q_actual - q_default)
        if self._default_joint_pos is None:
            self._default_joint_pos = extract_default_joint_pos(unwrapped, self.joint_names)

        delta_q = act_rad - self._default_joint_pos

        # Step-wise RMS Action by group
        a_hip = out_act[self._hip_idx]
        a_thigh = out_act[self._thigh_idx]
        a_calf = out_act[self._calf_idx]
        step_rms_a_hip = float(np.sqrt(np.mean(a_hip ** 2)))
        step_rms_a_thigh = float(np.sqrt(np.mean(a_thigh ** 2)))
        step_rms_a_calf = float(np.sqrt(np.mean(a_calf ** 2)))

        # Step-wise RMS Joint Displacement by group [rad]
        dq_hip = delta_q[self._hip_idx]
        dq_thigh = delta_q[self._thigh_idx]
        dq_calf = delta_q[self._calf_idx]
        step_rms_dq_hip_rad = float(np.sqrt(np.mean(dq_hip ** 2)))
        step_rms_dq_thigh_rad = float(np.sqrt(np.mean(dq_thigh ** 2)))
        step_rms_dq_calf_rad = float(np.sqrt(np.mean(dq_calf ** 2)))

        # Step-wise RMS Joint Displacement by group [deg]
        step_rms_dq_hip_deg = float(np.degrees(step_rms_dq_hip_rad))
        step_rms_dq_thigh_deg = float(np.degrees(step_rms_dq_thigh_rad))
        step_rms_dq_calf_deg = float(np.degrees(step_rms_dq_calf_rad))

        # Build row
        row = [elapsed_time, self._step_count]
        row.extend(float(v) for v in in_values)
        row.extend(float(v) for v in out_act)
        row.extend(float(v) for v in tgt_rad)
        row.extend(float(v) for v in act_rad)
        row.append(float(base_x))
        row.append(float(base_y))
        row.append(float(base_h))
        row.extend(int(bool(value)) for value in contact_states)
        row.extend(float(v) for v in flat_pos)
        row.extend(float(v) for v in flat_forces)
        row.extend(float(v) for v in flat_vels)
        # Added metrics
        row.extend(float(v) for v in delta_q)
        row.extend([step_rms_a_hip, step_rms_a_thigh, step_rms_a_calf])
        row.extend([step_rms_dq_hip_rad, step_rms_dq_thigh_rad, step_rms_dq_calf_rad])
        row.extend([step_rms_dq_hip_deg, step_rms_dq_thigh_deg, step_rms_dq_calf_deg])

        self._rows.append(row)
        self._step_count += 1

        if self._step_count >= self._target_steps:
            self.save()
            self._is_recording = False
            return True

        return False

    def compute_summary_metrics(self) -> dict[str, float | dict[str, float]]:
        """Compute aggregate RMS(a) and Δq summary statistics over the recorded session.

        Returns:
            Dictionary with overall and per-group RMS actions, displacements, and tracking errors.
        """
        if not self._rows:
            return {}

        data_matrix = np.array(self._rows, dtype=np.float64)
        headers = self.get_headers(input_dim=self._input_dim)
        col_map = {name: i for i, name in enumerate(headers)}

        # Extract matrices: (T, 12)
        out_act_cols = [col_map[f"out_action_{name}"] for name in self.joint_names]
        tgt_rad_cols = [col_map[f"action_joint_rad_{name}"] for name in self.joint_names]
        act_rad_cols = [col_map[f"actual_joint_rad_{name}"] for name in self.joint_names]
        delta_q_cols = [col_map[f"delta_q_rad_{name}"] for name in self.joint_names]

        actions_mat = data_matrix[:, out_act_cols]
        targets_mat = data_matrix[:, tgt_rad_cols]
        actuals_mat = data_matrix[:, act_rad_cols]
        delta_q_mat = data_matrix[:, delta_q_cols]
        error_mat = targets_mat - actuals_mat

        # Group actions
        a_hip = actions_mat[:, self._hip_idx]
        a_thigh = actions_mat[:, self._thigh_idx]
        a_calf = actions_mat[:, self._calf_idx]

        # Group displacements (rad)
        dq_hip = delta_q_mat[:, self._hip_idx]
        dq_thigh = delta_q_mat[:, self._thigh_idx]
        dq_calf = delta_q_mat[:, self._calf_idx]

        # Group tracking errors (rad)
        err_hip = error_mat[:, self._hip_idx]
        err_thigh = error_mat[:, self._thigh_idx]
        err_calf = error_mat[:, self._calf_idx]

        metrics: dict[str, float | dict[str, float]] = {
            # 1. RMS Actions
            "rms_a_hip": float(np.sqrt(np.mean(a_hip ** 2))),
            "rms_a_thigh": float(np.sqrt(np.mean(a_thigh ** 2))),
            "rms_a_calf": float(np.sqrt(np.mean(a_calf ** 2))),
            "rms_a_total": float(np.sqrt(np.mean(actions_mat ** 2))),
            # 2. RMS Displacement Δq = q_actual - q_default [rad]
            "rms_dq_hip_rad": float(np.sqrt(np.mean(dq_hip ** 2))),
            "rms_dq_thigh_rad": float(np.sqrt(np.mean(dq_thigh ** 2))),
            "rms_dq_calf_rad": float(np.sqrt(np.mean(dq_calf ** 2))),
            # 3. RMS Displacement Δq [deg]
            "rms_dq_hip_deg": float(np.degrees(np.sqrt(np.mean(dq_hip ** 2)))),
            "rms_dq_thigh_deg": float(np.degrees(np.sqrt(np.mean(dq_thigh ** 2)))),
            "rms_dq_calf_deg": float(np.degrees(np.sqrt(np.mean(dq_calf ** 2)))),
            # 4. Max Absolute Displacement |Δq| [deg]
            "max_abs_dq_hip_deg": float(np.degrees(np.max(np.abs(dq_hip)))),
            "max_abs_dq_thigh_deg": float(np.degrees(np.max(np.abs(dq_thigh)))),
            "max_abs_dq_calf_deg": float(np.degrees(np.max(np.abs(dq_calf)))),
            # 5. Mean Absolute Displacement |Δq| [deg]
            "mean_abs_dq_hip_deg": float(np.degrees(np.mean(np.abs(dq_hip)))),
            "mean_abs_dq_thigh_deg": float(np.degrees(np.mean(np.abs(dq_thigh)))),
            "mean_abs_dq_calf_deg": float(np.degrees(np.mean(np.abs(dq_calf)))),
            # 6. Target Action Displacement (Δq_target = 0.25 * a) [rad & deg]
            "rms_dq_tgt_hip_deg": float(np.degrees(0.25 * np.sqrt(np.mean(a_hip ** 2)))),
            "rms_dq_tgt_thigh_deg": float(np.degrees(0.25 * np.sqrt(np.mean(a_thigh ** 2)))),
            "rms_dq_tgt_calf_deg": float(np.degrees(0.25 * np.sqrt(np.mean(a_calf ** 2)))),
            # 7. Tracking Error e = q_target - q_actual [deg]
            "rms_err_hip_deg": float(np.degrees(np.sqrt(np.mean(err_hip ** 2)))),
            "rms_err_thigh_deg": float(np.degrees(np.sqrt(np.mean(err_thigh ** 2)))),
            "rms_err_calf_deg": float(np.degrees(np.sqrt(np.mean(err_calf ** 2)))),
        }

        # Per-leg breakdowns
        per_leg = {}
        for leg in ["FL", "FR", "RL", "RR"]:
            leg_hip_idx = [i for i, name in enumerate(self.joint_names) if name.startswith(leg) and ("hip_roll" in name or "hip_joint" in name)]
            leg_thigh_idx = [i for i, name in enumerate(self.joint_names) if name.startswith(leg) and ("hip_pitch" in name or "thigh" in name)]
            leg_calf_idx = [i for i, name in enumerate(self.joint_names) if name.startswith(leg) and ("knee" in name or "calf" in name)]

            a_h = actions_mat[:, leg_hip_idx[0]] if leg_hip_idx else np.array([0.0])
            a_t = actions_mat[:, leg_thigh_idx[0]] if leg_thigh_idx else np.array([0.0])
            a_c = actions_mat[:, leg_calf_idx[0]] if leg_calf_idx else np.array([0.0])

            dq_h = delta_q_mat[:, leg_hip_idx[0]] if leg_hip_idx else np.array([0.0])
            dq_t = delta_q_mat[:, leg_thigh_idx[0]] if leg_thigh_idx else np.array([0.0])
            dq_c = delta_q_mat[:, leg_calf_idx[0]] if leg_calf_idx else np.array([0.0])

            per_leg[f"{leg}_rms_a_hip"] = float(np.sqrt(np.mean(a_h ** 2)))
            per_leg[f"{leg}_rms_a_thigh"] = float(np.sqrt(np.mean(a_t ** 2)))
            per_leg[f"{leg}_rms_a_calf"] = float(np.sqrt(np.mean(a_c ** 2)))

            per_leg[f"{leg}_rms_dq_hip_deg"] = float(np.degrees(np.sqrt(np.mean(dq_h ** 2))))
            per_leg[f"{leg}_rms_dq_thigh_deg"] = float(np.degrees(np.sqrt(np.mean(dq_t ** 2))))
            per_leg[f"{leg}_rms_dq_calf_deg"] = float(np.degrees(np.sqrt(np.mean(dq_c ** 2))))

        metrics["per_leg"] = per_leg
        metrics["task_name"] = self._task_name
        metrics["checkpoint"] = str(self._checkpoint) if self._checkpoint else ""
        metrics["checkpoint_run"] = self.checkpoint_run_name
        metrics["checkpoint_model"] = self.checkpoint_model_name
        self._last_summary_dict = metrics
        return metrics

    def format_summary_metrics(self) -> str:
        """Format the summary metrics into a readable report for console and file logging."""
        m = self.compute_summary_metrics()
        if not m:
            return "[REC] No recorded data to summarize."

        pl = m.get("per_leg", {})
        duration = self._step_count * self._step_dt

        lines = [
            "",
            "=" * 72,
            f" [REC] Teleoperation Recording Metric Summary ({duration:.2f}s, {self._step_count} steps)",
        ]
        if self._task_name:
            lines.append(f" * Task: {self._task_name}")
        if self._checkpoint:
            lines.append(f" * Policy Checkpoint: {self.checkpoint_display}")
            if self.checkpoint_run_name:
                lines.append(f"   - Run Directory   : {self.checkpoint_run_name}")
            if self.checkpoint_model_name:
                lines.append(f"   - Model Checkpoint: {self.checkpoint_model_name}")
            lines.append(f"   - Checkpoint Path : {self._checkpoint}")
        lines.extend([
            "=" * 72,
            "",
            "1. Policy Action Root-Mean-Square (RMS):",
            f"   * RMS(a_hip)   = {m['rms_a_hip']:.4f}",
            f"   * RMS(a_thigh) = {m['rms_a_thigh']:.4f}",
            f"   * RMS(a_calf)  = {m['rms_a_calf']:.4f}",
            f"   * Overall Action RMS = {m['rms_a_total']:.4f}",
            "   [Per-Leg Action RMS Breakdown]",
            f"     FL: hip={pl.get('FL_rms_a_hip', 0.0):.3f}, thigh={pl.get('FL_rms_a_thigh', 0.0):.3f}, calf={pl.get('FL_rms_a_calf', 0.0):.3f}",
            f"     FR: hip={pl.get('FR_rms_a_hip', 0.0):.3f}, thigh={pl.get('FR_rms_a_thigh', 0.0):.3f}, calf={pl.get('FR_rms_a_calf', 0.0):.3f}",
            f"     RL: hip={pl.get('RL_rms_a_hip', 0.0):.3f}, thigh={pl.get('RL_rms_a_thigh', 0.0):.3f}, calf={pl.get('RL_rms_a_calf', 0.0):.3f}",
            f"     RR: hip={pl.get('RR_rms_a_hip', 0.0):.3f}, thigh={pl.get('RR_rms_a_thigh', 0.0):.3f}, calf={pl.get('RR_rms_a_calf', 0.0):.3f}",
            "",
            "2. Joint Displacement from Default Pose (Δq = q_actual - q_default):",
            f"   * RMS(Δq_hip)   = {m['rms_dq_hip_rad']:.4f} rad ({m['rms_dq_hip_deg']:6.2f} deg)  | Max: {m['max_abs_dq_hip_deg']:5.2f} deg, Mean: {m['mean_abs_dq_hip_deg']:5.2f} deg",
            f"   * RMS(Δq_thigh) = {m['rms_dq_thigh_rad']:.4f} rad ({m['rms_dq_thigh_deg']:6.2f} deg)  | Max: {m['max_abs_dq_thigh_deg']:5.2f} deg, Mean: {m['mean_abs_dq_thigh_deg']:5.2f} deg",
            f"   * RMS(Δq_calf)  = {m['rms_dq_calf_rad']:.4f} rad ({m['rms_dq_calf_deg']:6.2f} deg)  | Max: {m['max_abs_dq_calf_deg']:5.2f} deg, Mean: {m['mean_abs_dq_calf_deg']:5.2f} deg",
            "   [Per-Leg Δq RMS Breakdown (deg)]",
            f"     FL: hip={pl.get('FL_rms_dq_hip_deg', 0.0):5.2f} deg, thigh={pl.get('FL_rms_dq_thigh_deg', 0.0):5.2f} deg, calf={pl.get('FL_rms_dq_calf_deg', 0.0):5.2f} deg",
            f"     FR: hip={pl.get('FR_rms_dq_hip_deg', 0.0):5.2f} deg, thigh={pl.get('FR_rms_dq_thigh_deg', 0.0):5.2f} deg, calf={pl.get('FR_rms_dq_calf_deg', 0.0):5.2f} deg",
            f"     RL: hip={pl.get('RL_rms_dq_hip_deg', 0.0):5.2f} deg, thigh={pl.get('RL_rms_dq_thigh_deg', 0.0):5.2f} deg, calf={pl.get('RL_rms_dq_calf_deg', 0.0):5.2f} deg",
            f"     RR: hip={pl.get('RR_rms_dq_hip_deg', 0.0):5.2f} deg, thigh={pl.get('RR_rms_dq_thigh_deg', 0.0):5.2f} deg, calf={pl.get('RR_rms_dq_calf_deg', 0.0):5.2f} deg",
            "",
            "3. Policy Target Displacement (Δq_target = 0.25 * a):",
            f"   * RMS(Δq_tgt_hip)   = {m['rms_dq_tgt_hip_deg']:6.2f} deg",
            f"   * RMS(Δq_tgt_thigh) = {m['rms_dq_tgt_thigh_deg']:6.2f} deg",
            f"   * RMS(Δq_tgt_calf)  = {m['rms_dq_tgt_calf_deg']:6.2f} deg",
            "",
            "4. Joint Tracking Error (e = q_target - q_actual):",
            f"   * RMS(e_hip)   = {m['rms_err_hip_deg']:6.2f} deg",
            f"   * RMS(e_thigh) = {m['rms_err_thigh_deg']:6.2f} deg",
            f"   * RMS(e_calf)  = {m['rms_err_calf_deg']:6.2f} deg",
            "=" * 72,
            "",
        ])
        self._last_summary_text = "\n".join(lines)
        return self._last_summary_text

    def print_summary_metrics(self) -> None:
        """Print the formatted summary metrics to stdout."""
        print(self.format_summary_metrics())

    def save(self) -> Path | None:
        """Save collected data to CSV file and save accompanying summary text file.

        Returns:
            Path to saved CSV file, or None if no data to save.
        """
        if not self._rows:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tag = self.checkpoint_tag
        if tag:
            filename = f"teleop_record_{self._task_name}_{tag}_{self._start_timestamp}.csv"
        else:
            filename = f"teleop_record_{self._task_name}_{self._start_timestamp}.csv"
        file_path = self.output_dir / filename

        headers = self.get_headers(input_dim=self._input_dim)

        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self._rows)

        self._last_saved_file = file_path
        self._last_saved_steps = len(self._rows)

        # Save accompanying summary text file
        summary_text = self.format_summary_metrics()
        summary_path = file_path.with_suffix(".summary.txt")
        try:
            with open(summary_path, mode="w", encoding="utf-8") as f:
                f.write(summary_text)
        except Exception:
            pass

        return file_path

    def get_status_str(self) -> str:
        """Get formatted status string for display in teleop UI.

        Returns:
            Human readable status string.
        """
        if self._is_recording:
            curr_s = self._step_count * self._step_dt
            return (
                f"ACTIVE [ {curr_s:5.2f}s / {self.record_duration_s:5.2f}s | "
                f"{self._step_count:3d} / {self._target_steps:3d} steps ]"
            )
        if self._last_saved_file is not None:
            return f"COMPLETED -> {self._last_saved_file.name} ({self._last_saved_steps} steps)"
        return "READY (Press [R] or [C] to record 10s)"
