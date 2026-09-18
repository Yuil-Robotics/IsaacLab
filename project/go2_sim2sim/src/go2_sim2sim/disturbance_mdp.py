# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful disturbance events for recovery-oriented locomotion training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


_RECOVERY_TIME_LEFT_ATTR = "_robotlab_recovery_time_left"
_RECOVERY_EVENT_ID_ATTR = "_robotlab_recovery_event_id"


def _recovery_buffers(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lazily allocated recovery-window state."""
    time_left = getattr(env, _RECOVERY_TIME_LEFT_ATTR, None)
    event_id = getattr(env, _RECOVERY_EVENT_ID_ATTR, None)
    if time_left is None or time_left.shape != (env.num_envs,):
        time_left = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _RECOVERY_TIME_LEFT_ATTR, time_left)
    if event_id is None or event_id.shape != (env.num_envs,):
        event_id = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
        setattr(env, _RECOVERY_EVENT_ID_ATTR, event_id)
    return time_left, event_id


def get_recovery_window(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return recovery time remaining [s] and disturbance sequence identifiers.

    Args:
        env: Manager-based environment.

    Returns:
        A tuple containing per-environment recovery time remaining [s] and
        monotonically increasing disturbance identifiers.
    """
    return _recovery_buffers(env)


def mark_recovery_window(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    duration_s: float | torch.Tensor,
) -> None:
    """Open one finite recovery window for selected environments.

    Args:
        env: Manager-based environment.
        env_ids: Environment indices receiving a disturbance.
        duration_s: Recovery-window duration [s]. A scalar or one value per
            selected environment may be supplied.
    """
    if isinstance(duration_s, (int, float)) and duration_s <= 0.0:
        raise ValueError(f"Expected duration_s > 0, got {duration_s}.")
    time_left, event_id = _recovery_buffers(env)
    duration = torch.as_tensor(duration_s, device=env.device, dtype=time_left.dtype)
    if torch.any(duration <= 0.0):
        raise ValueError("Expected every recovery-window duration to be positive.")
    time_left[env_ids] = torch.maximum(time_left[env_ids], duration)
    event_id[env_ids] += 1


def push_by_setting_velocity_with_recovery_marker(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    recovery_window_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply a velocity impulse and open a finite recovery window.

    Args:
        env: Manager-based environment.
        env_ids: Environment indices receiving the impulse.
        velocity_range: Root linear velocity [m/s] and angular velocity
            [rad/s] sampling ranges.
        recovery_window_s: Time after the impulse classified as recovery [s].
        asset_cfg: Disturbed articulation configuration.
    """
    mdp.push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    mark_recovery_window(env, env_ids, recovery_window_s)


class TimedWrenchDisturbance(ManagerTermBase):
    """Apply finite-duration body-frame wrench pulses with a strength curriculum.

    Unlike :func:`isaaclab.envs.mdp.apply_external_force_torque`, this term
    explicitly clears every pulse after its sampled duration. The event manager
    calls the term once per policy step; internal per-environment timers decide
    when a pulse starts and stops.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize per-environment pulse timers and wrench buffers."""
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self._body_ids = asset_cfg.body_ids
        self._all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        self._time_to_next = torch.zeros(env.num_envs, device=env.device)
        self._pulse_time_left = torch.zeros(env.num_envs, device=env.device)
        self._force_b = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self._torque_b = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self._resample_interval(self._all_env_ids, cfg.params["pulse_interval_range_s"])

    @property
    def active_mask(self) -> torch.Tensor:
        """Whether a wrench pulse is active, shape ``(num_envs,)``."""
        return self._pulse_time_left > 0.0

    @property
    def force_b(self) -> torch.Tensor:
        """Current base force in the body frame [N], shape ``(num_envs, 3)``."""
        return self._force_b[:, 0]

    @property
    def torque_b(self) -> torch.Tensor:
        """Current base torque in the body frame [N·m], shape ``(num_envs, 3)``."""
        return self._torque_b[:, 0]

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        """Convert environment selection to a device tensor."""
        if env_ids is None:
            return self._all_env_ids
        if isinstance(env_ids, slice):
            return self._all_env_ids[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.int32)

    def _resample_interval(self, env_ids: torch.Tensor, interval_range_s: tuple[float, float]) -> None:
        """Sample time until the next pulse [s]."""
        lower, upper = interval_range_s
        self._time_to_next[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(lower, upper)

    def _set_wrench(self, env_ids: torch.Tensor, forces: torch.Tensor, torques: torch.Tensor) -> None:
        """Replace the permanent wrench for selected environments."""
        self.asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            body_ids=self._body_ids,
            env_ids=env_ids,
            is_global=False,
        )

    def _clear_wrench(self, env_ids: torch.Tensor) -> None:
        """Clear pulse state and the simulator wrench buffers."""
        if len(env_ids) == 0:
            return
        self._force_b[env_ids] = 0.0
        self._torque_b[env_ids] = 0.0
        self._pulse_time_left[env_ids] = 0.0
        zeros = torch.zeros((len(env_ids), 1, 3), device=self.device)
        self._set_wrench(env_ids, zeros, zeros)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear active pulses and resample their first trigger time."""
        resolved_ids = self._resolve_env_ids(env_ids)
        self._clear_wrench(resolved_ids)
        recovery_time_left, _ = _recovery_buffers(self._env)
        recovery_time_left[resolved_ids] = 0.0
        self._resample_interval(resolved_ids, self.cfg.params["pulse_interval_range_s"])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        pulse_interval_range_s: tuple[float, float],
        pulse_duration_range_s: tuple[float, float],
        horizontal_force_range_n: tuple[float, float],
        vertical_force_range_n: tuple[float, float],
        roll_pitch_torque_range_nm: tuple[float, float],
        yaw_torque_fraction: float,
        recovery_settle_time_s: float,
        curriculum_start_scale: float,
        curriculum_steps: int,
    ) -> None:
        """Advance pulse timers and apply or clear scheduled disturbances.

        Args:
            env: Manager-based environment.
            env_ids: Environments scheduled by the interval event manager.
            asset_cfg: Base rigid-body selection.
            pulse_interval_range_s: Time between pulse starts [s].
            pulse_duration_range_s: Wrench pulse duration [s].
            horizontal_force_range_n: Horizontal force-magnitude range [N].
            vertical_force_range_n: Vertical force-component range [N].
            roll_pitch_torque_range_nm: Roll/pitch torque-magnitude range [N·m].
            yaw_torque_fraction: Maximum yaw torque as a fraction of roll/pitch torque.
            recovery_settle_time_s: Recovery time retained after the pulse ends [s].
            curriculum_start_scale: Initial fraction of the configured wrench strength.
            curriculum_steps: Policy steps over which wrench strength reaches full scale.
        """
        del asset_cfg
        resolved_ids = self._resolve_env_ids(env_ids)
        if len(resolved_ids) == 0:
            return
        if not 0.0 < curriculum_start_scale <= 1.0:
            raise ValueError(f"Expected curriculum_start_scale in (0, 1], got {curriculum_start_scale}.")
        if curriculum_steps < 1:
            raise ValueError(f"Expected curriculum_steps >= 1, got {curriculum_steps}.")
        if recovery_settle_time_s <= 0.0:
            raise ValueError(f"Expected recovery_settle_time_s > 0, got {recovery_settle_time_s}.")

        dt = env.step_dt
        recovery_time_left, _ = _recovery_buffers(env)
        recovery_time_left[resolved_ids] = torch.clamp(recovery_time_left[resolved_ids] - dt, min=0.0)
        self._time_to_next[resolved_ids] -= dt
        was_active = self._pulse_time_left[resolved_ids] > 0.0
        self._pulse_time_left[resolved_ids] = torch.clamp(self._pulse_time_left[resolved_ids] - dt, min=0.0)
        ended_local = was_active & (self._pulse_time_left[resolved_ids] <= 0.0)
        if torch.any(ended_local):
            self._clear_wrench(resolved_ids[ended_local])

        start_local = (self._time_to_next[resolved_ids] <= 0.0) & ~self.active_mask[resolved_ids]
        start_ids = resolved_ids[start_local]
        if len(start_ids) == 0:
            return

        progress = min(float(env.common_step_counter) / float(curriculum_steps), 1.0)
        strength_scale = curriculum_start_scale + progress * (1.0 - curriculum_start_scale)
        count = len(start_ids)

        force_min, force_max = horizontal_force_range_n
        force_magnitude = torch.empty(count, device=self.device).uniform_(force_min, force_max) * strength_scale
        force_angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        self._force_b[start_ids, 0, 0] = force_magnitude * torch.cos(force_angle)
        self._force_b[start_ids, 0, 1] = force_magnitude * torch.sin(force_angle)
        vertical_min, vertical_max = vertical_force_range_n
        self._force_b[start_ids, 0, 2] = (
            torch.empty(count, device=self.device).uniform_(vertical_min, vertical_max) * strength_scale
        )

        torque_min, torque_max = roll_pitch_torque_range_nm
        torque_magnitude = torch.empty(count, device=self.device).uniform_(torque_min, torque_max) * strength_scale
        torque_angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        self._torque_b[start_ids, 0, 0] = torque_magnitude * torch.cos(torque_angle)
        self._torque_b[start_ids, 0, 1] = torque_magnitude * torch.sin(torque_angle)
        self._torque_b[start_ids, 0, 2] = (
            torch.empty(count, device=self.device).uniform_(-yaw_torque_fraction, yaw_torque_fraction)
            * torque_magnitude
        )

        duration_min, duration_max = pulse_duration_range_s
        self._pulse_time_left[start_ids] = torch.empty(count, device=self.device).uniform_(duration_min, duration_max)
        mark_recovery_window(
            env,
            start_ids,
            self._pulse_time_left[start_ids] + recovery_settle_time_s,
        )
        self._resample_interval(start_ids, pulse_interval_range_s)
        self._set_wrench(start_ids, self._force_b[start_ids], self._torque_b[start_ids])

        log = env.extras.setdefault("log", {})
        log["Disturbance/wrench_strength_scale"] = torch.tensor(strength_scale, device=self.device)
        log["Disturbance/active_wrench_ratio"] = torch.mean(self.active_mask.to(torch.float32))
        log["Disturbance/force_mean_n"] = torch.mean(torch.linalg.norm(self.force_b, dim=1))
        log["Disturbance/torque_mean_nm"] = torch.mean(torch.linalg.norm(self.torque_b, dim=1))
        log["Disturbance/recovery_window_ratio"] = torch.mean((recovery_time_left > 0.0).to(torch.float32))
