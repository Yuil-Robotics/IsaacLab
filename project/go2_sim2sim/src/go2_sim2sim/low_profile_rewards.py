# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward and constraint terms for low-profile flat-ground locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .robotlab_rewards import get_base_bottom_height

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor


def _as_torch(value) -> torch.Tensor:
    """Return a native tensor from a tensor or tensor proxy."""
    return value.torch if hasattr(value, "torch") else value


def _smooth_step(
    value: torch.Tensor,
    lower: float | torch.Tensor,
    upper: float | torch.Tensor,
) -> torch.Tensor:
    """Map values smoothly from zero to one over an interval."""
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and upper <= lower:
        raise ValueError(f"Expected upper ({upper}) to be greater than lower ({lower}).")
    phase = torch.clamp((value - lower) / (upper - lower), 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def _equivalent_command_speed(command: torch.Tensor, yaw_scale: float) -> torch.Tensor:
    """Return translation and scaled yaw command magnitude [m/s]."""
    return torch.sqrt(torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2]))


def _contact_time_target(
    command: torch.Tensor,
    yaw_scale: float,
    forward_minimum_contact_time: float,
    lateral_minimum_contact_time: float,
    turning_minimum_contact_time: float,
) -> torch.Tensor:
    """Blend direction-dependent minimum foot-contact times [s]."""
    component_magnitude = torch.stack(
        (torch.abs(command[:, 0]), torch.abs(command[:, 1]), torch.abs(yaw_scale * command[:, 2])), dim=1
    )
    component_target = torch.tensor(
        (forward_minimum_contact_time, lateral_minimum_contact_time, turning_minimum_contact_time),
        device=command.device,
        dtype=command.dtype,
    )
    return torch.sum(component_magnitude * component_target, dim=1) / torch.clamp(
        torch.sum(component_magnitude, dim=1), min=1.0e-6
    )


def _contact_time_target(
    command: torch.Tensor,
    yaw_scale: float,
    forward_time: float,
    lateral_time: float,
    turning_time: float,
) -> torch.Tensor:
    """Blend stance-time targets according to commanded motion components [s]."""
    components = torch.stack(
        [torch.abs(command[:, 0]), torch.abs(command[:, 1]), torch.abs(yaw_scale * command[:, 2])], dim=1
    )
    targets = torch.tensor([forward_time, lateral_time, turning_time], device=command.device, dtype=command.dtype)
    component_sum = torch.sum(components, dim=1)
    weighted_target = torch.sum(components * targets.unsqueeze(0), dim=1) / torch.clamp(component_sum, min=1.0e-6)
    return torch.where(component_sum > 0.0, weighted_target, forward_time)


def _as_torch(value) -> torch.Tensor:
    """Return a native tensor from either tensor or proxy-array data."""
    return value.torch if hasattr(value, "torch") else value


def _equivalent_command_speed(command: torch.Tensor, yaw_scale: float) -> torch.Tensor:
    """Return planar and yaw command magnitude expressed as speed [m/s]."""
    return torch.sqrt(torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2]))


def _contact_time_target(
    command: torch.Tensor,
    yaw_scale: float,
    forward_minimum_contact_time: float,
    lateral_minimum_contact_time: float,
    turning_minimum_contact_time: float,
) -> torch.Tensor:
    """Blend directional minimum contact times from command components [s]."""
    component_magnitudes = torch.stack(
        (torch.abs(command[:, 0]), torch.abs(command[:, 1]), torch.abs(yaw_scale * command[:, 2])), dim=1
    )
    directional_targets = command.new_tensor(
        [forward_minimum_contact_time, lateral_minimum_contact_time, turning_minimum_contact_time]
    )
    return torch.sum(component_magnitudes * directional_targets, dim=1) / torch.clamp(
        torch.sum(component_magnitudes, dim=1), min=1.0e-6
    )


def base_bottom_height_constraint_l2(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_height: float,
    lower_margin: float,
    upper_margin: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize low clearance mildly and clearance above the ceiling sharply.

    Each violation is normalized by its own margin before squaring. A smaller
    upper margin therefore makes ceiling violations much more expensive than
    equally sized lower-bound violations.

    Args:
        env: The RL environment.
        minimum_height: Minimum preferred base-bottom clearance [m].
        maximum_height: Maximum permitted base-bottom clearance [m].
        lower_margin: Lower-bound normalization distance [m].
        upper_margin: Upper-bound normalization distance [m].
        sensor_cfg: Downward-facing base ray-caster configuration.

    Returns:
        Dimensionless squared constraint violation, shape ``(num_envs,)``.

    Raises:
        ValueError: If the bounds or normalization margins are invalid.
    """
    if maximum_height <= minimum_height:
        raise ValueError(f"Expected maximum_height > minimum_height, got {maximum_height} and {minimum_height}.")
    if lower_margin <= 0.0 or upper_margin <= 0.0:
        raise ValueError(f"Expected positive lower_margin and upper_margin, got {lower_margin} and {upper_margin}.")

    fallback_height = 0.5 * (minimum_height + maximum_height)
    height = get_base_bottom_height(env, fallback_height, sensor_cfg)
    below = torch.clamp(minimum_height - height, min=0.0) / lower_margin
    above = torch.clamp(height - maximum_height, min=0.0) / upper_margin
    return torch.square(below) + torch.square(above)


def base_bottom_height_above_maximum(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate environments whose base-bottom clearance exceeds a ceiling.

    Args:
        env: The RL environment.
        maximum_height: Maximum permitted base-bottom clearance [m].
        sensor_cfg: Downward-facing base ray-caster configuration.

    Returns:
        Boolean ceiling-violation mask, shape ``(num_envs,)``.
    """
    height = get_base_bottom_height(env, maximum_height, sensor_cfg)
    return height > maximum_height


def directional_hip_roll_position_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    release_start: float,
    release_full: float,
    yaw_scale: float,
    stand_still_scale: float = 1.0,
    stand_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize hip-roll displacement unless lateral or yaw motion needs it.

    The penalty remains fully active for forward/backward commands. It fades
    continuously as commanded lateral velocity or yaw rate grows, avoiding a
    discontinuity between dedicated and mixed command modes.

    Args:
        env: The RL environment.
        command_name: Name of the velocity command term.
        asset_cfg: Articulation and hip-roll joints to regulate.
        release_start: Lateral/yaw equivalent speed where relief starts [m/s].
        release_full: Lateral/yaw equivalent speed where relief is complete [m/s].
        yaw_scale: Length scale converting yaw rate to equivalent speed [m].
        stand_still_scale: Additional scale at near-zero commands.
        stand_threshold: Equivalent command speed treated as standing [m/s].

    Returns:
        Gated hip-roll position deviation [rad], shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lateral_yaw_speed = torch.sqrt(torch.square(command[:, 1]) + torch.square(yaw_scale * command[:, 2]))
    release = _smooth_step(lateral_yaw_speed, release_start, release_full)
    total_speed = torch.sqrt(torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2]))
    standing_scale = torch.where(total_speed < stand_threshold, stand_still_scale, 1.0)
    deviation = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return deviation * (1.0 - release) * standing_scale


def directional_hip_roll_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    release_start: float,
    release_full: float,
    yaw_scale: float,
) -> torch.Tensor:
    """Penalize hip-roll motion unless commanded lateral or yaw motion needs it.

    Args:
        env: The RL environment.
        command_name: Name of the velocity command term.
        asset_cfg: Articulation and hip-roll joints to regulate.
        release_start: Lateral/yaw equivalent speed where relief starts [m/s].
        release_full: Lateral/yaw equivalent speed where relief is complete [m/s].
        yaw_scale: Length scale converting yaw rate to equivalent speed [m].

    Returns:
        Gated squared hip-roll velocity [rad²/s²], shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lateral_yaw_speed = torch.sqrt(torch.square(command[:, 1]) + torch.square(yaw_scale * command[:, 2]))
    release = _smooth_step(lateral_yaw_speed, release_start, release_full)
    joint_velocity_l2 = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return joint_velocity_l2 * (1.0 - release)


def swing_foot_minimum_clearance_l2(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    clearance_margin: float,
    swing_grace_time: float,
    swing_full_time: float,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize only swing feet below a minimum ground clearance.

    The term has no preferred or maximum swing height. Its penalty becomes
    active after liftoff and is exactly zero once a foot reaches the minimum
    clearance, preventing ground scuffing without shaping higher trajectories.

    Args:
        env: The RL environment.
        minimum_height: Minimum penalty-free swing-foot clearance [m].
        clearance_margin: Clearance-deficit normalization distance [m].
        swing_grace_time: Air time below which the penalty is inactive [s].
        swing_full_time: Air time where the penalty becomes fully active [s].
        asset_cfg: Robot articulation and foot-body configuration.
        contact_sensor_cfg: Foot contact-sensor configuration.
        height_sensor_cfg: Downward-facing base ray-caster configuration.

    Returns:
        Dimensionless summed swing-clearance violation, shape ``(num_envs,)``.

    Raises:
        ValueError: If clearance or swing-time parameters are invalid.
    """
    if minimum_height <= 0.0 or clearance_margin <= 0.0:
        raise ValueError(
            f"Expected positive minimum_height and clearance_margin, got {minimum_height} and {clearance_margin}."
        )
    if swing_grace_time < 0.0 or swing_full_time <= swing_grace_time:
        raise ValueError(
            f"Expected 0 <= swing_grace_time < swing_full_time, got {swing_grace_time} and {swing_full_time}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w
    if hasattr(feet_pos_w, "torch"):
        feet_pos_w = feet_pos_w.torch
    feet_z = feet_pos_w[:, asset_cfg.body_ids, 2]

    height_sensor = env.scene.sensors[height_sensor_cfg.name]
    sensor_pos_w = height_sensor.data.pos_w
    if hasattr(sensor_pos_w, "torch"):
        sensor_pos_w = sensor_pos_w.torch
    base_bottom_height = get_base_bottom_height(env, minimum_height, height_sensor_cfg)
    feet_height = base_bottom_height.unsqueeze(1) + feet_z - sensor_pos_w[:, 2].unsqueeze(1)

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    current_air_time = contact_sensor.data.current_air_time.torch[:, contact_sensor_cfg.body_ids]
    swing_gate = _smooth_step(current_air_time, swing_grace_time, swing_full_time)
    normalized_deficit = torch.clamp(minimum_height - feet_height, min=0.0) / clearance_margin
    return torch.sum(torch.square(normalized_deficit) * swing_gate, dim=1)


def foot_scuffing_below_clearance_l2(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    clearance_margin: float,
    base_height_fallback: float,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize horizontal foot speed only below a minimum clearance.

    Unlike an exponential height weighting, this bounded gate becomes exactly
    zero at :paramref:`minimum_height`. Raising a foot above that clearance
    therefore cannot further reduce the penalty.

    Args:
        env: The RL environment.
        minimum_height: Minimum penalty-free foot clearance [m].
        clearance_margin: Width of the smooth transition below the minimum [m].
        base_height_fallback: Base-bottom clearance used if ray data is unavailable [m].
        asset_cfg: Robot articulation and foot-body configuration.
        height_sensor_cfg: Downward-facing base ray-caster configuration.

    Returns:
        Gated squared horizontal foot speed [m²/s²], shape ``(num_envs,)``.

    Raises:
        ValueError: If a clearance parameter is invalid.
    """
    if minimum_height <= 0.0 or clearance_margin <= 0.0 or base_height_fallback <= 0.0:
        raise ValueError(
            "Expected positive minimum_height, clearance_margin, and base_height_fallback, "
            f"got {minimum_height}, {clearance_margin}, and {base_height_fallback}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w
    if hasattr(feet_pos_w, "torch"):
        feet_pos_w = feet_pos_w.torch
    feet_z = feet_pos_w[:, asset_cfg.body_ids, 2]

    feet_velocity_w = asset.data.body_lin_vel_w
    if hasattr(feet_velocity_w, "torch"):
        feet_velocity_w = feet_velocity_w.torch
    feet_speed_l2 = torch.sum(torch.square(feet_velocity_w[:, asset_cfg.body_ids, :2]), dim=2)

    height_sensor = env.scene.sensors[height_sensor_cfg.name]
    sensor_pos_w = height_sensor.data.pos_w
    if hasattr(sensor_pos_w, "torch"):
        sensor_pos_w = sensor_pos_w.torch
    base_bottom_height = get_base_bottom_height(env, base_height_fallback, height_sensor_cfg)
    feet_height = base_bottom_height.unsqueeze(1) + feet_z - sensor_pos_w[:, 2].unsqueeze(1)

    normalized_deficit = torch.clamp(
        (minimum_height - feet_height) / clearance_margin,
        min=0.0,
        max=1.0,
    )
    height_gate = normalized_deficit * normalized_deficit * (3.0 - 2.0 * normalized_deficit)
    return torch.sum(feet_speed_l2 * height_gate, dim=1)


class _FootContactDurationTerm(ManagerTermBase):
    """Track completed stance durations for contact-duration objectives."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if self.contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        self._sensor_body_ids = self.contact_sensor.find_sensors(cfg.params["foot_names"])[0]
        current_contact_time = self._get_contact_time()
        self._previous_contact_time = current_contact_time.clone()
        self._previous_in_contact = current_contact_time > 0.0
        self._latest_duration = torch.zeros_like(current_contact_time)
        self._latest_score = torch.zeros_like(current_contact_time)
        self._has_completed_contact = torch.zeros_like(current_contact_time, dtype=torch.bool)
        self._command_name = cfg.params["command_name"]
        self._previous_command = env.command_manager.get_command(self._command_name).clone()

    def _get_contact_time(self) -> torch.Tensor:
        """Return current contact times for the configured feet [s]."""
        return _as_torch(self.contact_sensor.data.current_contact_time)[:, self._sensor_body_ids]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset completed-contact history for selected environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        current_contact_time = self._get_contact_time()
        self._previous_contact_time[env_ids] = current_contact_time[env_ids]
        self._previous_in_contact[env_ids] = current_contact_time[env_ids] > 0.0
        self._latest_duration[env_ids] = 0.0
        self._latest_score[env_ids] = 0.0
        self._has_completed_contact[env_ids] = False
        self._previous_command[env_ids] = self._env.command_manager.get_command(self._command_name)[env_ids]

    def _evaluate(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        forward_minimum_contact_time: float,
        lateral_minimum_contact_time: float,
        turning_minimum_contact_time: float,
        contact_time_margin: float,
        command_change_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update completed-contact state and return reward and shortfall tensors."""
        if min(
            forward_minimum_contact_time,
            lateral_minimum_contact_time,
            turning_minimum_contact_time,
            contact_time_margin,
        ) <= 0.0:
            raise ValueError("Expected all contact-time targets and contact_time_margin to be positive.")
        if motion_full <= motion_start:
            raise ValueError(f"Expected motion_full > motion_start, got {motion_full} and {motion_start}.")

        command = env.command_manager.get_command(command_name)
        motion_gate = _smooth_step(_equivalent_command_speed(command, yaw_scale), motion_start, motion_full)
        command_delta = command - self._previous_command
        reset_state = (motion_gate <= 0.0) | (
            _equivalent_command_speed(command_delta, yaw_scale) > command_change_threshold
        )

        current_contact_time = self._get_contact_time()
        in_contact = current_contact_time > 0.0
        liftoff = self._previous_in_contact & ~in_contact & ~reset_state.unsqueeze(1)
        target = _contact_time_target(
            command,
            yaw_scale,
            forward_minimum_contact_time,
            lateral_minimum_contact_time,
            turning_minimum_contact_time,
        ).unsqueeze(1)
        last_contact_time = getattr(self.contact_sensor.data, "last_contact_time", None)
        if last_contact_time is None:
            completed_duration = self._previous_contact_time
        else:
            completed_duration = _as_torch(last_contact_time)[:, self._sensor_body_ids]
        completed_score = _smooth_step(completed_duration, target - contact_time_margin, target)
        self._latest_duration = torch.where(liftoff, completed_duration, self._latest_duration)
        self._latest_score = torch.where(liftoff, completed_score, self._latest_score)
        self._has_completed_contact |= liftoff

        self._latest_duration[reset_state] = 0.0
        self._latest_score[reset_state] = 0.0
        self._has_completed_contact[reset_state] = False
        self._previous_contact_time = current_contact_time.clone()
        self._previous_in_contact = in_contact
        self._previous_command = command.clone()

        completed_all_feet = torch.all(self._has_completed_contact, dim=1)
        mean_score = torch.mean(self._latest_score, dim=1)
        reward = mean_score * completed_all_feet * motion_gate
        shortfall = (1.0 - mean_score) * completed_all_feet * motion_gate

        valid_count = torch.sum(self._has_completed_contact)
        mean_duration = torch.sum(self._latest_duration * self._has_completed_contact) / torch.clamp(valid_count, min=1)
        success_ratio = torch.sum((self._latest_score >= 1.0) & self._has_completed_contact) / torch.clamp(
            valid_count, min=1
        )
        extras = getattr(env, "extras", None)
        if extras is not None:
            log = extras.setdefault("log", {})
            log["Metrics/contact_time_mean_s"] = mean_duration
            log["Metrics/contact_time_success_ratio"] = success_ratio
            log["Metrics/early_liftoff_ratio"] = 1.0 - success_ratio
            get_command_term = getattr(env.command_manager, "get_term", None)
            if get_command_term is not None:
                command_term = get_command_term(command_name)
                is_standing = getattr(
                    command_term, "is_standing_env", torch.zeros_like(motion_gate, dtype=torch.bool)
                )
                is_straight = getattr(command_term, "is_straight_env", torch.zeros_like(is_standing))
                is_turning = getattr(command_term, "is_turning_env", torch.zeros_like(is_standing))
                is_lateral = getattr(command_term, "is_lateral_env", torch.zeros_like(is_standing))
                mode_masks = {
                    "forward": is_straight & (command[:, 0] > 0.0),
                    "backward": is_straight & (command[:, 0] < 0.0),
                    "lateral": is_lateral,
                    "turning": is_turning,
                    "random": ~(is_standing | is_straight | is_turning | is_lateral),
                }
                completed_count_by_env = torch.sum(self._has_completed_contact, dim=1)
                duration_by_env = torch.sum(
                    self._latest_duration * self._has_completed_contact, dim=1
                ) / torch.clamp(completed_count_by_env, min=1)
                valid_env = (completed_count_by_env > 0) & (motion_gate > 0.0)
                for mode_name, mode_mask in mode_masks.items():
                    valid_mode = valid_env & mode_mask
                    mode_count = torch.sum(valid_mode)
                    log[f"Metrics/contact_time_{mode_name}_s"] = torch.sum(
                        duration_by_env * valid_mode
                    ) / torch.clamp(mode_count, min=1)
        return reward, shortfall


class FootContactDurationReward(_FootContactDurationTerm):
    """Reward feet that complete a direction-dependent minimum stance time."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        foot_names: Sequence[str],
        sensor_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        forward_minimum_contact_time: float,
        lateral_minimum_contact_time: float,
        turning_minimum_contact_time: float,
        contact_time_margin: float,
        command_change_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute the completed stance-duration reward.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            foot_names: Names of feet tracked by the contact sensor.
            sensor_cfg: Foot contact-sensor configuration.
            motion_start: Equivalent speed where the reward starts [m/s].
            motion_full: Equivalent speed where the reward is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to speed [m].
            forward_minimum_contact_time: Minimum sagittal stance duration [s].
            lateral_minimum_contact_time: Minimum lateral stance duration [s].
            turning_minimum_contact_time: Minimum turning stance duration [s].
            contact_time_margin: Transition width below the minimum duration [s].
            command_change_threshold: Command change that clears duration history [m/s].

        Returns:
            Completed-contact score in ``[0, 1]``, shape ``(num_envs,)``.
        """
        del foot_names, sensor_cfg
        reward, _ = self._evaluate(
            env,
            command_name,
            motion_start,
            motion_full,
            yaw_scale,
            forward_minimum_contact_time,
            lateral_minimum_contact_time,
            turning_minimum_contact_time,
            contact_time_margin,
            command_change_threshold,
        )
        return reward


class FootEarlyLiftoffPenalty(_FootContactDurationTerm):
    """Penalize feet whose completed stance is shorter than the minimum time."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        foot_names: Sequence[str],
        sensor_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        forward_minimum_contact_time: float,
        lateral_minimum_contact_time: float,
        turning_minimum_contact_time: float,
        contact_time_margin: float,
        command_change_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute the completed stance-duration shortfall.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            foot_names: Names of feet tracked by the contact sensor.
            sensor_cfg: Foot contact-sensor configuration.
            motion_start: Equivalent speed where the penalty starts [m/s].
            motion_full: Equivalent speed where the penalty is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to speed [m].
            forward_minimum_contact_time: Minimum sagittal stance duration [s].
            lateral_minimum_contact_time: Minimum lateral stance duration [s].
            turning_minimum_contact_time: Minimum turning stance duration [s].
            contact_time_margin: Transition width below the minimum duration [s].
            command_change_threshold: Command change that clears duration history [m/s].

        Returns:
            Completed-contact shortfall in ``[0, 1]``, shape ``(num_envs,)``.
        """
        del foot_names, sensor_cfg
        _, shortfall = self._evaluate(
            env,
            command_name,
            motion_start,
            motion_full,
            yaw_scale,
            forward_minimum_contact_time,
            lateral_minimum_contact_time,
            turning_minimum_contact_time,
            contact_time_margin,
            command_change_threshold,
        )
        return shortfall


class SwingFootClearanceBoundsPenalty(ManagerTermBase):
    """Keep swing clearance inside a band relative to the last support height."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if self.contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        self._foot_body_ids = asset_cfg.body_ids
        self._sensor_body_ids = sensor_cfg.body_ids
        self._last_contact_height = self._get_foot_height().clone()

    def _get_foot_height(self) -> torch.Tensor:
        """Return configured foot-body origin heights [m]."""
        return _as_torch(self.asset.data.body_pos_w)[:, self._foot_body_ids, 2]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset support-height references to current foot heights.

        Args:
            env_ids: Environment indices to reset.
        """
        current_height = self._get_foot_height()
        if env_ids is None:
            self._last_contact_height[:] = current_height
        else:
            self._last_contact_height[env_ids] = current_height[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        minimum_clearance: float,
        maximum_clearance: float,
        lower_margin: float,
        upper_margin: float,
        swing_grace_time: float,
        swing_full_time: float,
        upper_penalty_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute lower and upper swing-clearance violations.

        Args:
            env: The environment instance.
            asset_cfg: Robot articulation and foot-body configuration.
            sensor_cfg: Foot contact-sensor configuration.
            minimum_clearance: Minimum penalty-free swing clearance [m].
            maximum_clearance: Maximum penalty-free swing clearance [m].
            lower_margin: Lower violation normalization distance [m].
            upper_margin: Upper violation normalization distance [m].
            swing_grace_time: Air time below which the constraint is inactive [s].
            swing_full_time: Air time where the constraint is fully active [s].
            upper_penalty_scale: Relative scale applied to excessive clearance.

        Returns:
            Summed dimensionless clearance violation, shape ``(num_envs,)``.
        """
        del env, asset_cfg, sensor_cfg
        if maximum_clearance <= minimum_clearance:
            raise ValueError("Expected maximum_clearance to exceed minimum_clearance.")
        if min(lower_margin, upper_margin, upper_penalty_scale) <= 0.0:
            raise ValueError("Expected positive clearance margins and upper_penalty_scale.")
        if swing_grace_time < 0.0 or swing_full_time <= swing_grace_time:
            raise ValueError("Expected 0 <= swing_grace_time < swing_full_time.")

        foot_height = self._get_foot_height()
        contact_time = _as_torch(self.contact_sensor.data.current_contact_time)[:, self._sensor_body_ids]
        air_time = _as_torch(self.contact_sensor.data.current_air_time)[:, self._sensor_body_ids]
        in_contact = contact_time > 0.0
        self._last_contact_height = torch.where(in_contact, foot_height, self._last_contact_height)
        clearance = torch.clamp(foot_height - self._last_contact_height, min=0.0)
        swing_gate = _smooth_step(air_time, swing_grace_time, swing_full_time)

        lower = torch.clamp((minimum_clearance - clearance) / lower_margin, min=0.0, max=1.0)
        upper = torch.clamp((clearance - maximum_clearance) / upper_margin, min=0.0)
        penalty = (torch.square(lower) + upper_penalty_scale * torch.square(upper)) * swing_gate

        extras = getattr(self._env, "extras", None)
        if extras is not None:
            in_swing = air_time > 0.0
            swing_count = torch.sum(in_swing)
            log = extras.setdefault("log", {})
            log["Metrics/foot_clearance_mean_m"] = torch.sum(clearance * in_swing) / torch.clamp(swing_count, min=1)
            log["Metrics/foot_clearance_max_m"] = torch.max(torch.where(in_swing, clearance, 0.0))
            log["Metrics/foot_clearance_over_limit_ratio"] = torch.sum(
                in_swing & (clearance > maximum_clearance)
            ) / torch.clamp(swing_count, min=1)
        return torch.sum(penalty, dim=1)


def excessive_swing_time_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    maximum_swing_time: float,
    time_margin: float,
    motion_start: float,
    motion_full: float,
    yaw_scale: float,
) -> torch.Tensor:
    """Penalize feet that remain airborne beyond a maximum duration.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command term.
        sensor_cfg: Foot contact-sensor configuration.
        maximum_swing_time: Maximum penalty-free continuous air time [s].
        time_margin: Air-time violation normalization interval [s].
        motion_start: Equivalent speed where the penalty starts [m/s].
        motion_full: Equivalent speed where the penalty is fully active [m/s].
        yaw_scale: Length scale converting yaw rate to speed [m].

    Returns:
        Summed squared excessive-air-time violation, shape ``(num_envs,)``.
    """
    if maximum_swing_time <= 0.0 or time_margin <= 0.0:
        raise ValueError("Expected maximum_swing_time and time_margin to be positive.")
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    air_time = _as_torch(sensor.data.current_air_time)[:, sensor_cfg.body_ids]
    excess = torch.clamp((air_time - maximum_swing_time) / time_margin, min=0.0)
    command = env.command_manager.get_command(command_name)
    motion_gate = _smooth_step(_equivalent_command_speed(command, yaw_scale), motion_start, motion_full)
    return torch.sum(torch.square(excess), dim=1) * motion_gate


def stand_foot_motion_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    yaw_scale: float,
) -> torch.Tensor:
    """Penalize foot motion while the commanded base velocity is near zero.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command term.
        asset_cfg: Robot articulation and foot-body configuration.
        command_threshold: Equivalent speed treated as standing [m/s].
        yaw_scale: Length scale converting yaw rate to speed [m].

    Returns:
        Summed squared foot speed [m²/s²], shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_velocity = _as_torch(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    standing = _equivalent_command_speed(command, yaw_scale) < command_threshold
    return torch.sum(torch.square(foot_velocity), dim=(1, 2)) * standing


def stand_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    yaw_scale: float,
) -> torch.Tensor:
    """Penalize joint motion while the commanded base velocity is near zero.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command term.
        asset_cfg: Robot articulation and joint configuration.
        command_threshold: Equivalent speed treated as standing [m/s].
        yaw_scale: Length scale converting yaw rate to speed [m].

    Returns:
        Summed squared joint velocity [rad²/s²], shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_velocity = _as_torch(asset.data.joint_vel)[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command(command_name)
    standing = _equivalent_command_speed(command, yaw_scale) < command_threshold
    return torch.sum(torch.square(joint_velocity), dim=1) * standing


class TrotCadenceCeilingPenalty(ManagerTermBase):
    """Penalize trot cycles faster than a configured maximum frequency.

    A stride period is measured independently between consecutive touchdowns
    of each diagonal pair. Cadence at or below the ceiling has exactly zero
    penalty, so the term does not force the policy toward 3 Hz from below.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize diagonal-pair touchdown tracking.

        Args:
            cfg: Reward term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if self.contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        pair_names = cfg.params["synced_feet_pair_names"]
        if len(pair_names) != 2 or any(len(pair) != 2 for pair in pair_names):
            raise ValueError("Cadence limiting requires exactly two pairs containing two feet each.")
        self._pair_sensor_ids = [self.contact_sensor.find_sensors(pair)[0] for pair in pair_names]

        maximum_frequency = cfg.params["maximum_cycle_frequency"]
        interval_tolerance = cfg.params["interval_tolerance"]
        min_interval = cfg.params["min_touchdown_interval"]
        max_interval = cfg.params["max_touchdown_interval"]
        if maximum_frequency <= 0.0:
            raise ValueError(f"Expected maximum_cycle_frequency > 0, got {maximum_frequency}.")
        if interval_tolerance <= 0.0:
            raise ValueError(f"Expected interval_tolerance > 0, got {interval_tolerance}.")
        if min_interval < 0.0 or max_interval <= min_interval:
            raise ValueError(
                f"Expected 0 <= min_touchdown_interval < max_touchdown_interval, got {min_interval} and {max_interval}."
            )
        ceiling_interval = 1.0 / maximum_frequency
        if not min_interval <= ceiling_interval <= max_interval:
            raise ValueError(
                "Expected the cadence-ceiling interval to lie within "
                f"[{min_interval}, {max_interval}], got {ceiling_interval}."
            )

        self._step_dt = env.step_dt
        state_shape = (env.num_envs, len(self._pair_sensor_ids))
        self._elapsed_since_touchdown = torch.zeros(state_shape, device=env.device)
        self._latest_interval = torch.zeros(state_shape, device=env.device)
        self._latest_error = torch.zeros(state_shape, device=env.device)
        self._has_last_touchdown = torch.zeros(state_shape, device=env.device, dtype=torch.bool)
        self._has_completed_interval = torch.zeros(state_shape, device=env.device, dtype=torch.bool)
        self._previous_pair_contact = self._get_pair_contact().clone()
        self._command_name = cfg.params["command_name"]
        self._previous_command = env.command_manager.get_command(self._command_name).clone()

    def _get_pair_contact(self) -> torch.Tensor:
        """Return whether both feet in each diagonal pair are in contact."""
        contact_time = self.contact_sensor.data.current_contact_time.torch
        return torch.stack(
            [torch.all(contact_time[:, pair_ids] > 0.0, dim=1) for pair_ids in self._pair_sensor_ids],
            dim=1,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset cadence history for selected environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._elapsed_since_touchdown[env_ids] = 0.0
        self._latest_interval[env_ids] = 0.0
        self._latest_error[env_ids] = 0.0
        self._has_last_touchdown[env_ids] = False
        self._has_completed_interval[env_ids] = False
        self._previous_pair_contact[env_ids] = self._get_pair_contact()[env_ids]
        self._previous_command[env_ids] = self._env.command_manager.get_command(self._command_name)[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        synced_feet_pair_names: Sequence[Sequence[str]],
        sensor_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        maximum_cycle_frequency: float,
        interval_tolerance: float,
        min_touchdown_interval: float,
        max_touchdown_interval: float,
        command_change_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute a one-sided cadence-ceiling violation.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            synced_feet_pair_names: Names of the two diagonal foot pairs.
            sensor_cfg: Contact-sensor configuration.
            motion_start: Equivalent speed where cadence limiting starts [m/s].
            motion_full: Equivalent speed where cadence limiting is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to equivalent speed [m].
            maximum_cycle_frequency: Maximum penalty-free trot frequency [Hz].
            interval_tolerance: Exponential scale for overspeed period error [s].
            min_touchdown_interval: Minimum accepted same-pair touchdown interval [s].
            max_touchdown_interval: Maximum measured same-pair touchdown interval [s].
            command_change_threshold: Equivalent command change that resets history [m/s].

        Returns:
            Bounded cadence overspeed penalty in ``[0, 1]``, shape ``(num_envs,)``.
        """
        del synced_feet_pair_names, sensor_cfg
        command = env.command_manager.get_command(command_name)
        equivalent_speed = torch.sqrt(
            torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2])
        )
        motion_gate = _smooth_step(equivalent_speed, motion_start, motion_full)
        command_delta = command - self._previous_command
        equivalent_command_delta = torch.sqrt(
            torch.sum(torch.square(command_delta[:, :2]), dim=1) + torch.square(yaw_scale * command_delta[:, 2])
        )
        reset_state = (motion_gate <= 0.0) | (equivalent_command_delta > command_change_threshold)

        pair_contact = self._get_pair_contact()
        touchdown = (pair_contact & ~self._previous_pair_contact) & ~reset_state.unsqueeze(1)
        self._elapsed_since_touchdown = torch.where(
            reset_state.unsqueeze(1),
            torch.zeros_like(self._elapsed_since_touchdown),
            self._elapsed_since_touchdown + self._step_dt,
        )
        accepted_touchdown = touchdown & (
            ~self._has_last_touchdown | (self._elapsed_since_touchdown >= min_touchdown_interval)
        )
        completed_interval = accepted_touchdown & self._has_last_touchdown
        measured_interval = self._elapsed_since_touchdown
        current_interval = torch.clamp(measured_interval, max=max_touchdown_interval)
        ceiling_interval = 1.0 / maximum_cycle_frequency
        overspeed_interval = torch.clamp(ceiling_interval - current_interval, min=0.0)
        normalized_error = 1.0 - torch.exp(-0.5 * torch.square(overspeed_interval / interval_tolerance))
        self._latest_interval = torch.where(completed_interval, measured_interval, self._latest_interval)
        self._latest_error = torch.where(completed_interval, normalized_error, self._latest_error)
        self._has_last_touchdown |= accepted_touchdown
        self._has_completed_interval |= completed_interval
        self._elapsed_since_touchdown = torch.where(
            accepted_touchdown,
            torch.zeros_like(self._elapsed_since_touchdown),
            self._elapsed_since_touchdown,
        )

        self._elapsed_since_touchdown[reset_state] = 0.0
        self._latest_interval[reset_state] = 0.0
        self._latest_error[reset_state] = 0.0
        self._has_last_touchdown[reset_state] = False
        self._has_completed_interval[reset_state] = False
        self._previous_pair_contact = pair_contact
        self._previous_command = command.clone()

        valid_interval = self._has_completed_interval & (motion_gate.unsqueeze(1) > 0.0)
        cadence_hz = torch.where(
            valid_interval,
            torch.reciprocal(torch.clamp(self._latest_interval, min=self._step_dt)),
            torch.zeros_like(self._latest_interval),
        )
        valid_count = torch.sum(valid_interval)
        mean_cadence_hz = torch.sum(cadence_hz) / torch.clamp(valid_count, min=1)
        extras = getattr(env, "extras", None)
        if extras is not None:
            log = extras.setdefault("log", {})
            log["Metrics/cadence_hz"] = mean_cadence_hz
            log["Metrics/cadence_overspeed_ratio"] = torch.sum(
                valid_interval & (cadence_hz > maximum_cycle_frequency)
            ) / torch.clamp(valid_count, min=1)

            get_command_term = getattr(env.command_manager, "get_term", None)
            if get_command_term is not None:
                command_term = get_command_term(command_name)
                is_standing = getattr(command_term, "is_standing_env", torch.zeros_like(motion_gate, dtype=torch.bool))
                is_straight = getattr(command_term, "is_straight_env", torch.zeros_like(is_standing))
                is_turning = getattr(command_term, "is_turning_env", torch.zeros_like(is_standing))
                is_lateral = getattr(command_term, "is_lateral_env", torch.zeros_like(is_standing))
                mode_masks = {
                    "forward": is_straight & (command[:, 0] > 0.0),
                    "backward": is_straight & (command[:, 0] < 0.0),
                    "lateral": is_lateral,
                    "turning": is_turning,
                    "random": ~(is_standing | is_straight | is_turning | is_lateral),
                }
                for mode_name, mode_mask in mode_masks.items():
                    valid_mode_interval = valid_interval & mode_mask.unsqueeze(1)
                    mode_count = torch.sum(valid_mode_interval)
                    mode_cadence_hz = torch.sum(cadence_hz * valid_mode_interval) / torch.clamp(mode_count, min=1)
                    log[f"Metrics/cadence_hz_{mode_name}"] = mode_cadence_hz
        return torch.max(self._latest_error, dim=1).values * motion_gate


class TrotStrideLengthReward(ManagerTermBase):
    """Reward sufficient planar travel between same-pair trot touchdowns.

    The required stride length is the commanded planar speed divided by the
    configured maximum cadence. This couples stride distance to the 3 Hz
    ceiling without prescribing swing-foot height or rewarding arbitrarily
    large strides beyond the required distance.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize diagonal-pair stride tracking.

        Args:
            cfg: Reward term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if self.contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        pair_names = cfg.params["synced_feet_pair_names"]
        if len(pair_names) != 2 or any(len(pair) != 2 for pair in pair_names):
            raise ValueError("Stride tracking requires exactly two pairs containing two feet each.")
        self._pair_sensor_ids = [self.contact_sensor.find_sensors(pair)[0] for pair in pair_names]

        maximum_frequency = cfg.params["maximum_cycle_frequency"]
        minimum_stride_length = cfg.params["minimum_stride_length"]
        min_interval = cfg.params["min_touchdown_interval"]
        if maximum_frequency <= 0.0:
            raise ValueError(f"Expected maximum_cycle_frequency > 0, got {maximum_frequency}.")
        if minimum_stride_length <= 0.0:
            raise ValueError(f"Expected minimum_stride_length > 0, got {minimum_stride_length}.")
        if min_interval < 0.0:
            raise ValueError(f"Expected min_touchdown_interval >= 0, got {min_interval}.")

        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._step_dt = env.step_dt
        state_shape = (env.num_envs, len(self._pair_sensor_ids))
        self._elapsed_since_touchdown = torch.zeros(state_shape, device=env.device)
        self._last_touchdown_pos_w = torch.zeros((*state_shape, 2), device=env.device)
        self._latest_stride_length = torch.zeros(state_shape, device=env.device)
        self._latest_reward = torch.zeros(state_shape, device=env.device)
        self._has_last_touchdown = torch.zeros(state_shape, device=env.device, dtype=torch.bool)
        self._has_completed_stride = torch.zeros(state_shape, device=env.device, dtype=torch.bool)
        self._previous_pair_contact = self._get_pair_contact().clone()
        self._command_name = cfg.params["command_name"]
        self._previous_command = env.command_manager.get_command(self._command_name).clone()

    def _get_pair_contact(self) -> torch.Tensor:
        """Return whether both feet in each diagonal pair are in contact."""
        contact_time = self.contact_sensor.data.current_contact_time.torch
        return torch.stack(
            [torch.all(contact_time[:, pair_ids] > 0.0, dim=1) for pair_ids in self._pair_sensor_ids],
            dim=1,
        )

    def _get_base_xy(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Return world-frame base position in the horizontal plane [m]."""
        asset: Articulation = env.scene[self._asset_cfg.name]
        root_pos_w = asset.data.root_pos_w
        if hasattr(root_pos_w, "torch"):
            root_pos_w = root_pos_w.torch
        return root_pos_w[:, :2]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset stride history for selected environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._elapsed_since_touchdown[env_ids] = 0.0
        self._last_touchdown_pos_w[env_ids] = 0.0
        self._latest_stride_length[env_ids] = 0.0
        self._latest_reward[env_ids] = 0.0
        self._has_last_touchdown[env_ids] = False
        self._has_completed_stride[env_ids] = False
        self._previous_pair_contact[env_ids] = self._get_pair_contact()[env_ids]
        self._previous_command[env_ids] = self._env.command_manager.get_command(self._command_name)[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        synced_feet_pair_names: Sequence[Sequence[str]],
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        maximum_cycle_frequency: float,
        minimum_stride_length: float,
        min_touchdown_interval: float,
        command_change_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute reward for meeting the command-dependent stride distance.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            synced_feet_pair_names: Names of the two diagonal foot pairs.
            sensor_cfg: Contact-sensor configuration.
            asset_cfg: Robot articulation configuration.
            motion_start: Planar command speed where stride reward starts [m/s].
            motion_full: Planar command speed where stride reward is fully active [m/s].
            yaw_scale: Length scale converting yaw-rate changes to equivalent speed [m].
            maximum_cycle_frequency: Maximum intended trot frequency [Hz].
            minimum_stride_length: Minimum useful stride target at low speeds [m].
            min_touchdown_interval: Minimum accepted same-pair touchdown interval [s].
            command_change_threshold: Equivalent command change that resets history [m/s].

        Returns:
            Bounded stride-length reward in ``[0, 1]``, shape ``(num_envs,)``.
        """
        del synced_feet_pair_names, sensor_cfg, asset_cfg
        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)
        motion_gate = _smooth_step(command_speed, motion_start, motion_full)
        command_delta = command - self._previous_command
        equivalent_command_delta = torch.sqrt(
            torch.sum(torch.square(command_delta[:, :2]), dim=1) + torch.square(yaw_scale * command_delta[:, 2])
        )
        reset_state = (motion_gate <= 0.0) | (equivalent_command_delta > command_change_threshold)

        pair_contact = self._get_pair_contact()
        touchdown = (pair_contact & ~self._previous_pair_contact) & ~reset_state.unsqueeze(1)
        self._elapsed_since_touchdown = torch.where(
            reset_state.unsqueeze(1),
            torch.zeros_like(self._elapsed_since_touchdown),
            self._elapsed_since_touchdown + self._step_dt,
        )
        accepted_touchdown = touchdown & (
            ~self._has_last_touchdown | (self._elapsed_since_touchdown >= min_touchdown_interval)
        )
        completed_stride = accepted_touchdown & self._has_last_touchdown

        base_xy = self._get_base_xy(env)
        base_xy_by_pair = base_xy.unsqueeze(1).expand(-1, len(self._pair_sensor_ids), -1)
        measured_stride = torch.linalg.norm(base_xy_by_pair - self._last_touchdown_pos_w, dim=2)
        required_stride = torch.maximum(
            command_speed / maximum_cycle_frequency,
            torch.full_like(command_speed, minimum_stride_length),
        ).unsqueeze(1)
        progress = torch.clamp(
            (measured_stride - 0.5 * required_stride) / (0.5 * required_stride),
            0.0,
            1.0,
        )
        stride_reward = progress * progress * (3.0 - 2.0 * progress)

        self._latest_stride_length = torch.where(completed_stride, measured_stride, self._latest_stride_length)
        self._latest_reward = torch.where(completed_stride, stride_reward, self._latest_reward)
        self._last_touchdown_pos_w = torch.where(
            accepted_touchdown.unsqueeze(2), base_xy_by_pair, self._last_touchdown_pos_w
        )
        self._has_last_touchdown |= accepted_touchdown
        self._has_completed_stride |= completed_stride
        self._elapsed_since_touchdown = torch.where(
            accepted_touchdown,
            torch.zeros_like(self._elapsed_since_touchdown),
            self._elapsed_since_touchdown,
        )

        self._elapsed_since_touchdown[reset_state] = 0.0
        self._last_touchdown_pos_w[reset_state] = 0.0
        self._latest_stride_length[reset_state] = 0.0
        self._latest_reward[reset_state] = 0.0
        self._has_last_touchdown[reset_state] = False
        self._has_completed_stride[reset_state] = False
        self._previous_pair_contact = pair_contact
        self._previous_command = command.clone()

        valid_count = torch.sum(self._has_completed_stride)
        extras = getattr(env, "extras", None)
        if extras is not None:
            extras.setdefault("log", {})["Metrics/stride_length_m"] = torch.sum(
                self._latest_stride_length * self._has_completed_stride
            ) / torch.clamp(valid_count, min=1)

        both_pairs_valid = torch.all(self._has_completed_stride, dim=1)
        pair_mean_reward = torch.mean(self._latest_reward, dim=1)
        return pair_mean_reward * both_pairs_valid * motion_gate


class CommandedStrideProgressReward(ManagerTermBase):
    """Reward long command-aligned strides without using cadence as an objective."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize diagonal-pair touchdown and displacement tracking.

        Args:
            cfg: Reward term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if self.contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        pair_names = cfg.params["synced_feet_pair_names"]
        if len(pair_names) != 2 or any(len(pair) != 2 for pair in pair_names):
            raise ValueError("Stride tracking requires exactly two pairs containing two feet each.")
        self._pair_sensor_ids = [self.contact_sensor.find_sensors(pair)[0] for pair in pair_names]
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._step_dt = env.step_dt
        state_shape = (env.num_envs, len(self._pair_sensor_ids))
        self._elapsed_since_touchdown = torch.zeros(state_shape, device=env.device)
        self._last_touchdown_pos_w = torch.zeros((*state_shape, 2), device=env.device)
        self._latest_stride_length = torch.zeros(state_shape, device=env.device)
        self._latest_stride_target = torch.zeros(state_shape, device=env.device)
        self._latest_reward = torch.zeros(state_shape, device=env.device)
        self._latest_interval = torch.zeros(state_shape, device=env.device)
        self._has_last_touchdown = torch.zeros(state_shape, device=env.device, dtype=torch.bool)
        self._has_completed_stride = torch.zeros(state_shape, device=env.device, dtype=torch.bool)
        self._previous_pair_contact = self._get_pair_contact().clone()
        self._command_name = cfg.params["command_name"]
        self._previous_command = env.command_manager.get_command(self._command_name).clone()

    def _get_pair_contact(self) -> torch.Tensor:
        """Return whether both feet of each diagonal pair are in contact."""
        contact_time = _as_torch(self.contact_sensor.data.current_contact_time)
        return torch.stack(
            [torch.all(contact_time[:, pair_ids] > 0.0, dim=1) for pair_ids in self._pair_sensor_ids], dim=1
        )

    def _get_base_xy(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Return world-frame base position in the horizontal plane [m]."""
        asset: Articulation = env.scene[self._asset_cfg.name]
        return _as_torch(asset.data.root_pos_w)[:, :2]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset stride history for selected environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._elapsed_since_touchdown[env_ids] = 0.0
        self._last_touchdown_pos_w[env_ids] = 0.0
        self._latest_stride_length[env_ids] = 0.0
        self._latest_stride_target[env_ids] = 0.0
        self._latest_reward[env_ids] = 0.0
        self._latest_interval[env_ids] = 0.0
        self._has_last_touchdown[env_ids] = False
        self._has_completed_stride[env_ids] = False
        self._previous_pair_contact[env_ids] = self._get_pair_contact()[env_ids]
        self._previous_command[env_ids] = self._env.command_manager.get_command(self._command_name)[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        synced_feet_pair_names: Sequence[Sequence[str]],
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        stride_intercept: float,
        stride_speed_gain: float,
        minimum_stride_length: float,
        maximum_stride_length: float,
        lateral_scale: float,
        reward_start_ratio: float,
        min_touchdown_interval: float,
        command_change_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute reward for command-aligned displacement between touchdowns.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            synced_feet_pair_names: Names of the two diagonal foot pairs.
            sensor_cfg: Foot contact-sensor configuration.
            asset_cfg: Robot articulation configuration.
            motion_start: Planar speed where stride shaping starts [m/s].
            motion_full: Planar speed where stride shaping is fully active [m/s].
            yaw_scale: Length scale used only when detecting command changes [m].
            stride_intercept: Constant component of the target stride [m].
            stride_speed_gain: Target-stride gain with respect to command speed [s].
            minimum_stride_length: Minimum target stride [m].
            maximum_stride_length: Maximum target stride [m].
            lateral_scale: Target multiplier for pure lateral commands.
            reward_start_ratio: Target fraction where positive shaping begins.
            min_touchdown_interval: Minimum accepted pair touchdown interval [s].
            command_change_threshold: Command change that clears stride history [m/s].

        Returns:
            Bounded command-aligned stride reward in ``[0, 1]``, shape ``(num_envs,)``.
        """
        del synced_feet_pair_names, sensor_cfg, asset_cfg
        if maximum_stride_length <= minimum_stride_length or minimum_stride_length <= 0.0:
            raise ValueError("Expected 0 < minimum_stride_length < maximum_stride_length.")
        if stride_intercept < 0.0 or stride_speed_gain <= 0.0:
            raise ValueError("Expected non-negative stride_intercept and positive stride_speed_gain.")
        if not 0.0 < lateral_scale <= 1.0:
            raise ValueError(f"Expected lateral_scale in (0, 1], got {lateral_scale}.")
        if not 0.0 <= reward_start_ratio < 1.0:
            raise ValueError(f"Expected reward_start_ratio in [0, 1), got {reward_start_ratio}.")

        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)
        stride_gate = _smooth_step(command_speed, motion_start, motion_full)
        locomotion_gate = _smooth_step(
            _equivalent_command_speed(command, yaw_scale), motion_start, motion_full
        )
        command_delta = command - self._previous_command
        reset_state = (locomotion_gate <= 0.0) | (
            _equivalent_command_speed(command_delta, yaw_scale) > command_change_threshold
        )

        pair_contact = self._get_pair_contact()
        touchdown = (pair_contact & ~self._previous_pair_contact) & ~reset_state.unsqueeze(1)
        self._elapsed_since_touchdown = torch.where(
            reset_state.unsqueeze(1),
            torch.zeros_like(self._elapsed_since_touchdown),
            self._elapsed_since_touchdown + self._step_dt,
        )
        accepted_touchdown = touchdown & (
            ~self._has_last_touchdown | (self._elapsed_since_touchdown >= min_touchdown_interval)
        )
        completed_stride = accepted_touchdown & self._has_last_touchdown

        base_xy = self._get_base_xy(env)
        base_xy_by_pair = base_xy.unsqueeze(1).expand(-1, len(self._pair_sensor_ids), -1)
        command_direction_b = torch.cat(
            (command[:, :2] / torch.clamp(command_speed.unsqueeze(1), min=1.0e-6), torch.zeros_like(command[:, :1])),
            dim=1,
        )
        asset: Articulation = env.scene[self._asset_cfg.name]
        root_quat_w = _as_torch(asset.data.root_quat_w)
        command_direction_w = math_utils.quat_apply_yaw(root_quat_w, command_direction_b)[:, :2]
        displacement = base_xy_by_pair - self._last_touchdown_pos_w
        measured_stride = torch.clamp(
            torch.sum(displacement * command_direction_w.unsqueeze(1), dim=2), min=0.0
        )
        lateral_fraction = torch.abs(command[:, 1]) / torch.clamp(
            torch.abs(command[:, 0]) + torch.abs(command[:, 1]), min=1.0e-6
        )
        direction_scale = 1.0 - lateral_fraction * (1.0 - lateral_scale)
        required_stride = torch.clamp(
            stride_intercept + stride_speed_gain * command_speed,
            min=minimum_stride_length,
            max=maximum_stride_length,
        ) * direction_scale
        required_stride_by_pair = required_stride.unsqueeze(1)
        reward_start = reward_start_ratio * required_stride_by_pair
        progress = torch.clamp(
            (measured_stride - reward_start) / torch.clamp(required_stride_by_pair - reward_start, min=1.0e-6),
            0.0,
            1.0,
        )
        stride_reward = progress * progress * (3.0 - 2.0 * progress)

        self._latest_stride_length = torch.where(completed_stride, measured_stride, self._latest_stride_length)
        self._latest_stride_target = torch.where(
            completed_stride, required_stride_by_pair, self._latest_stride_target
        )
        self._latest_reward = torch.where(completed_stride, stride_reward, self._latest_reward)
        self._latest_interval = torch.where(
            completed_stride, self._elapsed_since_touchdown, self._latest_interval
        )
        self._last_touchdown_pos_w = torch.where(
            accepted_touchdown.unsqueeze(2), base_xy_by_pair, self._last_touchdown_pos_w
        )
        self._has_last_touchdown |= accepted_touchdown
        self._has_completed_stride |= completed_stride
        self._elapsed_since_touchdown = torch.where(
            accepted_touchdown, torch.zeros_like(self._elapsed_since_touchdown), self._elapsed_since_touchdown
        )

        self._elapsed_since_touchdown[reset_state] = 0.0
        self._last_touchdown_pos_w[reset_state] = 0.0
        self._latest_stride_length[reset_state] = 0.0
        self._latest_stride_target[reset_state] = 0.0
        self._latest_reward[reset_state] = 0.0
        self._latest_interval[reset_state] = 0.0
        self._has_last_touchdown[reset_state] = False
        self._has_completed_stride[reset_state] = False
        self._previous_pair_contact = pair_contact
        self._previous_command = command.clone()

        valid_stride = self._has_completed_stride & (stride_gate.unsqueeze(1) > 0.0)
        valid_stride_count = torch.sum(valid_stride)
        log = getattr(env, "extras", {}).setdefault("log", {})
        log["Metrics/stride_length_m"] = torch.sum(
            self._latest_stride_length * valid_stride
        ) / torch.clamp(valid_stride_count, min=1)
        log["Metrics/stride_target_m"] = torch.sum(
            self._latest_stride_target * valid_stride
        ) / torch.clamp(valid_stride_count, min=1)
        log["Metrics/stride_success_ratio"] = torch.sum(
            (self._latest_reward >= 1.0) & valid_stride
        ) / torch.clamp(valid_stride_count, min=1)
        valid_cycle = self._has_completed_stride & (locomotion_gate.unsqueeze(1) > 0.0)
        valid_cycle_count = torch.sum(valid_cycle)
        cadence_hz = torch.where(
            valid_cycle,
            torch.reciprocal(torch.clamp(self._latest_interval, min=self._step_dt)),
            torch.zeros_like(self._latest_interval),
        )
        log["Metrics/cadence_hz"] = torch.sum(cadence_hz) / torch.clamp(valid_cycle_count, min=1)

        get_command_term = getattr(env.command_manager, "get_term", None)
        if get_command_term is not None:
            command_term = get_command_term(command_name)
            is_standing = getattr(command_term, "is_standing_env", torch.zeros_like(stride_gate, dtype=torch.bool))
            is_straight = getattr(command_term, "is_straight_env", torch.zeros_like(is_standing))
            is_turning = getattr(command_term, "is_turning_env", torch.zeros_like(is_standing))
            is_lateral = getattr(command_term, "is_lateral_env", torch.zeros_like(is_standing))
            mode_masks = {
                "forward": is_straight & (command[:, 0] > 0.0),
                "backward": is_straight & (command[:, 0] < 0.0),
                "lateral": is_lateral,
                "turning": is_turning,
                "random": ~(is_standing | is_straight | is_turning | is_lateral),
            }
            for mode_name, mode_mask in mode_masks.items():
                valid_mode_cycle = valid_cycle & mode_mask.unsqueeze(1)
                mode_count = torch.sum(valid_mode_cycle)
                log[f"Metrics/cadence_hz_{mode_name}"] = torch.sum(cadence_hz * valid_mode_cycle) / torch.clamp(
                    mode_count, min=1
                )

        both_pairs_valid = torch.all(self._has_completed_stride, dim=1)
        return torch.mean(self._latest_reward, dim=1) * both_pairs_valid * stride_gate
