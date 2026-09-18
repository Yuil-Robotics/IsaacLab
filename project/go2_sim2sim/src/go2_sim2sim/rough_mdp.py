# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP reward terms for Unitree Go2 rough-terrain locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import GaitReward

from .robotlab_rewards import recovery_gate

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor


def _smooth_step(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Map a tensor smoothly from zero to one over the given interval."""
    if upper <= lower:
        raise ValueError(f"Expected upper ({upper}) to be greater than lower ({lower}).")
    phase = torch.clamp((value - lower) / (upper - lower), 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def _all_feet_air_gate(
    current_air_time: torch.Tensor,
    grace_time: float,
    full_time: float,
) -> torch.Tensor:
    """Measure sustained flight from the shortest current foot air time."""
    minimum_air_time = torch.min(current_air_time, dim=1).values
    return _smooth_step(minimum_air_time, grace_time, full_time)


def _locomotion_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    motion_start: float,
    motion_full: float,
    yaw_scale: float,
    recovery_velocity_threshold: float,
) -> torch.Tensor:
    """Compute a smooth locomotion gate from commanded and measured planar speed."""
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    command_speed = torch.sqrt(torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2]))
    body_speed = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    command_gate = _smooth_step(command_speed, motion_start, motion_full)
    recovery_gate = _smooth_step(
        body_speed,
        0.5 * recovery_velocity_threshold,
        recovery_velocity_threshold,
    )
    return torch.maximum(command_gate, recovery_gate)


def rough_air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    motion_start: float,
    motion_full: float,
    yaw_scale: float,
    recovery_velocity_threshold: float,
    command_name: str = "base_velocity",
    flight_grace_time: float = 0.04,
    flight_full_time: float = 0.10,
) -> torch.Tensor:
    """Blend stance support and rhythmic stepping based on locomotion demand.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot articulation.
        sensor_cfg: Scene entity configuration for the foot contact sensor.
        mode_time: Desired maximum air or contact phase duration [s].
        motion_start: Equivalent command speed where stepping starts to activate [m/s].
        motion_full: Equivalent command speed where stepping is fully active [m/s].
        yaw_scale: Length scale that converts yaw rate to equivalent foot speed [m].
        recovery_velocity_threshold: Planar body speed that fully activates recovery stepping [m/s].
        command_name: Name of the velocity command term.
        flight_grace_time: All-foot air time below which flight is ignored [s].
        flight_full_time: All-foot air time where stepping reward is fully suppressed [s].

    Returns:
        Per-environment reward tensor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    phase_time = torch.maximum(current_air_time, current_contact_time)
    stepping_reward = torch.where(phase_time < mode_time, torch.clamp(phase_time, max=mode_time), 0.0)
    flight_gate = _all_feet_air_gate(current_air_time, flight_grace_time, flight_full_time).unsqueeze(1)
    stepping_reward = stepping_reward * (1.0 - flight_gate)
    stance_reward = torch.clamp(current_contact_time - current_air_time, -mode_time, mode_time)
    motion_gate = _locomotion_gate(
        env,
        command_name,
        motion_start,
        motion_full,
        yaw_scale,
        recovery_velocity_threshold,
    ).unsqueeze(1)
    reward = torch.lerp(stance_reward, stepping_reward, motion_gate)
    return torch.sum(reward, dim=1)


def flight_phase_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    grace_time: float,
    full_time: float,
) -> torch.Tensor:
    """Penalize sustained phases where every selected foot is airborne.

    A short grace interval avoids reacting to one-frame contact transitions or
    brief loss of contact at terrain edges. The result increases smoothly to one
    so that a negative reward weight directly penalizes four-foot hopping.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the foot contact sensor.
        grace_time: All-foot air time below which no penalty is applied [s].
        full_time: All-foot air time where the penalty reaches one [s].

    Returns:
        Per-environment penalty tensor in the range ``[0, 1]``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    return _all_feet_air_gate(current_air_time, grace_time, full_time)


class RoughTerrainFootClearanceReward(ManagerTermBase):
    """Reward swing-foot clearance relative to its most recent support height.

    The reference height is updated while each foot is in contact. This makes the
    reward invariant to terrain and environment world-height offsets without
    exposing privileged terrain measurements to the policy.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize contact-height references.

        Args:
            cfg: Reward term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._foot_body_ids = asset_cfg.body_ids
        self._sensor_body_ids = sensor_cfg.body_ids
        self._last_contact_height = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset support-height references to the current foot heights.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            self._last_contact_height[:] = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2]
        else:
            current_height = self.asset.data.body_pos_w.torch[env_ids][:, self._foot_body_ids, 2]
            self._last_contact_height[env_ids] = current_height

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        target_height: float,
        std: float,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        recovery_velocity_threshold: float,
        command_name: str = "base_velocity",
        flight_grace_time: float = 0.04,
        flight_full_time: float = 0.10,
    ) -> torch.Tensor:
        """Compute terrain-offset-invariant swing clearance reward.

        Args:
            env: The environment instance.
            asset_cfg: Scene entity configuration for the robot feet.
            sensor_cfg: Scene entity configuration for the foot contact sensor.
            target_height: Desired swing clearance above the last support height [m].
            std: Exponential tracking width [m].
            motion_start: Equivalent command speed where the reward starts to activate [m/s].
            motion_full: Equivalent command speed where the reward is fully active [m/s].
            yaw_scale: Length scale that converts yaw rate to equivalent foot speed [m].
            recovery_velocity_threshold: Planar body speed that fully activates recovery stepping [m/s].
            command_name: Name of the velocity command term.
            flight_grace_time: All-foot air time below which flight is ignored [s].
            flight_full_time: All-foot air time where clearance reward is fully suppressed [s].

        Returns:
            Per-environment reward tensor in the range ``[0, 1]``.
        """
        del asset_cfg, sensor_cfg
        foot_height = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2]
        contact_time = self.contact_sensor.data.current_contact_time.torch[:, self._sensor_body_ids]
        air_time = self.contact_sensor.data.current_air_time.torch[:, self._sensor_body_ids]
        in_contact = contact_time > 0.0
        self._last_contact_height = torch.where(in_contact, foot_height, self._last_contact_height)

        clearance = torch.clamp(foot_height - self._last_contact_height, min=0.0)
        tracking_reward = torch.exp(-torch.square(clearance - target_height) / std**2)
        in_swing = air_time > 0.0
        swing_count = torch.sum(in_swing, dim=1)
        swing_reward = torch.sum(tracking_reward * in_swing, dim=1) / torch.clamp(swing_count, min=1)
        swing_reward = torch.where(swing_count > 0, swing_reward, 0.0)
        flight_gate = _all_feet_air_gate(air_time, flight_grace_time, flight_full_time)
        motion_gate = _locomotion_gate(
            env,
            command_name,
            motion_start,
            motion_full,
            yaw_scale,
            recovery_velocity_threshold,
        )
        return swing_reward * motion_gate * (1.0 - flight_gate)


class SwingFootApexHeightPenalty(ManagerTermBase):
    """Penalize completed swing trajectories whose apex is outside a height band.

    Unlike a dense clearance target, this term evaluates the maximum clearance
    reached during a complete swing. The foot is therefore free to descend
    naturally before touchdown instead of being rewarded for hovering near a
    target height throughout the entire swing.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-foot support-height and completed-swing state.

        Args:
            cfg: Reward term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if self.contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        self._foot_body_ids = asset_cfg.body_ids
        self._sensor_body_ids = sensor_cfg.body_ids

        foot_height = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2]
        contact = self.contact_sensor.data.current_contact_time.torch[:, self._sensor_body_ids] > 0.0
        self._last_contact_height = foot_height.clone()
        self._previous_contact = contact.clone()
        self._swing_apex = torch.zeros_like(foot_height)
        self._latest_apex = torch.zeros_like(foot_height)
        self._latest_bound_error = torch.zeros_like(foot_height)
        self._has_completed_swing = torch.zeros_like(contact)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset swing histories for selected environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        foot_height = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2]
        contact = self.contact_sensor.data.current_contact_time.torch[:, self._sensor_body_ids] > 0.0
        self._last_contact_height[env_ids] = foot_height[env_ids]
        self._previous_contact[env_ids] = contact[env_ids]
        self._swing_apex[env_ids] = 0.0
        self._latest_apex[env_ids] = 0.0
        self._latest_bound_error[env_ids] = 0.0
        self._has_completed_swing[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        minimum_clearance: float,
        maximum_clearance: float,
        lower_margin: float,
        upper_margin: float,
        symmetry_tolerance: float,
        symmetry_margin: float,
        symmetry_scale: float,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        recovery_velocity_threshold: float,
        command_name: str = "base_velocity",
        recovery_scale: float = 1.0,
    ) -> torch.Tensor:
        """Evaluate swing-apex bounds and inter-foot apex symmetry.

        Args:
            env: The environment instance.
            asset_cfg: Robot foot-body configuration.
            sensor_cfg: Foot contact-sensor configuration.
            minimum_clearance: Minimum penalty-free swing apex [m].
            maximum_clearance: Maximum penalty-free swing apex [m].
            lower_margin: Clearance deficit producing maximum lower penalty [m].
            upper_margin: Clearance excess producing maximum upper penalty [m].
            symmetry_tolerance: Penalty-free apex spread across feet [m].
            symmetry_margin: Additional apex spread producing maximum symmetry penalty [m].
            symmetry_scale: Relative scale of the apex-symmetry penalty.
            motion_start: Equivalent command speed where the term starts [m/s].
            motion_full: Equivalent command speed where the term is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to equivalent foot speed [m].
            recovery_velocity_threshold: Body speed fully activating recovery stepping [m/s].
            command_name: Velocity command term name.
            recovery_scale: Penalty multiplier at full disturbance recovery activation.

        Returns:
            Bounded swing-apex penalty per environment.
        """
        del asset_cfg, sensor_cfg
        if maximum_clearance <= minimum_clearance:
            raise ValueError("Expected maximum_clearance to exceed minimum_clearance.")
        if min(lower_margin, upper_margin, symmetry_margin, symmetry_scale) <= 0.0:
            raise ValueError("Expected positive apex margins and symmetry_scale.")
        if symmetry_tolerance < 0.0:
            raise ValueError("Expected symmetry_tolerance to be non-negative.")

        foot_height = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2]
        contact_time = self.contact_sensor.data.current_contact_time.torch[:, self._sensor_body_ids]
        in_contact = contact_time > 0.0
        liftoff = ~in_contact & self._previous_contact
        touchdown = in_contact & ~self._previous_contact

        clearance = torch.clamp(foot_height - self._last_contact_height, min=0.0)
        self._swing_apex = torch.where(liftoff, clearance, self._swing_apex)
        self._swing_apex = torch.where(
            ~in_contact,
            torch.maximum(self._swing_apex, clearance),
            self._swing_apex,
        )

        lower_error = torch.clamp(
            (minimum_clearance - self._swing_apex) / lower_margin,
            min=0.0,
            max=1.0,
        )
        upper_error = torch.clamp(
            (self._swing_apex - maximum_clearance) / upper_margin,
            min=0.0,
            max=1.0,
        )
        completed_bound_error = torch.square(lower_error) + torch.square(upper_error)
        self._latest_apex = torch.where(touchdown, self._swing_apex, self._latest_apex)
        self._latest_bound_error = torch.where(touchdown, completed_bound_error, self._latest_bound_error)
        self._has_completed_swing |= touchdown

        self._last_contact_height = torch.where(in_contact, foot_height, self._last_contact_height)
        self._swing_apex = torch.where(touchdown, torch.zeros_like(self._swing_apex), self._swing_apex)
        self._previous_contact = in_contact.clone()

        valid = self._has_completed_swing
        valid_count = torch.sum(valid, dim=1)
        valid_float = valid.to(foot_height.dtype)
        mean_apex = torch.sum(self._latest_apex * valid_float, dim=1) / torch.clamp(valid_count, min=1)
        apex_spread = torch.max(
            torch.where(valid, torch.abs(self._latest_apex - mean_apex.unsqueeze(1)), 0.0),
            dim=1,
        ).values
        symmetry_error = torch.clamp(
            (apex_spread - symmetry_tolerance) / symmetry_margin,
            min=0.0,
            max=1.0,
        )
        symmetry_error = torch.square(symmetry_error) * (valid_count >= 2)
        bound_error = torch.max(torch.where(valid, self._latest_bound_error, 0.0), dim=1).values

        motion_gate = _locomotion_gate(
            env,
            command_name,
            motion_start,
            motion_full,
            yaw_scale,
            recovery_velocity_threshold,
        )
        penalty = (bound_error + symmetry_scale * symmetry_error) * motion_gate
        if recovery_scale != 1.0:
            gate = recovery_gate(env, command_name)
            penalty = penalty * torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)

        extras = getattr(env, "extras", None)
        if extras is not None:
            log = extras.setdefault("log", {})
            completed_count = torch.sum(valid_float)
            log["Metrics/swing_apex_mean_m"] = torch.sum(self._latest_apex * valid_float) / torch.clamp(
                completed_count, min=1
            )
            log["Metrics/swing_apex_spread_m"] = torch.mean(apex_spread)
        return penalty


class TrotDiagonalPairTimingPenalty(ManagerTermBase):
    """Penalize phase-duration differences inside synchronized trot pairs."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Resolve synchronized foot-pair sensor indices.

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
            raise ValueError("Trot pair timing requires exactly two pairs containing two feet each.")
        self._pair_sensor_ids = [self.contact_sensor.find_sensors(pair)[0] for pair in pair_names]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
        allowed_time_error: float,
        error_margin: float,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        recovery_velocity_threshold: float,
        recovery_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute the worst normalized timing error within either diagonal pair.

        Args:
            env: The environment instance.
            command_name: Velocity command term name.
            synced_feet_pair_names: Names of the two synchronized foot pairs.
            sensor_cfg: Foot contact-sensor configuration.
            allowed_time_error: Penalty-free pair timing difference [s].
            error_margin: Additional timing error producing maximum penalty [s].
            motion_start: Equivalent command speed where the term starts [m/s].
            motion_full: Equivalent command speed where the term is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to equivalent foot speed [m].
            recovery_velocity_threshold: Body speed fully activating recovery stepping [m/s].
            recovery_scale: Penalty multiplier at full disturbance recovery activation.

        Returns:
            Pair timing penalty in ``[0, 1]`` per environment.
        """
        del synced_feet_pair_names, sensor_cfg
        if allowed_time_error < 0.0 or error_margin <= 0.0:
            raise ValueError("Expected non-negative allowed_time_error and positive error_margin.")
        air_time = self.contact_sensor.data.current_air_time.torch
        contact_time = self.contact_sensor.data.current_contact_time.torch
        pair_errors = []
        for foot_0, foot_1 in self._pair_sensor_ids:
            air_error = torch.abs(air_time[:, foot_0] - air_time[:, foot_1])
            contact_error = torch.abs(contact_time[:, foot_0] - contact_time[:, foot_1])
            pair_errors.append(torch.maximum(air_error, contact_error))
        worst_error = torch.max(torch.stack(pair_errors, dim=1), dim=1).values
        normalized_error = torch.clamp(
            (worst_error - allowed_time_error) / error_margin,
            min=0.0,
            max=1.0,
        )
        motion_gate = _locomotion_gate(
            env,
            command_name,
            motion_start,
            motion_full,
            yaw_scale,
            recovery_velocity_threshold,
        )
        extras = getattr(env, "extras", None)
        if extras is not None:
            extras.setdefault("log", {})["Metrics/trot_pair_timing_error_s"] = torch.mean(worst_error)
        penalty = torch.square(normalized_error) * motion_gate
        if recovery_scale != 1.0:
            gate = recovery_gate(env, command_name)
            penalty = penalty * torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)
        return penalty


def foot_air_time_symmetry_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    allowed_time_spread: float,
    spread_margin: float,
    motion_start: float,
    motion_full: float,
    yaw_scale: float,
    recovery_velocity_threshold: float,
    recovery_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize differences between the latest completed foot swing times.

    Args:
        env: The environment instance.
        command_name: Velocity command term name.
        sensor_cfg: Foot contact-sensor configuration.
        allowed_time_spread: Penalty-free max-to-min completed air-time spread [s].
        spread_margin: Additional spread producing maximum penalty [s].
        motion_start: Equivalent command speed where the term starts [m/s].
        motion_full: Equivalent command speed where the term is fully active [m/s].
        yaw_scale: Length scale converting yaw rate to equivalent foot speed [m].
        recovery_velocity_threshold: Body speed fully activating recovery stepping [m/s].
        recovery_scale: Penalty multiplier at full disturbance recovery activation.

    Returns:
        Swing-time asymmetry penalty in ``[0, 1]`` per environment.
    """
    if allowed_time_spread < 0.0 or spread_margin <= 0.0:
        raise ValueError("Expected non-negative allowed_time_spread and positive spread_margin.")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    valid = torch.all(last_air_time > 0.0, dim=1)
    spread = torch.max(last_air_time, dim=1).values - torch.min(last_air_time, dim=1).values
    normalized_error = torch.clamp(
        (spread - allowed_time_spread) / spread_margin,
        min=0.0,
        max=1.0,
    )
    motion_gate = _locomotion_gate(
        env,
        command_name,
        motion_start,
        motion_full,
        yaw_scale,
        recovery_velocity_threshold,
    )
    penalty = torch.square(normalized_error) * valid * motion_gate
    if recovery_scale != 1.0:
        gate = recovery_gate(env, command_name)
        penalty = penalty * torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)
    return penalty


class NearGroundFootDescentVelocityPenalty(ManagerTermBase):
    """Penalize fast downward swing-foot motion shortly before touchdown."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-foot support-height references.

        Args:
            cfg: Reward term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._foot_body_ids = asset_cfg.body_ids
        self._sensor_body_ids = sensor_cfg.body_ids
        self._last_contact_height = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2].clone()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset support-height references for selected environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        foot_height = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2]
        self._last_contact_height[env_ids] = foot_height[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        activation_clearance: float,
        downward_speed_threshold: float,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        recovery_velocity_threshold: float,
        command_name: str = "base_velocity",
    ) -> torch.Tensor:
        """Compute near-ground downward-speed excess.

        Args:
            env: The environment instance.
            asset_cfg: Robot foot-body configuration.
            sensor_cfg: Foot contact-sensor configuration.
            activation_clearance: Clearance below which descent shaping activates [m].
            downward_speed_threshold: Penalty-free downward foot speed [m/s].
            motion_start: Equivalent command speed where the term starts [m/s].
            motion_full: Equivalent command speed where the term is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to equivalent foot speed [m].
            recovery_velocity_threshold: Body speed fully activating recovery stepping [m/s].
            command_name: Velocity command term name.

        Returns:
            Summed squared downward-speed excess [(m/s)^2] per environment.
        """
        del asset_cfg, sensor_cfg
        if activation_clearance <= 0.0 or downward_speed_threshold < 0.0:
            raise ValueError("Expected positive activation_clearance and non-negative speed threshold.")
        foot_height = self.asset.data.body_pos_w.torch[:, self._foot_body_ids, 2]
        foot_velocity_z = self.asset.data.body_lin_vel_w.torch[:, self._foot_body_ids, 2]
        contact_time = self.contact_sensor.data.current_contact_time.torch[:, self._sensor_body_ids]
        air_time = self.contact_sensor.data.current_air_time.torch[:, self._sensor_body_ids]
        in_contact = contact_time > 0.0
        self._last_contact_height = torch.where(in_contact, foot_height, self._last_contact_height)
        clearance = torch.clamp(foot_height - self._last_contact_height, min=0.0)
        proximity = torch.clamp((activation_clearance - clearance) / activation_clearance, min=0.0, max=1.0)
        downward_speed = torch.clamp(-foot_velocity_z - downward_speed_threshold, min=0.0)
        penalty = torch.sum(torch.square(downward_speed) * torch.square(proximity) * (air_time > 0.0), dim=1)
        motion_gate = _locomotion_gate(
            env,
            command_name,
            motion_start,
            motion_full,
            yaw_scale,
            recovery_velocity_threshold,
        )
        return penalty * motion_gate


class RoughTerrainGaitReward(GaitReward):
    """Apply diagonal gait timing according to translational and yaw locomotion demand."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float = 0.0,
        flight_grace_time: float = 0.04,
        flight_full_time: float = 0.10,
    ) -> torch.Tensor:
        """Compute gait timing reward with a smooth planar-command gate.

        Args:
            env: The environment instance.
            std: Exponential timing-error scale [s²].
            max_err: Maximum timing error before clipping [s].
            velocity_threshold: Body velocity threshold used by the base gait reward [m/s].
            synced_feet_pair_names: Names of the two synchronized foot pairs.
            asset_cfg: Scene entity configuration for the robot articulation.
            sensor_cfg: Scene entity configuration for the contact sensor.
            motion_start: Equivalent command speed where gait shaping starts [m/s].
            motion_full: Equivalent command speed where gait shaping is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to equivalent foot speed [m].
            flight_grace_time: All-foot air time below which flight is ignored [s].
            flight_full_time: All-foot air time where gait reward is fully suppressed [s].

        Returns:
            Per-environment gait reward tensor in the range ``[0, 1]``.
        """
        timing_reward = super().__call__(
            env,
            std,
            max_err,
            velocity_threshold,
            synced_feet_pair_names,
            asset_cfg,
            sensor_cfg,
        )
        command = env.command_manager.get_command("base_velocity")
        gait_command_speed = torch.sqrt(
            torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2])
        )
        gait_foot_ids = self.synced_feet_pairs[0] + self.synced_feet_pairs[1]
        current_air_time = self.contact_sensor.data.current_air_time.torch[:, gait_foot_ids]
        flight_gate = _all_feet_air_gate(current_air_time, flight_grace_time, flight_full_time)
        return timing_reward * _smooth_step(gait_command_speed, motion_start, motion_full) * (1.0 - flight_gate)


class TrotCadenceUniformityPenalty(ManagerTermBase):
    """Penalize changes between consecutive diagonal-pair touchdown intervals."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize pair-contact history and per-environment interval state.

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
            raise ValueError("Cadence uniformity requires exactly two pairs containing two feet each.")
        self._pair_sensor_ids = [self.contact_sensor.find_sensors(pair)[0] for pair in pair_names]

        interval_tolerance = cfg.params["interval_tolerance"]
        min_interval = cfg.params["min_touchdown_interval"]
        max_interval = cfg.params["max_touchdown_interval"]
        if interval_tolerance <= 0.0:
            raise ValueError(f"Expected interval_tolerance > 0, got {interval_tolerance}.")
        if min_interval < 0.0 or max_interval <= min_interval:
            raise ValueError(
                f"Expected 0 <= min_touchdown_interval < max_touchdown_interval, got {min_interval} and {max_interval}."
            )

        self._step_dt = env.step_dt
        self._elapsed_since_touchdown = torch.zeros(env.num_envs, device=env.device)
        self._previous_interval = torch.zeros(env.num_envs, device=env.device)
        self._latest_error = torch.zeros(env.num_envs, device=env.device)
        self._has_last_touchdown = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._has_previous_interval = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._previous_pair_contact = self._get_pair_contact().clone()
        self._command_name = cfg.params["command_name"]
        self._previous_command = env.command_manager.get_command(self._command_name).clone()

    def _get_pair_contact(self) -> torch.Tensor:
        """Return whether both feet in each diagonal pair are currently in contact."""
        contact_time = self.contact_sensor.data.current_contact_time.torch
        return torch.stack(
            [torch.all(contact_time[:, pair_ids] > 0.0, dim=1) for pair_ids in self._pair_sensor_ids],
            dim=1,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset touchdown interval history for selected environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._elapsed_since_touchdown[env_ids] = 0.0
        self._previous_interval[env_ids] = 0.0
        self._latest_error[env_ids] = 0.0
        self._has_last_touchdown[env_ids] = False
        self._has_previous_interval[env_ids] = False
        self._previous_pair_contact[env_ids] = self._get_pair_contact()[env_ids]
        self._previous_command[env_ids] = self._env.command_manager.get_command(self._command_name)[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        interval_tolerance: float,
        min_touchdown_interval: float,
        max_touchdown_interval: float,
        command_change_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute normalized cadence non-uniformity from successive touchdown intervals.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            synced_feet_pair_names: Names of the two diagonal foot pairs.
            sensor_cfg: Scene entity configuration for the contact sensor.
            motion_start: Equivalent command speed where cadence tracking starts [m/s].
            motion_full: Equivalent command speed where cadence tracking is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to equivalent foot speed [m].
            interval_tolerance: Touchdown interval change producing unit pre-clamp error [s].
            min_touchdown_interval: Minimum accepted touchdown interval [s].
            max_touchdown_interval: Maximum interval included in the comparison [s].
            command_change_threshold: Equivalent command change that resets interval history [m/s].

        Returns:
            Cadence non-uniformity in the range ``[0, 1]``, shape ``(num_envs,)``.
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
        pair_touchdown = pair_contact & ~self._previous_pair_contact
        touchdown = torch.any(pair_touchdown, dim=1) & ~reset_state

        self._elapsed_since_touchdown = torch.where(
            reset_state,
            torch.zeros_like(self._elapsed_since_touchdown),
            self._elapsed_since_touchdown + self._step_dt,
        )
        accepted_touchdown = touchdown & (
            ~self._has_last_touchdown | (self._elapsed_since_touchdown >= min_touchdown_interval)
        )
        completed_interval = accepted_touchdown & self._has_last_touchdown
        current_interval = torch.clamp(self._elapsed_since_touchdown, max=max_touchdown_interval)
        compare_interval = completed_interval & self._has_previous_interval
        normalized_error = torch.clamp(
            torch.square((current_interval - self._previous_interval) / interval_tolerance),
            max=1.0,
        )
        self._latest_error = torch.where(compare_interval, normalized_error, self._latest_error)
        self._previous_interval = torch.where(completed_interval, current_interval, self._previous_interval)
        self._has_previous_interval |= completed_interval
        self._has_last_touchdown |= accepted_touchdown
        self._elapsed_since_touchdown = torch.where(
            accepted_touchdown,
            torch.zeros_like(self._elapsed_since_touchdown),
            self._elapsed_since_touchdown,
        )

        self._elapsed_since_touchdown[reset_state] = 0.0
        self._previous_interval[reset_state] = 0.0
        self._latest_error[reset_state] = 0.0
        self._has_last_touchdown[reset_state] = False
        self._has_previous_interval[reset_state] = False
        self._previous_pair_contact = pair_contact
        self._previous_command = command.clone()
        return self._latest_error * motion_gate


class TrotCadenceTargetPenalty(ManagerTermBase):
    """Penalize deviation from a target trot-cycle frequency.

    The period is measured independently between consecutive touchdowns of the
    same diagonal pair. This prevents a policy from satisfying the target with
    repeated events from only one pair or with unsynchronized four-beat steps.
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
            raise ValueError("Cadence targeting requires exactly two pairs containing two feet each.")
        self._pair_sensor_ids = [self.contact_sensor.find_sensors(pair)[0] for pair in pair_names]

        target_frequency = cfg.params["target_cycle_frequency"]
        interval_tolerance = cfg.params["interval_tolerance"]
        min_interval = cfg.params["min_touchdown_interval"]
        max_interval = cfg.params["max_touchdown_interval"]
        if target_frequency <= 0.0:
            raise ValueError(f"Expected target_cycle_frequency > 0, got {target_frequency}.")
        if interval_tolerance <= 0.0:
            raise ValueError(f"Expected interval_tolerance > 0, got {interval_tolerance}.")
        if min_interval < 0.0 or max_interval <= min_interval:
            raise ValueError(
                f"Expected 0 <= min_touchdown_interval < max_touchdown_interval, got {min_interval} and {max_interval}."
            )

        target_interval = 1.0 / target_frequency
        if not min_interval <= target_interval <= max_interval:
            raise ValueError(
                "Expected the target same-pair touchdown interval to lie within "
                f"[{min_interval}, {max_interval}], got {target_interval}."
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
        """Return whether both feet in each diagonal pair are currently in contact."""
        contact_time = self.contact_sensor.data.current_contact_time.torch
        return torch.stack(
            [torch.all(contact_time[:, pair_ids] > 0.0, dim=1) for pair_ids in self._pair_sensor_ids],
            dim=1,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset target-cadence history for selected environments.

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
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        target_cycle_frequency: float,
        interval_tolerance: float,
        min_touchdown_interval: float,
        max_touchdown_interval: float,
        command_change_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute target-cadence error from each pair's stride period.

        Args:
            env: The environment instance.
            command_name: Name of the velocity command term.
            synced_feet_pair_names: Names of the two diagonal foot pairs.
            sensor_cfg: Scene entity configuration for the contact sensor.
            motion_start: Equivalent command speed where cadence tracking starts [m/s].
            motion_full: Equivalent command speed where cadence tracking is fully active [m/s].
            yaw_scale: Length scale converting yaw rate to equivalent foot speed [m].
            target_cycle_frequency: Desired complete trot cycles per second [Hz].
            interval_tolerance: Exponential scale for touchdown interval error [s].
            min_touchdown_interval: Minimum accepted same-pair touchdown interval [s].
            max_touchdown_interval: Maximum measured same-pair touchdown interval [s].
            command_change_threshold: Equivalent command change that resets interval history [m/s].

        Returns:
            Smooth bounded target-cadence error in the range ``[0, 1]``, shape ``(num_envs,)``.
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
        pair_touchdown = pair_contact & ~self._previous_pair_contact
        touchdown = pair_touchdown & ~reset_state.unsqueeze(1)

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
        target_interval = 1.0 / target_cycle_frequency
        scaled_interval_error = (current_interval - target_interval) / interval_tolerance
        normalized_error = 1.0 - torch.exp(-0.5 * torch.square(scaled_interval_error))
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
                    mode_cadence_hz = torch.sum(cadence_hz * valid_mode_interval) / torch.clamp(
                        mode_count,
                        min=1,
                    )
                    log[f"Metrics/cadence_hz_{mode_name}"] = mode_cadence_hz
        return torch.max(self._latest_error, dim=1).values * motion_gate


class CommandSpeedTrotCadenceTargetPenalty(TrotCadenceTargetPenalty):
    """Penalize deviation from a command-speed-dependent trot cadence.

    Translational velocity commands are combined with yaw rate converted to an
    equivalent tangential foot speed. The resulting magnitude is mapped
    smoothly onto a bounded target cadence, while zero commands leave recovery
    stepping unconstrained.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize dynamic target-cadence tracking.

        Args:
            cfg: Reward term configuration.
            env: The RL environment instance.
        """
        ManagerTermBase.__init__(self, cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if self.contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        pair_names = cfg.params["synced_feet_pair_names"]
        if len(pair_names) != 2 or any(len(pair) != 2 for pair in pair_names):
            raise ValueError("Cadence targeting requires exactly two pairs containing two feet each.")
        self._pair_sensor_ids = [self.contact_sensor.find_sensors(pair)[0] for pair in pair_names]

        minimum_frequency = cfg.params["minimum_cycle_frequency"]
        maximum_frequency = cfg.params["maximum_cycle_frequency"]
        maximum_frequency_speed = cfg.params["maximum_frequency_speed"]
        interval_tolerance = cfg.params["interval_tolerance"]
        min_interval = cfg.params["min_touchdown_interval"]
        max_interval = cfg.params["max_touchdown_interval"]
        if minimum_frequency <= 0.0 or maximum_frequency < minimum_frequency:
            raise ValueError(
                "Expected 0 < minimum_cycle_frequency <= maximum_cycle_frequency, "
                f"got {minimum_frequency} and {maximum_frequency}."
            )
        if maximum_frequency_speed <= 0.0:
            raise ValueError(f"Expected maximum_frequency_speed > 0, got {maximum_frequency_speed}.")
        if interval_tolerance <= 0.0:
            raise ValueError(f"Expected interval_tolerance > 0, got {interval_tolerance}.")
        if min_interval < 0.0 or max_interval <= min_interval:
            raise ValueError(
                f"Expected 0 <= min_touchdown_interval < max_touchdown_interval, got {min_interval} and {max_interval}."
            )
        minimum_target_interval = 1.0 / maximum_frequency
        maximum_target_interval = 1.0 / minimum_frequency
        if min_interval > minimum_target_interval or maximum_target_interval > max_interval:
            raise ValueError(
                "Expected the dynamic target-period range to lie within "
                f"[{min_interval}, {max_interval}], got "
                f"[{minimum_target_interval}, {maximum_target_interval}]."
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

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
        motion_start: float,
        motion_full: float,
        yaw_scale: float,
        minimum_cycle_frequency: float,
        maximum_cycle_frequency: float,
        maximum_frequency_speed: float,
        interval_tolerance: float,
        min_touchdown_interval: float,
        max_touchdown_interval: float,
        command_change_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute cadence error against a target derived from command magnitude.

        Args:
            env: The RL environment instance.
            command_name: Name of the velocity command term.
            synced_feet_pair_names: Names of the two diagonal foot pairs.
            sensor_cfg: Scene entity configuration for the contact sensor.
            motion_start: Equivalent command speed where cadence shaping starts [m/s].
            motion_full: Equivalent command speed where cadence shaping is fully active [m/s].
            yaw_scale: Radius converting yaw rate to equivalent foot speed [m].
            minimum_cycle_frequency: Target cadence at zero equivalent speed [Hz].
            maximum_cycle_frequency: Maximum target cadence [Hz].
            maximum_frequency_speed: Equivalent speed reaching maximum cadence [m/s].
            interval_tolerance: Exponential touchdown-period error scale [s].
            min_touchdown_interval: Minimum accepted same-pair touchdown interval [s].
            max_touchdown_interval: Maximum measured same-pair touchdown interval [s].
            command_change_threshold: Equivalent command change that resets history [m/s].

        Returns:
            Smooth bounded target-cadence error in ``[0, 1]``, shape ``(num_envs,)``.
        """
        del synced_feet_pair_names, sensor_cfg
        command = env.command_manager.get_command(command_name)
        equivalent_speed = torch.sqrt(
            torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2])
        )
        target_phase = _smooth_step(equivalent_speed, 0.0, maximum_frequency_speed)
        target_frequency = minimum_cycle_frequency + target_phase * (maximum_cycle_frequency - minimum_cycle_frequency)
        target_interval = torch.reciprocal(target_frequency).unsqueeze(1)
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
        scaled_interval_error = (current_interval - target_interval) / interval_tolerance
        normalized_error = 1.0 - torch.exp(-0.5 * torch.square(scaled_interval_error))
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
        moving = motion_gate > 0.0
        moving_count = torch.sum(moving)
        extras = getattr(env, "extras", None)
        if extras is not None:
            log = extras.setdefault("log", {})
            log["Metrics/cadence_hz"] = torch.sum(cadence_hz) / torch.clamp(valid_count, min=1)
            log["Metrics/cadence_target_hz"] = torch.sum(target_frequency * moving) / torch.clamp(moving_count, min=1)
            cadence_error_hz = torch.abs(cadence_hz - target_frequency.unsqueeze(1))
            log["Metrics/cadence_target_error_hz"] = torch.sum(cadence_error_hz * valid_interval) / torch.clamp(
                valid_count, min=1
            )
        return torch.max(self._latest_error, dim=1).values * motion_gate


def straight_motion_hip_deviation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
    command_name: str = "base_velocity",
    vy_threshold: float = 0.1,
    wz_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize hip joint deviation from default pose when moving forward/backward.

    This term is conditionally active only when commanded lateral velocity |v_y| < vy_threshold [m/s]
    and commanded yaw rate |w_z| < wz_threshold [rad/s], enforcing straight-line sagittal leg stepping
    and eliminating lateral hip flaring/waddling during forward/backward locomotion.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the articulation.
        command_name: The name of the base velocity command term.
        vy_threshold: Lateral velocity threshold [m/s] below which straight motion is active.
        wz_threshold: Yaw angular velocity threshold [rad/s] below which straight motion is active.

    Returns:
        Penalty tensor [N] with values >= 0.0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    is_straight = (torch.abs(cmd[:, 1]) < vy_threshold) & (torch.abs(cmd[:, 2]) < wz_threshold)

    joint_dev = torch.abs(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    deviation = torch.sum(joint_dev, dim=1)
    return is_straight.float() * deviation


def straight_motion_hip_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
    command_name: str = "base_velocity",
    vy_threshold: float = 0.1,
    wz_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize hip joint velocities when moving forward/backward.

    Suppresses lateral hip oscillations and foot flaring during swing phase on straight locomotion.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the articulation.
        command_name: The name of the base velocity command term.
        vy_threshold: Lateral velocity threshold [m/s] below which straight motion is active.
        wz_threshold: Yaw angular velocity threshold [rad/s] below which straight motion is active.

    Returns:
        Penalty tensor [N] with values >= 0.0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    is_straight = (torch.abs(cmd[:, 1]) < vy_threshold) & (torch.abs(cmd[:, 2]) < wz_threshold)

    joint_vel_sq = torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids])
    vel_penalty = torch.sum(joint_vel_sq, dim=1)
    return is_straight.float() * vel_penalty


def get_ground_relative_base_height_tensor(
    scene,
    sensor_cfg: SceneEntityCfg | None = None,
    sensor_name: str = "height_scanner",
    asset_name: str = "robot",
    default_target_height: float = 0.33,
) -> torch.Tensor:
    """Compute robot base height [m] relative to ground directly beneath base_link (x, y) [N]."""
    try:
        robot = scene[asset_name]
    except (TypeError, KeyError, AttributeError):
        robot = getattr(scene, asset_name)
    base_z = robot.data.root_pos_w.torch[:, 2]

    if sensor_cfg is not None:
        if hasattr(scene, "sensors") and sensor_cfg.name in scene.sensors:
            sensor = scene.sensors[sensor_cfg.name]
            if hasattr(sensor.data, "ray_hits_w"):
                ray_hits_z = sensor.data.ray_hits_w.torch[..., 2]
                is_finite = torch.isfinite(ray_hits_z)
                if torch.all(is_finite):
                    ground_z = torch.mean(ray_hits_z, dim=1)
                else:
                    valid_counts = torch.sum(is_finite, dim=1)
                    safe_hits = torch.where(is_finite, ray_hits_z, torch.zeros_like(ray_hits_z))
                    fallback_z = base_z - default_target_height
                    ground_z = torch.where(
                        valid_counts > 0,
                        torch.sum(safe_hits, dim=1) / torch.clamp(valid_counts, min=1),
                        fallback_z,
                    )
                return base_z - ground_z
            else:
                feet_z = robot.data.body_pos_w.torch[:, sensor_cfg.body_ids, 2]
                ground_z = torch.min(feet_z, dim=1).values
                return base_z - ground_z
        elif hasattr(scene, "articulations") and sensor_cfg.name in scene.articulations:
            feet_z = scene[sensor_cfg.name].data.body_pos_w.torch[:, sensor_cfg.body_ids, 2]
            ground_z = torch.min(feet_z, dim=1).values
            return base_z - ground_z
        else:
            return base_z
    elif hasattr(scene, "sensors") and sensor_name in scene.sensors:
        sensor = scene.sensors[sensor_name]
        if hasattr(sensor.data, "ray_hits_w"):
            ray_hits_z = sensor.data.ray_hits_w.torch[..., 2]
            is_finite = torch.isfinite(ray_hits_z)
            if torch.all(is_finite):
                ground_z = torch.mean(ray_hits_z, dim=1)
            else:
                valid_counts = torch.sum(is_finite, dim=1)
                safe_hits = torch.where(is_finite, ray_hits_z, torch.zeros_like(ray_hits_z))
                fallback_z = base_z - default_target_height
                ground_z = torch.where(
                    valid_counts > 0,
                    torch.sum(safe_hits, dim=1) / torch.clamp(valid_counts, min=1),
                    fallback_z,
                )
            return base_z - ground_z

    if hasattr(robot, "find_bodies"):
        foot_ids = robot.find_bodies(".*_foot")[0]
        if len(foot_ids) > 0:
            feet_z = robot.data.body_pos_w.torch[:, foot_ids, 2]
            ground_z = torch.min(feet_z, dim=1).values
            return base_z - ground_z
    if hasattr(scene, "env_origins"):
        ground_z = scene.env_origins[:, 2]
        return base_z - ground_z
    return base_z


def get_ground_relative_base_height(
    scene,
    env_idx: int = 0,
    sensor_name: str = "height_scanner",
    asset_name: str = "robot",
) -> float:
    """Compute scalar base height [m] relative to ground for environment `env_idx`."""
    try:
        robot = scene[asset_name]
    except (TypeError, KeyError, AttributeError):
        robot = getattr(scene, asset_name)
    base_z = robot.data.root_pos_w.torch[env_idx, 2].item()
    if hasattr(scene, "sensors") and sensor_name in scene.sensors:
        scanner = scene.sensors[sensor_name]
        if hasattr(scanner.data, "ray_hits_w"):
            hits_z = scanner.data.ray_hits_w.torch[env_idx, :, 2]
            finite = hits_z[torch.isfinite(hits_z)]
            if len(finite) > 0:
                ground_z = torch.mean(finite).item()
                return base_z - ground_z
    if hasattr(robot, "find_bodies"):
        foot_ids = robot.find_bodies(".*_foot")[0]
        if len(foot_ids) > 0:
            feet_z = robot.data.body_pos_w.torch[env_idx, foot_ids, 2]
            ground_z = torch.min(feet_z).item()
            return base_z - ground_z
    if hasattr(scene, "env_origins"):
        ground_z = scene.env_origins[env_idx, 2].item()
        return base_z - ground_z
    return base_z


def get_ground_relative_foot_heights(
    scene,
    env_idx: int = 0,
    sensor_name: str = "height_scanner",
    asset_name: str = "robot",
    foot_names: Sequence[str] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
) -> list[float]:
    """Compute scalar foot clearances [m] relative to ground for environment `env_idx`.

    Args:
        scene: Scene instance containing robot and optional sensors.
        env_idx: Index of the environment.
        sensor_name: Name of the ray caster height scanner sensor.
        asset_name: Name of the robot asset in the scene.
        foot_names: Names of foot bodies to query in order.

    Returns:
        List of ground-relative vertical clearances [m] for each foot.
    """
    try:
        robot = scene[asset_name]
    except (TypeError, KeyError, AttributeError):
        robot = getattr(scene, asset_name)

    ground_z = 0.0
    if hasattr(scene, "sensors") and sensor_name in scene.sensors:
        scanner = scene.sensors[sensor_name]
        if hasattr(scanner.data, "ray_hits_w"):
            hits_z = scanner.data.ray_hits_w.torch[env_idx, :, 2]
            finite = hits_z[torch.isfinite(hits_z)]
            if len(finite) > 0:
                ground_z = torch.mean(finite).item()
    elif hasattr(scene, "env_origins"):
        ground_z = scene.env_origins[env_idx, 2].item()

    foot_heights: list[float] = []
    for name in foot_names:
        if hasattr(robot, "find_bodies"):
            matched = robot.find_bodies(name)[0]
            if len(matched) > 0:
                foot_z = robot.data.body_pos_w.torch[env_idx, matched[0], 2].item()
                foot_heights.append(foot_z - ground_z)
                continue
        foot_heights.append(0.0)
    return foot_heights


def rough_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize base height error from target height [m] using L2 squared kernel.

    On flat or rough terrain, evaluates base height relative to the ground beneath the robot.
    If feet bodies are provided, ground elevation is estimated from the lowest stance feet
    touching the ground. Otherwise, it falls back to RayCaster, env_origins Z, or world frame.

    Args:
        env: The environment instance.
        target_height: Desired height of the robot base link above the ground [m].
        asset_cfg: Scene entity configuration for the robot articulation.
        sensor_cfg: Scene entity configuration for the feet bodies or height sensor.

    Returns:
        Per-environment squared height error tensor [N].
    """
    current_height = get_ground_relative_base_height_tensor(
        env.scene,
        sensor_cfg=sensor_cfg,
        asset_name=asset_cfg.name,
        default_target_height=target_height,
    )
    return torch.square(current_height - target_height)
