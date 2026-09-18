# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward and curriculum functions adapted from the MoE-CTS RobotLab framework."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import disturbance_mdp

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, RayCaster


def _as_torch(value) -> torch.Tensor:
    """Return the torch view of an Isaac Lab proxy array or a tensor."""
    return getattr(value, "torch", value)


def recovery_gate(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    velocity_error_start: float = 0.25,
    velocity_error_full: float = 0.75,
    tilt_start: float = 0.12,
    tilt_full: float = 0.35,
    angular_velocity_start: float = 0.40,
    angular_velocity_full: float = 1.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Measure whether the robot is executing a disturbance-recovery maneuver.

    The gate is fully active during a short disturbance-marked window. Outside
    that window it activates only for substantial planar command-tracking
    error, body tilt, or roll/pitch angular velocity. This prevents ordinary
    locomotion error from permanently disabling posture and action regularizers.

    Args:
        env: The RL environment.
        command_name: Velocity command term name.
        velocity_error_start: Planar velocity error where recovery starts [m/s].
        velocity_error_full: Planar velocity error where recovery is fully active [m/s].
        tilt_start: Projected-gravity XY norm where recovery starts.
        tilt_full: Projected-gravity XY norm where recovery is fully active.
        angular_velocity_start: Roll/pitch angular speed where recovery starts [rad/s].
        angular_velocity_full: Roll/pitch angular speed where recovery is fully active [rad/s].
        asset_cfg: Robot articulation configuration.

    Returns:
        Smooth recovery activation in ``[0, 1]``, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    root_lin_vel_b = _as_torch(asset.data.root_lin_vel_b)
    root_ang_vel_b = _as_torch(asset.data.root_ang_vel_b)
    projected_gravity_b = _as_torch(asset.data.projected_gravity_b)

    def smooth_step(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        if upper <= lower:
            raise ValueError(f"Expected upper ({upper}) to be greater than lower ({lower}).")
        phase = torch.clamp((value - lower) / (upper - lower), 0.0, 1.0)
        return phase * phase * (3.0 - 2.0 * phase)

    velocity_error = torch.linalg.norm(root_lin_vel_b[:, :2] - command[:, :2], dim=1)
    tilt = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
    angular_velocity = torch.linalg.norm(root_ang_vel_b[:, :2], dim=1)
    state_gate = torch.maximum(
        smooth_step(velocity_error, velocity_error_start, velocity_error_full),
        smooth_step(tilt, tilt_start, tilt_full),
    )
    state_gate = torch.maximum(
        state_gate,
        smooth_step(angular_velocity, angular_velocity_start, angular_velocity_full),
    )
    recovery_time_left, _ = disturbance_mdp.get_recovery_window(env)
    event_gate = (recovery_time_left > 0.0).to(state_gate.dtype)
    gate = torch.maximum(state_gate, event_gate)
    extras = getattr(env, "extras", None)
    if extras is not None:
        log = extras.setdefault("log", {})
        log["Metrics/recovery_active_ratio"] = torch.mean((gate > 0.05).to(torch.float32))
        log["Metrics/recovery_event_window_ratio"] = torch.mean(event_gate)
        log["Metrics/recovery_state_trigger_ratio"] = torch.mean((state_gate > 0.05).to(torch.float32))
    return gate


def recovery_stability_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_scale: float,
    tilt_scale: float,
    angular_velocity_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize integrated velocity and attitude error while recovering.

    Args:
        env: The RL environment.
        command_name: Velocity command term name.
        velocity_scale: Planar velocity-error normalization [m/s].
        tilt_scale: Projected-gravity XY normalization.
        angular_velocity_scale: Roll/pitch angular-speed normalization [rad/s].
        asset_cfg: Robot articulation configuration.

    Returns:
        Bounded recovery-state error, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    velocity_error = torch.linalg.norm(_as_torch(asset.data.root_lin_vel_b)[:, :2] - command[:, :2], dim=1)
    tilt = torch.linalg.norm(_as_torch(asset.data.projected_gravity_b)[:, :2], dim=1)
    angular_velocity = torch.linalg.norm(_as_torch(asset.data.root_ang_vel_b)[:, :2], dim=1)
    normalized_error = (
        torch.square(velocity_error / velocity_scale)
        + torch.square(tilt / tilt_scale)
        + torch.square(angular_velocity / angular_velocity_scale)
    )
    return torch.clamp(normalized_error, max=4.0) * recovery_gate(env, command_name, asset_cfg=asset_cfg)


def _get_base_height(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Estimate base height above ground [m].

    If a height scanner is provided, this returns:
        base_height = base_z - estimated_ground_z
    Otherwise, it falls back to the world-frame root height (flat-ground assumption).

    Args:
        env: The RL environment.
        base_height_target: Target base height [m].
        asset_cfg: Scene entity configuration for the robot articulation. Defaults to 'robot'.
        sensor_cfg: Scene entity configuration for the height scanner. Defaults to None.

    Returns:
        Estimated base height above ground [m], shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2]

    if sensor_cfg is None or sensor_cfg.name not in env.scene.sensors:
        return base_z

    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    invalid = (
        torch.isnan(ray_hits_z).any(dim=1)
        | torch.isinf(ray_hits_z).any(dim=1)
        | (torch.max(torch.abs(ray_hits_z), dim=1).values > 1e6)
    )

    estimated_ground_z = torch.mean(ray_hits_z, dim=1)
    fallback_ground_z = base_z - base_height_target
    estimated_ground_z = torch.where(invalid, fallback_ground_z, estimated_ground_z)
    return base_z - estimated_ground_z


def get_base_bottom_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Measure ray clearance from the base underside to the ground [m].

    Args:
        env: The RL environment.
        target_height: Fallback clearance used when no valid ray is available [m].
        sensor_cfg: Scene entity configuration for the downward-facing base ray caster.

    Returns:
        Base-bottom clearance [m], shape ``(num_envs,)``.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_z = sensor.data.pos_w[:, 2].unsqueeze(1)
    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    clearance = sensor_z - ray_hits_z
    valid = torch.isfinite(clearance) & (clearance >= 0.0) & (clearance <= sensor.cfg.max_distance)
    valid_count = torch.sum(valid, dim=1)
    measured_height = torch.sum(torch.where(valid, clearance, 0.0), dim=1) / torch.clamp(valid_count, min=1)
    return torch.where(valid_count > 0, measured_height, target_height)


def get_base_bottom_height_from_reward(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Measure base-bottom clearance using the configured RobotLab height reward.

    This helper keeps diagnostic consumers aligned with
    :func:`base_bottom_height_l2`, :func:`base_bottom_height_range_l2`, and
    compatible range-based constraints, including their ray validity checks
    and fallback height. Environments that
    do not use a RobotLab base-bottom reward return ``None`` so callers can
    retain their task-specific measurement.

    Args:
        env: The RL environment.

    Returns:
        Base-bottom clearance [m], shape ``(num_envs,)``, or ``None`` when the
        configured reward does not use a supported base-bottom height reward.
    """
    rewards_cfg = getattr(getattr(env, "cfg", None), "rewards", None)
    reward_cfg = getattr(rewards_cfg, "base_height_l2", None)
    if reward_cfg is None:
        return None

    params = reward_cfg.params
    reward_func = getattr(reward_cfg, "func", None)
    if reward_func is base_bottom_height_l2:
        fallback_height = params["target_height"]
    elif reward_func is base_bottom_height_range_l2 or {
        "minimum_height",
        "maximum_height",
        "sensor_cfg",
    }.issubset(params):
        fallback_height = 0.5 * (params["minimum_height"] + params["maximum_height"])
    else:
        return None

    return get_base_bottom_height(
        env,
        target_height=fallback_height,
        sensor_cfg=params["sensor_cfg"],
    )


def base_bottom_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize ray-measured clearance from the base underside to the ground.

    Args:
        env: The RL environment.
        target_height: Target vertical clearance from the base underside to the ground [m].
        sensor_cfg: Scene entity configuration for the downward-facing base ray caster.

    Returns:
        Squared base-bottom clearance error [m²], shape ``(num_envs,)``.
    """
    measured_height = get_base_bottom_height(env, target_height, sensor_cfg)
    return torch.square(measured_height - target_height)


def base_bottom_height_range_l2(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_height: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize base-bottom clearance only outside an allowed height range.

    Args:
        env: The RL environment.
        minimum_height: Minimum penalty-free base-bottom clearance [m].
        maximum_height: Maximum penalty-free base-bottom clearance [m].
        sensor_cfg: Scene entity configuration for the downward-facing base ray caster.

    Returns:
        Squared distance to the nearest allowed height boundary [m²], shape
        ``(num_envs,)``. Heights inside the inclusive range receive zero.

    Raises:
        ValueError: If :paramref:`maximum_height` is not greater than
            :paramref:`minimum_height`.
    """
    if maximum_height <= minimum_height:
        raise ValueError(f"Expected maximum_height > minimum_height, got {maximum_height} and {minimum_height}.")

    fallback_height = 0.5 * (minimum_height + maximum_height)
    measured_height = get_base_bottom_height(env, fallback_height, sensor_cfg)
    below_range = torch.clamp(minimum_height - measured_height, min=0.0)
    above_range = torch.clamp(measured_height - maximum_height, min=0.0)
    return torch.square(below_range + above_range)


def joint_power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint power consumption [W], shape (num_envs,).

    Computed as: sum(|joint_vel * applied_torque|)

    Args:
        env: The RL environment.
        asset_cfg: Scene entity configuration for the articulation. Defaults to 'robot'.

    Returns:
        Total joint power consumption [W] per environment, shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def recovery_gated_action_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    recovery_scale: float = 0.30,
) -> torch.Tensor:
    """Penalize first-order action changes less strongly during recovery.

    Args:
        env: The RL environment.
        command_name: Velocity command term name.
        recovery_scale: Penalty multiplier at full recovery activation.

    Returns:
        Recovery-gated first-order action difference, shape ``(num_envs,)``.
    """
    penalty = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    gate = recovery_gate(env, command_name)
    return penalty * torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)


def action_soft_limit_l2(
    env: ManagerBasedRLEnv,
    soft_limit: float,
) -> torch.Tensor:
    """Penalize only policy actions that exceed a symmetric soft limit.

    Args:
        env: The RL environment.
        soft_limit: Penalty-free absolute policy-action limit.

    Returns:
        Squared action excess, shape ``(num_envs,)``.

    Raises:
        ValueError: If :paramref:`soft_limit` is not positive.
    """
    if soft_limit <= 0.0:
        raise ValueError(f"Expected soft_limit > 0, got {soft_limit}.")
    excess = torch.clamp(torch.abs(env.action_manager.action) - soft_limit, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def action_smoothness_l2(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    recovery_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize second-order action rate changes (action smoothness) using L2 norm.

    Computed as: sum((action - 2 * prev_action + prev_prev_action)^2)

    Args:
        env: The RL environment.
        command_name: Velocity command term name. ``None`` disables recovery gating.
        recovery_scale: Penalty multiplier at full recovery activation.

    Returns:
        Second-order action difference penalty per environment, shape (num_envs,).
    """
    action_manager = env.action_manager
    need_update = False
    if hasattr(action_manager, "prev_prev_action"):
        prev_prev_action = action_manager.prev_prev_action
    else:
        if not hasattr(env, "_prev_prev_action") or env._prev_prev_action.shape != action_manager.action.shape:
            env._prev_prev_action = torch.zeros_like(action_manager.action)
            env._action_smoothness_step_count = -1

        if hasattr(env, "episode_length_buf"):
            newly_reset = env.episode_length_buf <= 1
            if newly_reset.any():
                env._prev_prev_action[newly_reset] = 0.0

        prev_prev_action = env._prev_prev_action
        if env.common_step_counter != env._action_smoothness_step_count:
            env._action_smoothness_step_count = env.common_step_counter
            need_update = True

    diff = torch.square(action_manager.action - 2 * action_manager.prev_action + prev_prev_action)
    # Ignore initial steps where history is zero
    diff = diff * (action_manager.prev_action[:, :] != 0)
    diff = diff * (prev_prev_action[:, :] != 0)
    penalty = torch.sum(diff, dim=1)

    if not hasattr(action_manager, "prev_prev_action") and need_update:
        env._prev_prev_action = action_manager.prev_action.clone()

    if command_name is not None:
        gate = recovery_gate(env, command_name)
        penalty = penalty * torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)
    return penalty


def feet_regulation(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    recovery_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize fast horizontal foot motion near the ground to prevent scuffing/dragging.

    Feet that are close to the ground receive a larger penalty for lateral motion,
    while lifted feet during swing are penalized exponentially less.

    Args:
        env: The RL environment.
        base_height_target: Target base height [m].
        asset_cfg: Scene entity configuration for feet bodies. Defaults to 'robot'.
        sensor_cfg: Scene entity configuration for height scanner sensor. Defaults to None.
        command_name: Velocity command term name. ``None`` disables recovery gating.
        recovery_scale: Penalty multiplier at full recovery activation.

    Returns:
        Foot scuffing penalty per environment, shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_ids = asset_cfg.body_ids

    feet_pos_w = asset.data.body_pos_w[:, feet_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    feet_xy_vel_w = asset.data.body_lin_vel_w[:, feet_ids, :2]
    base_height = _get_base_height(
        env, base_height_target, asset_cfg=SceneEntityCfg(asset_cfg.name), sensor_cfg=sensor_cfg
    )

    gravity_w = torch.tensor(env.sim.cfg.gravity, device=env.device, dtype=feet_pos_w.dtype)
    down_w = gravity_w / torch.norm(gravity_w)

    delta_feet_w = feet_pos_w - base_pos_w
    feet2base_height = torch.sum(delta_feet_w * down_w.view(1, 1, 3), dim=-1)
    feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)

    penalty = (feet_xy_vel_w.pow(2).sum(dim=-1) * torch.exp(-feet_height / (0.050 * base_height_target))).sum(dim=-1)
    if command_name is not None:
        gate = recovery_gate(env, command_name)
        penalty = penalty * torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)
    return penalty


def feet_lateral_separation_l2(
    env: ManagerBasedRLEnv,
    minimum_separation: float,
    separation_margin: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize front or rear feet that become too close laterally.

    The signed left-to-right separations are measured along the robot heading
    frame's lateral axis. This makes the term independent of world heading and
    also penalizes left/right foot crossing. The configured body order must be
    ``FL, FR, RL, RR``.

    Args:
        env: The RL environment.
        minimum_separation: Penalty-free lateral separation [m].
        separation_margin: Separation shortfall [m] that produces a unit penalty.
        asset_cfg: Scene entity configuration containing the four feet in
            ``FL, FR, RL, RR`` order.

    Returns:
        Squared normalized separation shortfall per environment, shape
        ``(num_envs,)``.

    Raises:
        ValueError: If the separation parameters are invalid or four feet were
            not selected.
    """
    if minimum_separation < 0.0:
        raise ValueError(f"Expected non-negative minimum_separation, got {minimum_separation}.")
    if separation_margin <= 0.0:
        raise ValueError(f"Expected positive separation_margin, got {separation_margin}.")

    asset: RigidObject = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w
    root_quat_w = asset.data.root_quat_w
    if hasattr(feet_pos_w, "torch"):
        feet_pos_w = feet_pos_w.torch
    if hasattr(root_quat_w, "torch"):
        root_quat_w = root_quat_w.torch
    feet_pos_w = feet_pos_w[:, asset_cfg.body_ids, :]
    if feet_pos_w.shape[1] != 4:
        raise ValueError(
            f"feet_lateral_separation_l2 requires four feet in FL, FR, RL, RR order; received {feet_pos_w.shape[1]}."
        )

    pair_delta_w = torch.stack(
        (feet_pos_w[:, 0] - feet_pos_w[:, 1], feet_pos_w[:, 2] - feet_pos_w[:, 3]),
        dim=1,
    )
    yaw_quat_w = math_utils.yaw_quat(root_quat_w)
    pair_yaw_quat_w = yaw_quat_w.unsqueeze(1).expand(-1, 2, -1)
    pair_delta_b = math_utils.quat_apply_inverse(pair_yaw_quat_w, pair_delta_w)
    lateral_separation = pair_delta_b[:, :, 1]

    normalized_shortfall = torch.clamp(
        (minimum_separation - lateral_separation) / separation_margin,
        min=0.0,
    )
    penalty = torch.sum(torch.square(normalized_shortfall), dim=1)

    extras = getattr(env, "extras", None)
    if extras is not None:
        log = extras.setdefault("log", {})
        log["Metrics/front_feet_lateral_separation_m"] = torch.mean(lateral_separation[:, 0])
        log["Metrics/rear_feet_lateral_separation_m"] = torch.mean(lateral_separation[:, 1])
        log["Metrics/feet_lateral_separation_violation_ratio"] = torch.mean(
            (lateral_separation < minimum_separation).to(lateral_separation.dtype)
        )

    return penalty


def hip_pos_penalty_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 1.0,
    command_threshold: float = 0.1,
    recovery_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize hip-roll deviation during standing and pure straight motion.

    Lateral, turning, and general combined-motion modes receive no penalty so
    the policy can use hip roll freely when those motions require it.

    Args:
        env: The RL environment.
        command_name: Name of the velocity command term.
        asset_cfg: Scene entity configuration containing hip joints.
        stand_still_scale: Scale factor applied to the penalty while standing. Defaults to 1.0.
        command_threshold: Legacy command threshold retained for API compatibility. Defaults to 0.1.
        recovery_scale: Posture-penalty multiplier at full recovery activation.

    Returns:
        L1 joint position error penalty per environment [rad], shape (num_envs,).
    """
    del command_threshold
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    joint_deviation = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
        ord=1,
    )
    zero_penalty = torch.zeros_like(joint_deviation)
    penalty = torch.where(
        command_term.is_standing_env,
        stand_still_scale * joint_deviation,
        torch.where(command_term.is_straight_env, joint_deviation, zero_penalty),
    )
    if recovery_scale == 1.0:
        return penalty
    gate = recovery_gate(env, command_name, asset_cfg=SceneEntityCfg(asset_cfg.name))
    recovery_relief = torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)
    return penalty * recovery_relief


def joint_pos_penalty_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 1.0,
    velocity_threshold: float = 0.1,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint position deviation [rad] from default on thigh/calf joints.

    Args:
        env: The RL environment.
        command_name: Name of the velocity command term.
        asset_cfg: Scene entity configuration containing thigh and calf joints.
        stand_still_scale: Scale factor when standing still. Defaults to 1.0.
        velocity_threshold: Threshold on body velocity to consider robot in motion [m/s]. Defaults to 0.1.
        command_threshold: Threshold on command norm to consider robot commanded [m/s]. Defaults to 0.1.

    Returns:
        L1 joint position error penalty per environment [rad], shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
        ord=1,
    )
    return torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )


def stand_still_joint_pos_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    yaw_scale: float = 0.25,
    recovery_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize joint deviation from the default pose only at near-zero commands.

    Args:
        env: The RL environment.
        command_name: Name of the velocity command term.
        asset_cfg: Articulation and joints whose pose is regulated.
        command_threshold: Equivalent command speed below which the robot is standing [m/s].
        yaw_scale: Length scale converting yaw rate to equivalent foot speed [m].
        recovery_scale: Penalty multiplier at full recovery activation.

    Returns:
        Standing joint-position deviation [rad], shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    equivalent_speed = torch.sqrt(
        torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2])
    )
    joint_deviation = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    if recovery_scale == 1.0:
        return joint_deviation * (equivalent_speed < command_threshold)
    gate = recovery_gate(env, command_name, asset_cfg=SceneEntityCfg(asset_cfg.name))
    scale = torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)
    return joint_deviation * (equivalent_speed < command_threshold) * scale


def gradual_reward_weight_modification(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_weight: float,
    final_weight: float,
    start_it: int,
    end_it: int,
    num_steps_per_iter: int = 24,
) -> float:
    """Curriculum term that gradually linearly scales a reward weight between iterations.

    Args:
        env: The RL environment.
        env_ids: Affected environment indices.
        term_name: Name of the reward term to adjust.
        initial_weight: Initial reward term weight.
        final_weight: Final reward term weight.
        start_it: Iteration at which the ramp begins.
        end_it: Iteration at which the ramp completes.
        num_steps_per_iter: Number of environment steps per training iteration. Defaults to 24.

    Returns:
        The updated reward weight.
    """
    del env_ids  # Shared across all environments
    current_it = env.common_step_counter // num_steps_per_iter
    if current_it < start_it:
        new_weight = initial_weight
    elif current_it >= end_it:
        new_weight = final_weight
    else:
        progress = (current_it - start_it) / max(end_it - start_it, 1)
        new_weight = progress * (final_weight - initial_weight) + initial_weight

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return new_weight


class FeetImpactVelocityPenalty(ManagerTermBase):
    """Penalize vertical downward velocity of feet at the moment of touchdown.

    This penalizes the square of the downward velocity (v_z < 0) just prior to
    ground contact, encouraging the policy to decelerate feet before touchdown
    (soft touchdown).

    Args:
        cfg: Reward term configuration.
        env: The RL environment.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize foot velocity and contact history buffers."""
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._foot_body_ids = asset_cfg.body_ids
        self._sensor_body_ids = sensor_cfg.body_ids

        # Foot vertical velocity from previous step [m/s]
        self._prev_foot_vel_z = self.asset.data.body_lin_vel_w.torch[:, self._foot_body_ids, 2].clone()
        # Foot contact state from previous step (bool)
        contact_time = self.contact_sensor.data.current_contact_time.torch[:, self._sensor_body_ids]
        self._prev_contact = contact_time > 0.0

        # Persistent metrics buffers for logging
        device = self._prev_foot_vel_z.device
        self._last_td_speed_mean = torch.tensor(0.0, device=device)
        self._last_td_speed_max = torch.tensor(0.0, device=device)
        self._last_td_softness_ratio = torch.tensor(1.0, device=device)
        self._last_td_force_mean = torch.tensor(0.0, device=device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset history buffers for selected environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._prev_foot_vel_z[env_ids] = self.asset.data.body_lin_vel_w.torch[env_ids][:, self._foot_body_ids, 2]
        contact_time = self.contact_sensor.data.current_contact_time.torch[env_ids][:, self._sensor_body_ids]
        self._prev_contact[env_ids] = contact_time > 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        velocity_threshold: float = 0.2,
        command_name: str | None = None,
        recovery_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute touchdown impact velocity penalty.

        Args:
            env: The RL environment.
            asset_cfg: Scene entity configuration for robot feet.
            sensor_cfg: Scene entity configuration for contact sensor.
            velocity_threshold: Downward speed threshold [m/s] below which touchdown is considered soft.
            command_name: Velocity command term name. ``None`` disables recovery gating.
            recovery_scale: Penalty multiplier at full recovery activation.

        Returns:
            Squared excess touchdown vertical velocity per environment [(m/s)^2], shape (num_envs,).
        """
        del asset_cfg, sensor_cfg
        contact_time = self.contact_sensor.data.current_contact_time.torch[:, self._sensor_body_ids]
        current_contact = contact_time > 0.0

        # Touchdown transition: airborne on previous step, now in contact
        touchdown = current_contact & ~self._prev_contact

        # Downward speed just prior to touchdown [m/s] (positive when moving downwards)
        downward_speed = -self._prev_foot_vel_z
        violation = torch.clamp(downward_speed - velocity_threshold, min=0.0)
        penalty_per_foot = torch.square(violation)

        # Apply only to feet that touched down on this step
        touchdown_penalty = torch.where(touchdown, penalty_per_foot, torch.zeros_like(penalty_per_foot))
        total_penalty = torch.sum(touchdown_penalty, dim=1)
        if command_name is not None:
            gate = recovery_gate(env, command_name)
            total_penalty = total_penalty * torch.lerp(
                torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate
            )

        # Update buffers for next step
        self._prev_foot_vel_z = self.asset.data.body_lin_vel_w.torch[:, self._foot_body_ids, 2].clone()
        self._prev_contact = current_contact.clone()

        extras = getattr(env, "extras", None)
        if extras is not None:
            log = extras.setdefault("log", {})
            if torch.any(touchdown):
                td_speeds = torch.clamp(downward_speed[touchdown], min=0.0)
                self._last_td_speed_mean = torch.mean(td_speeds)
                self._last_td_speed_max = torch.max(td_speeds)
                self._last_td_softness_ratio = torch.mean((td_speeds <= velocity_threshold).to(torch.float32))
                if (
                    hasattr(self.contact_sensor.data, "net_forces_w")
                    and self.contact_sensor.data.net_forces_w is not None
                ):
                    net_forces = getattr(
                        self.contact_sensor.data.net_forces_w,
                        "torch",
                        self.contact_sensor.data.net_forces_w,
                    )
                    td_forces = torch.linalg.norm(net_forces[:, self._sensor_body_ids], dim=-1)[touchdown]
                    self._last_td_force_mean = torch.mean(td_forces)

            log["Metrics/feet_touchdown_speed_mean_mps"] = self._last_td_speed_mean
            log["Metrics/feet_touchdown_speed_max_mps"] = self._last_td_speed_max
            log["Metrics/feet_touchdown_softness_ratio"] = self._last_td_softness_ratio
            log["Metrics/feet_touchdown_force_mean_n"] = self._last_td_force_mean

        return total_penalty


class CapturePointFootPlacementReward(ManagerTermBase):
    """Reward recovery touchdowns near the velocity-error capture point."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize foot-contact history and resolved body indices."""
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._foot_body_ids = asset_cfg.body_ids
        self._sensor_body_ids = sensor_cfg.body_ids
        contact_time = _as_torch(self.contact_sensor.data.current_contact_time)
        self._previous_contact = contact_time[:, self._sensor_body_ids] > 0.0
        _, event_id = disturbance_mdp.get_recovery_window(env)
        self._rewarded_event_id = event_id.clone()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset touchdown history for selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        contact_time = _as_torch(self.contact_sensor.data.current_contact_time)
        self._previous_contact[env_ids] = contact_time[env_ids][:, self._sensor_body_ids] > 0.0
        _, event_id = disturbance_mdp.get_recovery_window(self._env)
        self._rewarded_event_id[env_ids] = event_id[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        nominal_height: float,
        placement_std: float,
    ) -> torch.Tensor:
        """Compute a touchdown reward from capture-point placement accuracy.

        Args:
            env: The RL environment.
            command_name: Velocity command term name.
            asset_cfg: Robot feet body selection.
            sensor_cfg: Foot contact-sensor selection.
            nominal_height: Nominal center-of-mass height used by the linear inverted-pendulum model [m].
            placement_std: Exponential capture-point distance scale [m].

        Returns:
            Recovery touchdown reward in ``[0, 1]``, shape ``(num_envs,)``.
        """
        del asset_cfg, sensor_cfg
        if nominal_height <= 0.0 or placement_std <= 0.0:
            raise ValueError("nominal_height and placement_std must be positive.")

        contact_time = _as_torch(self.contact_sensor.data.current_contact_time)
        contact = contact_time[:, self._sensor_body_ids] > 0.0
        touchdown = contact & ~self._previous_contact
        self._previous_contact = contact.clone()

        command = env.command_manager.get_command(command_name)
        command_direction_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=1)
        root_quat_w = _as_torch(self.asset.data.root_quat_w)
        command_velocity_w = math_utils.quat_apply_yaw(root_quat_w, command_direction_b)[:, :2]
        velocity_error_w = _as_torch(self.asset.data.root_lin_vel_w)[:, :2] - command_velocity_w
        natural_frequency = (9.81 / nominal_height) ** 0.5
        capture_point_w = _as_torch(self.asset.data.root_pos_w)[:, :2] + velocity_error_w / natural_frequency
        foot_pos_w = _as_torch(self.asset.data.body_pos_w)[:, self._foot_body_ids, :2]
        placement_error = torch.linalg.norm(foot_pos_w - capture_point_w.unsqueeze(1), dim=2)
        placement_reward = torch.exp(-torch.square(placement_error / placement_std))

        recovery_time_left, event_id = disturbance_mdp.get_recovery_window(env)
        new_event = (event_id != self._rewarded_event_id) & (recovery_time_left > 0.0)
        touchdown_count = torch.sum(touchdown, dim=1)
        rewarded_touchdown = new_event & (touchdown_count > 0)
        reward = torch.sum(placement_reward * touchdown, dim=1) / torch.clamp(touchdown_count, min=1)
        reward = torch.where(rewarded_touchdown, reward, 0.0)
        self._rewarded_event_id[rewarded_touchdown] = event_id[rewarded_touchdown]
        extras = getattr(env, "extras", None)
        if extras is not None:
            extras.setdefault("log", {})["Metrics/recovery_capture_rewarded_ratio"] = torch.mean(
                rewarded_touchdown.to(torch.float32)
            )
        return reward * recovery_gate(env, command_name)


feet_impact_vel_l2 = FeetImpactVelocityPenalty
"""Alias for FeetImpactVelocityPenalty."""


def feet_contact_force_limit(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 140.0,
    command_name: str | None = None,
    recovery_scale: float = 1.0,
    normalize: bool = False,
) -> torch.Tensor:
    """Penalize feet contact forces exceeding a maximum threshold.

    Args:
        env: The RL environment.
        sensor_cfg: Scene entity configuration for the contact sensor.
        threshold: Maximum allowed net contact force per foot [N]. Defaults to 140.0.
        command_name: Velocity command term name. ``None`` disables recovery gating.
        recovery_scale: Penalty multiplier at full recovery activation.
        normalize: Normalize force excess by :paramref:`threshold`.

    Returns:
        Excess contact force violation summed across feet [N], shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = getattr(
        contact_sensor.data.net_forces_w_history,
        "torch",
        contact_sensor.data.net_forces_w_history,
    )
    max_forces = torch.max(
        torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0]
    violation = torch.clamp(max_forces - threshold, min=0.0)
    if normalize:
        violation = violation / threshold

    extras = getattr(env, "extras", None)
    if extras is not None:
        log = extras.setdefault("log", {})
        log["Metrics/feet_contact_force_peak_n"] = torch.max(max_forces)
        log["Metrics/feet_contact_force_violation_ratio"] = torch.mean((max_forces > threshold).to(torch.float32))
        half_idx = max(1, max_forces.shape[1] // 2)
        front_force = torch.mean(max_forces[:, :half_idx])
        rear_force = torch.mean(max_forces[:, half_idx:])
        log["Metrics/feet_front_vs_rear_force_ratio"] = rear_force / torch.clamp(front_force, min=1.0)

    penalty = torch.sum(violation, dim=1)
    if command_name is not None:
        gate = recovery_gate(env, command_name)
        penalty = penalty * torch.lerp(torch.ones_like(gate), torch.full_like(gate, recovery_scale), gate)
    return penalty
