# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-success reward support for the Robotis HX5 cube task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import rotation_distance, scale, unscale


@torch.jit.script
def compute_joint_target_limits(
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    safety_margin_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return command limits inset from both physical joint limits."""
    joint_ranges = upper_limits - lower_limits
    margins = safety_margin_fraction * joint_ranges
    return lower_limits + margins, upper_limits - margins


@torch.jit.script
def limit_joint_target_rate(
    desired_targets: torch.Tensor,
    previous_targets: torch.Tensor,
    max_delta: float,
) -> torch.Tensor:
    """Limit joint-position target changes per policy step [rad]."""
    return torch.clamp(desired_targets, previous_targets - max_delta, previous_targets + max_delta)


@torch.jit.script
def update_cumulative_success_counts(
    cumulative_successes: torch.Tensor,
    cumulative_episodes: torch.Tensor,
    completed_episode_successes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update completed-episode success totals and return their cumulative rate."""
    cumulative_successes = cumulative_successes + completed_episode_successes.sum()
    cumulative_episodes = cumulative_episodes + completed_episode_successes.numel()
    success_rate = cumulative_successes.float() / torch.clamp_min(cumulative_episodes, 1).float()
    return cumulative_successes, cumulative_episodes, success_rate


@torch.jit.script
def apply_cube_pose_observation_error(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    position_bias: torch.Tensor,
    position_noise: torch.Tensor,
    rotation_bias: torch.Tensor,
    rotation_noise: torch.Tensor,
    previous_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a coherent position and SO(3) error to a cube pose observation.

    Args:
        object_pos: True cube positions [m], shape ``[num_envs, 3]``.
        object_rot: True cube orientations as ``xyzw`` quaternions.
        position_bias: Persistent position errors [m], shape ``[num_envs, 3]``.
        position_noise: Per-measurement position errors [m], shape ``[num_envs, 3]``.
        rotation_bias: Persistent orientation errors as ``xyzw`` quaternions.
        rotation_noise: Per-measurement orientation errors as ``xyzw`` quaternions.
        previous_rotation: Previous observed orientations as ``xyzw`` quaternions.

    Returns:
        Noisy cube positions [m] and normalized, sign-continuous ``xyzw`` quaternions.
    """
    observed_pos = object_pos + position_bias + position_noise
    observed_rot = quat_mul(rotation_noise, quat_mul(rotation_bias, object_rot))
    observed_rot = observed_rot / torch.clamp_min(torch.linalg.vector_norm(observed_rot, dim=-1, keepdim=True), 1.0e-8)
    flip_sign = torch.sum(observed_rot * previous_rotation, dim=-1, keepdim=True) < 0.0
    observed_rot = torch.where(flip_sign, -observed_rot, observed_rot)
    return observed_pos, observed_rot


@torch.jit.script
def reconstruct_cube_velocity_from_pose(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    previous_pos: torch.Tensor,
    previous_rot: torch.Tensor,
    history_valid: torch.Tensor,
    previous_linear_velocity: torch.Tensor,
    previous_angular_velocity: torch.Tensor,
    dt: torch.Tensor,
    low_pass_alpha: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct filtered cube velocities from consecutive pose observations.

    Args:
        object_pos: Current observed cube positions [m], shape ``[num_envs, 3]``.
        object_rot: Current observed cube orientations as ``xyzw`` quaternions.
        previous_pos: Previous observed cube positions [m], shape ``[num_envs, 3]``.
        previous_rot: Previous observed cube orientations as ``xyzw`` quaternions.
        history_valid: Whether each environment has a previous pose measurement.
        previous_linear_velocity: Previous filtered linear velocities [m/s].
        previous_angular_velocity: Previous filtered angular velocities [rad/s].
        dt: Time between pose measurements [s], shape ``[num_envs]``.
        low_pass_alpha: Weight assigned to the newest raw velocity.
        max_linear_speed: Maximum reconstructed linear speed [m/s].
        max_angular_speed: Maximum reconstructed angular speed [rad/s].

    Returns:
        Filtered linear velocities [m/s] and angular velocities [rad/s].
    """
    safe_dt = torch.clamp_min(dt, 1.0e-8)
    raw_linear_velocity = (object_pos - previous_pos) / safe_dt.unsqueeze(-1)

    rotation_delta = quat_mul(object_rot, quat_conjugate(previous_rot))
    rotation_delta = rotation_delta / torch.clamp_min(
        torch.linalg.vector_norm(rotation_delta, dim=-1, keepdim=True), 1.0e-8
    )
    rotation_delta = torch.where((rotation_delta[:, 3] < 0.0).unsqueeze(-1), -rotation_delta, rotation_delta)
    rotation_xyz = rotation_delta[:, :3]
    sin_half_angle = torch.linalg.vector_norm(rotation_xyz, dim=-1)
    angle = 2.0 * torch.atan2(sin_half_angle, torch.clamp_min(rotation_delta[:, 3], 1.0e-8))
    axis = rotation_xyz / torch.clamp_min(sin_half_angle.unsqueeze(-1), 1.0e-8)
    raw_angular_velocity = axis * (angle / safe_dt).unsqueeze(-1)
    raw_angular_velocity = torch.where(
        (sin_half_angle > 1.0e-7).unsqueeze(-1), raw_angular_velocity, torch.zeros_like(raw_angular_velocity)
    )

    linear_norm = torch.linalg.vector_norm(raw_linear_velocity, dim=-1, keepdim=True)
    linear_scale = torch.clamp(max_linear_speed / torch.clamp_min(linear_norm, 1.0e-8), max=1.0)
    raw_linear_velocity = raw_linear_velocity * linear_scale
    angular_norm = torch.linalg.vector_norm(raw_angular_velocity, dim=-1, keepdim=True)
    angular_scale = torch.clamp(max_angular_speed / torch.clamp_min(angular_norm, 1.0e-8), max=1.0)
    raw_angular_velocity = raw_angular_velocity * angular_scale

    linear_velocity = low_pass_alpha * raw_linear_velocity + (1.0 - low_pass_alpha) * previous_linear_velocity
    angular_velocity = low_pass_alpha * raw_angular_velocity + (1.0 - low_pass_alpha) * previous_angular_velocity
    linear_velocity = torch.where(history_valid.unsqueeze(-1), linear_velocity, torch.zeros_like(linear_velocity))
    angular_velocity = torch.where(history_valid.unsqueeze(-1), angular_velocity, torch.zeros_like(angular_velocity))
    return linear_velocity, angular_velocity


@torch.jit.script
def update_cube_pose_sample_hold_countdown(
    steps_until_update: torch.Tensor, sampled_intervals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance per-environment pose sample-and-hold update countdowns.

    Args:
        steps_until_update: Remaining policy steps before each new measurement.
        sampled_intervals: Newly sampled measurement intervals in policy steps.

    Returns:
        A measurement-update mask and the countdown values for the next policy step.
    """
    update_mask = steps_until_update <= 0
    next_countdown = torch.where(
        update_mask,
        sampled_intervals - 1,
        torch.clamp_min(steps_until_update - 1, 0),
    )
    return update_mask, next_countdown


@torch.jit.script
def compute_stable_success_termination(
    rot_dist: torch.Tensor,
    success_hold_count: torch.Tensor,
    success_tolerance: float,
    success_hold_steps: int,
) -> torch.Tensor:
    """Return environments that complete their required hold on this step."""
    return (torch.abs(rot_dist) <= success_tolerance) & (success_hold_count + 1 >= success_hold_steps)


@torch.jit.script
def compute_stable_success_rewards(
    reset_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    success_hold_count: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    success_hold_steps: int,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    success_requires_not_fallen: bool,
    av_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rewards after requiring consecutive in-tolerance policy steps."""
    goal_dist = torch.linalg.vector_norm(object_pos - target_pos, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    action_penalty = torch.sum(actions**2, dim=-1) * action_penalty_scale
    reward = dist_rew + rot_rew + action_penalty

    within_tolerance = torch.abs(rot_dist) <= success_tolerance
    success_hold_count = torch.where(
        within_tolerance,
        success_hold_count + 1,
        torch.zeros_like(success_hold_count),
    )
    fallen = goal_dist >= fall_dist
    goal_resets = success_hold_count >= success_hold_steps
    if success_requires_not_fallen:
        goal_resets = goal_resets & ~fallen
    successes = successes + goal_resets.to(successes.dtype)
    reward = torch.where(goal_resets, reward + reach_goal_bonus, reward)
    success_hold_count = torch.where(goal_resets, torch.zeros_like(success_hold_count), success_hold_count)

    reward = torch.where(fallen, reward + fall_penalty, reward)
    resets = fallen | (reset_buf != 0)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.to(successes.dtype))
    updated_consecutive_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, updated_consecutive_successes, success_hold_count


@torch.jit.script
def compute_single_goal_progress_rewards(
    reset_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    success_hold_count: torch.Tensor,
    best_rot_dist: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    object_angvel: torch.Tensor,
    hand_dof_vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    actions: torch.Tensor,
    raw_actions: torch.Tensor,
    previous_actions: torch.Tensor,
    action_history_valid: torch.Tensor,
    rotation_progress_scale: float,
    orientation_error_penalty_scale: float,
    distance_safe_radius: float,
    distance_penalty_scale: float,
    action_rate_penalty_scale: float,
    action_saturation_threshold: float,
    action_saturation_penalty_scale: float,
    joint_velocity_penalty_scale: float,
    success_tolerance: float,
    success_max_object_angvel: float,
    success_hold_steps: int,
    hold_reward_scale: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    time_penalty: float,
    timeout_penalty: float,
    av_factor: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute progress-based rewards for the terminal single-goal task."""
    goal_dist = torch.linalg.vector_norm(object_pos - target_pos, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)
    fallen = goal_dist >= fall_dist

    rotation_progress = torch.clamp_min(best_rot_dist - rot_dist, 0.0)
    rotation_progress_reward = rotation_progress * rotation_progress_scale
    best_rot_dist = torch.minimum(best_rot_dist, rot_dist)
    orientation_error_penalty = rot_dist * orientation_error_penalty_scale
    distance_excess = torch.clamp_min(goal_dist - distance_safe_radius, 0.0)
    distance_reward = distance_excess**2 * distance_penalty_scale

    action_delta = actions - previous_actions
    action_rate_penalty = torch.sum(action_delta**2, dim=-1) * action_rate_penalty_scale
    action_rate_penalty = torch.where(action_history_valid, action_rate_penalty, torch.zeros_like(action_rate_penalty))
    saturation_excess = torch.clamp_min(torch.abs(raw_actions) - action_saturation_threshold, 0.0)
    action_saturation_penalty = torch.sum(saturation_excess**2, dim=-1) * action_saturation_penalty_scale
    joint_velocity_penalty = torch.sum(hand_dof_vel**2, dim=-1) * joint_velocity_penalty_scale

    # Keep the legacy angular-velocity inputs in the function contract for
    # checkpoint/task compatibility, but success depends only on remaining
    # inside the orientation tolerance for the configured hold duration.
    _ = object_angvel
    _ = success_max_object_angvel
    stable_at_goal = torch.abs(rot_dist) <= success_tolerance
    success_hold_count = torch.where(
        stable_at_goal,
        success_hold_count + 1,
        torch.zeros_like(success_hold_count),
    )
    hold_reward = stable_at_goal.to(rot_dist.dtype) * hold_reward_scale
    goal_success = (success_hold_count >= success_hold_steps) & ~fallen
    successes = successes + goal_success.to(successes.dtype)
    success_bonus = goal_success.to(rot_dist.dtype) * reach_goal_bonus
    success_hold_count = torch.where(goal_success, torch.zeros_like(success_hold_count), success_hold_count)

    fall_reward = fallen.to(rot_dist.dtype) * fall_penalty
    step_time_penalty = torch.full_like(rot_dist, time_penalty)
    timeout_failure = (reset_buf != 0) & ~fallen & ~goal_success
    timeout_failure_penalty = timeout_failure.to(rot_dist.dtype) * timeout_penalty
    reward = (
        rotation_progress_reward
        + orientation_error_penalty
        + distance_reward
        + action_rate_penalty
        + action_saturation_penalty
        + joint_velocity_penalty
        + hold_reward
        + success_bonus
        + fall_reward
        + step_time_penalty
        + timeout_failure_penalty
    )

    resets = fallen | (reset_buf != 0)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.to(successes.dtype))
    updated_consecutive_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return (
        reward,
        goal_success,
        successes,
        updated_consecutive_successes,
        success_hold_count,
        best_rot_dist,
        rotation_progress_reward,
        orientation_error_penalty,
        distance_reward,
        action_rate_penalty,
        action_saturation_penalty,
        joint_velocity_penalty,
        hold_reward,
        success_bonus,
        fall_reward,
        timeout_failure_penalty,
    )


class StableSuccessRewardMixin:
    """Require a configurable hold duration before recognizing a goal."""

    success_hold_count: torch.Tensor

    def _extended_randomization_enabled(self) -> bool:
        """Return whether the A-task actuator and command randomization is enabled."""
        return bool(getattr(self.cfg, "enable_extended_sim2real_randomization", False))

    def _cube_pose_observation_noise_enabled(self) -> bool:
        """Return whether cube-pose-specific observation noise is enabled."""
        return bool(getattr(self.cfg, "enable_cube_pose_obs_noise", False))

    def _cube_pose_sample_hold_enabled(self) -> bool:
        """Return whether the cube pose is updated below the policy rate."""
        return bool(getattr(self.cfg, "enable_cube_pose_sample_hold", False))

    def _initialize_cube_pose_observation_noise(self) -> None:
        """Allocate and randomize persistent cube pose observation errors."""
        if hasattr(self, "cube_position_obs_bias"):
            return

        self.cube_position_obs_bias = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_orientation_obs_bias = torch.zeros((self.num_envs, 4), device=self.device)
        self.cube_orientation_obs_bias[:, 3] = 1.0
        self.cube_previous_orientation_obs = self.object_rot.clone()
        self._cube_position_obs_noise_std = torch.tensor(
            self.cfg.cube_position_obs_noise_std, dtype=torch.float32, device=self.device
        )
        self._cube_position_obs_bias_ranges = torch.tensor(
            self.cfg.cube_position_obs_bias_range, dtype=torch.float32, device=self.device
        )
        self._cube_pose_held_pos_obs = self.object_pos.clone()
        self._cube_pose_held_rot_obs = self.object_rot.clone()
        self._cube_pose_steps_until_update = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cube_pose_elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cube_pose_measurement_interval_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._cube_pose_measurement_updated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cube_velocity_previous_pos_obs = self.object_pos.clone()
        self._cube_velocity_previous_rot_obs = self.object_rot.clone()
        self._cube_linear_velocity_obs_filtered = torch.zeros_like(self.object_pos)
        self._cube_angular_velocity_obs_filtered = torch.zeros_like(self.object_pos)
        self._cube_velocity_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._randomize_cube_pose_observation_bias(env_ids)

    def _randomize_cube_pose_observation_bias(self, env_ids: torch.Tensor) -> None:
        """Sample persistent cube position and orientation errors for environments."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = len(env_ids)
        bias_low = self._cube_position_obs_bias_ranges[:, 0]
        bias_high = self._cube_position_obs_bias_ranges[:, 1]
        self.cube_position_obs_bias[env_ids] = bias_low + (bias_high - bias_low) * torch.rand(
            (count, 3), device=self.device
        )

        axes = torch.randn((count, 3), device=self.device)
        axes = axes / torch.clamp_min(torch.linalg.vector_norm(axes, dim=-1, keepdim=True), 1.0e-6)
        max_angle = float(self.cfg.cube_orientation_obs_bias_max)
        angles = (2.0 * torch.rand(count, device=self.device) - 1.0) * max_angle
        self.cube_orientation_obs_bias[env_ids] = quat_from_angle_axis(angles, axes)
        self.cube_previous_orientation_obs[env_ids] = self.object_rot[env_ids]

    def _get_policy_cube_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the internally consistent cube pose presented to the policy."""
        noise_enabled = self._cube_pose_observation_noise_enabled()
        sample_hold_enabled = self._cube_pose_sample_hold_enabled()
        velocity_from_pose_enabled = bool(getattr(self.cfg, "enable_cube_velocity_from_pose", False))
        if not noise_enabled and not sample_hold_enabled and not velocity_from_pose_enabled:
            return self.object_pos, self.object_rot

        self._initialize_cube_pose_observation_noise()
        sampled_intervals = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        if sample_hold_enabled:
            interval_low, interval_high = self.cfg.cube_pose_update_interval_steps_range
            sampled_intervals.random_(int(interval_low), int(interval_high) + 1)
            update_mask, next_countdown = update_cube_pose_sample_hold_countdown(
                self._cube_pose_steps_until_update, sampled_intervals
            )
        else:
            update_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            next_countdown = torch.zeros_like(self._cube_pose_steps_until_update)

        candidate_pos = self.object_pos
        candidate_rot = self.object_rot
        if noise_enabled:
            position_std = self._cube_position_obs_noise_std
            clip_sigma = float(self.cfg.cube_pose_obs_noise_clip_sigma)
            position_noise = torch.randn_like(self.object_pos) * position_std
            if clip_sigma > 0.0:
                position_noise = torch.maximum(position_noise, -clip_sigma * position_std)
                position_noise = torch.minimum(position_noise, clip_sigma * position_std)

            axes = torch.randn((self.num_envs, 3), device=self.device)
            axes = axes / torch.clamp_min(torch.linalg.vector_norm(axes, dim=-1, keepdim=True), 1.0e-6)
            rotation_std = float(self.cfg.cube_orientation_obs_noise_std)
            angles = torch.randn(self.num_envs, device=self.device) * rotation_std
            if clip_sigma > 0.0:
                angles = torch.clamp(angles, -clip_sigma * rotation_std, clip_sigma * rotation_std)
            rotation_noise = quat_from_angle_axis(angles, axes)

            candidate_pos, candidate_rot = apply_cube_pose_observation_error(
                self.object_pos,
                self.object_rot,
                self.cube_position_obs_bias,
                position_noise,
                self.cube_orientation_obs_bias,
                rotation_noise,
                self.cube_previous_orientation_obs,
            )

        self._cube_pose_held_pos_obs.copy_(
            torch.where(update_mask.unsqueeze(-1), candidate_pos, self._cube_pose_held_pos_obs)
        )
        self._cube_pose_held_rot_obs.copy_(
            torch.where(update_mask.unsqueeze(-1), candidate_rot, self._cube_pose_held_rot_obs)
        )
        self.cube_previous_orientation_obs.copy_(
            torch.where(update_mask.unsqueeze(-1), candidate_rot, self.cube_previous_orientation_obs)
        )

        self._cube_pose_elapsed_steps += 1
        elapsed_steps = torch.clamp_min(self._cube_pose_elapsed_steps, 1)
        self._cube_pose_measurement_interval_steps.copy_(
            torch.where(update_mask, elapsed_steps, self._cube_pose_measurement_interval_steps)
        )
        self._cube_pose_elapsed_steps.copy_(
            torch.where(update_mask, torch.zeros_like(self._cube_pose_elapsed_steps), self._cube_pose_elapsed_steps)
        )
        self._cube_pose_steps_until_update.copy_(next_countdown)
        self._cube_pose_measurement_updated.copy_(update_mask)

        update_count = torch.clamp_min(update_mask.sum(), 1)
        mean_interval_steps = (
            self._cube_pose_measurement_interval_steps * update_mask.to(dtype=torch.long)
        ).sum() / update_count
        log = self.extras.setdefault("log", {})
        log["cube_pose_measurement_update_rate"] = update_mask.float().mean()
        log["cube_pose_measurement_interval_steps"] = mean_interval_steps

        observed_pos = self._cube_pose_held_pos_obs
        observed_rot = self._cube_pose_held_rot_obs
        self.cube_object_pos_obs = observed_pos
        self.cube_object_rot_obs = observed_rot
        self.cube_pose_obs_position_error = torch.linalg.vector_norm(observed_pos - self.object_pos, dim=-1)
        self.cube_pose_obs_orientation_error = rotation_distance(observed_rot, self.object_rot)
        return observed_pos, observed_rot

    def _compute_policy_cube_velocity_from_pose(
        self, object_pos_obs: torch.Tensor, object_rot_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cube velocities reconstructed from the policy's pose observations."""
        if not bool(getattr(self.cfg, "enable_cube_velocity_from_pose", False)):
            return self.object_linvel, self.object_angvel

        self._initialize_cube_pose_observation_noise()
        policy_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        measurement_dt = self._cube_pose_measurement_interval_steps.to(dtype=object_pos_obs.dtype) * policy_dt
        candidate_linear_velocity, candidate_angular_velocity = reconstruct_cube_velocity_from_pose(
            object_pos_obs,
            object_rot_obs,
            self._cube_velocity_previous_pos_obs,
            self._cube_velocity_previous_rot_obs,
            self._cube_velocity_history_valid,
            self._cube_linear_velocity_obs_filtered,
            self._cube_angular_velocity_obs_filtered,
            measurement_dt,
            float(self.cfg.cube_velocity_obs_low_pass_alpha),
            float(self.cfg.cube_linear_velocity_obs_max),
            float(self.cfg.cube_angular_velocity_obs_max),
        )
        update_mask = self._cube_pose_measurement_updated
        linear_velocity = torch.where(
            update_mask.unsqueeze(-1), candidate_linear_velocity, self._cube_linear_velocity_obs_filtered
        )
        angular_velocity = torch.where(
            update_mask.unsqueeze(-1), candidate_angular_velocity, self._cube_angular_velocity_obs_filtered
        )
        self._cube_velocity_previous_pos_obs.copy_(
            torch.where(update_mask.unsqueeze(-1), object_pos_obs, self._cube_velocity_previous_pos_obs)
        )
        self._cube_velocity_previous_rot_obs.copy_(
            torch.where(update_mask.unsqueeze(-1), object_rot_obs, self._cube_velocity_previous_rot_obs)
        )
        self._cube_linear_velocity_obs_filtered.copy_(linear_velocity)
        self._cube_angular_velocity_obs_filtered.copy_(angular_velocity)
        self._cube_velocity_history_valid.copy_(self._cube_velocity_history_valid | update_mask)
        self.cube_object_linvel_obs = linear_velocity
        self.cube_object_angvel_obs = angular_velocity
        self.cube_velocity_obs_linear_error = torch.linalg.vector_norm(linear_velocity - self.object_linvel, dim=-1)
        self.cube_velocity_obs_angular_error = torch.linalg.vector_norm(angular_velocity - self.object_angvel, dim=-1)
        return linear_velocity, angular_velocity

    def _initialize_extended_randomization(self) -> None:
        """Allocate persistent buffers used by the A-task randomization."""
        if hasattr(self, "sim2real_joint_zero_offset"):
            return

        action_shape = (self.num_envs, len(self.actuated_dof_indices))
        self.sim2real_joint_zero_offset = torch.zeros(action_shape, device=self.device)
        self.sim2real_action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sim2real_previous_command = torch.zeros(action_shape, device=self.device)
        self.sim2real_previous_applied_action = torch.zeros(action_shape, device=self.device)
        self.sim2real_effort_scale = torch.ones(action_shape, device=self.device)
        self.sim2real_velocity_scale = torch.ones(action_shape, device=self.device)
        self.sim2real_action_hold_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.sim2real_action_sample_count = torch.zeros((), dtype=torch.long, device=self.device)

        joint_indices = torch.as_tensor(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self._sim2real_default_effort_limits = self.hand.data.joint_effort_limits.torch[:, joint_indices].clone()
        self._sim2real_default_velocity_limits = self.hand.data.joint_velocity_limits.torch[:, joint_indices].clone()

    def _randomize_extended_actuator_properties(self, env_ids: torch.Tensor) -> None:
        """Sample per-joint offsets and limits for the selected environments."""
        self._initialize_extended_randomization()
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = len(env_ids)
        joint_indices = torch.as_tensor(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        joint_ids_sim = joint_indices.to(dtype=torch.int32)
        env_ids_sim = env_ids.to(dtype=torch.int32)

        zero_low, zero_high = self.cfg.joint_zero_offset_range
        effort_low, effort_high = self.cfg.joint_effort_limit_scale_range
        velocity_low, velocity_high = self.cfg.joint_velocity_limit_scale_range
        shape = (count, len(self.actuated_dof_indices))
        self.sim2real_joint_zero_offset[env_ids] = zero_low + (zero_high - zero_low) * torch.rand(
            shape, device=self.device
        )
        self.sim2real_effort_scale[env_ids] = effort_low + (effort_high - effort_low) * torch.rand(
            shape, device=self.device
        )
        self.sim2real_velocity_scale[env_ids] = velocity_low + (velocity_high - velocity_low) * torch.rand(
            shape, device=self.device
        )

        delay_low, delay_high = self.cfg.action_delay_step_range
        self.sim2real_action_delay_steps[env_ids] = torch.randint(
            int(delay_low), int(delay_high) + 1, (count,), device=self.device
        )

        safe_lower, safe_upper = compute_joint_target_limits(
            self.hand_dof_lower_limits[env_ids][:, joint_indices],
            self.hand_dof_upper_limits[env_ids][:, joint_indices],
            float(getattr(self.cfg, "joint_limit_safety_margin_fraction", 0.0)),
        )
        neutral_action = unscale(self.cur_targets[env_ids][:, joint_indices], safe_lower, safe_upper)
        self.sim2real_previous_command[env_ids] = neutral_action
        self.sim2real_previous_applied_action[env_ids] = neutral_action

        effort_limits = self._sim2real_default_effort_limits[env_ids] * self.sim2real_effort_scale[env_ids]
        velocity_limits = self._sim2real_default_velocity_limits[env_ids] * self.sim2real_velocity_scale[env_ids]
        self.hand.write_joint_effort_limit_to_sim_index(
            limits=effort_limits,
            joint_ids=joint_ids_sim,
            env_ids=env_ids_sim,
        )
        self.hand.write_joint_velocity_limit_to_sim_index(
            limits=velocity_limits,
            joint_ids=joint_ids_sim,
            env_ids=env_ids_sim,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.raw_policy_actions = actions.clone()
        action_clip_value = getattr(self.cfg, "action_clip_value", None)
        command_actions = (
            torch.clamp(actions, -float(action_clip_value), float(action_clip_value))
            if action_clip_value is not None
            else actions
        )
        if getattr(self.cfg, "reward_mode", "proximity") == "single_goal_progress":
            if not hasattr(self, "reward_previous_actions"):
                self.reward_previous_actions = torch.zeros_like(actions)
                self.reward_action_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            elif hasattr(self, "actions"):
                self.reward_previous_actions.copy_(self.actions)
        if not self._extended_randomization_enabled():
            super()._pre_physics_step(command_actions)
            return

        self._initialize_extended_randomization()
        delayed_actions = torch.where(
            self.sim2real_action_delay_steps.unsqueeze(-1) > 0,
            self.sim2real_previous_command,
            command_actions,
        )
        hold_mask = torch.rand(self.num_envs, device=self.device) < float(self.cfg.action_hold_probability)
        applied_actions = torch.where(
            hold_mask.unsqueeze(-1),
            self.sim2real_previous_applied_action,
            delayed_actions,
        )
        self.sim2real_previous_command.copy_(command_actions)
        self.sim2real_previous_applied_action.copy_(applied_actions)
        self.sim2real_action_hold_count += hold_mask.sum()
        self.sim2real_action_sample_count += self.num_envs
        super()._pre_physics_step(applied_actions)

    def _apply_action(self) -> None:
        """Apply policy actions inside the configured joint safety limits."""
        joint_ids = torch.as_tensor(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        safe_lower, safe_upper = compute_joint_target_limits(
            self.hand_dof_lower_limits[:, joint_ids],
            self.hand_dof_upper_limits[:, joint_ids],
            float(getattr(self.cfg, "joint_limit_safety_margin_fraction", 0.0)),
        )
        targets = scale(self.actions, safe_lower, safe_upper)
        targets = self.cfg.act_moving_average * targets + (1.0 - self.cfg.act_moving_average) * self.prev_targets[
            :, joint_ids
        ]
        target_velocity_limit = float(getattr(self.cfg, "joint_target_velocity_limit", 0.0))
        if target_velocity_limit > 0.0:
            policy_dt = float(self.cfg.sim.dt * self.cfg.decimation)
            targets = limit_joint_target_rate(
                targets,
                self.prev_targets[:, joint_ids],
                target_velocity_limit * policy_dt,
            )
        targets = torch.maximum(targets, safe_lower)
        targets = torch.minimum(targets, safe_upper)
        self.cur_targets[:, joint_ids] = targets
        self.prev_targets[:, joint_ids] = targets

        applied_targets = targets
        if self._extended_randomization_enabled() and hasattr(self, "sim2real_joint_zero_offset"):
            applied_targets = targets + self.sim2real_joint_zero_offset
            applied_targets = torch.maximum(applied_targets, safe_lower)
            applied_targets = torch.minimum(applied_targets, safe_upper)
        self._set_joint_pos_target(target=applied_targets, joint_ids=self.actuated_dof_indices)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        rot_dist = rotation_distance(self.object_rot, self.goal_rot)
        # Cache pre-reset diagnostics. DirectRLEnv resets completed environments
        # before returning from step(), so playback code cannot recover these
        # terminal states from the post-step asset buffers.
        self._last_step_rot_dist = rot_dist.clone()
        self._last_step_object_linear_speed = torch.linalg.vector_norm(self.object_linvel, dim=-1)
        self._last_step_object_angular_speed = torch.linalg.vector_norm(self.object_angvel, dim=-1)
        self._last_step_joint_speed = torch.abs(self.hand_dof_vel).mean(dim=-1)
        self._last_step_joint_tracking_error = torch.abs(self.hand_dof_targets - self.hand_dof_pos).mean(dim=-1)
        if not getattr(self.cfg, "terminate_on_success", False):
            return terminated, truncated

        if not hasattr(self, "success_hold_count"):
            self.success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        reaches_required_hold = compute_stable_success_termination(
            rot_dist,
            self.success_hold_count,
            self.cfg.success_tolerance,
            int(self.cfg.success_hold_steps),
        )
        if getattr(self.cfg, "success_requires_not_fallen", False):
            reaches_required_hold = reaches_required_hold & ~terminated
        return terminated | reaches_required_hold, truncated

    def _get_rewards(self) -> torch.Tensor:
        if getattr(self.cfg, "reward_mode", "proximity") == "single_goal_progress":
            return self._get_single_goal_progress_rewards()

        hold_steps = int(getattr(self.cfg, "success_hold_steps", 1))
        if hold_steps <= 1:
            return super()._get_rewards()

        if not hasattr(self, "success_hold_count"):
            self.success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
            self.success_hold_count,
        ) = compute_stable_success_rewards(
            self.reset_buf,
            self.successes,
            self.consecutive_successes,
            self.success_hold_count,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            hold_steps,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            getattr(self.cfg, "success_requires_not_fallen", False),
            self.cfg.av_factor,
        )

        self.extras.setdefault("log", {})["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["log"]["success_hold_progress"] = self.success_hold_count.float().mean()

        if getattr(self.cfg, "reset_target_on_success", True):
            goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(goal_env_ids) > 0:
                self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_single_goal_progress_rewards(self) -> torch.Tensor:
        """Compute the A task reward without persistent proximity reward."""
        if not hasattr(self, "success_hold_count"):
            self.success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if not hasattr(self, "reward_best_rot_dist"):
            self.reward_best_rot_dist = rotation_distance(self.object_rot, self.goal_rot)
        if not hasattr(self, "reward_previous_actions"):
            self.reward_previous_actions = torch.zeros_like(self.actions)
            self.reward_action_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
            self.success_hold_count,
            self.reward_best_rot_dist,
            rotation_progress_reward,
            orientation_error_penalty,
            distance_reward,
            action_rate_penalty,
            action_saturation_penalty,
            joint_velocity_penalty,
            hold_reward,
            success_bonus,
            fall_reward,
            timeout_failure_penalty,
        ) = compute_single_goal_progress_rewards(
            self.reset_buf,
            self.successes,
            self.consecutive_successes,
            self.success_hold_count,
            self.reward_best_rot_dist,
            self.object_pos,
            self.object_rot,
            self.object_angvel,
            self.hand_dof_vel,
            self.in_hand_pos,
            self.goal_rot,
            self.actions,
            self.raw_policy_actions,
            self.reward_previous_actions,
            self.reward_action_history_valid,
            self.cfg.rotation_progress_scale,
            self.cfg.orientation_error_penalty_scale,
            self.cfg.distance_safe_radius,
            self.cfg.distance_penalty_scale,
            self.cfg.action_rate_penalty_scale,
            self.cfg.action_saturation_threshold,
            self.cfg.action_saturation_penalty_scale,
            self.cfg.joint_velocity_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.success_max_object_angvel,
            int(self.cfg.success_hold_steps),
            self.cfg.hold_reward_scale,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.time_penalty,
            self.cfg.timeout_penalty,
            self.cfg.av_factor,
        )
        self.reward_action_history_valid.fill_(True)

        log = self.extras.setdefault("log", {})
        log["consecutive_successes"] = self.consecutive_successes.mean()
        log["success_hold_progress"] = self.success_hold_count.float().mean()
        log["reward_rotation_progress"] = rotation_progress_reward.mean()
        log["reward_orientation_error"] = orientation_error_penalty.mean()
        log["reward_distance"] = distance_reward.mean()
        log["reward_action_rate"] = action_rate_penalty.mean()
        log["reward_action_saturation"] = action_saturation_penalty.mean()
        log["reward_joint_velocity"] = joint_velocity_penalty.mean()
        log["reward_hold"] = hold_reward.mean()
        log["reward_success"] = success_bonus.mean()
        log["reward_fall"] = fall_reward.mean()
        log["reward_timeout"] = timeout_failure_penalty.mean()
        log["reward_time"] = self.cfg.time_penalty
        log["in_hand_distance"] = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1).mean()
        log["object_linear_speed"] = torch.linalg.vector_norm(self.object_linvel, dim=-1).mean()
        log["object_angular_speed"] = torch.linalg.vector_norm(self.object_angvel, dim=-1).mean()
        log["action_saturation_rate"] = (
            torch.abs(self.raw_policy_actions) > self.cfg.action_saturation_threshold
        ).float().mean()
        log["applied_action_saturation_rate"] = (
            torch.abs(self.actions) > self.cfg.action_saturation_threshold
        ).float().mean()
        action_clip_value = float(getattr(self.cfg, "action_clip_value", 1.0))
        log["action_clip_rate"] = (torch.abs(self.raw_policy_actions) > action_clip_value).float().mean()
        log["raw_action_abs_mean"] = torch.abs(self.raw_policy_actions).mean()
        log["raw_action_abs_max"] = torch.abs(self.raw_policy_actions).max()
        if hasattr(self, "cube_pose_obs_position_error"):
            log["cube_pose_obs_position_error_m"] = self.cube_pose_obs_position_error.mean()
            log["cube_pose_obs_orientation_error_rad"] = self.cube_pose_obs_orientation_error.mean()
        if hasattr(self, "cube_velocity_obs_linear_error"):
            log["cube_velocity_obs_linear_error_m_s"] = self.cube_velocity_obs_linear_error.mean()
            log["cube_velocity_obs_angular_error_rad_s"] = self.cube_velocity_obs_angular_error.mean()
        if self._extended_randomization_enabled() and hasattr(self, "sim2real_joint_zero_offset"):
            log["sim2real_joint_zero_offset_abs"] = self.sim2real_joint_zero_offset.abs().mean()
            log["sim2real_effort_scale"] = self.sim2real_effort_scale.mean()
            log["sim2real_velocity_scale"] = self.sim2real_velocity_scale.mean()
            log["sim2real_action_delay_steps"] = self.sim2real_action_delay_steps.float().mean()
            log["sim2real_action_hold_rate"] = self.sim2real_action_hold_count.float() / torch.clamp_min(
                self.sim2real_action_sample_count, 1
            )
            if hasattr(self, "sim2real_cube_side_length"):
                log["sim2real_cube_side_length_mm"] = 1000.0 * self.sim2real_cube_side_length.mean()
                log["sim2real_cube_mass"] = self.sim2real_cube_mass.mean()
        return total_reward

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        completed_mask = self.episode_length_buf[env_ids_tensor] > 0
        super()._reset_idx(env_ids)
        if not hasattr(self, "cumulative_successes"):
            self.cumulative_successes = torch.zeros((), dtype=torch.long, device=self.device)
            self.cumulative_completed_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        completed_episode_successes = self._last_episode_success[env_ids_tensor][completed_mask].to(torch.long)
        (
            self.cumulative_successes,
            self.cumulative_completed_episodes,
            cumulative_success_rate,
        ) = update_cumulative_success_counts(
            self.cumulative_successes,
            self.cumulative_completed_episodes,
            completed_episode_successes,
        )
        log = self.extras.setdefault("log", {})
        log["Metrics/cumulative_success_rate"] = cumulative_success_rate
        log["Metrics/cumulative_successes"] = self.cumulative_successes
        log["Metrics/cumulative_completed_episodes"] = self.cumulative_completed_episodes
        safe_lower, safe_upper = compute_joint_target_limits(
            self.hand_dof_lower_limits[env_ids_tensor],
            self.hand_dof_upper_limits[env_ids_tensor],
            float(getattr(self.cfg, "joint_limit_safety_margin_fraction", 0.0)),
        )
        safe_dof_pos = torch.maximum(self.cur_targets[env_ids_tensor], safe_lower)
        safe_dof_pos = torch.minimum(safe_dof_pos, safe_upper)
        self.prev_targets[env_ids_tensor] = safe_dof_pos
        self.cur_targets[env_ids_tensor] = safe_dof_pos
        self.hand_dof_targets[env_ids_tensor] = safe_dof_pos
        self._set_joint_pos_target(target=safe_dof_pos, env_ids=env_ids_tensor)
        self._write_hand_joint_pos(position=safe_dof_pos, env_ids=env_ids_tensor)
        if self._extended_randomization_enabled():
            self._randomize_extended_actuator_properties(env_ids_tensor)
        if float(getattr(self.cfg, "reset_rotation_noise", 1.0)) == 0.0:
            # The upstream in-hand task always samples the object's initial X/Y
            # rotation. Restore the configured default orientation for this A/B
            # test while preserving the randomized position and zero velocity.
            object_pose = self.object.data.root_pose_w.torch[env_ids_tensor].clone()
            object_pose[:, 3:7] = self.object.data.default_root_pose.torch[env_ids_tensor, 3:7]
            self._write_obj_root_pose(root_pose=object_pose, env_ids=env_ids_tensor)
            self._compute_intermediate_values()
        if hasattr(self, "success_hold_count"):
            self.success_hold_count[env_ids_tensor] = 0
        if hasattr(self, "reward_best_rot_dist"):
            self.reward_best_rot_dist[env_ids_tensor] = rotation_distance(
                self.object_rot[env_ids_tensor], self.goal_rot[env_ids_tensor]
            )
        if hasattr(self, "reward_action_history_valid"):
            self.reward_action_history_valid[env_ids_tensor] = False
        if self._cube_pose_observation_noise_enabled() and hasattr(self, "cube_position_obs_bias"):
            self._randomize_cube_pose_observation_bias(env_ids_tensor)
        if hasattr(self, "_cube_pose_steps_until_update"):
            self._cube_pose_held_pos_obs[env_ids_tensor] = self.object_pos[env_ids_tensor]
            self._cube_pose_held_rot_obs[env_ids_tensor] = self.object_rot[env_ids_tensor]
            self.cube_previous_orientation_obs[env_ids_tensor] = self.object_rot[env_ids_tensor]
            self._cube_pose_steps_until_update[env_ids_tensor] = 0
            self._cube_pose_elapsed_steps[env_ids_tensor] = 0
            self._cube_pose_measurement_interval_steps[env_ids_tensor] = 1
            self._cube_pose_measurement_updated[env_ids_tensor] = False
        if hasattr(self, "_cube_velocity_history_valid"):
            self._cube_velocity_previous_pos_obs[env_ids_tensor] = self.object_pos[env_ids_tensor]
            self._cube_velocity_previous_rot_obs[env_ids_tensor] = self.object_rot[env_ids_tensor]
            self._cube_linear_velocity_obs_filtered[env_ids_tensor] = 0.0
            self._cube_angular_velocity_obs_filtered[env_ids_tensor] = 0.0
            self._cube_velocity_history_valid[env_ids_tensor] = False
