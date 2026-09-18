# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton runtime matching the Robotis HX5 V7 action and observation pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import convert_quat, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import (
    InHandManipulationEnv,
    randomize_rotation,
    rotation_distance,
    scale,
    unscale,
)

from .hx5_cube_env import Hx5CubeNewtonEnv, canonicalize_quaternions
from .hx5_cube_env_cfg import POLICY_OBSERVATION_DIM
from .hx5_cube_v7_newton_env_cfg import Hx5CubeV7NewtonEnvCfg

_V7_FINGERTIP_QUATERNION_REFERENCES_WXYZ = (
    (-0.5998666286, 0.1700966954, -0.1350484341, -0.7021100521),
    (-0.6458216310, 0.0580391921, -0.0303411949, -0.6781536937),
    (-0.7579505444, -0.1127609089, 0.1533686519, -0.5753964186),
    (-0.6639994979, 0.2804294229, -0.3597331345, -0.5023684502),
    (0.3511686623, -0.0230805315, 0.8686919212, 0.0539818257),
)


@torch.jit.script
def compute_v7_rewards(
    reset_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    raw_actions: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    action_penalty_scale: float,
    action_bound_threshold: float,
    action_bound_penalty_scale: float,
    success_tolerance: float,
    pos_success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the V7 continuous-goal reward and rotation-plus-position success."""
    goal_dist = torch.linalg.vector_norm(object_pos - target_pos, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)
    action_penalty = torch.sum(raw_actions**2, dim=-1) * action_penalty_scale
    action_bound_excess = torch.clamp(torch.abs(raw_actions) - action_bound_threshold, min=0.0)
    action_bound_penalty = torch.sum(action_bound_excess**2, dim=-1) * action_bound_penalty_scale
    reward = (
        goal_dist * dist_reward_scale
        + rot_reward_scale / (torch.abs(rot_dist) + rot_eps)
        + action_penalty
        + action_bound_penalty
    )

    goal_resets = (torch.abs(rot_dist) <= success_tolerance) & (goal_dist <= pos_success_tolerance)
    successes = successes + goal_resets.to(successes.dtype)
    reward = torch.where(goal_resets, reward + reach_goal_bonus, reward)

    fallen = goal_dist >= fall_dist
    reward = torch.where(fallen, reward + fall_penalty, reward)
    resets = fallen | (reset_buf != 0)
    num_resets = torch.sum(resets)
    finished_successes = torch.sum(successes * resets.to(successes.dtype))
    consecutive_successes = torch.where(
        num_resets > 0,
        av_factor * finished_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )
    return reward, goal_resets, successes, consecutive_successes


class Hx5CubeV7NewtonEnv(Hx5CubeNewtonEnv):
    """V7 policy playback environment on the Newton MJWarp backend."""

    cfg: Hx5CubeV7NewtonEnvCfg

    def __init__(self, cfg: Hx5CubeV7NewtonEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Newton normally folds all four environment decimation steps into one
        # backend call. V7 changes controller targets at physics-step cadence,
        # so keep Newton's four solver substeps per 120 Hz step but let
        # DirectRLEnv issue the four 120 Hz steps explicitly.
        self.sim.physics_manager.set_decimation(1)
        self._physics_handles_decimation = False

        self._fingertip_quaternion_references = torch.tensor(
            _V7_FINGERTIP_QUATERNION_REFERENCES_WXYZ,
            dtype=torch.float32,
            device=self.device,
        )
        self.raw_actions = torch.zeros_like(self.actions)
        self.clipped_actions = torch.zeros_like(self.actions)
        self._v7_joint_ids = torch.as_tensor(self.actuated_dof_indices, dtype=torch.long, device=self.device)

        self.action_pipeline_pending_targets = torch.zeros_like(self.cur_targets)
        self.action_pipeline_applied_targets = torch.zeros_like(self.cur_targets)
        self.action_pipeline_delay_remaining_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_pipeline_delay_steps = torch.zeros_like(self.action_pipeline_delay_remaining_steps)

        tau_min, tau_max = self.cfg.motor_target_time_constant_s_range
        self.motor_target_tau_s = tau_min + (tau_max - tau_min) * torch.rand(self.num_envs, device=self.device)
        physics_dt = float(self.cfg.sim.dt)
        self.motor_target_alpha = 1.0 - torch.exp(-physics_dt / torch.clamp(self.motor_target_tau_s, min=1.0e-8))
        self.motor_effective_targets = torch.zeros_like(self.cur_targets)

        self.policy_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        delay_min, delay_max = self.cfg.cube_pose_obs_delay_steps_range
        self._cube_pose_history_length = int(delay_max) + 1
        self._cube_pose_history_write_idx = 0
        initial_object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        initial_object_rot = self.object.data.root_quat_w.torch
        self._cube_pose_pos_history = initial_object_pos.unsqueeze(0).repeat(self._cube_pose_history_length, 1, 1)
        self._cube_pose_rot_history = initial_object_rot.unsqueeze(0).repeat(self._cube_pose_history_length, 1, 1)
        self.cube_pose_obs_delay_steps = torch.randint(
            int(delay_min), int(delay_max) + 1, (self.num_envs,), device=self.device
        )
        self._cube_velocity_prev_pos_obs = initial_object_pos.clone()
        self._cube_velocity_prev_rot_obs = initial_object_rot.clone()
        self._cube_velocity_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cube_object_linvel_obs_filtered = torch.zeros_like(initial_object_pos)
        self._cube_object_angvel_obs_filtered = torch.zeros_like(initial_object_pos)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.raw_actions = actions.clone()
        clip_value = float(self.cfg.action_clip_value)
        self.actions = torch.clamp(self.raw_actions, -clip_value, clip_value)
        self.clipped_actions = self.actions.clone()

        lower = self.hand_dof_lower_limits[:, self._v7_joint_ids]
        upper = self.hand_dof_upper_limits[:, self._v7_joint_ids]
        joint_range = upper - lower
        margin = float(self.cfg.action_joint_limit_margin) * joint_range
        safe_lower = lower + margin
        safe_upper = upper - margin
        desired_targets = scale(self.actions, safe_lower, safe_upper)
        desired_targets = (
            float(self.cfg.act_moving_average) * desired_targets
            + (1.0 - float(self.cfg.act_moving_average)) * self.prev_targets[:, self._v7_joint_ids]
        )
        max_delta = float(self.cfg.max_target_delta)
        desired_targets = torch.clamp(
            desired_targets,
            self.prev_targets[:, self._v7_joint_ids] - max_delta,
            self.prev_targets[:, self._v7_joint_ids] + max_delta,
        )
        desired_targets = torch.maximum(desired_targets, safe_lower)
        desired_targets = torch.minimum(desired_targets, safe_upper)
        self.cur_targets[:, self._v7_joint_ids] = desired_targets
        self.prev_targets[:, self._v7_joint_ids] = desired_targets

        delay_min, delay_max = self.cfg.action_pipeline_delay_physics_steps_range
        self.action_pipeline_delay_steps.random_(int(delay_min), int(delay_max) + 1)
        self.action_pipeline_delay_remaining_steps.copy_(self.action_pipeline_delay_steps)
        self.action_pipeline_pending_targets.copy_(self.cur_targets)

    def _apply_action(self) -> None:
        activate_mask = self.action_pipeline_delay_remaining_steps <= 0
        self.action_pipeline_applied_targets[activate_mask] = self.action_pipeline_pending_targets[activate_mask]
        self.action_pipeline_delay_remaining_steps.copy_(
            torch.clamp(self.action_pipeline_delay_remaining_steps - 1, min=0)
        )

        previous_effective = self.motor_effective_targets[:, self._v7_joint_ids]
        commanded = self.action_pipeline_applied_targets[:, self._v7_joint_ids]
        effective = previous_effective + self.motor_target_alpha.unsqueeze(-1) * (commanded - previous_effective)
        effective = torch.maximum(effective, self.hand_dof_lower_limits[:, self._v7_joint_ids])
        effective = torch.minimum(effective, self.hand_dof_upper_limits[:, self._v7_joint_ids])
        self.motor_effective_targets[:, self._v7_joint_ids] = effective
        self.hand_dof_targets[:, self._v7_joint_ids] = effective
        self._set_joint_pos_target(target=effective, joint_ids=self.actuated_dof_indices)

    def _get_rewards(self) -> torch.Tensor:
        total_reward, self.reset_goal_buf, self.successes[:], self.consecutive_successes[:] = compute_v7_rewards(
            self.reset_buf,
            self.successes,
            self.consecutive_successes,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.raw_actions,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.cfg.action_penalty_scale,
            self.cfg.action_bound_threshold,
            self.cfg.action_bound_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.pos_success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )
        log = self.extras.setdefault("log", {})
        log["consecutive_successes"] = self.consecutive_successes.mean()
        log["v7_position_error"] = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1).mean()
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = InHandManipulationEnv._get_dones(self)
        self._last_step_rot_dist = rotation_distance(self.object_rot, self.goal_rot).clone()
        self._last_step_pos_dist = torch.linalg.vector_norm(self.object_pos - self.in_hand_pos, dim=-1).clone()
        self._last_step_object_linear_speed = torch.linalg.vector_norm(self.object_linvel, dim=-1)
        self._last_step_object_angular_speed = torch.linalg.vector_norm(self.object_angvel, dim=-1)
        self._last_step_joint_speed = torch.abs(self.hand_dof_vel).mean(dim=-1)
        self._last_step_joint_tracking_error = torch.abs(self.hand_dof_targets - self.hand_dof_pos).mean(dim=-1)
        return terminated, truncated

    @staticmethod
    def _clip_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        return vector * torch.clamp(max_norm / torch.clamp(norm, min=1.0e-8), max=1.0)

    def _get_policy_cube_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        write_idx = self._cube_pose_history_write_idx
        self._cube_pose_pos_history[write_idx].copy_(self.object_pos)
        self._cube_pose_rot_history[write_idx].copy_(self.object_rot)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        read_idx = (write_idx - self.cube_pose_obs_delay_steps) % self._cube_pose_history_length
        object_pos_obs = self._cube_pose_pos_history[read_idx, env_ids].clone()
        object_rot_obs = self._cube_pose_rot_history[read_idx, env_ids].clone()
        self._cube_pose_history_write_idx = (write_idx + 1) % self._cube_pose_history_length

        clip_sigma = float(self.cfg.cube_pose_obs_noise_clip_sigma)
        pos_std = float(self.cfg.cube_position_obs_noise_std)
        position_noise = torch.clamp(
            torch.randn_like(object_pos_obs) * pos_std,
            -clip_sigma * pos_std,
            clip_sigma * pos_std,
        )
        object_pos_obs += position_noise

        rot_std = float(self.cfg.cube_orientation_obs_noise_std)
        axis = torch.randn((self.num_envs, 3), dtype=object_rot_obs.dtype, device=self.device)
        axis /= torch.clamp(torch.linalg.vector_norm(axis, dim=-1, keepdim=True), min=1.0e-6)
        angle = torch.clamp(
            torch.randn(self.num_envs, dtype=object_rot_obs.dtype, device=self.device) * rot_std,
            -clip_sigma * rot_std,
            clip_sigma * rot_std,
        )
        object_rot_obs = quat_mul(quat_from_angle_axis(angle, axis), object_rot_obs)
        object_rot_obs /= torch.clamp(torch.linalg.vector_norm(object_rot_obs, dim=-1, keepdim=True), min=1.0e-6)
        return object_pos_obs, object_rot_obs

    def _compute_policy_cube_velocity_from_pose(
        self, object_pos_obs: torch.Tensor, object_rot_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = self._cube_velocity_history_valid.clone()
        raw_lin = (object_pos_obs - self._cube_velocity_prev_pos_obs) / self.policy_dt
        q_delta = quat_mul(object_rot_obs, quat_conjugate(self._cube_velocity_prev_rot_obs))
        q_delta /= torch.clamp(torch.linalg.vector_norm(q_delta, dim=-1, keepdim=True), min=1.0e-8)
        q_delta = torch.where((q_delta[:, 3] < 0.0).unsqueeze(-1), -q_delta, q_delta)
        xyz = q_delta[:, :3]
        sin_half = torch.linalg.vector_norm(xyz, dim=-1)
        angle = 2.0 * torch.atan2(sin_half, torch.clamp(q_delta[:, 3], min=1.0e-8))
        axis = xyz / torch.clamp(sin_half.unsqueeze(-1), min=1.0e-8)
        raw_ang = axis * (angle / self.policy_dt).unsqueeze(-1)
        raw_ang = torch.where((sin_half > 1.0e-7).unsqueeze(-1), raw_ang, torch.zeros_like(raw_ang))
        raw_lin = torch.where(valid.unsqueeze(-1), raw_lin, torch.zeros_like(raw_lin))
        raw_ang = torch.where(valid.unsqueeze(-1), raw_ang, torch.zeros_like(raw_ang))
        raw_lin = self._clip_vector_norm(raw_lin, float(self.cfg.cube_linear_velocity_obs_max))
        raw_ang = self._clip_vector_norm(raw_ang, float(self.cfg.cube_angular_velocity_obs_max))

        alpha = float(self.cfg.cube_velocity_obs_low_pass_alpha)
        lin = alpha * raw_lin + (1.0 - alpha) * self._cube_object_linvel_obs_filtered
        ang = alpha * raw_ang + (1.0 - alpha) * self._cube_object_angvel_obs_filtered
        lin = torch.where(valid.unsqueeze(-1), lin, torch.zeros_like(lin))
        ang = torch.where(valid.unsqueeze(-1), ang, torch.zeros_like(ang))
        self._cube_object_linvel_obs_filtered.copy_(lin)
        self._cube_object_angvel_obs_filtered.copy_(ang)
        self._cube_velocity_prev_pos_obs.copy_(object_pos_obs)
        self._cube_velocity_prev_rot_obs.copy_(object_rot_obs)
        self._cube_velocity_history_valid.fill_(True)
        return lin, ang

    def compute_full_observations(self) -> torch.Tensor:
        """Build the V7 149-value delayed and noisy observation."""
        object_pos_obs, object_rot_obs = self._get_policy_cube_pose()
        object_linvel_obs, object_angvel_obs = self._compute_policy_cube_velocity_from_pose(
            object_pos_obs, object_rot_obs
        )
        object_rot_wxyz = convert_quat(object_rot_obs, to="wxyz")
        goal_rot_wxyz = convert_quat(self.goal_rot, to="wxyz")
        relative_rot_wxyz = convert_quat(quat_mul(object_rot_obs, quat_conjugate(self.goal_rot)), to="wxyz")
        fingertip_rot_wxyz = canonicalize_quaternions(
            convert_quat(self.fingertip_rot, to="wxyz"), self._fingertip_quaternion_references
        )
        observations = torch.cat(
            (
                unscale(
                    self.hand_dof_pos[:, self.actuated_dof_indices],
                    self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                    self.hand_dof_upper_limits[:, self.actuated_dof_indices],
                ),
                self.cfg.vel_obs_scale * self.hand_dof_vel[:, self.actuated_dof_indices],
                object_pos_obs,
                object_rot_wxyz,
                object_linvel_obs,
                self.cfg.vel_obs_scale * object_angvel_obs,
                self.in_hand_pos,
                goal_rot_wxyz,
                relative_rot_wxyz,
                self.fingertip_pos.reshape(self.num_envs, self.num_fingertips * 3),
                fingertip_rot_wxyz.reshape(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.reshape(self.num_envs, self.num_fingertips * 6),
                self.actions,
            ),
            dim=-1,
        )
        if observations.shape[-1] != POLICY_OBSERVATION_DIM:
            raise RuntimeError(
                f"V7 policy observation contract violation: expected {POLICY_OBSERVATION_DIM}, "
                f"got {observations.shape[-1]}."
            )
        return observations

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._last_episode_success[env_ids_tensor] = self.successes[env_ids_tensor] >= self.cfg.success_count_threshold
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (
            self._last_episode_success[env_ids_tensor].float().mean().item()
        )
        DirectRLEnv._reset_idx(self, env_ids_tensor)
        self._reset_target_pose(env_ids_tensor)

        object_default_pose = self.object.data.default_root_pose.torch.clone()[env_ids_tensor]
        object_default_vel = self.object.data.default_root_vel.torch.clone()[env_ids_tensor]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids_tensor), 3), device=self.device)
        object_default_pose[:, :3] += self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids_tensor]
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids_tensor), 2), device=self.device)
        object_default_pose[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids_tensor], self.y_unit_tensor[env_ids_tensor]
        )
        object_default_vel.zero_()
        self._write_obj_root_pose(root_pose=object_default_pose, env_ids=env_ids_tensor)
        self._write_obj_root_vel(root_velocity=object_default_vel, env_ids=env_ids_tensor)

        default_pos = self.hand.data.default_joint_pos.torch[env_ids_tensor]
        lower = self.hand_dof_lower_limits[env_ids_tensor]
        upper = self.hand_dof_upper_limits[env_ids_tensor]
        noise = sample_uniform(-1.0, 1.0, default_pos.shape, device=self.device)
        random_delta = torch.where(noise >= 0.0, noise * (upper - default_pos), noise * (default_pos - lower))
        dof_pos = torch.clamp(default_pos + self.cfg.reset_dof_pos_noise * random_delta, lower, upper)
        dof_vel = self.hand.data.default_joint_vel.torch[env_ids_tensor]

        self.prev_targets[env_ids_tensor] = dof_pos
        self.cur_targets[env_ids_tensor] = dof_pos
        self.hand_dof_targets[env_ids_tensor] = dof_pos
        self.action_pipeline_pending_targets[env_ids_tensor] = dof_pos
        self.action_pipeline_applied_targets[env_ids_tensor] = dof_pos
        self.action_pipeline_delay_remaining_steps[env_ids_tensor] = 0
        self.motor_effective_targets[env_ids_tensor] = dof_pos
        self._set_joint_pos_target(target=dof_pos, env_ids=env_ids_tensor)
        self._write_hand_joint_pos(position=dof_pos, env_ids=env_ids_tensor)
        self._write_hand_joint_vel(velocity=dof_vel, env_ids=env_ids_tensor)
        self.successes[env_ids_tensor] = 0
        self._compute_intermediate_values()

        self._cube_pose_pos_history[:, env_ids_tensor] = self.object_pos[env_ids_tensor].unsqueeze(0)
        self._cube_pose_rot_history[:, env_ids_tensor] = self.object_rot[env_ids_tensor].unsqueeze(0)
        delay_min, delay_max = self.cfg.cube_pose_obs_delay_steps_range
        self.cube_pose_obs_delay_steps[env_ids_tensor] = torch.randint(
            int(delay_min), int(delay_max) + 1, (len(env_ids_tensor),), device=self.device
        )
        self._cube_velocity_prev_pos_obs[env_ids_tensor] = self.object_pos[env_ids_tensor]
        self._cube_velocity_prev_rot_obs[env_ids_tensor] = self.object_rot[env_ids_tensor]
        self._cube_velocity_history_valid[env_ids_tensor] = False
        self._cube_object_linvel_obs_filtered[env_ids_tensor] = 0.0
        self._cube_object_angvel_obs_filtered[env_ids_tensor] = 0.0


__all__ = ["Hx5CubeV7NewtonEnv", "compute_v7_rewards"]
