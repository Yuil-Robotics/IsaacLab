# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX environment preserving the serialized Robotis benchmark policy contract."""

from __future__ import annotations

import torch

from isaaclab.utils.math import convert_quat, quat_conjugate, quat_mul

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale

from .hx5_cube_env_cfg import POLICY_OBSERVATION_DIM, RobotisHandEnvCfg
from .stable_success import StableSuccessRewardMixin

SPAWN_DROP_MAX_STEPS = 15
INITIAL_POSE_HOLD_STEPS = 15


class RobotisHandBenchmarkEnv(StableSuccessRewardMixin, InHandManipulationEnv):
    """Robotis PhysX task matching the July 2026 benchmark observation layout.

    The current shared IsaacLab task sorts joint and body indices and exposes
    quaternions in ``xyzw`` order. The serialized benchmark preserved its
    configured Robotis orders and trained on ``wxyz`` quaternions. This class
    restores that contract without changing the current simulator's internal
    quaternion convention.
    """

    cfg: RobotisHandEnvCfg

    def __init__(self, cfg: RobotisHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actuated_dof_indices = [self.hand.joint_names.index(name) for name in cfg.actuated_joint_names]
        self.finger_bodies = [self.hand.body_names.index(name) for name in cfg.fingertip_body_names]
        self.num_fingertips = len(self.finger_bodies)

    def compute_full_observations(self) -> torch.Tensor:
        """Build the benchmark's 149-value observation in its serialized order."""
        object_pos_obs, object_rot_obs = self._get_policy_cube_pose()
        object_linvel_obs, object_angvel_obs = self._compute_policy_cube_velocity_from_pose(
            object_pos_obs, object_rot_obs
        )
        object_rot_wxyz = convert_quat(object_rot_obs, to="wxyz")
        goal_rot_wxyz = convert_quat(self.goal_rot, to="wxyz")
        relative_rot_wxyz = convert_quat(quat_mul(object_rot_obs, quat_conjugate(self.goal_rot)), to="wxyz")
        fingertip_rot_wxyz = convert_quat(self.fingertip_rot, to="wxyz")

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
                f"Benchmark observation contract violation: expected {POLICY_OBSERVATION_DIM}, "
                f"got {observations.shape[-1]}."
            )
        return observations


class RobotisHandSingleGoalPlayEnv(RobotisHandBenchmarkEnv):
    """Single-goal PhysX playback environment with exclusive outcome statistics."""

    def __init__(self, cfg: RobotisHandEnvCfg, render_mode: str | None = None, **kwargs):
        self._play_step_count = 0
        self._play_episode_count = 0
        self._play_success_count = 0
        self._play_timeout_count = 0
        self._play_timeout_never_reached_count = 0
        self._play_timeout_hold_failed_count = 0
        self._play_drop_count = 0
        self._play_spawn_drop_count = 0
        self._play_policy_drop_count = 0
        self._play_action_limit_hits = 0
        self._play_action_samples = 0
        self._play_joint_speed_sum = 0.0
        self._play_joint_tracking_error_sum = 0.0
        self._play_control_samples = 0
        self._play_drop_linear_speed_sum = 0.0
        self._play_drop_angular_speed_sum = 0.0
        self._play_timeout_min_rot_dist_sum = 0.0
        self._play_timeout_max_hold_sum = 0
        super().__init__(cfg, render_mode, **kwargs)
        self._play_episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._play_current_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._play_max_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._play_min_rot_dist = torch.full((self.num_envs,), torch.inf, device=self.device)

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment and periodically print exclusive episode outcomes."""
        self._play_action_limit_hits += int((torch.abs(action) >= 0.99).sum().item())
        self._play_action_samples += action.numel()
        observations, rewards, terminated, truncated, extras = super().step(action)

        self._play_episode_steps += 1
        rot_dist = self._last_step_rot_dist
        within_tolerance = rot_dist <= self.cfg.success_tolerance
        self._play_min_rot_dist = torch.minimum(self._play_min_rot_dist, rot_dist)
        self._play_current_hold_steps = torch.where(
            within_tolerance,
            self._play_current_hold_steps + 1,
            torch.zeros_like(self._play_current_hold_steps),
        )
        self._play_max_hold_steps = torch.maximum(self._play_max_hold_steps, self._play_current_hold_steps)
        self._play_joint_speed_sum += float(self._last_step_joint_speed.sum().item())
        self._play_joint_tracking_error_sum += float(self._last_step_joint_tracking_error.sum().item())
        self._play_control_samples += self.num_envs
        done = terminated | truncated
        # The base task records this value immediately before resetting each
        # completed environment. Success gets priority when it coincides with
        # the episode time limit.
        successful = done & self._last_episode_success
        # A non-success termination is the task's out-of-reach/drop condition.
        # Give it priority over a simultaneous timeout so every episode belongs
        # to exactly one category.
        dropped = terminated & ~successful
        timed_out = done & ~successful & ~dropped
        spawn_dropped = dropped & (self._play_episode_steps <= SPAWN_DROP_MAX_STEPS)
        policy_dropped = dropped & ~spawn_dropped
        timeout_never_reached = timed_out & (self._play_max_hold_steps == 0)
        timeout_hold_failed = timed_out & ~timeout_never_reached

        self._play_step_count += 1
        self._play_success_count += int(successful.sum().item())
        self._play_drop_count += int(dropped.sum().item())
        self._play_spawn_drop_count += int(spawn_dropped.sum().item())
        self._play_policy_drop_count += int(policy_dropped.sum().item())
        self._play_timeout_count += int(timed_out.sum().item())
        self._play_timeout_never_reached_count += int(timeout_never_reached.sum().item())
        self._play_timeout_hold_failed_count += int(timeout_hold_failed.sum().item())
        self._play_episode_count += int(done.sum().item())
        self._play_drop_linear_speed_sum += float(self._last_step_object_linear_speed[dropped].sum().item())
        self._play_drop_angular_speed_sum += float(self._last_step_object_angular_speed[dropped].sum().item())
        self._play_timeout_min_rot_dist_sum += float(self._play_min_rot_dist[timed_out].sum().item())
        self._play_timeout_max_hold_sum += int(self._play_max_hold_steps[timed_out].sum().item())
        self._play_episode_steps[done] = 0
        self._play_current_hold_steps[done] = 0
        self._play_max_hold_steps[done] = 0
        self._play_min_rot_dist[done] = torch.inf

        stats_every = int(getattr(self.cfg, "play_stats_every_steps", 30))
        if (
            stats_every > 0
            and self._play_step_count % stats_every == 0
            and self._play_episode_count > 0
        ):
            success_rate = round(100.0 * self._play_success_count / self._play_episode_count, 2)
            timeout_rate = round(100.0 * self._play_timeout_count / self._play_episode_count, 2)
            # Use the residual after display rounding so the three printed
            # percentages always add up to exactly 100.00%.
            drop_rate = max(0.0, round(100.0 - success_rate - timeout_rate, 2))
            spawn_drop_rate = 100.0 * self._play_spawn_drop_count / self._play_episode_count
            policy_drop_rate = 100.0 * self._play_policy_drop_count / self._play_episode_count
            timeout_never_rate = 100.0 * self._play_timeout_never_reached_count / self._play_episode_count
            timeout_hold_rate = 100.0 * self._play_timeout_hold_failed_count / self._play_episode_count
            action_limit_rate = 100.0 * self._play_action_limit_hits / self._play_action_samples
            mean_joint_speed = self._play_joint_speed_sum / self._play_control_samples
            mean_tracking_error = self._play_joint_tracking_error_sum / self._play_control_samples
            mean_drop_linear_speed = self._play_drop_linear_speed_sum / max(self._play_drop_count, 1)
            mean_drop_angular_speed = self._play_drop_angular_speed_sum / max(self._play_drop_count, 1)
            mean_timeout_min_rot_dist = self._play_timeout_min_rot_dist_sum / max(self._play_timeout_count, 1)
            mean_timeout_max_hold = self._play_timeout_max_hold_sum / max(self._play_timeout_count, 1)
            cube_physics = ""
            if hasattr(self, "sim2real_cube_side_length"):
                cube_physics = (
                    f" | cube_side={1000.0 * self.sim2real_cube_side_length.mean().item():.3f}mm"
                    f" | cube_mass={1000.0 * self.sim2real_cube_mass.mean().item():.3f}g"
                )
            print(
                f"step={self._play_step_count} | envs={self.num_envs} "
                f"| episodes={self._play_episode_count} "
                f"| success_rate={success_rate:.2f}% "
                f"| timeout_rate={timeout_rate:.2f}% "
                f"(never_reached={timeout_never_rate:.2f}% | hold_failed={timeout_hold_rate:.2f}%) "
                f"| drop_rate={drop_rate:.2f}% "
                f"(spawn_drop_rate={spawn_drop_rate:.2f}% "
                f"| policy_drop_rate={policy_drop_rate:.2f}%) "
                f"| total=100.00%",
                flush=True,
            )
            print(
                f"  control | action_limit_rate={action_limit_rate:.2f}% "
                f"| mean_joint_speed={mean_joint_speed:.3f}rad/s "
                f"| mean_joint_tracking_error={mean_tracking_error:.4f}rad "
                f"| drop_linear_speed={mean_drop_linear_speed:.3f}m/s "
                f"| drop_angular_speed={mean_drop_angular_speed:.3f}rad/s "
                f"| timeout_min_theta={mean_timeout_min_rot_dist:.3f}rad "
                f"| timeout_max_hold={mean_timeout_max_hold:.2f}steps"
                f"{cube_physics}",
                flush=True,
            )

        return observations, rewards, terminated, truncated, extras


class RobotisHandSingleGoalHoldPlayEnv(RobotisHandSingleGoalPlayEnv):
    """Single-goal playback that holds each reset pose before enabling the policy."""

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Hold reset joint targets for 15 policy steps, then apply policy actions."""
        hold_actions = unscale(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        is_holding = (self._play_episode_steps < INITIAL_POSE_HOLD_STEPS).unsqueeze(-1)
        effective_actions = torch.where(is_holding, hold_actions, action)
        return super().step(effective_actions)
