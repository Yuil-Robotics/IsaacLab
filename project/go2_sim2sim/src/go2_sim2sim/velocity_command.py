# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity command visualization specialized for the Go2 playback task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg


class VisibleVelocityCommand(UniformVelocityCommand):
    """Sample dedicated locomotion modes and display velocity tracking."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator and dedicated motion-mode states.

        Args:
            cfg: Uniform velocity command configuration. Optional
                ``rel_straight_envs``, ``rel_turning_envs``, and
                ``rel_lateral_envs`` attributes set the fractions of all
                environments assigned pure forward/backward, pure yaw, and
                pure lateral commands, respectively.
            env: Manager-based environment.
        """
        straight_ratio = getattr(cfg, "rel_straight_envs", 0.0)
        turning_ratio = getattr(cfg, "rel_turning_envs", 0.0)
        lateral_ratio = getattr(cfg, "rel_lateral_envs", 0.0)
        if not 0.0 <= straight_ratio <= 1.0:
            raise ValueError(f"Expected rel_straight_envs in [0, 1], got {straight_ratio}.")
        if not 0.0 <= turning_ratio <= 1.0:
            raise ValueError(f"Expected rel_turning_envs in [0, 1], got {turning_ratio}.")
        if not 0.0 <= lateral_ratio <= 1.0:
            raise ValueError(f"Expected rel_lateral_envs in [0, 1], got {lateral_ratio}.")
        mode_ratio = cfg.rel_standing_envs + straight_ratio + turning_ratio + lateral_ratio
        if mode_ratio > 1.0:
            raise ValueError(
                "Expected the sum of standing, straight, turning, and lateral "
                f"environment ratios to be at most 1, got {mode_ratio}."
            )
        super().__init__(cfg, env)
        self.is_straight_env = torch.zeros_like(self.is_standing_env)
        self.is_turning_env = torch.zeros_like(self.is_standing_env)
        self.is_lateral_env = torch.zeros_like(self.is_standing_env)

    def _update_metrics(self) -> None:
        """Update tracking metrics and log reward-aligned mean base height [m]."""
        super()._update_metrics()
        if hasattr(self._env, "scene"):
            from .robotlab_rewards import get_base_bottom_height_from_reward
            from .rough_mdp import get_ground_relative_base_height_tensor

            heights = get_base_bottom_height_from_reward(self._env)
            if heights is None:
                heights = get_ground_relative_base_height_tensor(self._env.scene)
            mean_base_height = torch.mean(heights)
        else:
            mean_base_height = torch.mean(self.robot.data.root_pos_w.torch[:, 2])
        self._env.extras.setdefault("log", {})["Metrics/base_height"] = mean_base_height

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Sample mutually exclusive standing, straight, turning, lateral, and general commands.

        Turning environments track the sampled yaw-rate command for the entire
        resampling interval instead of converting a target-heading error into a
        yaw rate.
        """
        super()._resample_command(env_ids)
        straight_ratio = getattr(self.cfg, "rel_straight_envs", 0.0)
        turning_ratio = getattr(self.cfg, "rel_turning_envs", 0.0)
        lateral_ratio = getattr(self.cfg, "rel_lateral_envs", 0.0)
        if straight_ratio == 0.0 and turning_ratio == 0.0 and lateral_ratio == 0.0:
            self.is_straight_env[env_ids] = False
            self.is_turning_env[env_ids] = False
            self.is_lateral_env[env_ids] = False
            return

        non_standing_probability = 1.0 - self.cfg.rel_standing_envs
        straight_upper = straight_ratio / non_standing_probability
        turning_upper = (straight_ratio + turning_ratio) / non_standing_probability
        lateral_upper = (straight_ratio + turning_ratio + lateral_ratio) / non_standing_probability
        samples = torch.rand(len(env_ids), device=self.device)
        is_moving = ~self.is_standing_env[env_ids]
        self.is_straight_env[env_ids] = is_moving & (samples < straight_upper)
        self.is_turning_env[env_ids] = is_moving & (samples >= straight_upper) & (samples < turning_upper)
        self.is_lateral_env[env_ids] = is_moving & (samples >= turning_upper) & (samples < lateral_upper)
        # The parent class replaces yaw-rate commands with heading-error control
        # for heading environments. Exclude pure-turning environments so their
        # uniformly sampled yaw rate remains constant until the next resample.
        self.is_heading_env[env_ids] = self.is_heading_env[env_ids] & ~self.is_turning_env[env_ids]

    def _update_command(self) -> None:
        """Apply heading, standing, fixed-yaw-rate turning, and linear-motion modes."""
        super()._update_command()
        straight_env_ids = self.is_straight_env.nonzero(as_tuple=False).flatten()
        turning_env_ids = self.is_turning_env.nonzero(as_tuple=False).flatten()
        lateral_env_ids = self.is_lateral_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[straight_env_ids, 1:] = 0.0
        self.vel_command_b[turning_env_ids, :2] = 0.0
        self.vel_command_b[lateral_env_ids, 0] = 0.0
        self.vel_command_b[lateral_env_ids, 2] = 0.0

    def _resolve_velocity_to_arrow(self, vel_xy: torch.Tensor, wz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert linear velocity and yaw rate to 3D marker scale and orientation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(vel_xy.shape[0], 1)

        lin_norm = torch.linalg.norm(vel_xy, dim=1)
        is_pure_rotation = (lin_norm < 0.05) & (torch.abs(wz) > 0.05)

        # When turning in place (pure rotation), point arrow 90 deg left/right according to wz sign
        effective_x = torch.where(
            is_pure_rotation,
            torch.zeros_like(vel_xy[:, 0]),
            vel_xy[:, 0],
        )
        effective_y = torch.where(
            is_pure_rotation,
            torch.sign(wz) * torch.clamp(torch.abs(wz) * 0.5, min=0.3, max=1.0),
            vel_xy[:, 1] + 0.25 * wz,  # Dynamic turn tilt while walking
        )
        effective_mag = torch.where(
            is_pure_rotation,
            torch.clamp(torch.abs(wz) * 1.5, min=0.6, max=2.0),
            lin_norm * 3.0 + torch.abs(wz) * 0.5,
        )

        arrow_scale[:, 0] *= effective_mag
        heading_angle = torch.atan2(effective_y, effective_x)
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

    def _debug_vis_callback(self, event) -> None:
        """Update command and tracking arrows above the robot."""
        if not self.robot.is_initialized:
            return

        command_position_w = self.robot.data.root_pos_w.torch.clone()
        tracking_position_w = command_position_w.clone()
        command_position_w[:, 2] += 0.65
        tracking_position_w[:, 2] += 0.48

        command_scale, command_quat = self._resolve_velocity_to_arrow(self.command[:, :2], self.command[:, 2])
        tracking_scale, tracking_quat = self._resolve_velocity_to_arrow(
            self.robot.data.root_lin_vel_b.torch[:, :2],
            self.robot.data.root_ang_vel_b.torch[:, 2],
        )
        self.goal_vel_visualizer.visualize(command_position_w, command_quat, command_scale)
        self.current_vel_visualizer.visualize(tracking_position_w, tracking_quat, tracking_scale)
