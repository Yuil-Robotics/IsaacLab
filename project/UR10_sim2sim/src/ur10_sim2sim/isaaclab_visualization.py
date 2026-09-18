# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate-frame visualization for the project Isaac Lab task."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error


class UR10BasePoseCommand(UniformPoseCommand):
    """Pose command whose world marker respects the UR controller base frame.

    The upstream command visualizer interprets commands relative to
    ``base_link``. UR10e deployment commands are instead expressed relative to
    the UR controller ``base`` frame, which is rotated 180 degrees around Z.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._base_frame_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._base_frame_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
        self._base_frame_quat_b[:, 2] = 1.0

    def _update_metrics(self) -> None:
        base_pos_w, base_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self._base_frame_pos_b,
            self._base_frame_quat_b,
        )
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            base_pos_w,
            base_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w.torch[:, self.body_idx],
            self.robot.data.body_quat_w.torch[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.linalg.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.linalg.norm(rot_error, dim=-1)
        if self._track_success:
            self._succeeded |= self.metrics["position_error"] < self.cfg.position_success_threshold
