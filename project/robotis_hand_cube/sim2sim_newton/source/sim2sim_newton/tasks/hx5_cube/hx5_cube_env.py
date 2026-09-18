# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robotis HX5 cube environment with the trained policy's observation contract."""

from __future__ import annotations

import torch

from isaaclab.utils.math import convert_quat, quat_conjugate, quat_mul

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale

from .hx5_cube_newton_env_cfg import POLICY_ACTION_DIM, POLICY_OBSERVATION_DIM, ROBOT_BODY_COUNT, Hx5CubeNewtonEnvCfg
from .stable_success import StableSuccessRewardMixin

# Approximate training-distribution quaternion centers in wxyz order. They are
# used only to choose the equivalent q/-q representation consistently.
_FINGERTIP_QUATERNION_REFERENCES_WXYZ = (
    (-0.4998912215, 0.4242646992, -0.3392543197, -0.5963063240),
    (-0.6141346097, -0.2522679269, 0.2375784963, -0.6724558473),
    (-0.7253507972, -0.1840142608, 0.2383329421, -0.5732201934),
    (-0.7537555099, 0.0904104039, -0.1175142005, -0.5366834998),
    (0.2263838798, 0.1595797539, 0.9119610190, -0.0731724650),
)


def canonicalize_quaternions(quaternions: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """Choose quaternion signs closest to reference orientations.

    Args:
        quaternions: Quaternions, shape ``[..., quaternion_count, 4]``.
        references: Reference quaternions, shape ``[quaternion_count, 4]``.

    Returns:
        Equivalent normalized quaternions with stable signs.
    """
    normalized = torch.nn.functional.normalize(quaternions, dim=-1)
    references = torch.nn.functional.normalize(references, dim=-1)
    signs = torch.where((normalized * references).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    return normalized * signs


class Hx5CubeNewtonEnv(StableSuccessRewardMixin, InHandManipulationEnv):
    """Newton play environment matching the original Robotis policy layout."""

    cfg: Hx5CubeNewtonEnvCfg

    def __init__(self, cfg: Hx5CubeNewtonEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.hand.num_joints != POLICY_ACTION_DIM:
            raise ValueError(
                f"The policy requires exactly {POLICY_ACTION_DIM} hand joints, but the USD exposes "
                f"{self.hand.num_joints}: {self.hand.joint_names}"
            )
        if self.hand.num_bodies != ROBOT_BODY_COUNT:
            raise ValueError(
                f"The original HX5 asset requires {ROBOT_BODY_COUNT} rigid bodies, but the USD exposes "
                f"{self.hand.num_bodies}: {self.hand.body_names}"
            )

        # The original training task preserved these configured orders. The
        # current common task sorts indices, so restore the policy contract.
        self.actuated_dof_indices = [self.hand.joint_names.index(name) for name in cfg.actuated_joint_names]
        self.finger_bodies = [self.hand.body_names.index(name) for name in cfg.fingertip_body_names]
        self.num_fingertips = len(self.finger_bodies)

        self._fingertip_quaternion_references = torch.tensor(
            _FINGERTIP_QUATERNION_REFERENCES_WXYZ,
            dtype=torch.float32,
            device=self.device,
        )

    def compute_full_observations(self) -> torch.Tensor:
        """Build the exact 149-value observation expected by the policy."""
        object_pos_obs, object_rot_obs = self._get_policy_cube_pose()
        object_linvel_obs, object_angvel_obs = self._compute_policy_cube_velocity_from_pose(
            object_pos_obs, object_rot_obs
        )
        object_rot_wxyz = convert_quat(object_rot_obs, to="wxyz")
        goal_rot_wxyz = convert_quat(self.goal_rot, to="wxyz")
        relative_rot_wxyz = convert_quat(quat_mul(object_rot_obs, quat_conjugate(self.goal_rot)), to="wxyz")

        fingertip_rot_wxyz = convert_quat(self.fingertip_rot, to="wxyz")
        fingertip_rot_wxyz = canonicalize_quaternions(
            fingertip_rot_wxyz,
            self._fingertip_quaternion_references,
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
                f"Policy observation contract violation: expected {POLICY_OBSERVATION_DIM}, "
                f"got {observations.shape[-1]}."
            )
        return observations
