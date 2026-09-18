# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action smoothness, hip posture and terrain-aware foot regulation rewards."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .rewards import _get_base_height


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return squared second differences of actions, excluding reset history."""
    term = env.action_manager.get_term("joint_pos")
    diff = (term.raw_actions - 2 * term.previous + term.previous_previous).square()
    valid = (term.previous != 0) & (term.previous_previous != 0)
    return (diff * valid).sum(dim=-1)


def hip_pos_penalty_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize hip-roll offsets [rad], matching the reference's unit standing scale."""
    robot = env.scene[asset_cfg.name]
    return (
        (
            robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        )
        .abs()
        .sum(-1)
    )


def feet_regulation(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize horizontal foot speed near terrain, following the reference.

    Args:
        env: Environment.
        base_height_target: Nominal root clearance [m].
        asset_cfg: Robot and feet selection.
        sensor_cfg: Local ground height sensor.

    Returns:
        Height-weighted sum of squared horizontal foot speeds [m²/s²].
    """
    robot = env.scene[asset_cfg.name]
    height = _get_base_height(env, base_height_target, asset_cfg, sensor_cfg)
    feet_z = robot.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    base_z = robot.data.root_pos_w.torch[:, 2:3]
    clearance = (height[:, None] + feet_z - base_z).clamp(min=0)
    velocity = robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (velocity.square().sum(-1) * torch.exp(-clearance / (0.025 * base_height_target))).sum(-1)
