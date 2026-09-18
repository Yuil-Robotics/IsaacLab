# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for go2_moe_cts, ported from go2_rl_robotlab.

Key differences vs. stock IsaacLab locomotion:
- Fixed sigma tracking rewards (not dynamic sigma).
- 2× tracking weights for better velocity following.
- ``joint_pos_penalty_l1``: extra L1 joint-deviation penalty (improves stability).
- ``joint_acc_l2`` uses a low weight because IsaacLab runs at physics-step resolution.
- Height estimation via ray-caster when available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_base_height(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Estimate base height above the local ground plane.

    If a height scanner is provided, uses ray-caster mean to estimate ground z.
    Falls back to world-frame root z (flat ground assumption) otherwise.

    Args:
        env: The environment.
        base_height_target: Nominal height target [m] used only as fallback.
        asset_cfg: Config referencing the robot articulation.
        sensor_cfg: Optional config referencing a RayCaster sensor.

    Returns:
        Estimated base height above ground, shape [num_envs].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w.torch[:, 2]

    if sensor_cfg is None:
        return base_z

    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits_z = sensor.data.ray_hits_w.torch[..., 2]
    invalid = (
        torch.isnan(ray_hits_z).any(dim=1)
        | torch.isinf(ray_hits_z).any(dim=1)
        | (torch.max(torch.abs(ray_hits_z), dim=1).values > 1e6)
    )
    estimated_ground_z = torch.mean(ray_hits_z, dim=1)
    fallback_ground_z = base_z - base_height_target
    estimated_ground_z = torch.where(invalid, fallback_ground_z, estimated_ground_z)
    return base_z - estimated_ground_z


# ---------------------------------------------------------------------------
# Velocity tracking (fixed sigma)
# ---------------------------------------------------------------------------


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy) using exponential kernel.

    Uses a **fixed** sigma (``std``) rather than a dynamic one.

    Args:
        env: The environment.
        std: Standard deviation of the exponential kernel [m/s].
        command_name: Name of the velocity command.
        asset_cfg: Config referencing the robot articulation.

    Returns:
        Reward tensor, shape [num_envs].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b.torch[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of yaw angular velocity command using exponential kernel.

    Args:
        env: The environment.
        std: Standard deviation of the exponential kernel [rad/s].
        command_name: Name of the velocity command.
        asset_cfg: Config referencing the robot articulation.

    Returns:
        Reward tensor, shape [num_envs].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b.torch[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


# ---------------------------------------------------------------------------
# Base height
# ---------------------------------------------------------------------------


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward maintaining target base height [m] using exponential kernel.

    Args:
        env: The environment.
        target_height: Desired base height above ground [m].
        std: Standard deviation of the exponential kernel [m].
        asset_cfg: Config referencing the robot articulation.
        sensor_cfg: Optional config referencing a RayCaster for ground estimation.

    Returns:
        Reward tensor, shape [num_envs].
    """
    height = _get_base_height(env, target_height, asset_cfg, sensor_cfg)
    height_error = torch.square(height - target_height)
    return torch.exp(-height_error / std**2)


# ---------------------------------------------------------------------------
# Joint position penalty (go2_rl_robotlab addition)
# ---------------------------------------------------------------------------


def joint_pos_penalty_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 2.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalise L1 deviation of joint positions from default.

    Applies a higher penalty (``stand_still_scale`` multiplier) when the robot
    is stationary (small command AND small body velocity).

    This term is not in stock IsaacLab locomotion but is present in
    go2_rl_robotlab and improves sim-to-real transfer.

    Args:
        env: The environment.
        command_name: Name of the velocity command.
        asset_cfg: Config referencing joint IDs to penalise.
        stand_still_scale: Scale multiplier applied during standing.
        velocity_threshold: Body XY speed below which "standing" is declared [m/s].
        command_threshold: Command magnitude below which "standing" is declared [m/s or rad/s].

    Returns:
        Penalty tensor (positive), shape [num_envs].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)

    l1_dev = torch.linalg.norm(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids],
        dim=1,
        ord=1,
    )
    is_standing = (cmd < command_threshold) & (body_vel < velocity_threshold)
    scale = torch.where(is_standing, torch.full_like(l1_dev, stand_still_scale), torch.ones_like(l1_dev))
    return l1_dev * scale


# ---------------------------------------------------------------------------
# Energy / smoothness
# ---------------------------------------------------------------------------


def joint_power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise mechanical joint power consumption [W].

    Computed as the sum of |torque × velocity| across all tracked joints.

    Args:
        env: The environment.
        asset_cfg: Config referencing joint IDs to penalise.

    Returns:
        Power consumption tensor, shape [num_envs].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(
            asset.data.joint_vel.torch[:, asset_cfg.joint_ids] * asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


# ---------------------------------------------------------------------------
# Stance / gait
# ---------------------------------------------------------------------------


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise joint motion when the commanded velocity is near zero.

    Applies L1 joint deviation scaled by how upright the robot is
    (gravity projection in z).

    Args:
        env: The environment.
        command_name: Name of the velocity command.
        command_threshold: Command magnitude below which the penalty is active.
        asset_cfg: Config referencing the robot articulation.

    Returns:
        Penalty tensor (positive), shape [num_envs].
    """
    penalty = mdp.joint_deviation_l1(env, asset_cfg)
    cmd_small = torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    upright = torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0.0, 0.7) / 0.7
    return penalty * cmd_small.float() * upright
