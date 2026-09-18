# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for balancing the Go2 on its rear feet."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _contact_state(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Return whether each selected body is in contact above ``threshold`` [N]."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    return net_forces.norm(dim=-1).amax(dim=1) > threshold


def desired_contact_fraction(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward the fraction of selected bodies with contact force above ``threshold`` [N]."""
    return _contact_state(env, sensor_cfg, threshold).float().mean(dim=1)


def desired_contact_any(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward at least one selected body having contact above ``threshold`` [N]."""
    return _contact_state(env, sensor_cfg, threshold).any(dim=1).float()


def desired_air_fraction(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward the fraction of selected bodies with contact force below ``threshold`` [N]."""
    return (~_contact_state(env, sensor_cfg, threshold)).float().mean(dim=1)


def rear_stand_contact_pattern(
    env: ManagerBasedRLEnv,
    rear_sensor_cfg: SceneEntityCfg,
    front_sensor_cfg: SceneEntityCfg,
    forbidden_sensor_cfg: SceneEntityCfg,
    threshold: float,
    min_rear_contacts: int = 1,
) -> torch.Tensor:
    """Reward a rear-foot-supported contact pattern.

    At least ``min_rear_contacts`` rear feet must exceed ``threshold`` [N],
    while both front feet remain below it. The selected non-foot bodies must
    also remain below the threshold, preventing the thighs, knees, or lower
    legs from supporting the robot. Setting ``min_rear_contacts`` to one lets
    the robot lift and reposition either rear foot to recover its balance.
    """
    rear_contact = _contact_state(env, rear_sensor_cfg, threshold).sum(dim=1) >= min_rear_contacts
    front_air = (~_contact_state(env, front_sensor_cfg, threshold)).all(dim=1)
    forbidden_air = (~_contact_state(env, forbidden_sensor_cfg, threshold)).all(dim=1)
    return (rear_contact & front_air & forbidden_air).float()


def body_lin_vel_l2_when_rear_standing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rear_sensor_cfg: SceneEntityCfg,
    front_sensor_cfg: SceneEntityCfg,
    forbidden_sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize selected body motion only during a valid rear-foot stance.

    The selected body linear velocities [m/s] are squared and summed. Gating
    the penalty with the contact pattern lets the front legs move freely while
    rising but encourages them to become still after reaching the stance.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    standing = rear_stand_contact_pattern(
        env,
        rear_sensor_cfg,
        front_sensor_cfg,
        forbidden_sensor_cfg,
        threshold,
    )
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(body_vel), dim=(1, 2)) * standing


def joint_vel_l2_when_rear_standing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rear_sensor_cfg: SceneEntityCfg,
    front_sensor_cfg: SceneEntityCfg,
    forbidden_sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize selected joint motion [rad/s] only during a valid rear-foot stance."""
    asset: Articulation = env.scene[asset_cfg.name]
    standing = rear_stand_contact_pattern(
        env,
        rear_sensor_cfg,
        front_sensor_cfg,
        forbidden_sensor_cfg,
        threshold,
    )
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=1) * standing


def base_orientation_exp(
    env: ManagerBasedRLEnv,
    target_gravity: tuple[float, float, float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment with a target gravity direction in the base frame.

    Args:
        target_gravity: Unit gravity direction expressed in the base frame.
        std: Exponential-kernel width.
        asset_cfg: Articulation whose base orientation is evaluated.

    Returns:
        Orientation reward in the range ``[0, 1]``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = torch.tensor(target_gravity, device=env.device, dtype=torch.float32)
    target = target / torch.linalg.norm(target)
    error = torch.sum(torch.square(asset.data.projected_gravity_b.torch - target), dim=1)
    return torch.exp(-error / std**2)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward proximity to the target base height [m] with an exponential kernel.

    Args:
        target_height: Desired base height above the flat ground [m].
        std: Exponential-kernel width [m].
        asset_cfg: Articulation whose base height is evaluated.

    Returns:
        Height reward in the range ``[0, 1]``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_pos_w.torch[:, 2] - target_height)
    return torch.exp(-error / std**2)


def joint_pos_outside_range_l2(
    env: ManagerBasedRLEnv,
    lower: float,
    upper: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joint positions outside an allowed range [rad].

    The penalty is zero inside the range and grows quadratically outside it.
    This leaves the joints free to move within a useful balance range instead
    of pulling them toward one exact posture.

    Args:
        lower: Lower allowed joint position [rad].
        upper: Upper allowed joint position [rad].
        asset_cfg: Articulation and joints whose positions are evaluated.

    Returns:
        Summed squared distance outside the range for each environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    below = torch.clamp(lower - joint_pos, min=0.0)
    above = torch.clamp(joint_pos - upper, min=0.0)
    return torch.sum(torch.square(below) + torch.square(above), dim=1)
