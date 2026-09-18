# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overhead command and measured velocity arrows using a bundled mesh."""

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkersCfg


def arrow_cfg(name: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Configure a locally bundled arrow without physics or remote asset dependencies."""
    return VisualizationMarkersCfg(
        prim_path=f"/Visuals/Command/{name}",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=str(Path(__file__).parent / "assets" / "arrow_x.usda"),
                scale=(0.6, 0.15, 0.15),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=color),
            )
        },
    )


def planar_arrow(velocity: torch.Tensor, root_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal arrow scale [m] and quaternion for body XY velocity [m/s]."""
    speed = velocity.norm(dim=-1)
    scale = torch.full((len(speed), 3), 0.15, device=velocity.device)
    scale[:, 0] = (speed * 0.6).clamp(1e-5, 3.0)
    angle = torch.atan2(velocity[:, 1], velocity[:, 0])
    zero = torch.zeros_like(angle)
    rotation = math_utils.quat_from_euler_xyz(zero, zero, angle)
    return scale, math_utils.quat_mul(math_utils.yaw_quat(root_quat), rotation)


def yaw_arrow(rate: torch.Tensor, root_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a tangential turn arrow for yaw rate [rad/s], positive pointing left."""
    velocity = torch.stack((torch.zeros_like(rate), rate * 0.5), dim=-1)
    return planar_arrow(velocity, root_quat)


def tracking_arrow(
    vel_xy: torch.Tensor,
    wz: torch.Tensor,
    root_quat: torch.Tensor,
    default_scale: tuple[float, float, float] = (0.5, 1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the local sim2sim snapshot: velocity [m/s], yaw rate [rad/s], scale [m]."""
    arrow_scale = torch.tensor(default_scale, device=vel_xy.device).repeat(vel_xy.shape[0], 1)

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
    base_quat_w = root_quat
    arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
    return arrow_scale, arrow_quat


def tracking_arrow_cfg(name: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Match sim2sim arrow size [m] and material using the bundled mesh."""
    cfg = arrow_cfg(name, color)
    marker = cfg.markers["arrow"]
    marker.scale = (0.5, 1.0, 1.0)
    marker.visual_material.emissive_color = tuple(component * 0.35 for component in color)
    marker.visual_material.roughness = 0.8
    return cfg
