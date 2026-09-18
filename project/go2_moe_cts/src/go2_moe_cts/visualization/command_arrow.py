# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A single curved arrow representing a commanded planar twist."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .velocity import arrow_cfg


def command_curve(command: torch.Tensor, segments: int = 16) -> torch.Tensor:
    """Return body-frame curve points [m] for commands [vx m/s, vy m/s, wz rad/s].

    Preview a constant body-frame twist for 0.6 s. Pure rotation uses a symbolic
    0.22 m radius arc. Display length is capped at 3 m and turn at half a circle.
    """
    time = torch.linspace(0, 0.6, segments + 1, device=command.device, dtype=command.dtype)[None, :]
    velocity = command[:, :2]
    speed = velocity.norm(dim=-1)
    velocity = velocity * (5.0 / speed.clamp_min(5.0))[:, None]
    rate = command[:, 2].clamp(-torch.pi / 0.6, torch.pi / 0.6)[:, None]
    angle = rate * time
    safe_rate = torch.where(rate.abs() < 1e-5, torch.ones_like(rate), rate)
    forward = torch.where(rate.abs() < 1e-5, time.expand_as(angle), angle.sin() / safe_rate)
    lateral = torch.where(rate.abs() < 1e-5, torch.zeros_like(angle), (1 - angle.cos()) / safe_rate)
    x = forward * velocity[:, :1] - lateral * velocity[:, 1:2]
    y = lateral * velocity[:, :1] + forward * velocity[:, 1:2]
    turning = (speed < 0.02) & (command[:, 2].abs() > 0.02)
    x = torch.where(turning[:, None], 0.22 * angle.cos(), x)
    y = torch.where(turning[:, None], 0.22 * angle.sin(), y)
    return torch.stack((x, y, torch.zeros_like(x)), dim=-1)


class CommandArrow:
    """Draw one green curved shaft and one arrowhead above each robot."""

    def __init__(self):
        color = (0.0, 1.0, 0.0)
        self.shaft = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/combined_shaft",
                markers={
                    "shaft": sim_utils.CylinderCfg(
                        radius=0.012,
                        height=1.0,
                        axis="X",
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=color),
                    )
                },
            )
        )
        self.tip = VisualizationMarkers(arrow_cfg("combined_tip", color))

    def set_visibility(self, visible: bool) -> None:
        """Show or hide the entire arrow."""
        self.shaft.set_visibility(visible)
        self.tip.set_visibility(visible)

    def visualize(self, command: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor) -> None:
        """Update arrows using commands [m/s, m/s, rad/s], root position [m], and quaternion."""
        points = command_curve(command)
        n, point_count, _ = points.shape
        yaw = math_utils.yaw_quat(root_quat)
        rotation = yaw[:, None, :].expand(-1, point_count, -1).reshape(-1, 4)
        points = math_utils.quat_apply(rotation, points.reshape(-1, 3)).reshape(n, point_count, 3)
        points += root_pos[:, None, :]
        points[:, :, 2] += 0.65
        delta = points[:, 1:] - points[:, :-1]
        angles = torch.atan2(delta[..., 1], delta[..., 0])
        zeros = torch.zeros_like(angles)
        quats = math_utils.quat_from_euler_xyz(zeros, zeros, angles)
        scales = torch.ones_like(delta)
        scales[..., 0] = delta.norm(dim=-1).clamp_min(1e-5)
        stopped = (command[:, :2].norm(dim=-1) < 0.02) & (command[:, 2].abs() <= 0.02)
        scales[stopped] = 1e-5
        self.shaft.visualize(
            ((points[:, 1:] + points[:, :-1]) * 0.5).reshape(-1, 3), quats.reshape(-1, 4), scales.reshape(-1, 3)
        )
        tip_scale = torch.tensor([0.12, 0.24, 0.24], device=command.device).repeat(n, 1)
        tip_scale[stopped] = 1e-5
        self.tip.visualize(points[:, -1], quats[:, -1], tip_scale)
