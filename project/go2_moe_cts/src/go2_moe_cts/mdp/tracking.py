# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dimensionless command-following scores for terrain progression."""

import torch


def command_tracking_score(target: torch.Tensor, actual: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """Score signed velocity tracking in [0, 1].

    Args:
        target: Body-frame commands [vx m/s, vy m/s, wz rad/s].
        actual: Measured body-frame velocities in the same units.
        limits: Positive per-axis normalization limits in the same units.

    Returns:
        One minus relative vector error, clipped to [0, 1]. The denominator
        has a 0.1 floor in normalized units so zero commands remain evaluable.
    """
    reference = target / limits
    error = ((actual - target) / limits).norm(dim=-1)
    score = (1 - error / reference.norm(dim=-1).clamp_min(0.1)).clamp(0, 1)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
