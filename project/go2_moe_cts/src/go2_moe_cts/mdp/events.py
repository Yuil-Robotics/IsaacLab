# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motor calibration randomization for the CTS environment."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def randomize_motor_zero_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float] = (-0.035, 0.035),
) -> None:
    """Randomize encoder zero offsets relative to the nominal pose [rad].

    Args:
        env: Environment with the history-aware joint-position action.
        env_ids: Environments being reset, or all environments.
        offset_range: Uniform additive encoder offset bounds [rad].
    """
    term = env.action_manager.get_term("joint_pos")
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    nominal = term._asset.data.default_joint_pos.torch[:, term._joint_ids]
    noise = torch.empty_like(nominal[ids]).uniform_(*offset_range)
    term._offset[ids] = nominal[ids] + noise
