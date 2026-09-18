# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Current-frame and privileged CTS observation functions."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def single_policy_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Extract the latest 45D frame from the term-major policy history.

    Reusing the history preserves the same noise sample in actor and encoder.
    The policy observation group must precede this group in the configuration.
    """
    if not hasattr(env, "observation_manager"):
        return torch.zeros(env.num_envs, 45, device=env.device)
    history = env.observation_manager.compute_group("policy", update_history=False)
    terms = []
    offset = 0
    for width in (3, 3, 3, 12, 12, 12):
        end = offset + width * 10
        terms.append(history[:, end - width : end])
        offset = end
    return torch.cat(terms, dim=-1)


def joint_effort(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return applied joint torques [N·m] in policy joint order."""
    return env.scene[asset_cfg.name].data.applied_torque.torch[:, asset_cfg.joint_ids]


def foot_contact_force_norm(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return current contact-force magnitudes [N] for each foot."""
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return torch.linalg.vector_norm(forces, dim=-1)


def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return joint accelerations [rad/s²] in policy joint order."""
    return env.scene[asset_cfg.name].data.joint_acc.torch[:, asset_cfg.joint_ids]
