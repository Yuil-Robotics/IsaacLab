# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for go2_moe_cts.

The active policy history is managed by IsaacLab's observation manager. Legacy
small-scan and custom-history helpers remain available for API compatibility but
are not part of the current environment configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def height_scan_small(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_small"),
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the small (0.3×0.2 m) ray-caster, relative to base height.

    Args:
        env: The environment.
        sensor_cfg: Config referencing the small height-scanner sensor.
        offset: Vertical offset added to each reading [m].

    Returns:
        Flattened height readings, shape [num_envs, num_rays].
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene["robot"]
    base_z = asset.data.root_pos_w.torch[:, 2:3]
    heights = base_z - sensor.data.ray_hits_w.torch[..., 2] - offset
    return heights.clip(-1.0, 1.0)


def height_scan_large(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the large (1.5×0.9 m) ray-caster, relative to base height.

    Args:
        env: The environment.
        sensor_cfg: Config referencing the large height-scanner sensor.
        offset: Vertical offset added to each reading [m].

    Returns:
        Flattened height readings, shape [num_envs, num_rays].
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene["robot"]
    base_z = asset.data.root_pos_w.torch[:, 2:3]
    heights = base_z - sensor.data.ray_hits_w.torch[..., 2] - offset
    return heights.clip(-1.0, 1.0)


def foot_contact_binary(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot contact signal (1 = contact, 0 = no contact).

    Args:
        env: The environment.
        sensor_cfg: Config referencing the contact sensor with foot body names.
        threshold: Force threshold [N] for contact detection.

    Returns:
        Binary tensor, shape [num_envs, num_feet].
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    return (forces.norm(dim=-1).max(dim=1).values > threshold).float()


def history_obs(
    env: ManagerBasedRLEnv,
    history_length: int = 10,
) -> torch.Tensor:
    """Stacked proprioceptive observations over the last ``history_length`` steps.

    Concatenates the ``policy`` observation from the past N timesteps into a
    single flat vector. Requires the environment to maintain an ``obs_history``
    buffer attribute (populated in :meth:`Go2EnvCfg._populate_history`).

    This is intentionally kept as a separate obs group so that in Phase-2 the
    MoE encoder can receive it without touching the policy group.

    Args:
        env: The environment.
        history_length: Number of past timesteps to include.

    Returns:
        Flattened history tensor, shape [num_envs, history_length × policy_dim].
    """
    # env.obs_history is a deque (or circular buffer) maintained externally.
    # If not yet set up, return zeros as a safe fallback.
    if not hasattr(env, "obs_history") or env.obs_history is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    # Stack and flatten: [N, T, D] -> [N, T*D]
    stacked = torch.stack(list(env.obs_history), dim=1)  # [N, T, D]
    return stacked.flatten(start_dim=1)
