# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled domain randomization for the Robotis HX5 cube task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv


class RandomizeCubeSizeMassInertia(ManagerTermBase):
    """Randomize effective cube size while preserving a constant density.

    PhysX cannot change per-environment USD scales at runtime while replicated
    physics is enabled. This term therefore changes the collision rest offset
    to represent the sampled side length and computes mass and box inertia from
    that same length. Visual geometry remains at its nominal size.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize cached nominal collider properties.

        Args:
            cfg: Configuration of the randomization event.
            env: Environment containing the cube asset.
        """
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject = env.scene[self.asset_cfg.name]
        self.default_rest_offsets = wp.to_torch(self.asset.root_view.get_rest_offsets()).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        side_length_range: tuple[float, float],
        density: float,
        nominal_side_length: float,
    ) -> None:
        """Apply correlated collision size, mass, and inertia values.

        Args:
            env: Environment containing the cube.
            env_ids: Environments to randomize.
            asset_cfg: Scene entity selecting the cube.
            side_length_range: Minimum and maximum effective side length [m].
            density: Constant material density [kg/m³].
            nominal_side_length: Visual and nominal collision side length [m].
        """
        del env, asset_cfg
        if env_ids is None:
            env_ids_device = torch.arange(self.asset.num_instances, device=self.asset.device, dtype=torch.int32)
        else:
            env_ids_device = env_ids.to(device=self.asset.device, dtype=torch.int32)

        side_low, side_high = side_length_range
        side_lengths = side_low + (side_high - side_low) * torch.rand(
            len(env_ids_device), device=self.asset.device
        )
        masses = density * side_lengths**3
        diagonal_inertias = masses * side_lengths**2 / 6.0

        body_ids = torch.zeros(1, device=self.asset.device, dtype=torch.int32)
        self.asset.set_masses_index(
            masses=masses.unsqueeze(-1),
            body_ids=body_ids,
            env_ids=env_ids_device,
        )
        inertias = torch.zeros((len(env_ids_device), 1, 9), device=self.asset.device)
        inertias[:, 0, 0] = diagonal_inertias
        inertias[:, 0, 4] = diagonal_inertias
        inertias[:, 0, 8] = diagonal_inertias
        self.asset.set_inertias_index(
            inertias=inertias,
            body_ids=body_ids,
            env_ids=env_ids_device,
        )

        # A surface moves by half of the full side-length change.
        rest_offsets = self.default_rest_offsets.clone()
        env_ids_cpu = env_ids_device.cpu()
        surface_offsets = 0.5 * (side_lengths.cpu() - nominal_side_length)
        rest_offsets[env_ids_cpu] += surface_offsets.unsqueeze(-1)
        self.asset.root_view.set_rest_offsets(
            wp.from_torch(rest_offsets, dtype=wp.float32),
            wp.from_torch(env_ids_cpu, dtype=wp.int32),
        )

        if not hasattr(self._env, "sim2real_cube_side_length"):
            self._env.sim2real_cube_side_length = torch.full(
                (self.asset.num_instances,), nominal_side_length, device=self.asset.device
            )
            self._env.sim2real_cube_mass = torch.full(
                (self.asset.num_instances,), density * nominal_side_length**3, device=self.asset.device
            )
        self._env.sim2real_cube_side_length[env_ids_device.long()] = side_lengths
        self._env.sim2real_cube_mass[env_ids_device.long()] = masses
