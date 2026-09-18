# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project terrain importer that uses a pinned local ground-plane USD."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter

from .asset_cfg import LOCAL_GROUND_USD_PATH


class LocalPlaneTerrainImporter(TerrainImporter):
    """Load the standard Isaac ground plane from the project asset mirror."""

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)) -> None:
        """Add a local ground plane while preserving configured material values."""
        prim_path = self.cfg.prim_path + f"/{name}"
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        self.terrain_prim_paths.append(prim_path)

        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            color = material.get("diffuse_color", color)

        ground_plane_cfg = sim_utils.GroundPlaneCfg(
            usd_path=str(LOCAL_GROUND_USD_PATH),
            physics_material=self.cfg.physics_material,
            size=size,
            color=color,
        )
        ground_plane_cfg.func(prim_path, ground_plane_cfg)
