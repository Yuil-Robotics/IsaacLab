# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local visualization marker configurations."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMAND_ARROW_USD_PATH = PROJECT_ROOT / "assets" / "markers" / "arrow_x.usda"


def velocity_arrow_marker_cfg(
    prim_path: str,
    color: tuple[float, float, float],
) -> VisualizationMarkersCfg:
    """Create a local, high-visibility velocity arrow marker.

    Args:
        prim_path: Absolute USD prim path for the marker instances.
        color: RGB diffuse and emissive color.

    Returns:
        Marker configuration using the project-local arrow asset.
    """
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=str(COMMAND_ARROW_USD_PATH),
                scale=(0.5, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    emissive_color=tuple(component * 0.35 for component in color),
                    roughness=0.8,
                ),
            )
        },
    )


GOAL_VELOCITY_MARKER_CFG = velocity_arrow_marker_cfg(
    prim_path="/Visuals/Go2Sim2Sim/velocity_command",
    color=(0.1, 1.0, 0.1),
)
"""Green commanded planar-velocity arrow."""

CURRENT_VELOCITY_MARKER_CFG = velocity_arrow_marker_cfg(
    prim_path="/Visuals/Go2Sim2Sim/velocity_tracking",
    color=(0.1, 0.45, 1.0),
)
"""Blue current planar-velocity arrow."""
