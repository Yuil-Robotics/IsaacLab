# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""3D Duplex (2-Story Mezzanine) with Staircases scene configuration for Go2 LiDAR mapping."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


def add_3d_obstacle_map_to_scene(scene_cfg, prim_path_prefix: str = "/World/obstacles") -> None:
    """Attach 2-story duplex (mezzanine) structure with dual staircases to InteractiveSceneCfg.

    Layout Architecture (Spacious & Navigable):
      - Central Plaza (Ground Z = 0.0m): Wide open central hall (9m x 5.5m) around spawn (0,0)
        and stair entrances with generous spacing (> 2.8m - 3.5m clearance everywhere).
      - Dual Staircases (West & East): 10 steps (rise: 0.08m, run: 0.32m, width: 2.2m, angle: ~14.0°)
        allowing Unitree Go2 quadruped to smoothly climb up/down between 1st and 2nd floors.
      - 2nd Floor Mezzanine Deck (Z = 0.80m): West/East wings connected via a skybridge, with an open
        central atrium void overlooking the 1st floor.
      - Safety Railings: 0.45m high parapets around the 2nd floor edges and atrium.
      - Under-deck Clearance: 0.72m high pilotis colonnade with wide 5.0m central underpass.
    """
    wall_mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.85, 0.85, 0.88),
        roughness=0.6,
    )
    deck_mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.65, 0.68, 0.75),
        roughness=0.4,
    )
    stair_mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.82, 0.52, 0.25),
        roughness=0.5,
    )
    railing_mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.20, 0.25, 0.30),
        roughness=0.3,
    )
    pillar_mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.30, 0.55, 0.80),
        roughness=0.35,
    )
    collision_props = sim_utils.CollisionPropertiesCfg()

    # ----------------------------------------------------------------------------------------------
    # 1. Outer Arena Boundary Walls (16m x 16m Arena, Height 2.4m for 2-story coverage)
    # ----------------------------------------------------------------------------------------------
    outer_walls = [
        ("wall_north", (0.0, 8.0, 1.2), (16.0, 0.3, 2.4)),
        ("wall_south", (0.0, -8.0, 1.2), (16.0, 0.3, 2.4)),
        ("wall_east", (8.0, 0.0, 1.2), (0.3, 16.0, 2.4)),
        ("wall_west", (-8.0, 0.0, 1.2), (0.3, 16.0, 2.4)),
    ]
    for name, pos, size in outer_walls:
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=size,
                    visual_material=wall_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )

    # ----------------------------------------------------------------------------------------------
    # 2. Dual Staircases (West and East, 10 Steps: 0.08m Rise, 0.32m Run, 2.2m Width)
    #    Ascends from Y = -2.2m (Z = 0.0m) to Y = 1.0m (Z = 0.80m)
    # ----------------------------------------------------------------------------------------------
    num_steps = 10
    step_rise = 0.08   # 8 cm height per step (optimal for Go2 quadruped trotting)
    step_run = 0.32    # 32 cm depth per step
    step_width = 2.2   # 2.2 m wide for easy navigation
    start_y = -2.2

    # West Staircase (Center X = -4.0m)
    for k in range(num_steps):
        step_height = (k + 1) * step_rise
        step_y = start_y + (k + 0.5) * step_run
        step_z = step_height / 2.0
        name = f"stair_west_{k}"
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=(step_width, step_run, step_height),
                    visual_material=stair_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, step_y, step_z)),
            ),
        )

    # East Staircase (Center X = +4.0m)
    for k in range(num_steps):
        step_height = (k + 1) * step_rise
        step_y = start_y + (k + 0.5) * step_run
        step_z = step_height / 2.0
        name = f"stair_east_{k}"
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=(step_width, step_run, step_height),
                    visual_material=stair_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(4.0, step_y, step_z)),
            ),
        )

    # Stair Side Guardrails
    stair_guardrails = [
        ("guard_west_outer", (-5.15, -0.6, 0.60), (0.10, 3.20, 0.90)),
        ("guard_west_inner", (-2.85, -0.6, 0.60), (0.10, 3.20, 0.90)),
        ("guard_east_inner", (2.85, -0.6, 0.60), (0.10, 3.20, 0.90)),
        ("guard_east_outer", (5.15, -0.6, 0.60), (0.10, 3.20, 0.90)),
    ]
    for name, pos, size in stair_guardrails:
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=size,
                    visual_material=railing_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )

    # ----------------------------------------------------------------------------------------------
    # 3. 2nd Floor Mezzanine Decks (Top Surface at Z = 0.80m, Deck Thickness = 0.08m)
    # ----------------------------------------------------------------------------------------------
    mezzanine_decks = [
        ("deck_west", (-4.675, 4.425, 0.76), (6.35, 6.85, 0.08)),
        ("deck_east", (4.675, 4.425, 0.76), (6.35, 6.85, 0.08)),
        ("deck_skybridge", (0.0, 6.425, 0.76), (3.0, 2.85, 0.08)),
    ]
    for name, pos, size in mezzanine_decks:
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=size,
                    visual_material=deck_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )

    # ----------------------------------------------------------------------------------------------
    # 4. 2nd Floor Safety Railings / Parapets (Height 0.45m above 2nd floor, Z: 0.80m to 1.25m)
    # ----------------------------------------------------------------------------------------------
    second_floor_railings = [
        # Front Mezzanine Edge Railings (leaving gaps at X=[-5.1, -2.9] and X=[2.9, 5.1] for stairs)
        ("railing_front_w_outer", (-6.475, 1.0, 1.025), (2.75, 0.12, 0.45)),
        ("railing_front_w_inner", (-2.175, 1.0, 1.025), (1.35, 0.12, 0.45)),
        ("railing_front_e_inner", (2.175, 1.0, 1.025), (1.35, 0.12, 0.45)),
        ("railing_front_e_outer", (6.475, 1.0, 1.025), (2.75, 0.12, 0.45)),
        # Central Atrium Void Railings (Overlooking 1st floor)
        ("railing_atrium_west", (-1.5, 3.0, 1.025), (0.12, 4.0, 0.45)),
        ("railing_atrium_east", (1.5, 3.0, 1.025), (0.12, 4.0, 0.45)),
        ("railing_bridge_front", (0.0, 5.0, 1.025), (3.0, 0.12, 0.45)),
    ]
    for name, pos, size in second_floor_railings:
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=size,
                    visual_material=railing_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )

    # ----------------------------------------------------------------------------------------------
    # 5. Under-Mezzanine Support Columns (1st Floor Under-Deck, Height 0.72m, Radius 0.16m)
    #    Arranged with wide 5.0m central underpass clearance
    # ----------------------------------------------------------------------------------------------
    under_deck_pillars = [
        ("pillar_uw1", (-5.2, 3.5, 0.36), 0.16, 0.72),
        ("pillar_uw2", (-5.2, 6.2, 0.36), 0.16, 0.72),
        ("pillar_uw3", (-2.5, 3.5, 0.36), 0.16, 0.72),
        ("pillar_uw4", (-2.5, 6.2, 0.36), 0.16, 0.72),
        ("pillar_ue1", (5.2, 3.5, 0.36), 0.16, 0.72),
        ("pillar_ue2", (5.2, 6.2, 0.36), 0.16, 0.72),
        ("pillar_ue3", (2.5, 3.5, 0.36), 0.16, 0.72),
        ("pillar_ue4", (2.5, 6.2, 0.36), 0.16, 0.72),
        ("pillar_um1", (-1.5, 5.0, 0.36), 0.16, 0.72),
        ("pillar_um2", (1.5, 5.0, 0.36), 0.16, 0.72),
    ]
    for name, pos, radius, height in under_deck_pillars:
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCylinderCfg(
                    radius=radius,
                    height=height,
                    visual_material=pillar_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )

    # ----------------------------------------------------------------------------------------------
    # 6. 2nd Floor Room Partitions (Height 1.2m above 2nd floor deck, Z: 0.80m to 2.0m)
    #    Positioned with 2.5m+ wide aisles around them
    # ----------------------------------------------------------------------------------------------
    second_floor_walls = [
        ("wall_2f_west", (-5.2, 4.5, 1.40), (0.20, 3.2, 1.20)),
        ("wall_2f_east", (5.2, 4.5, 1.40), (0.20, 3.2, 1.20)),
    ]
    for name, pos, size in second_floor_walls:
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=size,
                    visual_material=wall_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )

    # ----------------------------------------------------------------------------------------------
    # 7. 1st Floor South Ground Rooms (Y < -4.0) with Wide Corridors & Spacious Central Plaza
    #    Leaving a completely open 9m x 5.5m turning plaza around spawn (0,0) and stair entries
    # ----------------------------------------------------------------------------------------------
    ground_partitions = [
        ("ground_wall_sw", (-5.2, -5.5, 0.9), (0.25, 3.2, 1.8)),
        ("ground_wall_se", (5.2, -5.5, 0.9), (0.25, 3.2, 1.8)),
        ("ground_wall_mid", (0.0, -5.8, 0.9), (2.8, 0.25, 1.8)),
    ]
    for name, pos, size in ground_partitions:
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCuboidCfg(
                    size=size,
                    visual_material=wall_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )

    ground_pillars = [
        ("ground_pillar_sw", (-5.2, -2.5, 0.9), 0.20, 1.8),
        ("ground_pillar_se", (5.2, -2.5, 0.9), 0.20, 1.8),
    ]
    for name, pos, radius, height in ground_pillars:
        setattr(
            scene_cfg,
            name,
            AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{name}",
                spawn=sim_utils.MeshCylinderCfg(
                    radius=radius,
                    height=height,
                    visual_material=pillar_mat,
                    collision_props=collision_props,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )

    # ----------------------------------------------------------------------------------------------
    # 8. Dynamically register obstacle prim path in LiDAR and Camera raycasters if sensor is present
    # ----------------------------------------------------------------------------------------------
    for sensor_name in ("lidar", "camera"):
        if hasattr(scene_cfg, sensor_name) and getattr(scene_cfg, sensor_name) is not None:
            sensor = getattr(scene_cfg, sensor_name)
            if hasattr(sensor, "mesh_prim_paths"):
                obs_pattern = f"{prim_path_prefix}/.*"
                if obs_pattern not in sensor.mesh_prim_paths:
                    sensor.mesh_prim_paths.append(obs_pattern)
