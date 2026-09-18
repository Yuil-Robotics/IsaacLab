# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""3D Point Cloud to ROS 2 2D Occupancy Grid and 3D Traversability Elevation Map Converter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPS_DIR = PROJECT_ROOT / "logs" / "maps"


def get_latest_ply() -> Path | None:
    """Find the most recently created PLY map in logs/maps/."""
    if not MAPS_DIR.exists():
        return None
    ply_files = sorted(MAPS_DIR.glob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ply_files[0] if ply_files else None


def load_ply_points(filepath: Path) -> np.ndarray:
    """Load (x, y, z) points from ASCII PLY file."""
    positions = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        header = True
        for line in f:
            line_str = line.strip()
            if header:
                if line_str == "end_header":
                    header = False
                continue
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(positions, dtype=np.float32)


def convert_ply_to_3d_traversability_map(
    ply_path: Path,
    output_dir: Path | None = None,
    map_name: str = "go2_map_3d",
    resolution: float = 0.05,
    arena_padding: float = 0.5,
    inscribed_radius: float = 0.42,  # Go2 body width & hip envelope (~42cm safety radius)
    inflation_radius: float = 0.90,  # 90cm repulsive decay field for center-corridor routing
    cost_scaling_factor: float = 3.5,
) -> Path:
    """Convert 3D PLY point cloud into a continuous 3D Multi-Layer Traversability & Elevation Costmap.

    Applies robot body footprint-aware inflation to walls, guardrails, and cliff drop edges
    so paths stay centered and never hug edges or corners.
    """
    if output_dir is None:
        output_dir = ply_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Analyzing 3D Traversability Map from: {ply_path.name}")
    points = load_ply_points(ply_path)
    if len(points) == 0:
        raise ValueError("Point cloud has 0 points.")

    all_pts_2d = points[:, :2]
    min_x = float(np.floor(all_pts_2d[:, 0].min() - arena_padding))
    max_x = float(np.ceil(all_pts_2d[:, 0].max() + arena_padding))
    min_y = float(np.floor(all_pts_2d[:, 1].min() - arena_padding))
    max_y = float(np.ceil(all_pts_2d[:, 1].max() + arena_padding))

    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    print(f"[INFO] 3D Map Grid Dimensions: {width} x {height} ({width * resolution:.2f}m x {height * resolution:.2f}m)")

    gx = np.clip(np.floor((points[:, 0] - min_x) / resolution).astype(int), 0, width - 1)
    gy = np.clip(np.floor((points[:, 1] - min_y) / resolution).astype(int), 0, height - 1)
    z = points[:, 2]

    # 1. 1F Ground Surface (z < 0.12m)
    is_surface_1f = np.zeros((height, width), dtype=bool)
    is_surface_1f[gy[z < 0.12], gx[z < 0.12]] = True

    # 2. Stair Traversable Surface (0.04m <= z <= 0.82m)
    m_stairs = (z >= 0.04) & (z <= 0.82) & (points[:, 1] >= -2.3) & (points[:, 1] <= 1.1) & (np.abs(np.abs(points[:, 0]) - 4.0) <= 1.1)
    is_surface_1f[gy[m_stairs], gx[m_stairs]] = True
    # Fill small laser scan line gaps on solid floor
    is_surface_1f = binary_closing(is_surface_1f, structure=np.ones((5, 5)))

    # 3. 2F Deck Surface (0.75m <= z <= 0.88m, y >= 0.9m)
    is_surface_2f = np.zeros((height, width), dtype=bool)
    m_deck_2f = (z >= 0.75) & (z <= 0.88) & (points[:, 1] >= 0.9)
    is_surface_2f[gy[m_deck_2f], gx[m_deck_2f]] = True
    is_surface_2f[gy[m_stairs], gx[m_stairs]] = True
    is_surface_2f = binary_closing(is_surface_2f, structure=np.ones((5, 5)))

    # 4. Obstacles on 1F (walls, pillars with 0.15m <= z <= 0.70m and outer/cliff boundaries)
    is_obs_1f = np.zeros((height, width), dtype=bool)
    m_obs_1f = (z >= 0.15) & (z <= 0.70) & ~m_stairs
    is_obs_1f[gy[m_obs_1f], gx[m_obs_1f]] = True
    is_obs_1f[~is_surface_1f] = True

    # 5. Obstacles on 2F (railings, 2F walls with 0.88m < z <= 2.3m and void/balcony drop edges)
    is_obs_2f = np.zeros((height, width), dtype=bool)
    m_obs_2f = (z > 0.88) & (z <= 2.3) & ~m_stairs
    is_obs_2f[gy[m_obs_2f], gx[m_obs_2f]] = True
    is_obs_2f[~is_surface_2f] = True

    # 6. Nav2 Continuous Exponential Inflation Layer (Inscribed margin = 0.42m)
    dist_1f = distance_transform_edt(~is_obs_1f) * resolution
    cost_1f = np.zeros((height, width), dtype=np.float32)
    cost_1f[dist_1f <= inscribed_radius] = 254.0
    decay_mask_1f = (dist_1f > inscribed_radius) & (dist_1f <= inflation_radius)
    cost_1f[decay_mask_1f] = 253.0 * np.exp(-cost_scaling_factor * (dist_1f[decay_mask_1f] - inscribed_radius))

    dist_2f = distance_transform_edt(~is_obs_2f) * resolution
    cost_2f = np.zeros((height, width), dtype=np.float32)
    cost_2f[dist_2f <= inscribed_radius] = 254.0
    decay_mask_2f = (dist_2f > inscribed_radius) & (dist_2f <= inflation_radius)
    cost_2f[decay_mask_2f] = 253.0 * np.exp(-cost_scaling_factor * (dist_2f[decay_mask_2f] - inscribed_radius))

    # 7. Surface Elevations
    elev_1f = np.zeros((height, width), dtype=np.float32)
    elev_2f = np.full((height, width), 0.80, dtype=np.float32)
    for i in np.where(m_stairs)[0]:
        elev_1f[gy[i], gx[i]] = max(elev_1f[gy[i], gx[i]], z[i])
        elev_2f[gy[i], gx[i]] = max(elev_2f[gy[i], gx[i]], z[i])

    npz_path = output_dir / f"{map_name}.npz"
    np.savez_compressed(
        npz_path,
        elev_1f=elev_1f,
        elev_2f=elev_2f,
        is_surface_1f=is_surface_1f,
        is_surface_2f=is_surface_2f,
        cost_1f=cost_1f,
        cost_2f=cost_2f,
        origin_x=min_x,
        origin_y=min_y,
        resolution=resolution,
        width=width,
        height=height,
    )
    print(f"[INFO] Continuous 3D Traversability Elevation Map saved: {npz_path}")
    return npz_path


def convert_ply_to_ros2_map(
    ply_path: Path,
    output_dir: Path | None = None,
    map_name: str = "go2_map",
    resolution: float = 0.05,
    min_obstacle_height: float = 0.12,
    max_obstacle_height: float = 1.80,
    arena_padding: float = 0.5,
) -> tuple[Path, Path]:
    """Convert 3D point cloud PLY to ROS 2 standard 2D Costmap (PGM + YAML) and 3D Traversability Map."""
    if output_dir is None:
        output_dir = ply_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 3D continuous traversability elevation map
    convert_ply_to_3d_traversability_map(
        ply_path=ply_path,
        output_dir=output_dir,
        map_name=f"{map_name}_3d",
        resolution=resolution,
        arena_padding=arena_padding,
    )

    print(f"[INFO] Loading 3D point cloud: {ply_path}")
    points = load_ply_points(ply_path)
    print(f"[INFO] Total loaded points: {len(points):,}")

    if len(points) == 0:
        raise ValueError("Point cloud has 0 points.")

    ground_mask = points[:, 2] < min_obstacle_height
    obstacle_mask = (points[:, 2] >= min_obstacle_height) & (points[:, 2] <= max_obstacle_height)

    obstacle_pts = points[obstacle_mask]
    ground_pts = points[ground_mask]

    all_pts_2d = points[:, :2]
    min_x = np.floor(all_pts_2d[:, 0].min() - arena_padding)
    max_x = np.ceil(all_pts_2d[:, 0].max() + arena_padding)
    min_y = np.floor(all_pts_2d[:, 1].min() - arena_padding)
    max_y = np.ceil(all_pts_2d[:, 1].max() + arena_padding)

    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    grid = np.full((height, width), 205, dtype=np.uint8)

    if len(ground_pts) > 0:
        gx = np.clip(np.floor((ground_pts[:, 0] - min_x) / resolution).astype(int), 0, width - 1)
        gy = np.clip(np.floor((ground_pts[:, 1] - min_y) / resolution).astype(int), 0, height - 1)
        grid[gy, gx] = 254

    if len(obstacle_pts) > 0:
        ox = np.clip(np.floor((obstacle_pts[:, 0] - min_x) / resolution).astype(int), 0, width - 1)
        oy = np.clip(np.floor((obstacle_pts[:, 1] - min_y) / resolution).astype(int), 0, height - 1)
        grid[oy, ox] = 0

    pgm_image = np.flipud(grid)

    pgm_path = output_dir / f"{map_name}.pgm"
    with open(pgm_path, "wb") as f:
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        f.write(header)
        f.write(pgm_image.tobytes())

    yaml_path = output_dir / f"{map_name}.yaml"
    origin_x = float(min_x)
    origin_y = float(min_y)
    origin_z = 0.0

    yaml_content = f"""image: {pgm_path.name}
mode: trinary
resolution: {resolution:.4f}
origin: [{origin_x:.4f}, {origin_y:.4f}, {origin_z:.4f}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[Map Converted Successfully!]")
    print(f"   - PGM Image: {pgm_path}")
    print(f"   - YAML File: {yaml_path}")
    return pgm_path, yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 3D PLY point cloud to ROS 2 2D Occupancy Grid and 3D Map.")
    parser.add_argument("ply_file", nargs="?", type=Path, default=None, help="Input .ply file")
    parser.add_argument("--res", type=float, default=0.05, help="Grid resolution in meters (default: 0.05)")
    parser.add_argument("--name", default="go2_map", help="Output map name (default: go2_map)")
    args = parser.parse_args()

    ply_path = args.ply_file
    if ply_path is None:
        ply_path = get_latest_ply()
        if ply_path is None:
            print(f"[ERROR] No .ply map files found in {MAPS_DIR}")
            return
        print(f"[INFO] Auto-selected latest PLY map: {ply_path.name}")

    convert_ply_to_ros2_map(
        ply_path=ply_path,
        map_name=args.name,
        resolution=args.res,
    )


if __name__ == "__main__":
    main()
