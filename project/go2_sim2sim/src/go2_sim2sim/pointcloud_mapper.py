# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-time voxel point cloud mapper and PLY exporter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def create_map_marker_cfg(prim_path: str = "/Visuals/AccumulatedMap") -> VisualizationMarkersCfg:
    """Create visualization markers for the accumulated 3D point cloud map."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.9, 0.4),
                    roughness=0.5,
                ),
            )
        },
    )


class PointCloudMapper:
    """High-performance real-time voxel 3D point cloud accumulator and exporter."""

    def __init__(
        self,
        voxel_size: float = 0.06,
        max_visual_points: int = 3500,
        output_dir: Path | str | None = None,
    ):
        self.voxel_size = voxel_size
        self.max_visual_points = max_visual_points
        self.output_dir = Path(output_dir) if output_dir else Path("logs/maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Fast voxel storage
        self._voxel_set: set[tuple[int, int, int]] = set()
        self._points_list: list[np.ndarray] = []
        self._visualizer: VisualizationMarkers | None = None
        self._vis_enabled = True
        self._dirty = False

    def init_visualizer(self, prim_path: str = "/Visuals/AccumulatedMap") -> None:
        """Initialize the real-time USD point cloud map visualizer."""
        marker_cfg = create_map_marker_cfg(prim_path)
        self._visualizer = VisualizationMarkers(marker_cfg)

    def toggle_visualization(self) -> bool:
        """Toggle in-sim USD point cloud rendering on/off."""
        self._vis_enabled = not self._vis_enabled
        if not self._vis_enabled and self._visualizer is not None:
            try:
                self._visualizer.visualize(torch.empty((0, 3), dtype=torch.float32, device="cuda:0"))
            except Exception:
                pass
        return self._vis_enabled

    def add_points(self, points: torch.Tensor | np.ndarray) -> int:
        """Add new 3D hit points from LiDAR scan in world frame.

        Args:
            points: Tensor or ndarray of shape (N, 3) in world coordinates.

        Returns:
            Current total unique voxel point count.
        """
        if isinstance(points, torch.Tensor):
            pts = points.detach().cpu().numpy()
        else:
            pts = np.asarray(points)

        # Filter out inf / nan / invalid points
        valid_mask = np.isfinite(pts).all(axis=-1)
        valid_pts = pts[valid_mask]
        if len(valid_pts) == 0:
            return len(self._points_list)

        # Filter points within reasonable arena range
        dist_sq = valid_pts[:, 0] ** 2 + valid_pts[:, 1] ** 2
        in_range = (dist_sq < 250.0) & (valid_pts[:, 2] > -0.5) & (valid_pts[:, 2] < 3.5)
        valid_pts = valid_pts[in_range]
        if len(valid_pts) == 0:
            return len(self._points_list)

        # Fast Voxel hashing
        voxel_indices = np.floor(valid_pts / self.voxel_size).astype(np.int32)
        added = False
        for idx, pt in zip(voxel_indices, valid_pts):
            key = (int(idx[0]), int(idx[1]), int(idx[2]))
            if key not in self._voxel_set:
                self._voxel_set.add(key)
                self._points_list.append(pt)
                added = True

        if added:
            self._dirty = True

        return len(self._points_list)

    def update_visualization(self) -> None:
        """Update USD point cloud markers in Isaac Sim with throttled point count."""
        if not self._vis_enabled or self._visualizer is None or not self._dirty or len(self._points_list) == 0:
            return

        self._dirty = False
        num_pts = len(self._points_list)

        if num_pts <= self.max_visual_points:
            pts_vis = np.array(self._points_list, dtype=np.float32)
        else:
            # Uniform decimation for fast, non-blocking rendering
            step = num_pts / self.max_visual_points
            indices = (np.arange(self.max_visual_points) * step).astype(np.int32)
            pts_vis = np.array([self._points_list[i] for i in indices], dtype=np.float32)

        positions = torch.tensor(pts_vis, dtype=torch.float32, device="cuda:0")
        try:
            self._visualizer.visualize(positions)
        except Exception:
            pass

    def get_points(self) -> np.ndarray:
        """Return all unique mapped 3D points as an (N, 3) numpy array."""
        if not self._points_list:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(self._points_list, dtype=np.float32)

    def save_ply(self, filename: str | None = None) -> str:
        """Export accumulated 3D map to a colored PLY file.

        Args:
            filename: Target file path (auto-generated if None).

        Returns:
            Absolute path of the saved PLY file.
        """
        pts = self.get_points()
        if len(pts) == 0:
            print("[WARN] Point cloud map is empty. Nothing saved.")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(self.output_dir / f"go2_scanned_map_{timestamp}.ply")

        # Color map by height (Z-gradient: blue floor -> green -> yellow -> red)
        z = pts[:, 2]
        z_min, z_max = z.min(), max(z.max(), z.min() + 0.1)
        z_norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)

        # Simple Turbo / Jet colormap
        r = np.clip(1.5 - np.abs(z_norm * 4.0 - 3.0), 0.0, 1.0) * 255
        g = np.clip(1.5 - np.abs(z_norm * 4.0 - 2.0), 0.0, 1.0) * 255
        b = np.clip(1.5 - np.abs(z_norm * 4.0 - 1.0), 0.0, 1.0) * 255
        colors = np.stack([r, g, b], axis=-1).astype(np.uint8)

        num_points = len(pts)
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {num_points}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]

        with open(filename, "w") as f:
            f.write("\n".join(header) + "\n")
            for (x, y, z_val), (cr, cg, cb) in zip(pts, colors):
                f.write(f"{x:.4f} {y:.4f} {z_val:.4f} {cr} {cg} {cb}\n")

        print(f"\n[Saved 3D Map] {num_points:,} points exported to: {filename}")
        return filename

    def clear(self) -> None:
        """Clear all accumulated points."""
        self._voxel_set.clear()
        self._points_list.clear()
        self._dirty = False
        if self._visualizer is not None:
            try:
                empty_positions = torch.empty((0, 3), dtype=torch.float32, device="cuda:0")
                self._visualizer.visualize(empty_positions)
            except Exception:
                pass
        print("\n[Map Cleared] Reset 3D map buffer.")
