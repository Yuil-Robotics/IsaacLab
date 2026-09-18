# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""3D Multi-Layer Traversability Graph & Path Planning Stack using NetworkX and SciPy."""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


class ElevationCostmap3D:
    """ROS 2 Nav2-Grade Continuous 3D Elevation and Inflation Costmap."""

    def __init__(
        self,
        map_path: Path | str,
        inscribed_radius: float = 0.26,
        inflation_radius: float = 0.45,
        cost_scaling_factor: float = 5.0,
        robot_radius: float | None = None,
    ):
        map_path = Path(map_path)
        self.inscribed_radius = robot_radius if robot_radius is not None else inscribed_radius
        self.inflation_radius = inflation_radius
        self.cost_scaling_factor = cost_scaling_factor

        if map_path.suffix == ".npz":
            npz_path = map_path
        elif map_path.suffix == ".yaml":
            npz_path = map_path.parent / "go2_map_3d.npz"
        else:
            npz_path = map_path.parent / f"{map_path.stem}_3d.npz"

        if not npz_path.exists():
            from .map_converter import convert_ply_to_3d_traversability_map, get_latest_ply

            latest_ply = get_latest_ply()
            if latest_ply is not None:
                print(f"[INFO] Building 3D Traversability Map from {latest_ply.name}...")
                convert_ply_to_3d_traversability_map(
                    latest_ply,
                    npz_path.parent,
                    map_name=npz_path.stem,
                    inscribed_radius=self.inscribed_radius,
                    inflation_radius=self.inflation_radius,
                    cost_scaling_factor=self.cost_scaling_factor,
                )
            else:
                raise FileNotFoundError(f"3D Map file not found: {npz_path}")

        data = np.load(npz_path)
        self.elev_1f = data["elev_1f"]
        self.elev_2f = data["elev_2f"]
        self.is_surface_1f = data["is_surface_1f"]
        self.is_surface_2f = data["is_surface_2f"]
        self.cost_1f = data["cost_1f"]
        self.cost_2f = data["cost_2f"]
        self.origin_x = float(data["origin_x"])
        self.origin_y = float(data["origin_y"])
        self.resolution = float(data["resolution"])
        self.width = int(data["width"])
        self.height = int(data["height"])

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(math.floor((x - self.origin_x) / self.resolution))
        gy = int(math.floor((y - self.origin_y) / self.resolution))
        return np.clip(gx, 0, self.width - 1), np.clip(gy, 0, self.height - 1)

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return (gx + 0.5) * self.resolution + self.origin_x, (gy + 0.5) * self.resolution + self.origin_y

    def get_cost(self, gx: int, gy: int, layer: int = 0) -> float:
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return 254.0
        if layer == 0:
            return float(self.cost_1f[gy, gx])
        else:
            return float(self.cost_2f[gy, gx])

    def get_elevation(self, gx: int, gy: int, layer: int = 0) -> float:
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return 0.0
        if layer == 0:
            return float(self.elev_1f[gy, gx])
        else:
            return float(self.elev_2f[gy, gx])


# Backward compatibility alias
GridCostmap = ElevationCostmap3D


class NetworkX3DNavPlanner:
    """Industrial 3D Topological Multi-Layer Path Planner using NetworkX and SciPy KDTree."""

    def __init__(
        self,
        costmap: ElevationCostmap3D,
        max_step_height: float = 0.095,  # Max allowable vertical step riser in meters
    ):
        self.costmap = costmap
        self.max_step_height = max_step_height
        self.graph = nx.Graph()
        self._build_3d_navgraph()

    def _build_3d_navgraph(self) -> None:
        """Construct a 3D traversability graph across multi-level floors and staircases."""
        w, h = self.costmap.width, self.costmap.height
        nodes_3d = []

        # 1. Add traversable 3D vertices
        for gy in range(h):
            for gx in range(w):
                wx, wy = self.costmap.grid_to_world(gx, gy)
                # 1F Layer
                if self.costmap.is_surface_1f[gy, gx] and self.costmap.cost_1f[gy, gx] < 254.0:
                    z = float(self.costmap.elev_1f[gy, gx])
                    cost = float(self.costmap.cost_1f[gy, gx])
                    nid = (gx, gy, 0)
                    self.graph.add_node(nid, pos=(wx, wy, z), cost=cost)
                    nodes_3d.append((wx, wy, z, nid))

                # 2F Layer
                if self.costmap.is_surface_2f[gy, gx] and self.costmap.cost_2f[gy, gx] < 254.0:
                    z = float(self.costmap.elev_2f[gy, gx])
                    cost = float(self.costmap.cost_2f[gy, gx])
                    nid = (gx, gy, 1)
                    self.graph.add_node(nid, pos=(wx, wy, z), cost=cost)
                    nodes_3d.append((wx, wy, z, nid))

        # 2. Add 8-connectivity edges based on physical traversability
        motions = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (-1, -1, math.sqrt(2)),
        ]

        for u in self.graph.nodes:
            ugx, ugy, _ = u
            ux, uy, uz = self.graph.nodes[u]["pos"]
            ucost = self.graph.nodes[u]["cost"]

            for dx, dy, _ in motions:
                ngx, ngy = ugx + dx, ugy + dy
                if not (0 <= ngx < w and 0 <= ngy < h):
                    continue

                for vlayer in (0, 1):
                    v = (ngx, ngy, vlayer)
                    if v in self.graph:
                        vx, vy, vz = self.graph.nodes[v]["pos"]
                        vcost = self.graph.nodes[v]["cost"]
                        dz = abs(vz - uz)
                        if dz <= self.max_step_height:
                            # Edge weight = 3D distance * Nav2 repulsive cost factor + slope climbing penalty
                            d3d = math.sqrt((vx - ux) ** 2 + (vy - uy) ** 2 + dz**2)
                            weight = d3d * (1.0 + (ucost + vcost) / 24.0) + 3.0 * dz
                            self.graph.add_edge(u, v, weight=weight)

        # 3. Build SciPy KDTree for nearest-neighbor projection
        self.kdtree_pts = np.array([[n[0], n[1], n[2]] for n in nodes_3d], dtype=np.float32)
        self.kdtree_ids = [n[3] for n in nodes_3d]
        self.kdtree = cKDTree(self.kdtree_pts)
        print(
            f"[INFO] NetworkX 3D NavGraph ready: {self.graph.number_of_nodes():,} nodes, "
            f"{self.graph.number_of_edges():,} edges"
        )

    def plan(
        self,
        start_w: tuple[float, float] | tuple[float, float, float],
        goal_w: tuple[float, float] | tuple[float, float, float],
    ) -> list[tuple[float, float, float]]:
        """Plan shortest collision-free 3D path using NetworkX A* and SciPy KDTree."""
        sx, sy = float(start_w[0]), float(start_w[1])
        sz = float(start_w[2]) if len(start_w) >= 3 else 0.0
        gx, gy = float(goal_w[0]), float(goal_w[1])
        gz = float(goal_w[2]) if len(goal_w) >= 3 else 0.0

        if len(self.kdtree_pts) == 0:
            return []

        # Find nearest 3D graph vertices via KDTree
        _, s_idx = self.kdtree.query([sx, sy, sz], k=1)
        _, g_idx = self.kdtree.query([gx, gy, gz], k=1)
        start_node = self.kdtree_ids[int(s_idx)]
        goal_node = self.kdtree_ids[int(g_idx)]

        def dist_heuristic(u, v) -> float:
            p1 = self.graph.nodes[u]["pos"]
            p2 = self.graph.nodes[v]["pos"]
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + ((p1[2] - p2[2]) * 3.0) ** 2)

        try:
            path_nodes = nx.astar_path(
                self.graph,
                start_node,
                goal_node,
                heuristic=dist_heuristic,
                weight="weight",
            )
            raw_path_3d = [self.graph.nodes[nid]["pos"] for nid in path_nodes]
            # Smooth along graph nodes without shortcutting through void / railings
            return self._smooth_graph_path(raw_path_3d, window=5)
        except nx.NetworkXNoPath:
            print("[WARN] NetworkX 3D Planner: No path found between vertices.")
            return []

    def _smooth_graph_path(
        self, path: list[tuple[float, float, float]], window: int = 5
    ) -> list[tuple[float, float, float]]:
        """Smooth graph trajectory along valid physical surface without jumping off edges."""
        if len(path) <= window:
            return path

        smoothed = []
        for i in range(len(path)):
            i_min = max(0, i - window // 2)
            i_max = min(len(path), i + window // 2 + 1)
            sub = path[i_min:i_max]
            avg_x = float(np.mean([p[0] for p in sub]))
            avg_y = float(np.mean([p[1] for p in sub]))
            avg_z = float(np.mean([p[2] for p in sub]))
            smoothed.append((avg_x, avg_y, avg_z))

        return smoothed


# Alias for seamless replacement
AStarPlanner3D = NetworkX3DNavPlanner


class PurePursuitController3D:
    """3D Monotonic Regulated Pure Pursuit Controller for Quadruped Locomotion."""

    def __init__(
        self,
        lookahead_dist: float = 0.55,
        max_lin_vel: float = 0.55,
        stair_climb_vel: float = 0.45,
        max_ang_vel: float = 0.90,
        goal_tolerance: float = 0.30,
    ):
        self.lookahead_dist = lookahead_dist
        self.max_lin_vel = max_lin_vel
        self.stair_climb_vel = stair_climb_vel
        self.max_ang_vel = max_ang_vel
        self.goal_tolerance = goal_tolerance
        self.current_idx = 0

    def reset(self) -> None:
        """Reset forward path progress index."""
        self.current_idx = 0

    def compute_command(
        self,
        robot_pos: tuple[float, float] | tuple[float, float, float] | np.ndarray,
        robot_yaw: float,
        path: list[tuple[float, float]] | list[tuple[float, float, float]],
    ) -> tuple[float, float, float, bool]:
        """Compute (vx, vy, wz, is_goal_reached) with strictly monotonic forward progress."""
        if not path or self.current_idx >= len(path):
            return 0.0, 0.0, 0.0, True

        rx, ry = float(robot_pos[0]), float(robot_pos[1])
        rz = float(robot_pos[2]) if len(robot_pos) >= 3 else 0.0

        goal_dist_xy = math.hypot(path[-1][0] - rx, path[-1][1] - ry)
        goal_dist_z = abs((path[-1][2] if len(path[-1]) >= 3 else 0.0) - rz)

        if goal_dist_xy < self.goal_tolerance and goal_dist_z < 0.35:
            return 0.0, 0.0, 0.0, True

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        # 1. Monotonic forward progress: search only forward from self.current_idx
        search_end = min(len(path), self.current_idx + 40)
        best_idx = self.current_idx
        min_d3d = float("inf")

        for i in range(self.current_idx, search_end):
            pt = path[i]
            dx = pt[0] - rx
            dy = pt[1] - ry
            dz = (pt[2] - rz) if len(pt) >= 3 else 0.0
            d3d = math.sqrt(dx * dx + dy * dy + 3.0 * dz * dz)

            forward_proj = dx * cos_yaw + dy * sin_yaw
            if forward_proj < -0.05 and d3d < 0.40:
                best_idx = max(best_idx, i + 1)
            elif d3d < min_d3d:
                min_d3d = d3d
                best_idx = i

        self.current_idx = min(len(path) - 1, max(self.current_idx, best_idx))

        # 2. Lookahead target waypoint ahead of current progress
        target_pt = path[-1]
        for i in range(self.current_idx, len(path)):
            pt = path[i]
            dx = pt[0] - rx
            dy = pt[1] - ry
            d = math.hypot(dx, dy)
            if d >= self.lookahead_dist:
                target_pt = pt
                break

        # 3. Heading angle error
        dx = target_pt[0] - rx
        dy = target_pt[1] - ry
        target_yaw = math.atan2(dy, dx)
        alpha = math.atan2(math.sin(target_yaw - robot_yaw), math.cos(target_yaw - robot_yaw))

        # 4. Check stair state
        is_on_stairs = False
        if len(target_pt) >= 3:
            dz = abs(target_pt[2] - rz)
            is_on_stairs = dz > 0.06 or (0.05 < rz < 0.75)

        align_factor = max(0.30, math.cos(alpha))
        base_vel = self.stair_climb_vel if is_on_stairs else self.max_lin_vel
        vx = base_vel * align_factor

        if goal_dist_xy < 0.70 and not is_on_stairs:
            vx *= max(0.35, goal_dist_xy / 0.70)

        wz = float(np.clip(1.25 * alpha, -self.max_ang_vel, self.max_ang_vel))
        return float(vx), 0.0, float(wz), False


# Backward compatibility alias
PurePursuitController = PurePursuitController3D
