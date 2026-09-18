# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Front-facing 3D LiDAR configuration for Unitree Go2 (Forward/Elevation/Azimuth FOV)."""

from __future__ import annotations

import copy
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.utils.math import quat_apply

# 🟢 초록색 라이다 포인트 클라우드 마커
GREEN_RAY_CASTER_MARKER_CFG = copy.deepcopy(RAY_CASTER_MARKER_CFG)
GREEN_RAY_CASTER_MARKER_CFG.prim_path = "/Visuals/LidarGreenHits"
GREEN_RAY_CASTER_MARKER_CFG.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.2)
GREEN_RAY_CASTER_MARKER_CFG.markers["hit"].visual_material.emissive_color = (0.0, 0.6, 0.1)
GREEN_RAY_CASTER_MARKER_CFG.markers["hit"].radius = 0.02

# 🔵 파란색 뎁스 카메라 포인트 클라우드 마커
BLUE_DEPTH_CAMERA_MARKER_CFG = copy.deepcopy(RAY_CASTER_MARKER_CFG)
BLUE_DEPTH_CAMERA_MARKER_CFG.prim_path = "/Visuals/DepthBlueHits"
BLUE_DEPTH_CAMERA_MARKER_CFG.markers["hit"].visual_material.diffuse_color = (0.0, 0.4, 1.0)
BLUE_DEPTH_CAMERA_MARKER_CFG.markers["hit"].visual_material.emissive_color = (0.0, 0.2, 0.8)
BLUE_DEPTH_CAMERA_MARKER_CFG.markers["hit"].radius = 0.015


from isaaclab.sensors.camera.utils import create_pointcloud_from_depth


def depth_image_to_world_point_cloud(
    depth: torch.Tensor,
    cam_pos_w: torch.Tensor,
    cam_quat_w: torch.Tensor,
    intrinsic_matrix: torch.Tensor | None = None,
    hfov_deg: float = 87.0,
    subsample: int = 16,
    min_range: float = 0.3,
    max_range: float = 3.5,
) -> torch.Tensor:
    """Convert depth image tensor into 3D world coordinate point cloud (Blue Depth Points).

    Args:
        depth: Depth tensor from IsaacLab camera (H, W) or (1, H, W, 1).
        cam_pos_w: Camera world position [3].
        cam_quat_w: Camera world quaternion [4] (x, y, z, w) in ROS frame.
        intrinsic_matrix: Optional 3x3 camera calibration matrix.
        hfov_deg: Horizontal FOV in degrees (D435i default: 87.0°).
        subsample: Spatial sampling stride (default 16 for high FPS).
        min_range: Minimum depth distance in meters (default 0.3m).
        max_range: Maximum depth distance in meters (default 3.5m).
    """
    if not isinstance(depth, torch.Tensor):
        depth_t = torch.as_tensor(depth)
    else:
        depth_t = depth

    depth_t = depth_t.squeeze()
    if depth_t.dim() != 2:
        return torch.empty((0, 3), device=depth_t.device)

    device = depth_t.device
    H, W = depth_t.shape[0], depth_t.shape[1]
    sub_depth = depth_t[::subsample, ::subsample].clone()

    # ⭐ 허공(Sky/무한대) 및 유효 범위(0.3m ~ 3.5m)를 벗어난 광선 제거 (NaN 처리 -> 점 생성 안 함)
    sub_depth[torch.isinf(sub_depth)] = float("nan")
    sub_depth[sub_depth < min_range] = float("nan")
    sub_depth[sub_depth > max_range] = float("nan")

    if intrinsic_matrix is not None:
        K = torch.as_tensor(intrinsic_matrix, device=device, dtype=torch.float32).clone()
    else:
        fx = (W / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        K = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fx, H / 2.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )

    # Subsample intrinsic parameters
    K[0, 0] /= subsample
    K[1, 1] /= subsample
    K[0, 2] /= subsample
    K[1, 2] /= subsample

    pts_world = create_pointcloud_from_depth(
        K, sub_depth, position=cam_pos_w, orientation=cam_quat_w
    )
    return pts_world


# --------------------------------------------------------------------------------------------------
# RAY PATTERN GENERATOR (정면 기준 상하 수직 / 좌우 수평 각도 정방향 투사)
# --------------------------------------------------------------------------------------------------

def front_lidar_pattern(
    cfg: patterns.LidarPatternCfg, device: str = "cuda:0"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate forward-facing LiDAR rays with precise Vertical (Pitch) and Horizontal (Yaw) FOVs.

    방향 및 각도 규칙 (로봇 정면 기준):
      - Vertical (Pitch, v):
          - 음수 (예: -45° ~ -10°): 전방 아래쪽 바닥을 향함 (Down / Ground)
          - 0°: 수평 정면 (Horizon)
          - 양수 (예: +10° ~ +30°): 전방 위쪽을 향함 (Up / Sky / Obstacle Top)
      - Horizontal (Yaw, h):
          - 음수 (예: -90° ~ -10°): 우측 방향 (Right)
          - 0°: 정면 중앙 (Center Forward)
          - 양수 (예: +10° ~ +90°): 좌측 방향 (Left)

    방향 벡터 수식 (단위 구면 좌표계 -> 직교 좌표계):
      x = cos(v) * cos(h)   (전방 +X)
      y = cos(v) * sin(h)   (좌우 +/-Y)
      z = sin(v)            (상하 +/-Z)
    """
    # 1. 수직 각도 (Pitch): 아래(음수)부터 위(양수)까지 균등 분할
    vertical_angles = torch.linspace(
        cfg.vertical_fov_range[0], cfg.vertical_fov_range[1], cfg.channels, device=device
    )
    # 2. 수평 각도 (Yaw): 우측(음수)부터 좌측(양수)까지 스텝 단위 생성
    horizontal_angles = torch.arange(
        cfg.horizontal_fov_range[0], cfg.horizontal_fov_range[1] + 1e-4, cfg.horizontal_res, device=device
    )

    v_rad = torch.deg2rad(vertical_angles)
    h_rad = torch.deg2rad(horizontal_angles)

    v_grid, h_grid = torch.meshgrid(v_rad, h_rad, indexing="ij")

    # 3. 로봇 정면 방향(+X)으로 투사되는 단위 벡터 계산
    x = torch.cos(v_grid) * torch.cos(h_grid)
    y = torch.cos(v_grid) * torch.sin(h_grid)
    z = torch.sin(v_grid)

    ray_dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    ray_starts = torch.zeros_like(ray_dirs)
    return ray_starts, ray_dirs


# 하위 호환용 별칭
dual_sector_lidar_pattern = front_lidar_pattern
front_conical_lidar_pattern = front_lidar_pattern


# --------------------------------------------------------------------------------------------------
# SENSOR CONFIGURATION
# --------------------------------------------------------------------------------------------------

def create_lidar_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/base",
    debug_vis: bool = False,
    max_distance: float = 15.0,
    channels: int = 16,                                         # [하드웨어 스펙: 수직 채널 수]
    horizontal_res: float = 2.0,                               # [하드웨어 스펙: 수평 각도 분해능 (deg)]
    vertical_fov_range: tuple[float, float] = (-7.0, 52.0),     # [하드웨어 스펙: 상하 화각 범위 (-7° ~ +52°)]
    horizontal_fov_range: tuple[float, float] = (-180.0, 180.0),# [하드웨어 스펙: 좌우 화각 범위 (-180° ~ +180°)]
    pos: tuple[float, float, float] = (0.335, 0.0, 0.045),        #  [장착 위치 (X, Y, Z)]
    rot: tuple[float, float, float, float] = (0.0, 0.5736, 0.0, 0.8192),  # [하향 70° 틸트 장착 회전 쿼터니언 (x, y, z, w)]
) -> MultiMeshRayCasterCfg:
    """Create the Go2 3D LiDAR configuration."""
    quat_sq_norm = rot[0] ** 2 + rot[1] ** 2 + rot[2] ** 2 + rot[3] ** 2
    valid_rot = (0.0, 0.0, 0.0, 1.0) if quat_sq_norm < 1e-5 else rot

    return MultiMeshRayCasterCfg(
        prim_path=prim_path,
        update_period=0.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=pos,
            rot=valid_rot,
        ),
        # 충돌 대상: 기본 바닥(/World/ground). 장애물 맵 로드 시 자동으로 /World/obstacles/.* 추가됨
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            func=front_lidar_pattern,
            channels=channels,
            vertical_fov_range=vertical_fov_range,
            horizontal_fov_range=horizontal_fov_range,
            horizontal_res=horizontal_res,
        ),
        max_distance=max_distance,
        debug_vis=debug_vis,
        visualizer_cfg=GREEN_RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster"),
    )


# Alias for backwards compatibility
create_downward_lidar_cfg = create_lidar_cfg
create_depth_camera_cfg = lambda **kwargs: create_d435i_camera_cfg(**kwargs)


# --------------------------------------------------------------------------------------------------
# INTEL REALSENSE D435i DEPTH CAMERA CONFIGURATION (High-Precision RayCaster)
# --------------------------------------------------------------------------------------------------

def create_d435i_camera_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/base",
    debug_vis: bool = True,
    min_distance: float = 0.3,
    max_distance: float = 3.5,
    channels: int = 24,                                            # [D435i 수직 해상도 채널]
    horizontal_res: float = 2.0,                                   # [D435i 수평 각도 분해능 (deg)]
    vertical_fov_range: tuple[float, float] = (-29.0, 29.0),       # [하드웨어 스펙: VFOV 58° (-29° ~ +29°)]
    horizontal_fov_range: tuple[float, float] = (-43.5, 43.5),     # [하드웨어 스펙: HFOV 87° (-43.5° ~ +43.5°)]
    pos: tuple[float, float, float] = (0.335, 0.0, -0.015),        #  [장착 위치 (X, Y, Z)] - 코 전방 중앙
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),#  [장착 회전 쿼터니언 (x, y, z, w)]
) -> MultiMeshRayCasterCfg:
    """Create Intel RealSense D435i Depth RayCaster Configuration.

    Intel RealSense D435i 광학 하드웨어 스펙:
      - Depth Field of View (FOV): 87° ± 3° (Horizontal) x 58° ± 1° (Vertical)
      - Depth Stream Range: 0.3m ~ 3.5m (최소 0.3m ~ 최대 3.5m)
      - Visualization: Blue Point Cloud (/Visuals/DepthBlueHits)
    """
    quat_sq_norm = rot[0] ** 2 + rot[1] ** 2 + rot[2] ** 2 + rot[3] ** 2
    valid_rot = (0.0, 0.0, 0.0, 1.0) if quat_sq_norm < 1e-5 else rot

    # 최소 감지 거리(min_distance, 0.3m)부터 최대 거리(max_distance, 3.5m)까지 유효한 광선 거리 계산
    effective_max_distance = max_distance - min_distance if max_distance > min_distance else max_distance

    def _d435i_ray_pattern(cfg: patterns.LidarPatternCfg, device: str = "cuda:0") -> tuple[torch.Tensor, torch.Tensor]:
        starts, dirs = front_lidar_pattern(cfg, device)
        if min_distance > 0.0:
            starts = dirs * min_distance
        return starts, dirs

    return MultiMeshRayCasterCfg(
        prim_path=prim_path,
        update_period=0.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=pos,
            rot=valid_rot,
        ),
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            func=_d435i_ray_pattern,
            channels=channels,
            vertical_fov_range=vertical_fov_range,
            horizontal_fov_range=horizontal_fov_range,
            horizontal_res=horizontal_res,
        ),
        max_distance=effective_max_distance,
        debug_vis=debug_vis,
        visualizer_cfg=BLUE_DEPTH_CAMERA_MARKER_CFG.replace(prim_path="/Visuals/DepthBlueHits"),
    )
