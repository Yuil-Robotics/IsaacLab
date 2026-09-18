# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim with native real-time 3D LiDAR ray visualizer for Go2 USD editing."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open Go2 USD with live LiDAR ray visualizer.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force Kit GUI visualizer window
args_cli.headless = False
args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Gf, UsdGeom, Vt
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sensors import patterns
from go2_sim2sim.sensor_patterns import front_lidar_pattern

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USD_PATH = PROJECT_ROOT / "assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"


def main():
    sim_cfg = SimulationCfg(dt=0.02)
    sim = SimulationContext(sim_cfg)

    print(f"\n[INFO] Opening Go2 USD in Isaac Sim GUI: {USD_PATH}")
    omni.usd.get_context().open_stage(str(USD_PATH))
    stage = omni.usd.get_context().get_stage()

    # Standard Go2 LiDAR hardware specification pattern
    pattern_cfg = patterns.LidarPatternCfg(
        func=front_lidar_pattern,
        channels=16,
        vertical_fov_range=(-7.0, 52.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=4.0,  # 분해능 4도로 부드러운 실시간 뷰포트 렌더링
    )
    _, local_ray_dirs = front_lidar_pattern(pattern_cfg, device="cpu")

    # 거리별 샘플 포인트 (0.6m, 1.2m, 2.0m 거리의 궤적을 표시)
    distances = torch.tensor([0.6, 1.2, 2.0], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    rays_expanded = local_ray_dirs.unsqueeze(1) * distances
    local_points = rays_expanded.reshape(-1, 3).numpy()

    # USD 네이티브 초록색 3D 포인트 프리뷰 생성
    preview_path = "/Visuals_RayPreview"
    points_prim = UsdGeom.Points.Define(stage, preview_path)
    points_prim.GetWidthsAttr().Set(Vt.FloatArray([0.015] * len(local_points)))
    points_prim.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 1.0, 0.2)]))

    print("\n=======================================================")
    print("Go2 USD Editor & Live LiDAR Visualizer")
    print("=======================================================")
    print("🟢 초록색 광선 점들이 실시간으로 라이다의 스캔 각도/범위를 보여줍니다.")
    print("1. Stage 창에서 '/go2_description/base/lidar' 선택")
    print("2. 키보드 'W'(이동) 또는 'E'(회전)로 기즈모를 드래그하면 광선이 실시간으로 따라 움직입니다.")
    print("3. 조절이 끝나면 창을 닫으시면 USD 파일에 자동 저장됩니다.")
    print("=======================================================\n")

    while simulation_app.is_running():
        # 실시간으로 lidar Prim의 위치 및 회전 읽기
        lidar_prim = stage.GetPrimAtPath("/go2_description/base/lidar")
        if lidar_prim.IsValid():
            xformable = UsdGeom.Xformable(lidar_prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0.0)

            # 월드 변환 행렬 적용
            world_pts = []
            for pt in local_points:
                p_vec = Gf.Vec3f(float(pt[0]), float(pt[1]), float(pt[2]))
                transformed_p = world_transform.Transform(p_vec)
                world_pts.append(Gf.Vec3f(transformed_p[0], transformed_p[1], transformed_p[2]))

            points_prim.GetPointsAttr().Set(Vt.Vec3fArray(world_pts))

        sim.step(render=True)

    # 임시 시각화 프리뷰는 저장 전 제거하여 USD 파일 깔끔하게 유지
    try:
        if stage.GetPrimAtPath(preview_path).IsValid():
            stage.RemovePrim(preview_path)

        if stage and stage.GetRootLayer():
            stage.GetRootLayer().Save()
            print(f"\n[INFO] USD 파일 저장 완료: {USD_PATH}")
    except Exception as e:
        print(f"\n[WARN] 저장 중 에러: {e}")

    simulation_app.close()


if __name__ == "__main__":
    main()
