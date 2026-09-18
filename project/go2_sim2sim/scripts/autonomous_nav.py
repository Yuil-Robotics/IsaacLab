#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Autonomous Navigation for Unitree Go2 using Scanned Costmap, A* Planner, and RL Policy."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go2_sim2sim.isaaclab_cfg import PLAY_TASK_ID, register_tasks  # noqa: E402
from go2_sim2sim.map_scene import add_3d_obstacle_map_to_scene  # noqa: E402
from go2_sim2sim.navigation_planner import (  # noqa: E402
    AStarPlanner3D,
    ElevationCostmap3D,
    PurePursuitController3D,
)
from go2_sim2sim.sensor_patterns import create_lidar_cfg  # noqa: E402

register_tasks()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=PLAY_TASK_ID, help="Gymnasium task ID.")
parser.add_argument("--checkpoint", type=Path, required=True, help="Policy checkpoint path.")
parser.add_argument(
    "--map_yaml", type=Path, default=PROJECT_ROOT / "logs/maps/go2_map.yaml", help="Path to costmap YAML."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulation environments.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument(
    "--lidar_vis", "--lidar-vis", action="store_true", default=False, help="Enable LiDAR ray visualization."
)
parser.add_argument(
    "--overview", action="store_true", default=False, help="Enable secondary top-down overview viewport."
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.5,
    help="Simulation playback speed multiplier (default: 1.5, 0 for max uncapped speed).",
)
parser.add_argument(
    "--real_time", action="store_true", default=True, help="Run simulation at wall-clock real-time rate."
)
add_launcher_args(parser)

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

if getattr(args_cli, "headless", False):
    args_cli.visualizer = ["none"]
elif not getattr(args_cli, "visualizer", None):
    args_cli.visualizer = ["kit"]


def create_nav_markers_cfg() -> tuple[VisualizationMarkersCfg, VisualizationMarkersCfg]:
    """Create visual markers for target goal and planned path."""
    goal_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/NavGoal",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=0.25,
                height=0.8,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.0),  # Bright Yellow
                    roughness=0.2,
                ),
            )
        },
    )
    path_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/NavPath",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.4),  # Bright Green
                    roughness=0.3,
                ),
            )
        },
    )
    return goal_cfg, path_cfg


def setup_dual_viewports_and_cameras() -> None:
    """Setup dual viewport: Main (3rd-Person Follow) + Overview (Top-Down whole map)."""
    try:
        import omni.kit.app

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.set_extension_enabled_immediate("omni.kit.viewport.utility", True)
        ext_mgr.set_extension_enabled_immediate("omni.kit.viewport.window", True)

        import omni.kit.viewport.utility as vp_utils
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()

        # Overview Top-Down Camera
        cam_prim = UsdGeom.Camera.Define(stage, "/World/OverviewCamera")
        cam_prim.GetFocalLengthAttr().Set(18.0)
        cam_prim.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 22.0))
        cam_prim.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        vp_win = vp_utils.create_viewport_window("Global Map Overview", width=640, height=480)
        if vp_win and hasattr(vp_win, "viewport_api"):
            vp_win.viewport_api.set_active_camera("/World/OverviewCamera")
            print("[INFO] Dual Viewports initialized: 1. Main (3rd-Person Follow) | 2. Global Map Overview")
    except Exception as e:
        print(f"[WARN] Dual viewport skipped: {e}")


def main() -> None:  # noqa: C901
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    map_yaml = args_cli.map_yaml.expanduser().resolve()
    if not map_yaml.is_file():
        print(f"[INFO] {map_yaml} not found. Converting latest 3D PLY map...")
        from go2_sim2sim.map_converter import convert_ply_to_ros2_map, get_latest_ply

        ply = get_latest_ply()
        if ply is None:
            raise FileNotFoundError("No map file found. Please run teleop_mapping.sh first.")
        _, map_yaml = convert_ply_to_ros2_map(ply)

    print(f"[INFO] Loading 3D Traversability & Elevation Map: {map_yaml}")
    costmap = ElevationCostmap3D(
        map_yaml,
        inscribed_radius=0.42,
        inflation_radius=0.90,
        cost_scaling_factor=3.5,
    )
    planner = AStarPlanner3D(costmap)
    controller = PurePursuitController3D(
        lookahead_dist=0.55,
        max_lin_vel=0.55,
        stair_climb_vel=0.45,
        max_ang_vel=0.90,
        goal_tolerance=0.30,
    )

    # 3D Preset Destination Goals across 1F Ground, Stairs, and 2F Mezzanine
    PRESET_GOALS = {
        "1": (-6.5, -5.5, 0.0, "1F Far-Left Outer Room (Beyond West Wall)"),
        "2": (6.5, -5.5, 0.0, "1F Far-Right Outer Room (Beyond East Wall)"),
        "3": (0.0, 3.8, 0.0, "1F Underpass Plaza"),
        "4": (0.0, 0.0, 0.0, "1F Central Plaza"),
        "5": (-4.0, 4.5, 0.80, "2F West Deck (Up West Stairs)"),
        "6": (4.0, 4.5, 0.80, "2F East Deck (Up East Stairs)"),
        "7": (0.0, 6.2, 0.80, "2F Skybridge Overlook"),
        "8": (0.0, -2.0, 0.0, "1F Stairway South Entrance"),
    }

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

    env_cfg.episode_length_s = 1.0e9
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None

    if args_cli.lidar_vis:
        env_cfg.scene.lidar = create_lidar_cfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=True,
        )

    add_3d_obstacle_map_to_scene(env_cfg.scene, "/World/obstacles")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = str(checkpoint.parent)

        if args_cli.overview:
            setup_dual_viewports_and_cameras()

        env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        configure_seed(args_cli.seed, True)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        # Visual Markers for Navigation
        goal_cfg, path_cfg = create_nav_markers_cfg()
        goal_vis = VisualizationMarkers(goal_cfg)
        path_vis = VisualizationMarkers(path_cfg)

        unwrapped = vec_env.unwrapped
        command_term = unwrapped.command_manager.get_term("base_velocity")

        obs = vec_env.get_observations()
        sim_dt = unwrapped.step_dt
        step_idx = 0

        current_goal: tuple[float, float] | None = None
        current_path: list[tuple[float, float]] = []
        nav_status = "IDLE (Press [1]-[8] to set goal)"
        smoothed_eye = None
        smoothed_target = None

        print("\n" + "=" * 70)
        print("\n" + "=" * 70)
        print("Go2 Autonomous Navigation Stack Active")
        print("=" * 70)
        print("  [1] ~ [8]       : Select Preset Goal (Rooms A-D / Corridors)")
        print("  [SPACE] / [K]   : Emergency Stop / Cancel Navigation")
        for k, (gx, gy, gz, name) in PRESET_GOALS.items():
            print(f"    [{k}] {name:<28}: ({gx:+4.1f}m, {gy:+4.1f}m, Z={gz:+.2f}m)")
        print("=" * 70 + "\n")

        # Isaac Sim Window Keyboard Listener
        selected_goal_key = [None]
        try:
            import carb
            import carb.input
            import omni.appwindow

            appwindow = omni.appwindow.get_default_app_window()
            input_iface = carb.input.acquire_input_interface()
            keyboard = appwindow.get_keyboard()

            carb_num_keys = {}
            for num in range(1, 9):
                key_attr = getattr(carb.input.KeyboardInput, f"KEY_{num}", None)
                numpad_attr = getattr(carb.input.KeyboardInput, f"NUMPAD_{num}", None)
                if key_attr is not None:
                    carb_num_keys[key_attr] = str(num)
                if numpad_attr is not None:
                    carb_num_keys[numpad_attr] = str(num)

            for attr_name, key_val in [
                ("EQUAL", "+"),
                ("MINUS", "-"),
                ("NUMPAD_ADD", "+"),
                ("NUMPAD_SUBTRACT", "-"),
                ("KEY_0", "0"),
                ("NUMPAD_0", "0"),
            ]:
                attr = getattr(carb.input.KeyboardInput, attr_name, None)
                if attr is not None:
                    carb_num_keys[attr] = key_val

            def _on_nav_key(event, *args, **kwargs):
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    if event.input in (carb.input.KeyboardInput.SPACE, carb.input.KeyboardInput.K):
                        selected_goal_key[0] = "stop"
                    elif event.input in carb_num_keys:
                        selected_goal_key[0] = carb_num_keys[event.input]

            _kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_nav_key)
            print("[INFO] Isaac Sim window keyboard listener active: [1]-[8] Goal, [+/-] Speed, [SPACE] Stop.")
        except Exception as e:
            print(f"[WARN] Kit window keyboard listener not active ({e}). Terminal input active.")

        sim_speed = max(0.0, float(args_cli.speed))

        old_term = None
        if sys.stdin.isatty():
            try:
                old_term = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                old_term = None

        try:
            while True:
                step_start_time = time.time()
                step_idx += 1

                # Check carb window key
                ch = ""
                if selected_goal_key[0] is not None:
                    ch = selected_goal_key[0]
                    selected_goal_key[0] = None

                # Poll terminal keyboard input
                if not ch and sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.0)[0]:
                    try:
                        ch = sys.stdin.read(1).lower()
                    except Exception:
                        ch = ""

                # Handle Speed Adjustment Hotkeys
                if ch in ("+", "="):
                    sim_speed = round(min(5.0, (sim_speed if sim_speed > 0 else 1.0) + 0.5), 1)
                    print(f"\n[Simulation Speed] Set to {sim_speed:.1f}x")
                    ch = ""
                elif ch in ("-", "_"):
                    sim_speed = round(max(0.5, (sim_speed if sim_speed > 0 else 1.0) - 0.5), 1)
                    print(f"\n[Simulation Speed] Set to {sim_speed:.1f}x")
                    ch = ""
                elif ch == "0":
                    sim_speed = 0.0
                    print("\n[Simulation Speed] Set to MAX (Uncapped GPU Speed)")
                    ch = ""

                # Extract robot position and heading (Yaw)
                step_idx += 1
                robot_data = unwrapped.scene["robot"].data
                pos_w = robot_data.root_pos_w[0].cpu().numpy()
                quat_w = robot_data.root_quat_w[0].cpu().numpy()  # PhysX quaternion format: [x, y, z, w]

                # Canonical 3D Euler Yaw from [x, y, z, w] quaternion
                x, y, z, w = float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3])
                siny_cosp = 2.0 * (w * z + x * y)
                cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                if ch in PRESET_GOALS:
                    gx, gy, gz, gname = PRESET_GOALS[ch]
                    current_goal = (gx, gy, gz)
                    controller.reset()
                    # Plan 3D global path from current robot position
                    current_path = planner.plan((float(pos_w[0]), float(pos_w[1]), float(pos_w[2])), current_goal)
                    nav_status = f"NAVIGATING to {gname}"
                    print(
                        f"\n[New 3D Goal Set] {gname} ({gx:+.1f}m, {gy:+.1f}m, Z={gz:+.2f}m) | "
                        f"Path Points: {len(current_path)}"
                    )

                    # Update 3D Visual Goal Marker
                    gpos_t = torch.tensor([[gx, gy, gz + 0.40]], dtype=torch.float32, device="cuda:0")
                    goal_vis.visualize(gpos_t)

                    # Update 3D Path Waypoint Markers
                    if current_path:
                        pts_3d = [[px, py, pz + 0.06] for px, py, pz in current_path]
                        path_vis.visualize(torch.tensor(pts_3d, dtype=torch.float32, device="cuda:0"))
                elif ch in (" ", "k", "s", "stop"):
                    current_goal = None
                    current_path = []
                    controller.reset()
                    nav_status = "STOPPED / IDLE"
                    goal_vis.visualize(torch.zeros((0, 3), dtype=torch.float32, device="cuda:0"))
                    path_vis.visualize(torch.zeros((0, 3), dtype=torch.float32, device="cuda:0"))
                    print("\n[Navigation Stopped by User]")

                # True simulation clock fast-forward via non-rendering physics substepping
                n_substeps = max(1, int(round(sim_speed))) if sim_speed > 0 else 4

                for sub_idx in range(n_substeps):
                    step_idx += 1
                    robot_data = unwrapped.scene["robot"].data
                    pos_w = robot_data.root_pos_w[0].cpu().numpy()
                    quat_w = robot_data.root_quat_w[0].cpu().numpy()
                    x, y, z, w = float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3])
                    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

                    # Off-track replanning
                    if current_goal is not None and current_path and step_idx % 50 == 0:
                        min_path_dist = (
                            min(
                                math.hypot(pt[0] - pos_w[0], pt[1] - pos_w[1])
                                for pt in current_path[controller.current_idx :]
                            )
                            if controller.current_idx < len(current_path)
                            else float("inf")
                        )
                        if min_path_dist > 1.2:
                            controller.reset()
                            new_path = planner.plan((float(pos_w[0]), float(pos_w[1]), float(pos_w[2])), current_goal)
                            if new_path:
                                current_path = new_path
                                pts_3d = [[px, py, pz + 0.06] for px, py, pz in current_path]
                                path_vis.visualize(torch.tensor(pts_3d, dtype=torch.float32, device="cuda:0"))

                    # Compute Pure Pursuit velocity commands (stable nominal gait velocities)
                    cmd_vx, cmd_vy, cmd_wz = 0.0, 0.0, 0.0
                    if current_path and current_goal is not None:
                        vx, vy, wz, reached = controller.compute_command(
                            (float(pos_w[0]), float(pos_w[1]), float(pos_w[2])), yaw, current_path
                        )
                        if reached:
                            nav_status = "GOAL REACHED!"
                            print(
                                f"\n[Success] Reached 3D destination: ({current_goal[0]:+.1f}, "
                                f"{current_goal[1]:+.1f}, Z={current_goal[2]:+.2f}m)"
                            )
                            current_goal = None
                            current_path = []
                            controller.reset()
                            goal_vis.visualize(torch.zeros((0, 3), dtype=torch.float32, device="cuda:0"))
                            path_vis.visualize(torch.zeros((0, 3), dtype=torch.float32, device="cuda:0"))
                        else:
                            cmd_vx, cmd_vy, cmd_wz = vx, vy, wz

                    # Send commands to RL policy
                    cmd_tensor = torch.tensor(
                        [[cmd_vx, cmd_vy, cmd_wz]], dtype=torch.float32, device=vec_env.unwrapped.device
                    )
                    command_term.vel_command_b[:, :3] = cmd_tensor
                    command_term.command[:, :3] = cmd_tensor
                    command_term.is_heading_env.zero_()
                    command_term.is_standing_env.zero_()
                    if hasattr(command_term, "is_straight_env"):
                        command_term.is_straight_env.zero_()
                    if hasattr(command_term, "is_turning_env"):
                        command_term.is_turning_env.zero_()
                    if hasattr(command_term, "is_lateral_env"):
                        command_term.is_lateral_env.zero_()

                    # Render ONLY the final sub-step in this frame loop for rock-solid 60 FPS
                    is_final_substep = sub_idx == n_substeps - 1
                    if hasattr(unwrapped, "render_enabled"):
                        unwrapped.render_enabled = is_final_substep

                    # Step simulation
                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, _, _, _ = vec_env.step(actions)

                # Prune passed waypoints visually
                if current_path and controller.current_idx > 0 and step_idx % 4 == 0:
                    rem_pts = current_path[controller.current_idx :]
                    if rem_pts:
                        pts_3d = [[px, py, pz + 0.06] for px, py, pz in rem_pts]
                        path_vis.visualize(torch.tensor(pts_3d, dtype=torch.float32, device="cuda:0"))

                # Export live state for WebGL 3D Viewer (at ~5 Hz)
                if step_idx % 10 == 0:
                    try:
                        import json

                        rem_path = current_path[controller.current_idx :] if current_path else []
                        live_data = {
                            "connected": True,
                            "timestamp": time.time(),
                            "robot": {
                                "x": float(pos_w[0]),
                                "y": float(pos_w[1]),
                                "z": float(pos_w[2]),
                                "yaw": float(yaw),
                            },
                            "goal": {
                                "x": float(current_goal[0]) if current_goal else None,
                                "y": float(current_goal[1]) if current_goal else None,
                                "z": float(current_goal[2]) if current_goal else None,
                            },
                            "path": [
                                [round(float(pt[0]), 3), round(float(pt[1]), 3), round(float(pt[2]), 3)]
                                for pt in rem_path
                            ],
                            "status": nav_status,
                            "stats": {
                                "rem_dist": round(
                                    float(math.hypot(current_goal[0] - pos_w[0], current_goal[1] - pos_w[1])), 2
                                )
                                if current_goal
                                else 0.0,
                                "vx": round(float(cmd_vx), 2),
                                "wz": round(float(cmd_wz), 2),
                            },
                        }
                        live_path = PROJECT_ROOT / "logs" / "maps" / "nav_live_state.json"
                        tmp_path = PROJECT_ROOT / "logs" / "maps" / "nav_live_state.tmp"
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(live_data, f)
                        tmp_path.replace(live_path)
                    except Exception:
                        pass

                # Smooth 3rd-Person Follow Camera
                try:
                    cam_eye = np.array(
                        [
                            pos_w[0] - 2.2 * math.cos(yaw),
                            pos_w[1] - 2.2 * math.sin(yaw),
                            pos_w[2] + 0.95,
                        ]
                    )
                    cam_target = np.array(
                        [
                            pos_w[0] + 0.8 * math.cos(yaw),
                            pos_w[1] + 0.8 * math.sin(yaw),
                            pos_w[2] + 0.25,
                        ]
                    )
                    if smoothed_eye is None:
                        smoothed_eye = cam_eye.copy()
                        smoothed_target = cam_target.copy()
                    else:
                        smoothed_eye = 0.88 * smoothed_eye + 0.12 * cam_eye
                        smoothed_target = 0.88 * smoothed_target + 0.12 * cam_target

                    unwrapped.sim.set_camera_view(eye=smoothed_eye, target=smoothed_target)
                except Exception:
                    pass

                # Terminal Status HUD
                dist_str = (
                    f"{math.hypot(current_goal[0] - pos_w[0], current_goal[1] - pos_w[1]):.2f}m"
                    if current_goal
                    else "--"
                )
                speed_str = f"{sim_speed:.1f}x" if sim_speed > 0 else "MAX"
                print(
                    f"\r[{nav_status}] Rem: {dist_str:<6} | Pos: ({pos_w[0]:+4.1f}, {pos_w[1]:+4.1f}) | "
                    f"vx: {cmd_vx:+4.2f}m/s | Speed: {speed_str} ([+/-]) | [1]-[8]: Goal | [SPACE]: Stop",
                    end="",
                    flush=True,
                )

                # Slow-motion sleep for sim_speed < 1.0
                if 0.0 < sim_speed < 1.0:
                    sleep_time = (sim_dt / sim_speed) - (time.time() - step_start_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received.")
        finally:
            if old_term is not None and sys.stdin.isatty():
                with contextlib.suppress(Exception):
                    termios.tcsetattr(sys.stdin, termios.TCSANOW, old_term)
            with contextlib.suppress(Exception):
                os.system("stty sane 2>/dev/null")
            try:
                import json

                live_path = PROJECT_ROOT / "logs" / "maps" / "nav_live_state.json"
                with open(live_path, "w", encoding="utf-8") as f:
                    json.dump({"connected": False}, f)
            except Exception:
                pass
            vec_env.close()


if __name__ == "__main__":
    main()
