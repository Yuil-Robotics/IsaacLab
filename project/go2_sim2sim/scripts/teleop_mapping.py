#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate a Yuil Dog RL policy in a 3D obstacle map with optional point-cloud mapping."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
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

from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go2_sim2sim.asset_cfg import YUIL_DOG_BASE_PRIM_PATH  # noqa: E402
from go2_sim2sim.isaaclab_cfg import register_tasks  # noqa: E402
from go2_sim2sim.map_scene import add_3d_obstacle_map_to_scene  # noqa: E402
from go2_sim2sim.pointcloud_mapper import PointCloudMapper  # noqa: E402
from go2_sim2sim.rough_mdp import get_ground_relative_base_height  # noqa: E402
from go2_sim2sim.sensor_patterns import (  # noqa: E402
    create_d435i_camera_cfg,
    create_lidar_cfg,
)
from go2_sim2sim.teleop_input import LongitudinalHoldCommand  # noqa: E402
from go2_sim2sim.yuil_dog_cfg import YUIL_DOG_ROUGH_PLAY_TASK_ID  # noqa: E402

register_tasks()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=YUIL_DOG_ROUGH_PLAY_TASK_ID, help="Gymnasium task ID.")
parser.add_argument("--checkpoint", type=Path, required=True, help="Policy checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulation environments.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument("--vx", type=float, default=0.5, help="Linear velocity magnitude along X [m/s].")
parser.add_argument(
    "--vx_fast",
    "--vx-fast",
    type=float,
    default=1.0,
    help="Linear X speed after holding W/S [m/s].",
)
parser.add_argument(
    "--vx_hold_time",
    "--vx-hold-time",
    type=float,
    default=1.0,
    help="W/S hold duration before accelerating [s].",
)
parser.add_argument("--vy", type=float, default=0.5, help="Linear velocity magnitude along Y [m/s].")
parser.add_argument("--wz", type=float, default=1.0, help="Angular yaw velocity magnitude [rad/s].")
parser.add_argument("--voxel_size", type=float, default=0.06, help="Mapping voxel resolution in meters.")
parser.add_argument(
    "--lidar_vis", "--lidar-vis", action="store_true", default=False, help="Enable LiDAR point-cloud mapping."
)
parser.add_argument(
    "--camera_vis",
    "--camera-vis",
    "--cam-vis",
    action="store_true",
    default=False,
    help="Enable D435i depth point-cloud mapping.",
)
parser.add_argument("--real-time", "--real_time", action="store_true", default=True, help="Run in real-time step rate.")
parser.add_argument("--gui", action="store_true", default=True, help="Enable Kit GUI visualizer.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# Enable cameras subsystem when camera_vis is requested
if getattr(args_cli, "camera_vis", False):
    args_cli.enable_cameras = True

# Ensure Kit GUI visualizer is default unless headless is specified
if getattr(args_cli, "headless", False):
    args_cli.visualizer = ["none"]
elif not getattr(args_cli, "visualizer", None):
    args_cli.visualizer = ["kit"]


def _restore_term(settings):
    if settings is not None and sys.stdin.isatty():
        with contextlib.suppress(Exception):
            termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)
    if sys.stdin.isatty():
        with contextlib.suppress(Exception):
            os.system("stty sane 2>/dev/null")


def _resolve_sensor_prim_path(task_id: str) -> str:
    """Return the robot base prim used to mount optional mapping sensors."""
    if "Yuil-Dog" in task_id:
        return YUIL_DOG_BASE_PRIM_PATH
    return "{ENV_REGEX_NS}/Robot/base"


class MappingKeyboardTeleop:
    """Keyboard teleoperator supporting locomotion and mapping actions (Save PLY / Clear)."""

    def __init__(
        self,
        vx: float = 0.5,
        vx_fast: float = 1.0,
        vx_hold_time: float = 1.0,
        vy: float = 0.5,
        wz: float = 1.0,
        device: str = "cuda:0",
        mapper: PointCloudMapper | None = None,
    ):
        self.vx_step = vx
        self.vx_fast = max(vx_fast, vx)
        self.vx_hold_time = vx_hold_time
        self.vy_step = vy
        self.wz_step = wz
        self.device = device
        self.mapper = mapper
        self.current_cmd = np.zeros(3, dtype=np.float32)
        self._longitudinal_command = LongitudinalHoldCommand(
            base_speed=self.vx_step,
            fast_speed=self.vx_fast,
            hold_duration=self.vx_hold_time,
        )

        # 1. Terminal keyboard setup
        self._term_settings = None
        if sys.stdin.isatty():
            try:
                import atexit

                self._term_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                atexit.register(_restore_term, self._term_settings)
            except Exception:
                pass

        # 2. Omniverse Kit GUI Window keyboard setup
        self._input = None
        self._keyboard_sub = None
        try:
            import carb
            import carb.input
            import omni.appwindow

            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()

            self._forward_keys = {carb.input.KeyboardInput.W, carb.input.KeyboardInput.UP}
            self._backward_keys = {carb.input.KeyboardInput.S, carb.input.KeyboardInput.DOWN}
            self._key_mapping = {
                carb.input.KeyboardInput.A: np.array([0.0, self.vy_step, 0.0], dtype=np.float32),
                carb.input.KeyboardInput.LEFT: np.array([0.0, self.vy_step, 0.0], dtype=np.float32),
                carb.input.KeyboardInput.D: np.array([0.0, -self.vy_step, 0.0], dtype=np.float32),
                carb.input.KeyboardInput.RIGHT: np.array([0.0, -self.vy_step, 0.0], dtype=np.float32),
                carb.input.KeyboardInput.Q: np.array([0.0, 0.0, self.wz_step], dtype=np.float32),
                carb.input.KeyboardInput.Z: np.array([0.0, 0.0, self.wz_step], dtype=np.float32),
                carb.input.KeyboardInput.E: np.array([0.0, 0.0, -self.wz_step], dtype=np.float32),
                carb.input.KeyboardInput.X: np.array([0.0, 0.0, -self.wz_step], dtype=np.float32),
            }

            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_carb_keyboard_event,
            )
            print("[INFO] Isaac Sim window keyboard listener active.")
        except Exception as e:
            print(f"[WARN] Kit window keyboard listener not active ({e}). Terminal input active.")

        self._print_controls()

    def _print_controls(self):
        print("\n" + "=" * 70)
        print("Yuil Dog RL Mapping Teleoperation Active")
        print("=" * 70)
        print(
            f"  [W] / [Up]      : Forward   (+{self.vx_step:.2f}, hold {self.vx_hold_time:.1f}s: "
            f"+{self.vx_fast:.2f} m/s)"
        )
        print(
            f"  [S] / [Down]    : Backward  (-{self.vx_step:.2f}, hold {self.vx_hold_time:.1f}s: "
            f"-{self.vx_fast:.2f} m/s)"
        )
        print(f"  [A] / [Left]    : Left      (+{self.vy_step:.2f} m/s)")
        print(f"  [D] / [Right]   : Right     (-{self.vy_step:.2f} m/s)")
        print(f"  [Q] / [Z]       : Turn Left (+{self.wz_step:.2f} rad/s)")
        print(f"  [E] / [X]       : Turn Right(-{self.wz_step:.2f} rad/s)")
        print("  [SPACE] / [K]   : Stop (0.0 m/s)")
        if self.mapper is not None:
            print("  [P]             : Save 3D Map to PLY file")
            print("  [C]             : Clear 3D Map buffer")
            print("  [V]             : Toggle In-Sim Map Visualizer (Boost FPS)")
        else:
            print("  LiDAR / vision point-cloud mapping: disabled")
        print("=" * 70 + "\n")

    def _on_carb_keyboard_event(self, event, *args, **kwargs):
        import carb.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input in (carb.input.KeyboardInput.SPACE, carb.input.KeyboardInput.K, carb.input.KeyboardInput.L):
                self.current_cmd.fill(0.0)
                self._longitudinal_command.stop()
            elif event.input in self._forward_keys:
                self._longitudinal_command.press(event.input, 1)
            elif event.input in self._backward_keys:
                self._longitudinal_command.press(event.input, -1)
            elif event.input == carb.input.KeyboardInput.P:
                if self.mapper is not None:
                    self.mapper.save_ply()
            elif event.input == carb.input.KeyboardInput.C:
                if self.mapper is not None:
                    self.mapper.clear()
            elif event.input == carb.input.KeyboardInput.V:
                if self.mapper is not None:
                    is_on = self.mapper.toggle_visualization()
                    state_str = "ON" if is_on else "OFF (Max FPS)"
                    print(f"\n[In-Sim Visualizer] {state_str}")
            elif event.input in self._key_mapping:
                self.current_cmd += self._key_mapping[event.input]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in self._forward_keys or event.input in self._backward_keys:
                self._longitudinal_command.release(event.input)
            elif event.input in self._key_mapping:
                self.current_cmd -= self._key_mapping[event.input]

    def _poll_terminal(self):
        """Read pending terminal keys non-blockingly."""
        if not sys.stdin.isatty():
            return

        while select.select([sys.stdin], [], [], 0.0)[0]:
            try:
                ch = sys.stdin.read(1)
            except Exception:
                break

            if ch == "\x1b":
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        if select.select([sys.stdin], [], [], 0.0)[0]:
                            ch3 = sys.stdin.read(1)
                            if ch3 == "A":  # Up Arrow
                                self._longitudinal_command.terminal_press(1)
                            elif ch3 == "B":  # Down Arrow
                                self._longitudinal_command.terminal_press(-1)
                            elif ch3 == "C":  # Right Arrow
                                self.current_cmd[1] = -self.vy_step
                            elif ch3 == "D":  # Left Arrow
                                self.current_cmd[1] = self.vy_step
                continue

            ch_lower = ch.lower()
            if ch_lower == "w":
                self._longitudinal_command.terminal_press(1)
            elif ch_lower == "s":
                self._longitudinal_command.terminal_press(-1)
            elif ch_lower == "a":
                self.current_cmd[1] = self.vy_step
            elif ch_lower == "d":
                self.current_cmd[1] = -self.vy_step
            elif ch_lower == "q":
                self.current_cmd[2] = self.wz_step
            elif ch_lower == "e":
                self.current_cmd[2] = -self.wz_step
            elif ch_lower in (" ", "k", "x", "r"):
                self.current_cmd.fill(0.0)
                self._longitudinal_command.stop()
            elif ch_lower == "p":
                if self.mapper is not None:
                    self.mapper.save_ply()
            elif ch_lower == "c":
                if self.mapper is not None:
                    self.mapper.clear()
            elif ch_lower == "v":
                if self.mapper is not None:
                    is_on = self.mapper.toggle_visualization()
                    state_str = "ON" if is_on else "OFF (Max FPS)"
                    print(f"\n[In-Sim Visualizer] {state_str}")

    def get_command(self) -> torch.Tensor:
        self._poll_terminal()
        self.current_cmd[0] = self._longitudinal_command.velocity()
        cmd = np.clip(self.current_cmd, [-1.0, -1.0, -3.0], [1.0, 1.0, 3.0])
        return torch.tensor(cmd, dtype=torch.float32, device=self.device)

    def close(self):
        if self._input is not None and self._keyboard_sub is not None:
            with contextlib.suppress(Exception):
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        _restore_term(self._term_settings)


def setup_dual_viewports_and_cameras() -> None:
    """Setup a 2-screen split: Main Viewport (3rd person follow) and Overview Viewport (Top-Down whole map)."""
    try:
        import omni.kit.app

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.set_extension_enabled_immediate("omni.kit.viewport.utility", True)
        ext_mgr.set_extension_enabled_immediate("omni.kit.viewport.window", True)

        import omni.kit.viewport.utility as vp_utils
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()

        # 1. Define Overview Top-Down Camera
        cam_prim = UsdGeom.Camera.Define(stage, "/World/OverviewCamera")
        cam_prim.GetFocalLengthAttr().Set(18.0)
        cam_prim.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 22.0))
        cam_prim.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        # 2. Create second viewport window for Overview
        vp_win = vp_utils.create_viewport_window("Global Map Overview", width=640, height=480)
        if vp_win and hasattr(vp_win, "viewport_api"):
            vp_win.viewport_api.set_active_camera("/World/OverviewCamera")
            print("[INFO] Dual Viewports initialized: 1. Main (3rd-Person Follow) | 2. Global Map Overview")
    except Exception as e:
        print(f"[WARN] Dual viewport configuration skipped: {e}")


def main() -> None:
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Disable heading controller for direct velocity teleoperation
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.ranges.heading = None
    # Disable episode time_out so mapping runs indefinitely without resets
    env_cfg.episode_length_s = 1.0e9
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    # Disable random push / force disturbances during mapping
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None

    # Keep LiDAR and vision point-cloud support available but disabled by default.
    env_cfg.scene.lidar = None
    env_cfg.scene.camera = None
    sensor_prim_path = _resolve_sensor_prim_path(args_cli.task)
    if args_cli.lidar_vis:
        env_cfg.scene.lidar = create_lidar_cfg(
            prim_path=sensor_prim_path,
            debug_vis=True,
        )

    # Configure D435i Depth Camera (Blue point cloud)
    if getattr(args_cli, "camera_vis", False):
        env_cfg.scene.camera = create_d435i_camera_cfg(
            prim_path=sensor_prim_path,
            debug_vis=True,
        )

    # Ensure height_scanner is present on base prim for ground-relative height measurement
    if env_cfg.scene.height_scanner is None:
        from isaaclab.sensors.ray_caster import RayCasterCfg
        from isaaclab.sensors.ray_caster.patterns import grid_pattern
        from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg

        env_cfg.scene.height_scanner = RayCasterCfg(
            prim_path=sensor_prim_path,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=GridPatternCfg(func=grid_pattern, resolution=0.1, size=[0.1, 0.1]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
    env_cfg.observations.policy.height_scan = None

    # Attach 3D obstacle map (walls, rooms, pillars, boxes) into the scene & register to LiDAR
    add_3d_obstacle_map_to_scene(env_cfg.scene, "/World/obstacles")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = str(checkpoint.parent)

        # Initialize 2-screen split viewports (3rd person + Global Overview)
        setup_dual_viewports_and_cameras()

        agent_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            agent_cfg.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        configure_seed(args_cli.seed, True)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        # Construct the mapper only when a perception source is explicitly enabled.
        mapper = None
        if args_cli.lidar_vis or args_cli.camera_vis:
            mapper = PointCloudMapper(
                voxel_size=args_cli.voxel_size,
                output_dir=PROJECT_ROOT / "logs" / "maps",
            )
            mapper.init_visualizer("/Visuals/AccumulatedMap")
        else:
            print("[INFO] LiDAR and vision point-cloud mapping are disabled.")

        teleop = MappingKeyboardTeleop(
            vx=args_cli.vx,
            vx_fast=args_cli.vx_fast,
            vx_hold_time=args_cli.vx_hold_time,
            vy=args_cli.vy,
            wz=args_cli.wz,
            device=vec_env.unwrapped.device,
            mapper=mapper,
        )

        unwrapped = vec_env.unwrapped
        command_term = unwrapped.command_manager.get_term("base_velocity")
        lidar_sensor = unwrapped.scene.sensors.get("lidar", None)

        obs = vec_env.get_observations()
        sim_dt = unwrapped.step_dt
        step_idx = 0
        last_hud_time = 0.0
        smoothed_eye = None
        smoothed_target = None

        print("[INFO] Mapping loop running. Drive Yuil Dog around the 3D environment.")

        try:
            while True:
                step_start_time = time.time()
                step_idx += 1
                cmd_tensor = teleop.get_command()

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

                # Step simulation
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, _, _ = vec_env.step(actions)

                # Smooth 3rd-person follow camera tracking Yuil Dog in Main Viewport
                try:
                    robot_data = unwrapped.scene["robot"].data
                    pos_w = robot_data.root_pos_w[0].cpu().numpy()
                    quat_w = robot_data.root_quat_w[0].cpu().numpy()  # [w, x, y, z]

                    w, x, y, z = quat_w[0], quat_w[1], quat_w[2], quat_w[3]
                    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

                    cam_dist = 2.2
                    cam_height = 0.95
                    target_dist = 0.8

                    cam_eye = np.array(
                        [
                            pos_w[0] - cam_dist * np.cos(yaw),
                            pos_w[1] - cam_dist * np.sin(yaw),
                            pos_w[2] + cam_height,
                        ]
                    )
                    cam_target = np.array(
                        [
                            pos_w[0] + target_dist * np.cos(yaw),
                            pos_w[1] + target_dist * np.sin(yaw),
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

                # 3. Accumulate 3D Green LiDAR scan into PointCloudMapper
                if mapper is not None and lidar_sensor is not None:
                    # ray_hits_w shape: [num_envs, num_rays, 3]
                    hit_points = lidar_sensor.data.ray_hits_w[0]
                    mapper.add_points(hit_points)

                # 4. 🔵 Accumulate Real-Time Blue D435i Depth Ray Hits into SLAM Map
                if mapper is not None and args_cli.camera_vis and "camera" in unwrapped.scene.sensors:
                    cam_sensor = unwrapped.scene["camera"]
                    cam_hits = cam_sensor.data.ray_hits_w[0]
                    mapper.add_points(cam_hits)

                # Update 3D visualization every 15 steps (~3.3 Hz) to preserve 60 FPS
                if mapper is not None and step_idx % 15 == 0:
                    mapper.update_visualization()

                # Update terminal HUD periodically
                now = time.time()
                if now - last_hud_time > 0.15:
                    last_hud_time = now
                    base_height = get_ground_relative_base_height(unwrapped.scene)
                    if mapper is None:
                        status = (
                            f"\r[Teleop] vx: {cmd_tensor[0]:+5.2f} | vy: {cmd_tensor[1]:+5.2f} | "
                            f"wz: {cmd_tensor[2]:+5.2f} | base_height: {base_height:.3f} m | "
                            "Point clouds: disabled"
                        )
                    else:
                        status = (
                            f"\r[3D Map] Voxel Points: {len(mapper.get_points()):,} | "
                            f"vx: {cmd_tensor[0]:+5.2f} | vy: {cmd_tensor[1]:+5.2f} | "
                            f"wz: {cmd_tensor[2]:+5.2f} | base_height: {base_height:.3f} m | "
                            "[P]: Save PLY | [C]: Clear | [V]: Toggle Vis"
                        )
                    print(status, end="", flush=True)

                if args_cli.real_time:
                    sleep_time = sim_dt - (time.time() - step_start_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received.")
        finally:
            if mapper is not None:
                print("\n[INFO] Auto-saving scanned 3D map on exit...")
                mapper.save_ply()
            teleop.close()
            vec_env.close()


if __name__ == "__main__":
    main()
