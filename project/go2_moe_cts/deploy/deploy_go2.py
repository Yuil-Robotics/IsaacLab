# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Sim2Sim deployment for go2_moe_cts.

Runs the exported policy.pt in MuJoCo at 50 Hz.
Policy obs order must match IsaacLab training exactly:
    base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
    + joint_pos_rel(12) + joint_vel(12) + last_action(12)
    = 45 dim/frame, with a term-major 10-frame history = 450 dim

Usage
-----
    python deploy/deploy_go2.py                       # use default config
    python deploy/deploy_go2.py --config custom.yaml  # override config

Controller (gamepad):
    LY: lin_vel_x  LX: lin_vel_y  RX: ang_vel_z
    Auto-detected; falls back to default_command if not connected.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT_DIR = Path(__file__).parent.parent


def load_config(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        raw = f.read().replace("${ROOT_DIR}", str(ROOT_DIR))
    return yaml.safe_load(raw)


def load_policy(policy_path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    return policy


def get_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Project world gravity [0,0,-1] into body frame from quaternion [w,x,y,z]."""
    w, x, y, z = quat
    gravity_world = np.array([0.0, 0.0, -1.0])
    # Rotate gravity vector from world to body frame (inverse rotation)
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )
    return R.T @ gravity_world  # body = R^T @ world


class Go2MuJoCoDeployer:
    """Sim2Sim deployment controller for Unitree Go2 in MuJoCo."""

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.num_actions = cfg.get("num_actions", 12)
        self.num_single_obs = cfg.get("num_single_obs", 45)
        self.history_length = cfg.get("history_length", 10)
        self.num_obs = cfg.get("num_obs", self.num_single_obs * self.history_length)
        self.obs_clip = cfg.get("obs_clip", 100.0)
        self.obs_scales = cfg.get("obs_scales", {})
        self.dt_sim = cfg.get("simulation_dt", 0.002)
        self.control_decimation = cfg.get("control_decimation", 10)
        self.kp = cfg.get("kp", 25.0)
        self.kd = cfg.get("kd", 0.5)

        # Go2 default joint positions (FL/FR/RL/RR × hip/thigh/calf, leg-major order)
        self.default_joint_pos = np.array(
            [
                0.1,
                0.8,
                -1.5,  # FL
                -0.1,
                0.8,
                -1.5,  # FR
                0.1,
                1.0,
                -1.5,  # RL
                -0.1,
                1.0,
                -1.5,  # RR
            ],
            dtype=np.float32,
        )

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.feature_dims = (3, 3, 3, self.num_actions, self.num_actions, self.num_actions)
        if sum(self.feature_dims) != self.num_single_obs:
            raise ValueError(
                f"Invalid observation layout: feature dimensions sum to {sum(self.feature_dims)}, "
                f"but num_single_obs is {self.num_single_obs}."
            )
        if self.num_obs != self.num_single_obs * self.history_length:
            raise ValueError(
                f"num_obs must equal num_single_obs * history_length "
                f"({self.num_single_obs * self.history_length}), got {self.num_obs}."
            )
        self.obs_history = np.zeros(self.num_obs, dtype=np.float32)
        self._history_initialized = False
        self.command = np.array(
            [
                cfg.get("default_command", {}).get("lin_vel_x", 0.5),
                cfg.get("default_command", {}).get("lin_vel_y", 0.0),
                cfg.get("default_command", {}).get("ang_vel_z", 0.0),
            ],
            dtype=np.float32,
        )

        # Load policy
        self.policy = load_policy(cfg["policy_path"], device)
        print(f"[INFO] Policy loaded from: {cfg['policy_path']}")

    def build_obs(
        self,
        ang_vel: np.ndarray,  # [3] rad/s (body frame)
        quat: np.ndarray,  # [4] w,x,y,z
        joint_pos: np.ndarray,  # [12]
        joint_vel: np.ndarray,  # [12]
    ) -> torch.Tensor:
        """Assemble one frame and update the term-major policy history."""
        proj_grav = get_projected_gravity(quat)
        joint_pos_rel = joint_pos - self.default_joint_pos

        scale = self.obs_scales
        single_obs = np.concatenate(
            [
                np.clip(ang_vel, -self.obs_clip, self.obs_clip) * scale.get("angular_velocity", 0.25),
                np.clip(proj_grav, -self.obs_clip, self.obs_clip) * scale.get("projected_gravity", 1.0),
                np.clip(self.command, -self.obs_clip, self.obs_clip) * scale.get("command", 1.0),
                np.clip(joint_pos_rel, -self.obs_clip, self.obs_clip) * scale.get("joint_position", 1.0),
                np.clip(joint_vel, -self.obs_clip, self.obs_clip) * scale.get("joint_velocity", 0.05),
                np.clip(self.last_action, -self.obs_clip, self.obs_clip) * scale.get("action", 1.0),
            ]
        ).astype(np.float32)
        if single_obs.shape[0] != self.num_single_obs:
            raise RuntimeError(
                f"Single observation dimension mismatch: {single_obs.shape[0]} vs {self.num_single_obs}."
            )

        history_offset = 0
        single_offset = 0
        for feature_dim in self.feature_dims:
            history_end = history_offset + feature_dim * self.history_length
            single_end = single_offset + feature_dim
            if not self._history_initialized:
                self.obs_history[history_offset:history_end] = np.tile(
                    single_obs[single_offset:single_end], self.history_length
                )
            else:
                block = self.obs_history[history_offset:history_end]
                block[:-feature_dim] = block[feature_dim:].copy()
                block[-feature_dim:] = single_obs[single_offset:single_end]
            history_offset = history_end
            single_offset = single_end
        self._history_initialized = True

        return torch.from_numpy(self.obs_history.copy()).unsqueeze(0).to(self.device)

    def compute_action(self, obs_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action = self.policy(obs_tensor).squeeze(0).cpu().numpy()
        self.last_action = action.copy()
        return action

    def action_to_torque(self, action: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        """Convert normalised action to joint torques via PD control."""
        target_pos = self.default_joint_pos + action * 0.25
        torque = self.kp * (target_pos - joint_pos) - self.kd * joint_vel
        return torque.clip(-23.5, 23.5)  # Isaac Lab Go2 DCMotor effort limit

    def run(self):
        """Main simulation loop."""
        try:
            import mujoco
        except ImportError:
            print("[ERROR] MuJoCo not installed. Run: pip install mujoco")
            sys.exit(1)

        xml_path = self.cfg["xml_path"]
        print(f"[INFO] Loading MuJoCo scene: {xml_path}")
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # Set initial state
        init_pos = self.cfg.get("base_init_pos", [0.0, 0.0, 0.42])
        init_quat = self.cfg.get("base_init_quat", [1.0, 0.0, 0.0, 0.0])
        data.qpos[:3] = init_pos
        data.qpos[3:7] = init_quat  # MuJoCo: [w, x, y, z]
        data.qpos[7:] = self.default_joint_pos
        mujoco.mj_forward(model, data)

        step = 0
        duration = self.cfg.get("simulation_duration", 60.0)
        max_steps = int(duration / self.dt_sim)

        print("[INFO] Starting simulation. Press Ctrl+C to stop.")
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and step < max_steps:
                    step_start = time.time()

                    if step % self.control_decimation == 0:
                        # Build obs (using MuJoCo sensor data)
                        ang_vel = (
                            data.sensor("angular-velocity").data.copy()
                            if "angular-velocity" in [model.sensor(i).name for i in range(model.nsensor)]
                            else data.qvel[3:6].copy()
                        )
                        quat = data.qpos[3:7].copy()  # [w, x, y, z]
                        joint_pos = data.qpos[7:].copy()
                        joint_vel = data.qvel[6:].copy()
                        obs = self.build_obs(ang_vel, quat, joint_pos, joint_vel)
                        action = self.compute_action(obs)
                        torque = self.action_to_torque(action, joint_pos, joint_vel)
                        data.ctrl[:] = torque

                    mujoco.mj_step(model, data)
                    viewer.sync()

                    elapsed = time.time() - step_start
                    sleep_time = self.dt_sim - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    step += 1

        except KeyboardInterrupt:
            print("\n[INFO] Simulation stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Go2 MuJoCo Sim2Sim deployment.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "deploy" / "configs" / "go2.yaml"),
        help="Path to deployment config YAML.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    deployer = Go2MuJoCoDeployer(cfg, device=args.device)
    deployer.run()


if __name__ == "__main__":
    main()
