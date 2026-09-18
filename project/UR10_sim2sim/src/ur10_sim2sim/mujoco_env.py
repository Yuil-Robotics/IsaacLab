# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo rollout environment matching the Isaac Lab UR10e Reach contract."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .config import ACTUATOR_NAMES, JOINT_NAMES, Sim2SimConfig
from .math_utils import quat_from_euler_xyz, quat_xyzw_to_wxyz
from .policy import TorchScriptPolicy, build_observation


class UR10eMuJoCoEnv:
    """Minimal single-robot MuJoCo environment for policy inference."""

    def __init__(
        self,
        model_path: str | Path,
        policy: TorchScriptPolicy,
        config: Sim2SimConfig | None = None,
        target_pos: np.ndarray | None = None,
        target_rpy: np.ndarray | None = None,
        action_clip: float | None = None,
    ):
        self.config = config or Sim2SimConfig()
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.policy = policy
        self.action_clip = action_clip
        self.target_pos = np.array(target_pos if target_pos is not None else self.config.target_pos, dtype=np.float64)
        target_rpy_value = target_rpy if target_rpy is not None else self.config.target_rpy
        self.target_quat_xyzw = quat_from_euler_xyz(np.asarray(target_rpy_value))
        self.target_quat_wxyz = quat_xyzw_to_wxyz(self.target_quat_xyzw)
        self._joint_qpos_ids = np.array(
            [self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINT_NAMES], dtype=int
        )
        self._joint_dof_ids = np.array(
            [self.model.jnt_dofadr[self.model.joint(name).id] for name in JOINT_NAMES], dtype=int
        )
        self._actuator_ids = np.array([self.model.actuator(name).id for name in ACTUATOR_NAMES], dtype=int)
        self._target_mocap_id = self.model.body("target").mocapid[0]
        self.reset()

    def reset(self) -> None:
        """Reset to the same nominal joint state used by Isaac Lab."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos_ids] = self.config.initial_joint_pos
        self.data.ctrl[self._actuator_ids] = self.config.initial_joint_pos
        self.data.mocap_pos[self._target_mocap_id] = self.target_pos
        self.data.mocap_quat[self._target_mocap_id] = self.target_quat_wxyz
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, position: np.ndarray, rpy: np.ndarray | None = None) -> None:
        """Set the commanded target pose and update its MuJoCo marker."""
        target_pos = np.asarray(position, dtype=np.float64)
        if target_pos.shape != (3,) or not np.all(np.isfinite(target_pos)):
            raise ValueError(f"Expected a finite target position with shape (3,), got {target_pos}.")
        self.target_pos = target_pos.copy()
        if rpy is not None:
            target_rpy = np.asarray(rpy, dtype=np.float64)
            if target_rpy.shape != (3,) or not np.all(np.isfinite(target_rpy)):
                raise ValueError(f"Expected finite target RPY angles with shape (3,), got {target_rpy}.")
            self.target_quat_xyzw = quat_from_euler_xyz(target_rpy)
            self.target_quat_wxyz = quat_xyzw_to_wxyz(self.target_quat_xyzw)
        self.data.mocap_pos[self._target_mocap_id] = self.target_pos
        self.data.mocap_quat[self._target_mocap_id] = self.target_quat_wxyz
        mujoco.mj_forward(self.model, self.data)

    def observation(self) -> np.ndarray:
        """Return the policy observation in Isaac Lab term order."""
        return build_observation(
            self.data.qpos[self._joint_qpos_ids],
            self.data.qvel[self._joint_dof_ids],
            self.target_pos,
            self.target_quat_xyzw,
        )

    def policy_step(self) -> dict[str, np.ndarray | float]:
        """Apply one policy command and advance two MuJoCo physics steps."""
        observation = self.observation()
        action = self.policy(observation)
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        joint_pos = self.data.qpos[self._joint_qpos_ids].copy()
        joint_target = joint_pos + self.config.action_scale * action
        self.data.ctrl[self._actuator_ids] = joint_target
        for _ in range(self.config.policy_decimation):
            mujoco.mj_step(self.model, self.data)
        return {
            "time": float(self.data.time),
            "observation": observation,
            "action": action.copy(),
            "joint_pos": self.data.qpos[self._joint_qpos_ids].copy(),
            "joint_vel": self.data.qvel[self._joint_dof_ids].copy(),
            "joint_target": joint_target,
            "ee_pos": self.data.site("wrist_3_site").xpos.copy(),
            "ee_quat": self.data.xquat[self.model.body("wrist_3_link").id].copy(),
            "target_pos": self.target_pos.copy(),
            "target_quat_xyzw": self.target_quat_xyzw.copy(),
        }
