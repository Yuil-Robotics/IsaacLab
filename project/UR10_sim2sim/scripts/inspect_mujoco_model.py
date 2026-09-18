#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the UR10e MuJoCo model and verify the transfer contract."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ur10_sim2sim.config import ACTUATOR_NAMES, JOINT_NAMES, Sim2SimConfig  # noqa: E402


def main() -> None:
    """Load the model and validate its names, rates, and nominal state."""
    model_path = PROJECT_ROOT / "assets" / "ur10e.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    cfg = Sim2SimConfig()

    actual_joints = tuple(model.joint(index).name for index in range(model.njnt))
    actual_actuators = tuple(model.actuator(index).name for index in range(model.nu))
    assert actual_joints == JOINT_NAMES, (actual_joints, JOINT_NAMES)
    assert actual_actuators == ACTUATOR_NAMES, (actual_actuators, ACTUATOR_NAMES)
    assert np.isclose(model.opt.timestep, cfg.physics_dt)

    qpos_ids = np.array([model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES])
    data.qpos[qpos_ids] = cfg.initial_joint_pos
    data.ctrl[:] = cfg.initial_joint_pos
    mujoco.mj_forward(model, data)

    print(f"Model: {model_path}")
    print(f"Physics timestep: {model.opt.timestep:.12f} s ({1.0 / model.opt.timestep:.1f} Hz)")
    print(f"Policy timestep:  {cfg.policy_dt:.12f} s ({1.0 / cfg.policy_dt:.1f} Hz)")
    print(f"Joint order:      {actual_joints}")
    print(f"Actuator order:   {actual_actuators}")
    print(f"Initial q:        {data.qpos[qpos_ids]}")
    print(f"Initial EE pos:   {data.site('wrist_3_site').xpos}")
    print("MuJoCo model contract: OK")


if __name__ == "__main__":
    main()
