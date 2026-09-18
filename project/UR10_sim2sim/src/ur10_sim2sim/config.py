# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Shared configuration for the UR10e sim-to-sim transfer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

ACTUATOR_NAMES = tuple(name.replace("_joint", "_actuator") for name in JOINT_NAMES)


@dataclass(frozen=True)
class Sim2SimConfig:
    """Numerical contract shared by Isaac Lab and MuJoCo."""

    physics_dt: float = 1.0 / 120.0
    policy_decimation: int = 2
    action_scale: float = 0.0625
    initial_joint_pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [np.pi, -np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0, 0.0],
            dtype=np.float64,
        )
    )
    target_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.8875, -0.225, 0.2], dtype=np.float64)
    )
    target_rpy: np.ndarray = field(
        default_factory=lambda: np.array([np.pi, 0.0, -np.pi / 2.0], dtype=np.float64)
    )

    @property
    def policy_dt(self) -> float:
        """Policy update period in seconds."""
        return self.physics_dt * self.policy_decimation
