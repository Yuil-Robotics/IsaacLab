# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yuil Dog joint ordering, position limits, frame paths, and initial pose."""

YUIL_DOG_BASE_PRIM_PATH = "{ENV_REGEX_NS}/Robot/Geometry/base/base_link"
"""Yuil Dog base rigid-body prim path inside each environment namespace."""

YUIL_DOG_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_roll",
    "FL_hip_pitch",
    "FL_knee_pitch",
    "FR_hip_roll",
    "FR_hip_pitch",
    "FR_knee_pitch",
    "RL_hip_roll",
    "RL_hip_pitch",
    "RL_knee_pitch",
    "RR_hip_roll",
    "RR_hip_pitch",
    "RR_knee_pitch",
)
"""Canonical leg-major Yuil Dog policy and model-contract joint order."""

YUIL_DOG_FOOT_NAMES: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
"""Yuil Dog foot body names."""

YUIL_DOG_TROT_FOOT_PAIRS: tuple[tuple[str, str], tuple[str, str]] = (
    ("FL_foot", "RR_foot"),
    ("FR_foot", "RL_foot"),
)
"""Yuil Dog synchronized trot foot pairs."""

YUIL_DOG_JOINT_POSITION_CLIP: dict[str, tuple[float, float]] = {
    "FL_hip_roll": (-0.38, 0.46),
    "FR_hip_roll": (-0.46, 0.38),
    "RL_hip_roll": (-0.46, 0.38),
    "RR_hip_roll": (-0.38, 0.46),
    ".*L_hip_pitch": (-1.80, 1.20),
    ".*R_hip_pitch": (-1.20, 1.80),
    ".*L_knee_pitch": (0.05, 1.57),
    ".*R_knee_pitch": (-1.57, -0.05),
}
"""Joint position target limits [rad]."""

YUIL_DOG_ROOT_HEIGHT: float = 0.325
"""Initial and target base-link origin height above ground [m]."""

YUIL_DOG_DEFAULT_JOINT_POS: dict[str, float] = {
    "FL_hip_roll": -0.05,
    "FR_hip_roll": 0.05,
    "RL_hip_roll": 0.05,
    "RR_hip_roll": -0.05,
    ".*L_hip_pitch": -0.7,
    ".*R_hip_pitch": 0.7,
    ".*L_knee_pitch": 0.7,
    ".*R_knee_pitch": -0.7,
}
"""Nominal crouched joint pose [rad], also used as the action offset."""
