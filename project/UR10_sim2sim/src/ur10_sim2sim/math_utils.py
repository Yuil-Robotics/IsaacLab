# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Small quaternion helpers matching Isaac Lab and MuJoCo conventions."""

from __future__ import annotations

import numpy as np


def quat_from_euler_xyz(rpy: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles to an Isaac Lab ``xyzw`` quaternion."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quat = np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )
    if quat[3] < 0.0:
        quat = -quat
    return quat


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert an Isaac Lab quaternion to MuJoCo scalar-first order."""
    quat = np.asarray(quat)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion final dimension 4, got {quat.shape}.")
    return quat[..., [3, 0, 1, 2]]
