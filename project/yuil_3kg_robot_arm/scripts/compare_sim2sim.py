#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare PhysX and Newton/MJWarp JSON results from ``sim2sim_probe.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("physx", type=Path)
parser.add_argument("newton", type=Path)
args = parser.parse_args()


def _maximum_absolute_difference(lhs: dict, rhs: dict, key: str) -> float:
    """Return the maximum absolute difference for one numeric result field."""
    return float(np.max(np.abs(np.asarray(lhs[key], dtype=float) - np.asarray(rhs[key], dtype=float))))


def main() -> None:
    """Print model-contract and dynamic-response differences."""
    physx = json.loads(args.physx.read_text(encoding="utf-8"))
    newton = json.loads(args.newton.read_text(encoding="utf-8"))
    if physx["usd_path"] != newton["usd_path"]:
        raise ValueError("The probes did not use the same USD file.")
    if physx["joint_names"] != newton["joint_names"]:
        raise ValueError("The probes produced different joint orders.")
    if physx["gravity_m_s2"] != newton["gravity_m_s2"]:
        raise ValueError("The probes used different gravity vectors.")
    if physx["physics_dt_s"] != newton["physics_dt_s"]:
        raise ValueError("The probes used different physics time steps.")
    if physx["duration_s"] != newton["duration_s"]:
        raise ValueError("The probes used different durations.")

    print(f"USD: {physx['usd_path']}")
    print(f"Gravity [m/s^2]: {tuple(physx['gravity_m_s2'])}")
    print(f"Joint order: {tuple(physx['joint_names'])}")
    for key in (
        "body_mass_kg",
        "body_com_m",
        "body_inertia_kg_m2",
        "joint_limits_rad",
        "joint_stiffness",
        "joint_damping",
        "joint_friction",
    ):
        print(f"{key} max abs diff: {_maximum_absolute_difference(physx, newton, key):.9g}")

    print("\nDynamic response max abs differences")
    for key in (
        "final_position_rad",
        "final_velocity_rad_s",
        "final_position_error_rad",
        "overshoot_rad",
        "tail_rms_velocity_rad_s",
    ):
        print(f"{key}: {_maximum_absolute_difference(physx, newton, key):.9g}")


if __name__ == "__main__":
    main()
