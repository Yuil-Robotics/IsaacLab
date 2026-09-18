#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the company URDF without launching Isaac Sim."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yuil_3kg_robot_arm.asset_cfg import (  # noqa: E402
    END_EFFECTOR_LINK_NAME,
    JOINT_NAMES,
    ROBOT_URDF_PATH,
    ROBOT_USD_PATH,
)


def main() -> None:
    """Print robot structure and validate the configured training contract."""
    robot = ET.parse(ROBOT_URDF_PATH).getroot()
    links = {link.attrib["name"] for link in robot.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in robot.findall("joint")}

    print(f"Robot: {robot.attrib['name']}")
    print(f"URDF:  {ROBOT_URDF_PATH}")
    print(f"USD:   {ROBOT_USD_PATH}")
    print(f"Links: {len(links)}")
    print(f"Joints: {len(joints)}")
    print(f"Configured end effector: {END_EFFECTOR_LINK_NAME}")
    print("\nControlled joints:")
    for index, joint_name in enumerate(JOINT_NAMES):
        joint = joints[joint_name]
        limit = joint.find("limit")
        axis = joint.find("axis")
        print(
            f"  {index}: {joint_name}, axis={axis.attrib['xyz']}, "
            f"range=[{limit.attrib['lower']}, {limit.attrib['upper']}] rad, "
            f"effort={limit.attrib['effort']}, velocity={limit.attrib['velocity']} rad/s"
        )

    if END_EFFECTOR_LINK_NAME not in links:
        raise ValueError(f"Configured end-effector link is missing: {END_EFFECTOR_LINK_NAME}")
    missing_joints = set(JOINT_NAMES) - joints.keys()
    if missing_joints:
        raise ValueError(f"Configured joints are missing from the URDF: {sorted(missing_joints)}")
    if not ROBOT_USD_PATH.is_file():
        raise FileNotFoundError(f"Converted USD is missing: {ROBOT_USD_PATH}")
    print("\nURDF/USD contract validation passed.")


if __name__ == "__main__":
    main()
