# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab asset configuration for the company-provided Yuil arm."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_PACKAGE_ROOT = (
    PROJECT_ROOT / "Yuil 3kg cobot_fixed_20260401" / "YBL-A603_250924.SLDASM"
)
ROBOT_URDF_PATH = ROBOT_PACKAGE_ROOT / "urdf" / "YBL-A603_250924.SLDASM.urdf"
ROBOT_USD_PATH = (
    ROBOT_PACKAGE_ROOT
    / "urdf"
    / "YBL-A603_250924.SLDASM"
    / "YBL-A603_250924.SLDASM.usd"
)

JOINT_NAMES = (
    "j1_link_rev",
    "j2_link_rev",
    "j3_adapter_rev",
    "j4_link_rev",
    "j5_link_rev",
    "j6_link_rev",
)
BASE_LINK_NAME = "base_link"
END_EFFECTOR_LINK_NAME = "j6_adapter_endplate"

# Gravity-on position-drive gains. With a relative action scale of 0.05 rad,
# the shoulder/elbow stiffness must provide enough torque to oppose gravity
# within the policy's command range. Damping is critically damped using the
# effective joint inertias inferred from the original 25 rad/s drive values.
DRIVE_STIFFNESS = {
    "j1_link_rev": 80.0,
    "j2_link_rev": 600.0,
    "j3_adapter_rev": 600.0,
    "j4_link_rev": 120.0,
    "j5_link_rev": 60.0,
    "j6_link_rev": 15.0,
}
DRIVE_DAMPING = {
    "j1_link_rev": 1.730,
    "j2_link_rev": 10.563,
    "j3_adapter_rev": 25.258,
    "j4_link_rev": 5.657,
    "j5_link_rev": 2.268,
    "j6_link_rev": 0.362,
}
DRIVE_EFFORT_LIMIT = {
    "j1_link_rev": 101.0,
    "j2_link_rev": 101.0,
    "j3_adapter_rev": 101.0,
    "j4_link_rev": 63.0,
    "j5_link_rev": 63.0,
    "j6_link_rev": 63.0,
}
DRIVE_VELOCITY_LIMIT = {joint_name: 3.141592654 for joint_name in JOINT_NAMES}

if not ROBOT_USD_PATH.is_file():
    raise FileNotFoundError(f"Yuil robot USD does not exist: {ROBOT_USD_PATH}")


YUIL_3KG_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={joint_name: 0.0 for joint_name in JOINT_NAMES},
        joint_vel={joint_name: 0.0 for joint_name in JOINT_NAMES},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(JOINT_NAMES),
            effort_limit_sim=DRIVE_EFFORT_LIMIT,
            velocity_limit_sim=DRIVE_VELOCITY_LIMIT,
            stiffness=DRIVE_STIFFNESS,
            damping=DRIVE_DAMPING,
            friction=None,
            armature=None,
        ),
    },
)
