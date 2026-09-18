# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project robot asset paths and modular articulation configurations."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from .sim2real_cfg import Sim2RealDCMotorCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets"
REMOTE_GO2_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
)
REMOTE_GROUND_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd"
)
LOCAL_GO2_USD_PATH = (
    ASSET_ROOT / "Assets" / "Isaac" / "6.0" / "Isaac" / "IsaacLab" / "Robots" / "Unitree" / "Go2" / "go2.usd"
)
LOCAL_GROUND_USD_PATH = (
    ASSET_ROOT / "Assets" / "Isaac" / "6.0" / "Isaac" / "Environments" / "Grid" / "default_environment.usd"
)

GO2_USD_PATH = str(LOCAL_GO2_USD_PATH) if LOCAL_GO2_USD_PATH.is_file() else REMOTE_GO2_USD_URL
GO2_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)
"""Canonical Go2 policy and model-contract joint order."""

GO2_FOOT_NAMES: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GO2_TROT_FOOT_PAIRS: tuple[tuple[str, str], tuple[str, str]] = (
    ("FL_foot", "RR_foot"),
    ("FR_foot", "RL_foot"),
)

GO2_CFG = copy.deepcopy(UNITREE_GO2_CFG)
GO2_CFG.spawn.usd_path = GO2_USD_PATH
"""Go2 articulation using the downloaded project asset when available."""

# -----------------------------------------------------------------------------
# Yuil Dog Configuration
# -----------------------------------------------------------------------------
YUIL_DOG_DEFAULT_USD = ASSET_ROOT / "yuil_dog" / "yuil_dog.usda"
YUIL_DOG_USD_PATH = os.getenv("YUIL_DOG_USD_PATH", str(YUIL_DOG_DEFAULT_USD))
"""Path to the stable Yuil Dog USD entrypoint."""

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

YUIL_DOG_ACTUATOR_CFG: dict[str, DCMotorCfg] = {
    "hip_roll": DCMotorCfg(
        joint_names_expr=[".*_hip_roll"],
        effort_limit=30.0,
        saturation_effort=60.0,
        velocity_limit=10.0,
        effort_limit_sim=60.0,
        velocity_limit_sim=11.0,
        stiffness=60.0,
        damping=3.5,
        armature=0.0043,
        friction=0.45,
        dynamic_friction=0.27,
        viscous_friction=0.003,
    ),
    "hip_pitch": DCMotorCfg(
        joint_names_expr=[".*_hip_pitch"],
        effort_limit=60.0,
        saturation_effort=115.0,
        velocity_limit=20.0,
        effort_limit_sim=115.0,
        velocity_limit_sim=22.0,
        stiffness=85.0,
        damping=2.2,
        armature=0.010,
        friction=0.90,
        dynamic_friction=0.53,
        viscous_friction=0.005,
    ),
    "knee_pitch": DCMotorCfg(
        joint_names_expr=[".*_knee_pitch"],
        effort_limit=60.0,
        saturation_effort=115.0,
        velocity_limit=20.0,
        effort_limit_sim=115.0,
        velocity_limit_sim=22.0,
        stiffness=80.0,
        damping=2.2,
        armature=0.010,
        friction=0.90,
        dynamic_friction=0.53,
        viscous_friction=0.005,
    ),
}
"""Nominal RS03 and RS04 actuator parameters without domain randomization."""

YUIL_DOG_SIM2REAL_ACTUATOR_CFG: dict[str, Sim2RealDCMotorCfg] = {
    "hip_roll": Sim2RealDCMotorCfg(
        joint_names_expr=[".*_hip_roll"],
        effort_limit=30.0,
        saturation_effort=60.0,
        velocity_limit=20.0,
        effort_limit_sim=60.0,
        velocity_limit_sim=22.0,
        stiffness=60.0,
        damping=3.5,
        armature=0.0043,
        friction=0.45,
        dynamic_friction=0.27,
        viscous_friction=0.003,
        min_delay=0,
        max_delay=3,
        effort_scale_range=(0.85, 1.15),
        motor_strength_range=(0.9, 1.1),
    ),
    "hip_pitch": Sim2RealDCMotorCfg(
        joint_names_expr=[".*_hip_pitch"],
        effort_limit=60.0,
        saturation_effort=120.0,
        velocity_limit=20.0,
        effort_limit_sim=120.0,
        velocity_limit_sim=22.0,
        stiffness=85.0,
        damping=2.2,
        armature=0.010,
        friction=0.90,
        dynamic_friction=0.53,
        viscous_friction=0.005,
        min_delay=0,
        max_delay=3,
        effort_scale_range=(0.85, 1.15),
        motor_strength_range=(0.9, 1.1),
    ),
    "knee_pitch": Sim2RealDCMotorCfg(
        joint_names_expr=[".*_knee_pitch"],
        effort_limit=60.0,
        saturation_effort=120.0,
        velocity_limit=20.0,
        effort_limit_sim=120.0,
        velocity_limit_sim=22.0,
        stiffness=80.0,
        damping=2.2,
        armature=0.010,
        friction=0.90,
        dynamic_friction=0.53,
        viscous_friction=0.005,
        min_delay=0,
        max_delay=3,
        effort_scale_range=(0.85, 1.15),
        motor_strength_range=(0.9, 1.1),
    ),
}
"""Sim2Real RS03 and RS04 actuator parameters with randomized delay and motor strength."""

YUIL_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=YUIL_DOG_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    articulation_root_prim_path="/Geometry/base/base_link",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_hip_roll": 0.0,
            ".*L_hip_pitch": -0.5,
            ".*R_hip_pitch": 0.5,
            ".*L_knee_pitch": 0.5,
            ".*R_knee_pitch": -0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=YUIL_DOG_ACTUATOR_CFG,
)
"""Yuil Dog articulation with nominal RS03 and RS04 DC-motor models."""

# -----------------------------------------------------------------------------
# Custom Robot Configuration (URDF -> USD replacement target)
# -----------------------------------------------------------------------------
CUSTOM_ROBOT_DEFAULT_USD = ASSET_ROOT / "custom_robot" / "robot.usd"
CUSTOM_ROBOT_USD_PATH = os.getenv("CUSTOM_ROBOT_USD_PATH", os.getenv("ROBOT_USD_PATH", str(CUSTOM_ROBOT_DEFAULT_USD)))

CUSTOM_ROBOT_JOINT_NAMES: tuple[str, ...] = GO2_JOINT_NAMES
"""Default joint order for custom quadruped, overridable via environment or config."""

CUSTOM_ROBOT_FOOT_NAMES: tuple[str, ...] = GO2_FOOT_NAMES
"""Default foot body names for custom quadruped."""

CUSTOM_ROBOT_TROT_FOOT_PAIRS: tuple[tuple[str, str], tuple[str, str]] = GO2_TROT_FOOT_PAIRS
"""Default synchronized trot foot pairs for custom quadruped."""

CUSTOM_ROBOT_CFG = copy.deepcopy(UNITREE_GO2_CFG)
CUSTOM_ROBOT_CFG.spawn.usd_path = CUSTOM_ROBOT_USD_PATH
"""Custom robot articulation referencing the converted USD asset."""


def get_active_robot_type() -> str:
    """Return the active robot type identifier ('go2' or 'custom')."""
    return os.getenv("ROBOT_TYPE", "go2").strip().lower()


def get_active_robot_cfg(robot_type: str | None = None) -> ArticulationCfg:
    """Resolve the active robot ArticulationCfg."""
    r_type = (robot_type or get_active_robot_type()).lower()
    if r_type == "custom":
        return copy.deepcopy(CUSTOM_ROBOT_CFG)
    return copy.deepcopy(GO2_CFG)


def get_active_joint_names(robot_type: str | None = None) -> tuple[str, ...]:
    """Resolve the active robot joint names sequence."""
    r_type = (robot_type or get_active_robot_type()).lower()
    if r_type == "custom":
        raw = os.getenv("CUSTOM_ROBOT_JOINT_NAMES")
        if raw:
            return tuple(x.strip() for x in raw.split(",") if x.strip())
        return CUSTOM_ROBOT_JOINT_NAMES
    return GO2_JOINT_NAMES


def get_active_foot_names(robot_type: str | None = None) -> tuple[str, ...]:
    """Resolve the active robot foot body names."""
    r_type = (robot_type or get_active_robot_type()).lower()
    if r_type == "custom":
        raw = os.getenv("CUSTOM_ROBOT_FOOT_NAMES")
        if raw:
            return tuple(x.strip() for x in raw.split(",") if x.strip())
        return CUSTOM_ROBOT_FOOT_NAMES
    return GO2_FOOT_NAMES


def get_active_trot_foot_pairs(
    robot_type: str | None = None,
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Resolve the synchronized trot foot pairs for the active robot."""
    r_type = (robot_type or get_active_robot_type()).lower()
    if r_type == "custom":
        return CUSTOM_ROBOT_TROT_FOOT_PAIRS
    return GO2_TROT_FOOT_PAIRS
