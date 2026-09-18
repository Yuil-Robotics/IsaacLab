# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Yuil Dog articulation using bundled USD assets and nominal RS03/RS04 motors."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from .actuators_cfg import YUIL_DOG_ACTUATOR_CFG
from .joints_cfg import YUIL_DOG_DEFAULT_JOINT_POS, YUIL_DOG_ROOT_HEIGHT

YUIL_DOG_DEFAULT_USD = Path(__file__).resolve().parent / "usd" / "yuil_dog.usda"
"""Bundled entrypoint; all USD layers and meshes are packaged below this directory."""
YUIL_DOG_USD_PATH = os.getenv("YUIL_DOG_USD_PATH", str(YUIL_DOG_DEFAULT_USD))
"""Optional explicit USD override; defaults to the bundled robot."""

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
        pos=(0.0, 0.0, YUIL_DOG_ROOT_HEIGHT),
        joint_pos=YUIL_DOG_DEFAULT_JOINT_POS.copy(),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=YUIL_DOG_ACTUATOR_CFG,
)
"""Yuil Dog articulation with nominal RS03 and RS04 DC-motor models."""
