# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Nominal RS03/RS04 motors: stiffness [N·m/rad], damping [N·m·s/rad]."""

from isaaclab.actuators import DCMotorCfg

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
