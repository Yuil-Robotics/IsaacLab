# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the received assets, Newton configuration, and TorchScript contract."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pxr import Usd, UsdGeom, UsdPhysics

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from sim2sim_newton.tasks.hx5_cube.hx5_cube_newton_env_cfg import (  # noqa: E402
    NEWTON_ACTUATOR_ARMATURE,
    NEWTON_CUBE_USD,
    NEWTON_ROBOT_USD,
    POLICY_ACTION_DIM,
    POLICY_OBSERVATION_DIM,
    ROBOT_BODY_COUNT,
    Hx5CubeNewtonEnvCfg,
)
from sim2sim_newton.tasks.hx5_cube.hx5_cube_env_cfg import (  # noqa: E402
    DEX_CUBE_DIAGONAL_INERTIA,
    DEX_CUBE_MASS,
    ORIGINAL_ROBOT_USD,
)

POLICY_PATH = PROJECT_DIR.parent / "Robotis_left_benchmark_2026-07-30_16-01-04" / "exported" / "policy.pt"


def validate_assets() -> None:
    """Validate the original source and the Newton wrapper topology."""
    required_paths = (ORIGINAL_ROBOT_USD, NEWTON_ROBOT_USD, NEWTON_CUBE_USD, POLICY_PATH)
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(f"- {path}" for path in missing_paths))

    hand_stage = Usd.Stage.Open(str(NEWTON_ROBOT_USD))
    if hand_stage is None:
        raise RuntimeError(f"Could not open the Newton hand wrapper: {NEWTON_ROBOT_USD}")
    joint_count = sum(prim.IsA(UsdPhysics.RevoluteJoint) for prim in hand_stage.Traverse())
    body_count = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in hand_stage.Traverse())
    valid_collider_count = sum(
        prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI) for prim in hand_stage.Traverse()
    )
    invalid_collider_count = sum(
        not prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI) for prim in hand_stage.Traverse()
    )
    if (joint_count, body_count, valid_collider_count, invalid_collider_count) != (
        POLICY_ACTION_DIM,
        ROBOT_BODY_COUNT,
        ROBOT_BODY_COUNT,
        0,
    ):
        raise RuntimeError(
            "Unexpected hand wrapper topology: "
            f"joints={joint_count}, bodies={body_count}, valid_colliders={valid_collider_count}, "
            f"invalid_colliders={invalid_collider_count}."
        )

    cube_stage = Usd.Stage.Open(str(NEWTON_CUBE_USD))
    if cube_stage is None:
        raise RuntimeError(f"Could not open the Newton cube wrapper: {NEWTON_CUBE_USD}")
    cube_mass_api = UsdPhysics.MassAPI(cube_stage.GetDefaultPrim())
    cube_mass = float(cube_mass_api.GetMassAttr().Get())
    cube_inertia = torch.tensor(cube_mass_api.GetDiagonalInertiaAttr().Get())
    expected_inertia = torch.full((3,), DEX_CUBE_DIAGONAL_INERTIA)
    if abs(cube_mass - DEX_CUBE_MASS) > 1.0e-7 or not torch.allclose(cube_inertia, expected_inertia):
        raise RuntimeError(f"Unexpected cube mass properties: mass={cube_mass}, inertia={cube_inertia.tolist()}.")


def main() -> None:
    """Check configuration values and exported policy tensor dimensions."""
    validate_assets()
    cfg = Hx5CubeNewtonEnvCfg()
    policy = torch.jit.load(str(POLICY_PATH), map_location="cpu").eval()
    normalizer_mean = policy.normalizer._mean
    if normalizer_mean.shape != (1, POLICY_OBSERVATION_DIM):
        raise RuntimeError(f"Expected policy normalizer (1, {POLICY_OBSERVATION_DIM}), got {normalizer_mean.shape}.")
    sample_observations = torch.zeros((1, POLICY_OBSERVATION_DIM), dtype=torch.float32)
    with torch.inference_mode():
        actions = policy(sample_observations)

    if actions.shape != (1, POLICY_ACTION_DIM):
        raise RuntimeError(f"Expected policy output (1, {POLICY_ACTION_DIM}), got {actions.shape}.")
    if not torch.isfinite(actions).all():
        raise RuntimeError("The policy returned non-finite values for a zero observation.")
    actuator = cfg.robot_cfg.actuators["hand"]
    if cfg.seed != 42 or actuator.armature != NEWTON_ACTUATOR_ARMATURE:
        raise RuntimeError(f"Unexpected Newton task seed/armature: seed={cfg.seed}, armature={actuator.armature}.")

    print(
        "validation=passed | backend=Newton MJWarp | hand=21_bodies/20_joints/21_colliders | "
        f"observation={POLICY_OBSERVATION_DIM} | action={POLICY_ACTION_DIM} | "
        f"cube_mass={DEX_CUBE_MASS:.6g} kg | cube_inertia={DEX_CUBE_DIAGONAL_INERTIA:.9g} kg*m^2 | "
        f"dt={cfg.sim.dt:.8f} s | decimation={cfg.decimation} | control_rate=30 Hz"
    )


if __name__ == "__main__":
    main()
