#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a custom robot URDF file into a layered USD asset for IsaacLab simulation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "custom_robot"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for URDF to USD conversion."""
    parser = argparse.ArgumentParser(
        description="Convert custom robot URDF into a simulation-ready layered USD asset."
    )
    parser.add_argument(
        "--urdf",
        "--urdf_path",
        type=Path,
        required=True,
        help="Path to the input URDF file.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where converted USD and payloads will be saved (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--usd_filename",
        "--usd-filename",
        type=str,
        default="robot.usd",
        help="Target USD filename (default: robot.usd).",
    )
    parser.add_argument(
        "--robot_type",
        "--robot-type",
        type=str,
        default="Quadruped",
        choices=["Quadruped", "Default", "Humanoid", "Wheeled", "Manipulator"],
        help="Robot type schema for USD (default: Quadruped).",
    )
    parser.add_argument(
        "--fix_base",
        "--fix-base",
        action="store_true",
        default=False,
        help="Fix root link to world (default: False for mobile/quadruped robots).",
    )
    parser.add_argument(
        "--merge_fixed_joints",
        "--merge-fixed-joints",
        action="store_true",
        default=True,
        help="Merge links connected by fixed joints (default: True).",
    )
    parser.add_argument(
        "--collision_type",
        "--collision-type",
        type=str,
        default="Convex Hull",
        choices=["Convex Hull", "Convex Decomposition", "Bounding Sphere", "Bounding Cube"],
        help="Collision mesh approximation (default: Convex Hull).",
    )
    parser.add_argument(
        "--self_collision",
        "--self-collision",
        action="store_true",
        default=False,
        help="Enable self-collisions between robot links (default: False).",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=25.0,
        help="Default joint drive stiffness [Nm/rad] (default: 25.0).",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.5,
        help="Default joint drive damping [Nm/(rad/s)] (default: 0.5).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate resulting USD structure and spawnability (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    """Main conversion entry point."""
    args = parse_args()

    # Clear sys.argv before instantiating SimulationApp so Kit doesn't see our custom CLI args
    sys.argv = [sys.argv[0]]

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    try:
        import isaaclab.sim as sim_utils  # noqa: E402
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

        urdf_path = args.urdf.resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Input URDF file not found: {urdf_path}")

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("[URDF Converter] Starting URDF to USD Conversion...")
        print("=" * 70)
        print(f"  Input URDF:    {urdf_path}")
        print(f"  Output Dir:    {output_dir}")
        print(f"  USD Filename:  {args.usd_filename}")
        print(f"  Robot Type:    {args.robot_type}")
        print(f"  Fix Base:      {args.fix_base}")
        print(f"  Drive Gains:   stiffness={args.stiffness}, damping={args.damping}")
        print(f"  Collision:     {args.collision_type}")

        converter_cfg = UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(output_dir),
            usd_file_name=args.usd_filename,
            force_usd_conversion=True,
            fix_base=args.fix_base,
            merge_fixed_joints=args.merge_fixed_joints,
            collision_type=args.collision_type,
            self_collision=args.self_collision,
            robot_type=args.robot_type,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=args.stiffness,
                    damping=args.damping,
                ),
            ),
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )

        converter = UrdfConverter(converter_cfg)
        generated_usd_path = converter.usd_path
        print(f"[URDF Converter] Success! Generated USD: {generated_usd_path}")

        # Create convenience symlink at output_dir / "robot.usd" if output_dir matches default
        convenience_link = output_dir / "robot.usd"
        if Path(generated_usd_path) != convenience_link:
            try:
                if convenience_link.is_symlink() or convenience_link.exists():
                    convenience_link.unlink()
                convenience_link.symlink_to(Path(generated_usd_path).resolve())
                print(f"[URDF Converter] Linked default asset: {convenience_link} -> {generated_usd_path}")
            except Exception as e:
                print(f"[URDF Converter] Note: Could not create symlink at {convenience_link}: {e}")

        if args.validate:
            from pxr import Usd, UsdPhysics

            print("\n[URDF Validation] Inspecting USD hierarchy...")
            stage = Usd.Stage.Open(generated_usd_path)
            if not stage:
                raise RuntimeError(f"Failed to open generated USD stage: {generated_usd_path}")

            revolute_joints: list[str] = []
            prismatic_joints: list[str] = []
            rigid_bodies: list[str] = []

            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    revolute_joints.append(prim.GetName())
                elif prim.IsA(UsdPhysics.PrismaticJoint):
                    prismatic_joints.append(prim.GetName())
                elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_bodies.append(prim.GetName())

            print(f"  - Rigid Bodies ({len(rigid_bodies)}): {rigid_bodies}")
            print(f"  - Revolute Joints ({len(revolute_joints)}): {revolute_joints}")
            if prismatic_joints:
                print(f"  - Prismatic Joints ({len(prismatic_joints)}): {prismatic_joints}")

            # Verify spawnability
            test_prim_path = "/World/ValidationRobot"
            sim_utils.create_prim(test_prim_path, usd_path=generated_usd_path)
            test_prim = stage.GetPrimAtPath(test_prim_path)
            if not test_prim.IsValid():
                raise RuntimeError(f"Failed to spawn test instance at {test_prim_path}")
            print("  - Stage Spawnability Test: PASSED")
            print("[URDF Validation] Asset is verified and ready for IsaacLab simulation!")

        print("\n" + "=" * 70)
        print("To use your custom robot with Go2 Sim2Sim training/play:")
        print(f"  export ROBOT_TYPE=custom")
        print(f"  export CUSTOM_ROBOT_USD_PATH=\"{generated_usd_path}\"")
        if 'revolute_joints' in locals() and revolute_joints:
            print(f"  export CUSTOM_ROBOT_JOINT_NAMES=\"{','.join(revolute_joints)}\"")
        print("=" * 70 + "\n")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
