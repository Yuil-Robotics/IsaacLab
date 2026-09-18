# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the temporary Newton HX5 USD with a fixed, pose-neutral root."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_DIR = Path(__file__).resolve().parents[1]
ISAACLAB_DIR = PROJECT_DIR.parents[2]
DEFAULT_INPUT = PROJECT_DIR.parent / "sim2sim-mujoco" / "model" / "MJCF" / "hx5_d20_left.xml"
DEFAULT_OUTPUT = PROJECT_DIR / "assets" / "hx5_d20_left_fixed" / "hx5_d20_left_2.usd"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source HX5 MJCF path.")
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination USD path.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402


def main() -> None:
    """Convert the MJCF and restore the training asset's root semantics."""
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"HX5 MJCF was not found: {input_path}")

    converter = MjcfConverter(
        MjcfConverterCfg(
            asset_path=str(input_path),
            usd_dir=str(output_path.parent),
            force_usd_conversion=False,
            self_collision=True,
            fix_base=True,
        )
    )

    stage = Usd.Stage.Open(converter.usd_path)
    stage.SetEditTarget(stage.GetRootLayer())
    root = stage.GetDefaultPrim()
    root_link = stage.GetPrimAtPath(root.GetPath().AppendPath("Geometry/hx5_d20_left_base"))
    if not root_link.IsValid():
        raise RuntimeError("Converted USD does not contain Geometry/hx5_d20_left_base.")

    root_link.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.0))
    root_link.GetAttribute("xformOp:orient").Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    fixed_joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]
    if not fixed_joints:
        raise RuntimeError("MJCF conversion did not create the required world fixed joint.")
    for fixed_joint_prim in fixed_joints:
        fixed_joint = UsdPhysics.FixedJoint(fixed_joint_prim)
        fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    stage.GetRootLayer().Save()

    local_transform = UsdGeom.Xformable(root_link).GetLocalTransformation()
    print(f"generated_usd={converter.usd_path}")
    print(f"root_link_transform={local_transform}")
    print(f"fixed_joints={[str(prim.GetPath()) for prim in fixed_joints]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
