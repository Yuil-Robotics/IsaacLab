# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create a Newton-compatible wrapper around the original Robotis HX5 USD."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_DIR
    / "robotis_hand"
    / "robotis_hand_description"
    / "urdf"
    / "hx5_d20_left_2"
    / "hx5_d20_left_2"
    / "hx5_d20_left_2.usd"
)
DEFAULT_OUTPUT = PROJECT_DIR / "assets" / "hx5_d20_left_original_newton" / "hx5_d20_left.usda"
DEFAULT_CUBE_OUTPUT = PROJECT_DIR / "assets" / "dex_cube_original_newton" / "dex_cube.usda"
DEFAULT_CUBE_SOURCE_DIR = PROJECT_DIR / "assets" / "dex_cube_original_newton" / "source"

DEX_CUBE_SOURCE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Blocks/DexCube"
)
DEX_CUBE_SOURCE_FILES = (
    "dex_cube_instanceable.usd",
    "Props/instanceable_meshes.usd",
    "Materials/dex_cube_mod.png",
)
DEX_CUBE_SCALE = 0.8
DEX_CUBE_MASS = 0.077
DEX_CUBE_SIDE_LENGTH = 0.06 * DEX_CUBE_SCALE
DEX_CUBE_DENSITY = DEX_CUBE_MASS / DEX_CUBE_SIDE_LENGTH**3
DEX_CUBE_DIAGONAL_INERTIA = DEX_CUBE_MASS * DEX_CUBE_SIDE_LENGTH**2 / 6.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Original Robotis HX5 USD path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Newton wrapper USD path.")
    parser.add_argument(
        "--cube_output",
        type=Path,
        default=DEFAULT_CUBE_OUTPUT,
        help="Newton DexCube wrapper USD path.",
    )
    parser.add_argument(
        "--cube_source_dir",
        type=Path,
        default=DEFAULT_CUBE_SOURCE_DIR,
        help="Directory used for the local Isaac DexCube source layers.",
    )
    return parser.parse_args()


def create_wrapper(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    """Create a wrapper that moves collision schemas from Xforms to their meshes.

    Args:
        input_path: Original layered Robotis USD path.
        output_path: Destination wrapper USD path.

    Returns:
        Counts of revolute joints, rigid bodies, and collision meshes.
    """
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Original Robotis USD was not found: {input_path}")

    source_stage = Usd.Stage.Open(str(input_path))
    if source_stage is None or not source_stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Could not open a valid default prim from: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetMetadata("upAxis", source_stage.GetMetadata("upAxis"))
    stage.SetMetadata("metersPerUnit", source_stage.GetMetadata("metersPerUnit"))

    source_root = source_stage.GetDefaultPrim()
    root = UsdGeom.Xform.Define(stage, source_root.GetPath()).GetPrim()
    relative_source = Path(os.path.relpath(input_path, output_path.parent)).as_posix()
    root.GetReferences().AddReference(relative_source)
    stage.SetDefaultPrim(root)

    # The PhysX asset attaches collision schemas to an Xform inside each
    # instance. Newton expects the schema on the actual GPrim. De-instance only
    # collision trees so the stronger wrapper layer can relocate those schemas.
    collision_instance_paths = [
        prim.GetPath() for prim in stage.Traverse() if prim.IsInstance() and prim.GetName() == "collisions"
    ]
    for prim_path in collision_instance_paths:
        stage.GetPrimAtPath(prim_path).SetInstanceable(False)

    collision_mesh_count = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Gprim):
            continue

        collision_meshes = [child for child in Usd.PrimRange(prim) if child.IsA(UsdGeom.Mesh)]
        if len(collision_meshes) != 1:
            raise RuntimeError(f"Expected one collision mesh below {prim.GetPath()}, found {len(collision_meshes)}.")

        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        collision_mesh = collision_meshes[0]
        UsdPhysics.CollisionAPI.Apply(collision_mesh)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collision_mesh)
        mesh_collision.CreateApproximationAttr("convexHull")
        collision_mesh_count += 1

    stage.GetRootLayer().Save()

    revolute_joint_count = sum(prim.IsA(UsdPhysics.RevoluteJoint) for prim in stage.Traverse())
    rigid_body_count = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse())
    if (revolute_joint_count, rigid_body_count, collision_mesh_count) != (20, 21, 21):
        raise RuntimeError(
            "Unexpected converted asset counts: "
            f"joints={revolute_joint_count}, bodies={rigid_body_count}, colliders={collision_mesh_count}."
        )
    return revolute_joint_count, rigid_body_count, collision_mesh_count


def download_cube_sources(source_dir: Path) -> Path:
    """Download the small Isaac DexCube layers and texture when not local.

    Args:
        source_dir: Destination directory for the source USD layers.

    Returns:
        Path to the local DexCube root layer.
    """
    source_dir = source_dir.expanduser().resolve()
    for relative_path in DEX_CUBE_SOURCE_FILES:
        destination = source_dir / relative_path
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination.with_suffix(destination.suffix + ".download")
        try:
            urllib.request.urlretrieve(f"{DEX_CUBE_SOURCE_URL}/{relative_path}", temporary_path)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return source_dir / DEX_CUBE_SOURCE_FILES[0]


def create_cube_wrapper(source_path: Path, output_path: Path) -> None:
    """Create a scaled DexCube wrapper with the training-time mass and inertia.

    Args:
        source_path: Local Isaac DexCube root USD layer.
        output_path: Destination wrapper USD path.
    """
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Isaac DexCube USD was not found: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/DexCube").GetPrim()
    relative_source = Path(os.path.relpath(source_path, output_path.parent)).as_posix()
    root.GetReferences().AddReference(relative_source)
    root.GetAttribute("xformOp:scale").Set(Gf.Vec3d(DEX_CUBE_SCALE))
    stage.SetDefaultPrim(root)

    # Newton does not currently discover the collider through the instance
    # proxy used by the Isaac DexCube. De-instance just that subtree.
    stage.OverridePrim("/DexCube/collisions").SetInstanceable(False)

    # The source asset authors mass but leaves inertia for PhysX to infer. The
    # final 48 mm cube inertia is written explicitly so Newton uses the same
    # physical body instead of its small-sphere fallback.
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(DEX_CUBE_MASS)
    mass_api.CreateDensityAttr(DEX_CUBE_DENSITY)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(DEX_CUBE_DIAGONAL_INERTIA))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    stage.GetRootLayer().Save()


def main() -> None:
    """Generate and report the Newton-compatible hand and cube wrappers."""
    args = parse_args()
    counts = create_wrapper(args.input, args.output)
    cube_source = download_cube_sources(args.cube_source_dir)
    create_cube_wrapper(cube_source, args.cube_output)
    print(
        f"generated_usd={args.output.expanduser().resolve()} | "
        f"revolute_joints={counts[0]} | rigid_bodies={counts[1]} | collision_meshes={counts[2]}"
    )
    print(
        f"generated_cube_usd={args.cube_output.expanduser().resolve()} | "
        f"side_length={DEX_CUBE_SIDE_LENGTH:.6g} m | mass={DEX_CUBE_MASS:.6g} kg | "
        f"diagonal_inertia={DEX_CUBE_DIAGONAL_INERTIA:.9g} kg*m^2"
    )


if __name__ == "__main__":
    main()
