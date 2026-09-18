#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the Yuil Dog URDF into a validated, layered USD asset."""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_ROOT = PROJECT_ROOT / "Yuil_dog_description" / "dog_description"
DEFAULT_URDF_PATH = DESCRIPTION_ROOT / "urdf" / "TOTAL ASSY_4차_URDF_sample.urdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "yuil_dog"
DEFAULT_ENTRYPOINT_NAME = "yuil_dog.usda"
ROS_PACKAGE_NAME = "dog_description"
EXPECTED_REVOLUTE_JOINTS = 12
HIGH_POLYGON_THRESHOLD = 50_000
COLLISION_APPROXIMATION_TOKENS = {
    "Convex Hull": "convexHull",
    "Convex Decomposition": "convexDecomposition",
    "Bounding Sphere": "boundingSphere",
    "Bounding Cube": "boundingCube",
}


@dataclass(frozen=True)
class UrdfSummary:
    """Summary of the URDF checks performed before conversion."""

    robot_name: str
    link_count: int
    joint_count: int
    revolute_joint_names: tuple[str, ...]
    mesh_paths: tuple[Path, ...]
    collision_mesh_paths: tuple[Path, ...]
    shared_visual_collision_mesh_count: int
    high_polygon_collision_meshes: tuple[tuple[Path, int], ...]
    warnings: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf_path",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help=f"Input URDF path (default: {DEFAULT_URDF_PATH}).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the layered USD asset (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--entrypoint_name",
        default=DEFAULT_ENTRYPOINT_NAME,
        help=f"Stable USD entrypoint filename (default: {DEFAULT_ENTRYPOINT_NAME}).",
    )
    parser.add_argument(
        "--collision_type",
        choices=list(COLLISION_APPROXIMATION_TOKENS),
        default="Convex Decomposition",
        help="Collision approximation used for mesh collision geometry (default: Convex Decomposition).",
    )
    parser.add_argument(
        "--self_collision",
        action="store_true",
        help="Enable articulation self-collision (default: disabled).",
    )
    parser.add_argument(
        "--fix_base",
        action="store_true",
        help="Fix the robot base to the world (default: disabled for locomotion training).",
    )
    fixed_joint_group = parser.add_mutually_exclusive_group()
    fixed_joint_group.add_argument(
        "--merge_fixed_joints",
        dest="merge_fixed_joints",
        action="store_true",
        help="Merge links connected by fixed joints.",
    )
    fixed_joint_group.add_argument(
        "--preserve_fixed_joints",
        dest="merge_fixed_joints",
        action="store_false",
        help="Preserve fixed foot and motor links (default; required by foot rewards).",
    )
    parser.set_defaults(merge_fixed_joints=False)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat URDF preflight warnings as errors.",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Keep intermediate URDF importer artifacts for debugging.",
    )
    return parser.parse_args()


def _resolve_mesh_path(mesh_uri: str, urdf_path: Path) -> Path:
    """Resolve a URDF mesh URI to a local path.

    Args:
        mesh_uri: Mesh URI from the URDF.
        urdf_path: Absolute path of the URDF containing the URI.

    Returns:
        Resolved local mesh path.

    Raises:
        ValueError: If a ``package://`` URI names an unexpected ROS package.
    """
    package_prefix = "package://"
    if mesh_uri.startswith(package_prefix):
        package_and_path = mesh_uri.removeprefix(package_prefix)
        package_name, separator, relative_path = package_and_path.partition("/")
        if not separator or package_name != ROS_PACKAGE_NAME:
            raise ValueError(f"Unsupported mesh URI '{mesh_uri}'; expected package://{ROS_PACKAGE_NAME}/...")
        return (DESCRIPTION_ROOT / relative_path).resolve()

    mesh_path = Path(mesh_uri)
    if not mesh_path.is_absolute():
        mesh_path = urdf_path.parent / mesh_path
    return mesh_path.resolve()


def _binary_stl_triangle_count(mesh_path: Path) -> int | None:
    """Return the triangle count for a binary STL, if detectable.

    Args:
        mesh_path: STL file to inspect.

    Returns:
        Triangle count for a valid binary STL, otherwise ``None``.
    """
    if mesh_path.suffix.lower() != ".stl" or mesh_path.stat().st_size < 84:
        return None
    with mesh_path.open("rb") as mesh_file:
        header = mesh_file.read(84)
    triangle_count = struct.unpack("<I", header[80:84])[0]
    expected_size = 84 + triangle_count * 50
    return triangle_count if mesh_path.stat().st_size == expected_size else None


def inspect_urdf(urdf_path: Path) -> UrdfSummary:
    """Validate URDF structure and collect conversion warnings.

    Args:
        urdf_path: URDF file to inspect.

    Returns:
        Parsed URDF summary.

    Raises:
        FileNotFoundError: If the URDF or a referenced mesh is missing.
        ValueError: If the XML or robot structure is invalid.
    """
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {urdf_path}")

    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid URDF XML: {error}") from error

    if root.tag != "robot" or not root.get("name"):
        raise ValueError("URDF root must be a <robot> element with a non-empty name.")

    links = root.findall("link")
    joints = root.findall("joint")
    link_names = {link.get("name") for link in links}
    if None in link_names or len(link_names) != len(links):
        raise ValueError("Every URDF link must have a unique, non-empty name.")

    joint_names = [joint.get("name") for joint in joints]
    if None in joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("Every URDF joint must have a unique, non-empty name.")

    warnings: list[str] = []
    revolute_joint_names: list[str] = []
    for joint in joints:
        joint_name = joint.get("name", "<unnamed>")
        joint_type = joint.get("type")
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.get("link") if parent is not None else None
        child_name = child.get("link") if child is not None else None
        if parent_name not in link_names or child_name not in link_names:
            raise ValueError(f"Joint '{joint_name}' references a missing parent or child link.")
        if joint_type in {"revolute", "prismatic"}:
            limit = joint.find("limit")
            if limit is None or any(limit.get(field) is None for field in ("lower", "upper", "effort", "velocity")):
                raise ValueError(f"Movable joint '{joint_name}' requires lower, upper, effort, and velocity limits.")
        if joint_type == "revolute":
            revolute_joint_names.append(joint_name)
            limit = joint.find("limit")
            if limit is not None:
                lower = float(limit.get("lower", "nan"))
                upper = float(limit.get("upper", "nan"))
                if lower <= -3.14 and upper >= 3.14:
                    warnings.append(f"Joint '{joint_name}' still uses an approximately ±pi placeholder limit.")

    if len(revolute_joint_names) != EXPECTED_REVOLUTE_JOINTS:
        warnings.append(
            f"Expected {EXPECTED_REVOLUTE_JOINTS} revolute joints for Yuil Dog, found {len(revolute_joint_names)}."
        )

    mesh_paths: list[Path] = []
    collision_mesh_paths: list[Path] = []
    visual_mesh_uris: set[str] = set()
    collision_mesh_uris: set[str] = set()
    for link in links:
        link_name = link.get("name", "<unnamed>")
        if link.find("inertial") is None and (link.findall("visual") or link.findall("collision")):
            warnings.append(f"Link '{link_name}' has geometry but no inertial properties.")

        for mesh in link.findall("./visual/geometry/mesh"):
            mesh_uri = mesh.get("filename")
            if mesh_uri:
                visual_mesh_uris.add(mesh_uri)
                mesh_paths.append(_resolve_mesh_path(mesh_uri, urdf_path))
        for mesh in link.findall("./collision/geometry/mesh"):
            mesh_uri = mesh.get("filename")
            if mesh_uri:
                collision_mesh_uris.add(mesh_uri)
                resolved_path = _resolve_mesh_path(mesh_uri, urdf_path)
                mesh_paths.append(resolved_path)
                collision_mesh_paths.append(resolved_path)

    missing_meshes = sorted({path for path in mesh_paths if not path.is_file()})
    if missing_meshes:
        missing_list = "\n  - ".join(str(path) for path in missing_meshes)
        raise FileNotFoundError(f"URDF references missing meshes:\n  - {missing_list}")

    shared_mesh_uris = visual_mesh_uris & collision_mesh_uris
    high_polygon_collision_meshes: list[tuple[Path, int]] = []
    for mesh_path in sorted(set(collision_mesh_paths)):
        triangle_count = _binary_stl_triangle_count(mesh_path)
        if triangle_count is not None and triangle_count > HIGH_POLYGON_THRESHOLD:
            high_polygon_collision_meshes.append((mesh_path, triangle_count))
    return UrdfSummary(
        robot_name=root.get("name", ""),
        link_count=len(links),
        joint_count=len(joints),
        revolute_joint_names=tuple(revolute_joint_names),
        mesh_paths=tuple(sorted(set(mesh_paths))),
        collision_mesh_paths=tuple(sorted(set(collision_mesh_paths))),
        shared_visual_collision_mesh_count=len(shared_mesh_uris),
        high_polygon_collision_meshes=tuple(high_polygon_collision_meshes),
        warnings=tuple(warnings),
    )


def _print_preflight(summary: UrdfSummary, collision_type: str) -> None:
    """Print a concise URDF preflight report.

    Args:
        summary: Parsed URDF summary.
        collision_type: Selected importer collision approximation.
    """
    print("[Yuil Dog] URDF preflight")
    print(f"  Robot:               {summary.robot_name}")
    print(f"  Links / joints:      {summary.link_count} / {summary.joint_count}")
    print(f"  Revolute joints:     {len(summary.revolute_joint_names)}")
    print(f"  Unique mesh files:   {len(summary.mesh_paths)}")
    print(f"  Collision strategy:  {collision_type}")
    for warning in summary.warnings:
        print(f"  WARNING: {warning}")
    if summary.shared_visual_collision_mesh_count:
        label = "NOTE" if collision_type == "Convex Decomposition" else "WARNING"
        print(
            f"  {label}: {summary.shared_visual_collision_mesh_count} detailed meshes are shared by visual and "
            "collision geometry."
        )
    if summary.high_polygon_collision_meshes:
        mesh_details = ", ".join(
            f"{path.name} ({triangle_count:,} triangles)"
            for path, triangle_count in summary.high_polygon_collision_meshes
        )
        label = "NOTE" if collision_type == "Convex Decomposition" else "WARNING"
        print(f"  {label}: High-polygon collision sources detected: {mesh_details}.")
    if collision_type == "Convex Decomposition" and summary.high_polygon_collision_meshes:
        print(
            "  NOTE: Detailed sources improve collision fidelity but increase first-load cooking time and asset size."
        )
    sys.stdout.flush()


def _strict_preflight_issues(summary: UrdfSummary, collision_type: str) -> list[str]:
    """Return URDF issues that should stop strict conversion.

    Args:
        summary: Parsed URDF summary.
        collision_type: Selected collision approximation.

    Returns:
        List of strict validation failures.
    """
    issues = list(summary.warnings)
    if collision_type != "Convex Decomposition":
        if summary.shared_visual_collision_mesh_count:
            issues.append(
                f"{summary.shared_visual_collision_mesh_count} detailed meshes are shared by visual and collision "
                "geometry without convex decomposition."
            )
        if summary.high_polygon_collision_meshes:
            issues.append(
                f"{len(summary.high_polygon_collision_meshes)} high-polygon meshes are used as collision sources "
                "without convex decomposition."
            )
    return issues


def _create_stable_entrypoint(generated_usd_path: Path, entrypoint_path: Path) -> None:
    """Create a stable USD file that references the importer-generated asset.

    Args:
        generated_usd_path: Layered USD produced by the URDF importer.
        entrypoint_path: Stable USD entrypoint to create.
    """
    from pxr import Usd, UsdGeom

    if entrypoint_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"USD entrypoint must use a .usd, .usda, or .usdc suffix: {entrypoint_path.name}")
    if entrypoint_path.resolve() == generated_usd_path.resolve():
        return

    entrypoint_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(entrypoint_path))
    root_prim = stage.DefinePrim("/YuilDog", "Xform")
    relative_asset_path = os.path.relpath(generated_usd_path, entrypoint_path.parent)
    root_prim.GetReferences().AddReference(relative_asset_path)
    stage.SetDefaultPrim(root_prim)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().Save()


def _author_collision_approximation(generated_usd_path: Path, collision_type: str) -> int:
    """Author the requested collision approximation in the layered collision asset.

    Isaac Sim's asset-transformer currently emits ``convexHull`` in the final
    ``instances.usda`` layer even when the URDF importer receives another supported
    collision type. This function applies the requested approximation to that final
    layer so the setting used by PhysX matches the conversion request.

    Args:
        generated_usd_path: Layered USD produced by the URDF importer.
        collision_type: Requested collision approximation name.

    Returns:
        Number of collision mesh attributes updated.

    Raises:
        RuntimeError: If the collision layer cannot be opened or has no approximation attributes.
    """
    from pxr import Usd

    instances_path = generated_usd_path.parent / "payloads" / "instances.usda"
    stage = Usd.Stage.Open(str(instances_path))
    if stage is None:
        raise RuntimeError(f"Could not open generated collision layer: {instances_path}")

    approximation = COLLISION_APPROXIMATION_TOKENS[collision_type]
    updated_count = 0
    for prim in stage.TraverseAll():
        approximation_attribute = prim.GetAttribute("physics:approximation")
        if approximation_attribute.IsValid() and approximation_attribute.HasAuthoredValueOpinion():
            approximation_attribute.Set(approximation)
            updated_count += 1
    if updated_count == 0:
        print("[Yuil Dog] All collisions are analytical primitives (no mesh collision approximations needed)")
    else:
        stage.GetRootLayer().Save()
        print(f"[Yuil Dog] Authored {approximation} on {updated_count} collision meshes")
    return updated_count


def _author_contact_sensor_api(generated_usd_path: Path) -> int:
    """Author PhysxContactReportAPI on all rigid bodies in the PhysX layer.

    When robot links are hierarchically nested as produced by URDF importers,
    default contact sensor activation traversals can skip child links.
    Authoring the PhysX contact report and rigid body APIs directly into the
    physx payload layer ensures all links report contacts in simulation.

    Args:
        generated_usd_path: Layered USD produced by the URDF importer.

    Returns:
        Number of rigid bodies configured for contact reporting.
    """
    from pxr import Sdf

    physics_path = generated_usd_path.parent / "payloads" / "Physics" / "physics.usda"
    physx_path = generated_usd_path.parent / "payloads" / "Physics" / "physx.usda"

    physics_layer = Sdf.Layer.FindOrOpen(str(physics_path))
    physx_layer = Sdf.Layer.FindOrOpen(str(physx_path))
    if physics_layer is None or physx_layer is None:
        raise RuntimeError(f"Could not open physics layers in {generated_usd_path.parent}")

    rb_paths: list[Sdf.Path] = []

    def walk(spec: Sdf.PrimSpec) -> None:
        if spec.HasInfo("apiSchemas"):
            tokens = spec.GetInfo("apiSchemas").prependedItems
            if "PhysicsRigidBodyAPI" in tokens:
                rb_paths.append(spec.path)
        for child in spec.nameChildren:
            walk(child)

    for root_spec in physics_layer.rootPrims:
        walk(root_spec)

    if not rb_paths:
        raise RuntimeError(f"No rigid bodies found in physics layer: {physics_path}")

    for path in rb_paths:
        spec = physx_layer.GetPrimAtPath(path)
        if spec is None:
            spec = Sdf.CreatePrimInLayer(physx_layer, path)
            spec.specifier = Sdf.SpecifierOver

        schemas_list = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
        prepended = list(schemas_list.prependedItems)
        for schema_name in ["PhysxRigidBodyAPI", "PhysxContactReportAPI"]:
            if schema_name not in prepended:
                prepended.append(schema_name)
        schemas_list.prependedItems = prepended
        spec.SetInfo("apiSchemas", schemas_list)

        cr_attr = spec.attributes.get("physxContactReport:threshold")
        if cr_attr is None:
            cr_attr = Sdf.AttributeSpec(spec, "physxContactReport:threshold", Sdf.ValueTypeNames.Float)
        cr_attr.default = 0.0

        sb_attr = spec.attributes.get("physxRigidBody:sleepThreshold")
        if sb_attr is None:
            sb_attr = Sdf.AttributeSpec(spec, "physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float)
        sb_attr.default = 0.0

    physx_layer.Save()
    print(f"[Yuil Dog] Authored PhysxContactReportAPI on {len(rb_paths)} rigid bodies in physx.usda")
    return len(rb_paths)


def _validate_usd(
    usd_path: Path,
    expected_joint_names: tuple[str, ...],
    expected_collision_type: str,
) -> None:
    """Validate that a USD entrypoint composes into the expected articulation.

    Args:
        usd_path: USD entrypoint to inspect.
        expected_joint_names: Revolute joints expected from the URDF.
        expected_collision_type: Collision approximation selected for conversion.

    Raises:
        RuntimeError: If the stage or articulation structure is invalid.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open generated USD: {usd_path}")
    if not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Generated USD has no valid default prim: {usd_path}")

    revolute_joint_names: set[str] = set()
    rigid_body_count = 0
    articulation_root_count = 0
    collision_count = 0
    collision_approximations: set[str] = set()
    contact_report_count = 0
    traversal_predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange.Stage(stage, traversal_predicate):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joint_names.add(prim.GetName())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_count += 1
            applied = prim.GetAppliedSchemas()
            api_schemas = prim.GetMetadata("apiSchemas")
            explicit_items = api_schemas.explicitItems if api_schemas else ()
            prepended_items = api_schemas.prependedItems if api_schemas else ()
            if (
                "PhysxContactReportAPI" in applied
                or "PhysxContactReportAPI" in explicit_items
                or "PhysxContactReportAPI" in prepended_items
            ):
                contact_report_count += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_root_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
        approximation_attribute = prim.GetAttribute("physics:approximation")
        if approximation_attribute.IsValid() and approximation_attribute.HasAuthoredValueOpinion():
            collision_approximations.add(str(approximation_attribute.Get()))

    missing_joints = sorted(set(expected_joint_names) - revolute_joint_names)
    extra_joints = sorted(revolute_joint_names - set(expected_joint_names))
    if missing_joints or extra_joints:
        raise RuntimeError(f"USD revolute joints differ from URDF: missing={missing_joints}, extra={extra_joints}")
    if rigid_body_count == 0:
        raise RuntimeError("Generated USD contains no rigid bodies.")
    if articulation_root_count != 1:
        raise RuntimeError(f"Expected one articulation root, found {articulation_root_count}.")
    if collision_count == 0:
        raise RuntimeError("Generated USD contains no collision geometry.")
    expected_approximation = COLLISION_APPROXIMATION_TOKENS[expected_collision_type]
    mesh_approximations = {approx for approx in collision_approximations if approx not in {"none", "None"}}
    if mesh_approximations and mesh_approximations != {expected_approximation}:
        raise RuntimeError(
            "Generated USD collision approximation differs from the requested setting: "
            f"expected={expected_approximation}, found={sorted(collision_approximations)}"
        )
    if contact_report_count < rigid_body_count:
        raise RuntimeError(
            f"Not all rigid bodies have PhysxContactReportAPI: {contact_report_count} / {rigid_body_count}"
        )

    print("[Yuil Dog] USD validation")
    print(f"  Articulation roots:  {articulation_root_count}")
    print(f"  Rigid bodies:        {rigid_body_count}")
    print(f"  Contact reporters:   {contact_report_count} / {rigid_body_count}")
    print(f"  Revolute joints:     {len(revolute_joint_names)}")
    print(f"  Collision shapes:    {collision_count}")
    print(f"  Collision mode:      {expected_approximation}")


def main() -> None:
    """Convert and validate the Yuil Dog asset."""
    args = parse_args()
    urdf_path = args.urdf_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    entrypoint_path = output_dir / args.entrypoint_name

    summary = inspect_urdf(urdf_path)
    _print_preflight(summary, args.collision_type)
    strict_issues = _strict_preflight_issues(summary, args.collision_type)
    if args.strict and strict_issues:
        issue_list = "\n  - ".join(strict_issues)
        raise RuntimeError(f"Strict URDF preflight failed:\n  - {issue_list}")

    # Prevent Isaac Sim/Kit from parsing this script's application-specific arguments.
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    exit_code = 0
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".yuil_dog_conversion_", dir=output_dir) as temporary_dir:
            converter_cfg = UrdfConverterCfg(
                asset_path=str(urdf_path),
                usd_dir=temporary_dir,
                force_usd_conversion=True,
                fix_base=args.fix_base,
                merge_fixed_joints=args.merge_fixed_joints,
                collision_from_visuals=False,
                collision_type=args.collision_type,
                self_collision=args.self_collision,
                robot_type="Quadruped",
                ros_package_paths=[{"name": ROS_PACKAGE_NAME, "path": str(DESCRIPTION_ROOT)}],
                joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none"),
                run_asset_transformer=True,
                run_multi_physics_conversion=True,
                debug_mode=args.debug_mode,
            )
            converter = UrdfConverter(converter_cfg)
            temporary_usd_path = Path(converter.usd_path).resolve()
            if not temporary_usd_path.is_file():
                raise RuntimeError(f"URDF converter did not create the expected USD: {temporary_usd_path}")
            _author_collision_approximation(temporary_usd_path, args.collision_type)
            _author_contact_sensor_api(temporary_usd_path)
            _validate_usd(temporary_usd_path, summary.revolute_joint_names, args.collision_type)

            generated_dir = output_dir / "generated"
            replacement_dir = output_dir / ".generated_next"
            if replacement_dir.exists():
                shutil.rmtree(replacement_dir)
            shutil.copytree(temporary_usd_path.parent, replacement_dir)
            if generated_dir.exists():
                shutil.rmtree(generated_dir)
            replacement_dir.rename(generated_dir)
            generated_usd_path = generated_dir / temporary_usd_path.name

        _create_stable_entrypoint(generated_usd_path, entrypoint_path)
        _validate_usd(entrypoint_path, summary.revolute_joint_names, args.collision_type)
        print("[Yuil Dog] Conversion complete")
        print(f"  Generated asset:     {generated_usd_path}")
        print(f"  Training entrypoint: {entrypoint_path}")
        print("  Joint drives:        disabled in USD; configure them in YUIL_DOG_CFG")
        sys.stdout.flush()
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
