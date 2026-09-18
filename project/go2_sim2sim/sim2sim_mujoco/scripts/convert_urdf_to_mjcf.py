#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the validated Yuil Dog URDF into a reproducible MuJoCo model."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT_DIR.parent
DESCRIPTION_DIR = PROJECT_DIR / "Yuil_dog_description" / "dog_description"
DEFAULT_INPUT = DESCRIPTION_DIR / "urdf" / "TOTAL ASSY_4차_URDF_sample.urdf"
DEFAULT_STAGED_URDF = ROOT_DIR / "model" / "urdf" / "yuil_dog.urdf"
DEFAULT_OUTPUT = ROOT_DIR / "model" / "MJCF" / "yuil_dog.xml"
VISUAL_MESH_DIR = ROOT_DIR / "model" / "meshes"
ROS_MESH_PREFIX = "package://dog_description/meshes/"
JOINT_DYNAMICS = {
    "hip_roll": {"armature": "0.0043", "damping": "0.003", "frictionloss": "0.27"},
    "hip_pitch": {"armature": "0.01", "damping": "0.005", "frictionloss": "0.53"},
    "knee_pitch": {"armature": "0.01", "damping": "0.005", "frictionloss": "0.53"},
}


def write_xml(tree: ET.ElementTree, path: Path) -> None:
    """Write deterministic UTF-8 XML with a final newline."""
    with path.open("wb") as output_file:
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        output_file.write(b"\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source Yuil Dog URDF.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination MJCF XML.")
    return parser.parse_args()


def stage_visual_mesh(source_path: Path) -> Path:
    """Stage one source STL for direct use by MuJoCo.

    Args:
        source_path: Original binary STL mesh.

    Returns:
        Staged STL path.
    """
    output_path = VISUAL_MESH_DIR / source_path.name
    if output_path.is_file() and output_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
        return output_path
    VISUAL_MESH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    print(f"visual_mesh={output_path.name} bytes={output_path.stat().st_size}")
    return output_path


def prepare_urdf(input_path: Path, staged_path: Path) -> None:
    """Rewrite ROS mesh URIs and add MuJoCo URDF compiler settings.

    Args:
        input_path: Source URDF containing ROS package mesh URIs.
        staged_path: MuJoCo-readable URDF destination.
    """
    root = ET.parse(input_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected a URDF <robot> root, got <{root.tag}>.")
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None:
            continue
        if not filename.startswith(ROS_MESH_PREFIX):
            raise ValueError(f"Unsupported mesh URI: {filename}")
        source_path = DESCRIPTION_DIR / "meshes" / filename.removeprefix(ROS_MESH_PREFIX)
        if not source_path.is_file():
            raise FileNotFoundError(f"Visual mesh was not found: {source_path}")
        mesh.set("filename", stage_visual_mesh(source_path).name)

    for extension in root.findall("mujoco"):
        root.remove(extension)
    relative_mesh_dir = Path("../meshes")
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        {
            "meshdir": relative_mesh_dir.as_posix(),
            "discardvisual": "false",
            "fusestatic": "false",
            "balanceinertia": "true",
        },
    )
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    write_xml(ET.ElementTree(root), staged_path)


def add_task_world(output_path: Path) -> None:
    """Add the flat task world and deterministic contact filtering to MJCF.

    Args:
        output_path: Canonical MJCF emitted by MuJoCo.
    """
    tree = ET.parse(output_path)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.attrib.update(
        {
            "timestep": "0.005",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "iterations": "50",
        }
    )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3"})

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Converted MJCF has no <worldbody>.")
    robot_root = worldbody.find("body[@name='base']")
    if robot_root is None:
        raise ValueError("Converted MJCF has no root body named 'base'.")
    if robot_root.find("freejoint") is None:
        robot_root.insert(0, ET.Element("freejoint", {"name": "base_free_joint"}))
    base_link = robot_root.find("body[@name='base_link']")
    if base_link is None:
        raise ValueError("Converted MJCF has no body named 'base_link'.")
    for camera in base_link.findall("camera"):
        base_link.remove(camera)
    ET.SubElement(
        base_link,
        "camera",
        {
            "name": "tracking",
            "mode": "trackcom",
            "pos": "0 -2 0.7",
            "xyaxes": "1 0 0 0 0.33 0.94",
        },
    )
    ET.SubElement(
        base_link,
        "camera",
        {
            "name": "base_chase",
            "mode": "fixed",
            "pos": "-1.8 0 0.5",
            "xyaxes": "0 -1 0 0.259 0 0.966",
        },
    )
    ET.SubElement(
        base_link,
        "camera",
        {
            "name": "base_side",
            "mode": "fixed",
            "pos": "0 -1.8 0.4",
            "xyaxes": "1 0 0 0 0.259 0.966",
        },
    )
    floor = ET.Element(
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "size": "0 0 0.05",
            "rgba": "0.2 0.2 0.2 1",
            "friction": "1 0.005 0.0001",
            "condim": "3",
            "contype": "2",
            "conaffinity": "1",
        },
    )
    worldbody.insert(0, floor)

    for geom in worldbody.findall(".//geom"):
        if geom is floor or geom.get("contype", "1") == "0":
            continue
        geom.set("contype", "1")
        geom.set("conaffinity", "0")
        geom.set("friction", "1 0.005 0.0001")
        geom.set("condim", "3")
        geom.set("group", "3")
        geom.set("rgba", "0.25 0.7 0.25 0")

    for joint in worldbody.findall(".//joint"):
        name = joint.get("name", "")
        for suffix, dynamics in JOINT_DYNAMICS.items():
            if name.endswith(suffix):
                joint.attrib.update(dynamics)
                break

    ET.indent(root)
    write_xml(tree, output_path)


def validate_model(output_path: Path) -> None:
    """Compile the generated MJCF and verify its basic topology."""
    model = mujoco.MjModel.from_xml_path(str(output_path))
    joint_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in range(model.njnt)}
    required = {
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
    }
    missing = sorted(required - joint_names)
    free_joint_count = int((model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum())
    if missing or free_joint_count != 1:
        raise ValueError(f"Generated model contract failed: missing={missing}, free_joints={free_joint_count}.")
    print(f"generated_mjcf={output_path}")
    print(f"nbody={model.nbody} njnt={model.njnt} nq={model.nq} nv={model.nv} ngeom={model.ngeom}")


def main() -> None:
    """Generate and validate the Yuil Dog MJCF."""
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Source URDF was not found: {input_path}")
    staged_path = DEFAULT_STAGED_URDF.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_urdf(input_path, staged_path)
    model = mujoco.MjModel.from_xml_path(str(staged_path))
    mujoco.mj_saveLastXML(str(output_path), model)
    add_task_world(output_path)
    validate_model(output_path)


if __name__ == "__main__":
    main()
