#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

show_help() {
  cat <<'EOF'
Usage: convert_urdf_to_usd.sh --urdf <PATH_TO_URDF> [options]

Required:
  --urdf PATH            Path to the robot URDF file

Options:
  --output_dir DIR       Output directory (default: assets/custom_robot)
  --usd_filename NAME    Target USD filename (default: robot.usd)
  --robot_type TYPE      Quadruped, Default, Humanoid, Wheeled (default: Quadruped)
  --stiffness N          Joint stiffness [Nm/rad] (default: 25.0)
  --damping N            Joint damping [Nm/(rad/s)] (default: 0.5)
  --collision_type TYPE  Convex Hull, Convex Decomposition (default: Convex Hull)
  --self_collision       Enable self collisions
  --fix_base             Fix base link to world (default: false)
  -h, --help             Show this help message and exit
EOF
}

if [[ $# -eq 0 ]]; then
  show_help
  exit 0
fi

prepare_project_environment
exec "${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/convert_urdf_to_usd.py" "$@"
