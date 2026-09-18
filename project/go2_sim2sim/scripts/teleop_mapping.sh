#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NUM_ENVS="1"
PHYSICS="physx"
VISUALIZER="${GO2_PLAY_VISUALIZER:-kit}"
CHECKPOINT_PATH=""
TELEOP_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/yuil_dog_rough_leg_major"
VX="0.5"
VX_FAST="1.0"
VX_HOLD_TIME="1.0"
VY="0.5"
WZ="1.0"
VOXEL_SIZE="0.06"
LIDAR_VIS=false
CAMERA_VIS=false
FORWARDED_ARGS=()

show_help() {
  cat <<'EOF'
Usage: teleop_mapping.sh [options]

Yuil Dog RL Teleoperation / Optional Mapping Controls:
  W / Up Arrow      : Forward   (+0.5, hold 1 s: +1.0 m/s)
  S / Down Arrow    : Backward  (-0.5, hold 1 s: -1.0 m/s)
  A / Left Arrow    : Left      (+vy)
  D / Right Arrow   : Right     (-vy)
  Q                 : Turn Left (+wz)
  E                 : Turn Right(-wz)
  SPACE / K         : Stop (0.0 m/s)
  P                 : Save current 3D map to logs/maps/*.ply
  C                 : Clear 3D map buffer

Options:
  --gui                Enable Kit GUI visualizer (default)
  --headless           Disable visualizer (headless mode)
  --voxel-size SIZE    3D mapping voxel resolution in meters (default: 0.06)
  --vx SPEED           Linear velocity magnitude for X in m/s (default: 0.5)
  --vx-fast SPEED      X speed after holding W/S in m/s (default: 1.0)
  --vx-hold-time SEC   W/S hold duration before accelerating (default: 1.0)
  --vy SPEED           Linear velocity magnitude for Y in m/s (default: 0.5)
  --lidar-vis          Enable LiDAR mapping and raw ray visualization
  --no-lidar-vis       Disable LiDAR mapping (default)
  --camera-vis         Enable D435i depth-camera point-cloud mapping
  --no-camera-vis      Disable D435i point cloud (default)
  --checkpoint PATH    Explicit checkpoint (default: highest iteration)
  --num_envs N         Number of evaluation environments (default: 1)
  --real-time          Enforce real-time execution step rate
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gui)
      VISUALIZER="kit"
      shift
      ;;
    --headless)
      VISUALIZER="none"
      shift
      ;;
    --voxel-size|--voxel_size)
      [[ $# -ge 2 ]] || { echo "Missing value for --voxel-size" >&2; exit 2; }
      VOXEL_SIZE="$2"
      shift 2
      ;;
    --vx)
      [[ $# -ge 2 ]] || { echo "Missing value for --vx" >&2; exit 2; }
      VX="$2"
      shift 2
      ;;
    --vx=*)
      VX="${1#*=}"
      shift
      ;;
    --vx-fast|--vx_fast)
      [[ $# -ge 2 ]] || { echo "Missing value for --vx-fast" >&2; exit 2; }
      VX_FAST="$2"
      shift 2
      ;;
    --vx-fast=*|--vx_fast=*)
      VX_FAST="${1#*=}"
      shift
      ;;
    --vx-hold-time|--vx_hold_time)
      [[ $# -ge 2 ]] || { echo "Missing value for --vx-hold-time" >&2; exit 2; }
      VX_HOLD_TIME="$2"
      shift 2
      ;;
    --vx-hold-time=*|--vx_hold_time=*)
      VX_HOLD_TIME="${1#*=}"
      shift
      ;;
    --vy)
      [[ $# -ge 2 ]] || { echo "Missing value for --vy" >&2; exit 2; }
      VY="$2"
      shift 2
      ;;
    --vy=*)
      VY="${1#*=}"
      shift
      ;;
    --wz)
      [[ $# -ge 2 ]] || { echo "Missing value for --wz" >&2; exit 2; }
      WZ="$2"
      shift 2
      ;;
    --wz=*)
      WZ="${1#*=}"
      shift
      ;;
    --num_envs)
      [[ $# -ge 2 ]] || { echo "Missing value for --num_envs" >&2; exit 2; }
      NUM_ENVS="$2"
      shift 2
      ;;
    --num_envs=*)
      NUM_ENVS="${1#*=}"
      shift
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; exit 2; }
      CHECKPOINT_PATH="$(realpath -- "$2")"
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT_PATH="$(realpath -- "${1#*=}")"
      shift
      ;;
    --no-lidar-vis)
      LIDAR_VIS=false
      shift
      ;;
    --lidar-vis)
      LIDAR_VIS=true
      shift
      ;;
    --camera-vis|--cam-vis)
      CAMERA_VIS=true
      shift
      ;;
    --no-camera-vis|--no-cam-vis)
      CAMERA_VIS=false
      shift
      ;;
    --real-time)
      FORWARDED_ARGS+=("--real-time")
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num_envs must be a positive integer." >&2; exit 2; }
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(select_highest_iteration_checkpoint "${TELEOP_CHECKPOINT_ROOT}")"
fi
[[ -f "${CHECKPOINT_PATH}" ]] || { echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2; exit 2; }

echo \
  "[Yuil Dog Mapping] num_envs=${NUM_ENVS}, visualizer=${VISUALIZER}, voxel_size=${VOXEL_SIZE}m," \
  "vx=${VX}m/s, vy=${VY}m/s, wz=${WZ}rad/s, lidar_vis=${LIDAR_VIS}, camera_vis=${CAMERA_VIS}," \
  "checkpoint=${CHECKPOINT_PATH}"

export ROBOT_TYPE="yuil_dog"
ensure_go2_asset
prepare_project_environment

LIDAR_ARG=()
if [[ "${LIDAR_VIS}" == true ]]; then
  LIDAR_ARG+=("--lidar_vis")
fi
CAMERA_ARG=()
if [[ "${CAMERA_VIS}" == true ]]; then
  CAMERA_ARG+=("--camera_vis")
fi

trap 'stty sane 2>/dev/null || true' EXIT INT TERM

"${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/teleop_mapping.py" \
  --task "Isaac-Velocity-Rough-Yuil-Dog-Play-v0" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num_envs "${NUM_ENVS}" \
  --visualizer "${VISUALIZER}" \
  --voxel_size "${VOXEL_SIZE}" \
  --vx "${VX}" \
  --vx-fast "${VX_FAST}" \
  --vx-hold-time "${VX_HOLD_TIME}" \
  --vy "${VY}" \
  --wz "${WZ}" \
  "${LIDAR_ARG[@]}" \
  "${CAMERA_ARG[@]}" \
  "${FORWARDED_ARGS[@]}"

stty sane 2>/dev/null || true
