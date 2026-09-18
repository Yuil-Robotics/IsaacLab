#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TASK="Isaac-Velocity-Rough-Yuil-Dog-Play-v0"
TELEOP_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/yuil_dog_rough_leg_major"
ROBOT_TYPE="yuil_dog"
NUM_ENVS="1"
PHYSICS="physx"
VISUALIZER="${GO2_PLAY_VISUALIZER:-kit}"
CHECKPOINT_PATH=""
VX="0.5"
VX_FAST="1.0"
VX_HOLD_TIME="1.0"
VY="0.5"
WZ="1.0"
LIDAR_VIS=false
CAMERA_VIS=false
FORWARDED_ARGS=()

show_help() {
  cat <<'EOF'
Usage: teleop_policy.sh [options]

Keyboard Controls:
  W / Up Arrow      : Forward   (+0.5, hold 1 s: +1.0 m/s)
  S / Down Arrow    : Backward  (-0.5, hold 1 s: -1.0 m/s)
  A / Left Arrow    : Left      (+vy)
  D / Right Arrow   : Right     (-vy)
  Q                 : Turn Left (+wz)
  E                 : Turn Right(-wz)
  SPACE / K         : Stop (0.0 m/s)
  R / C             : Record 10s CSV (input 45, output 12, joint rad)
  P                 : Push Robot (Manual external velocity impulse)

Options:
  --low-profile        Teleoperate with the Yuil Dog low-profile flat policy
  --robotlab           Teleoperate with the Yuil Dog flat RobotLab policy
  --rough              Teleoperate with the Yuil Dog rough policy (default)
  --flat               Teleoperate with the legacy Go2 flat policy
  --gui                Enable Kit GUI visualizer (default)
  --headless           Disable visualizer (headless mode)
  --push               Enable automatic periodic push disturbances (default: on)
  --no-push            Disable automatic periodic push disturbances
  --push-vel SPEED     Push velocity magnitude in m/s (default: 0.5)
  --push-interval-min S Min interval between automatic pushes (default: 10.0)
  --push-interval-max S Max interval between automatic pushes (default: 15.0)
  --vx SPEED           Linear velocity magnitude for X in m/s (default: 0.5)
  --vx-fast SPEED      X speed after holding W/S in m/s (default: 1.0)
  --vx-hold-time SEC   W/S hold duration before accelerating (default: 1.0)
  --vy SPEED           Linear velocity magnitude for Y in m/s (default: 0.5)
  --wz SPEED           Yaw angular velocity magnitude in rad/s (default: 1.0)
  --lidar-vis          Enable LiDAR point cloud visualization
  --camera-vis         Enable Intel RealSense D435i depth point cloud
  --checkpoint PATH    Explicit checkpoint (default: highest iteration)
  --num_envs N         Number of evaluation environments (default: 1)
  --real-time          Enforce real-time execution step rate
  --record-dir PATH    Directory to save teleop CSV recordings
  --record-duration S  Recording duration in seconds of sim time (default: 10.0)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --low-profile|--low_profile)
      TASK="Isaac-Velocity-Flat-Yuil-Dog-Low-Profile-Play-v0"
      TELEOP_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/yuil_dog_flat_low_profile"
      ROBOT_TYPE="yuil_dog"
      shift
      ;;
    --robotlab)
      TASK="Isaac-Velocity-Flat-Yuil-Dog-RobotLab-Play-v0"
      TELEOP_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/yuil_dog_flat_robotlab"
      ROBOT_TYPE="yuil_dog"
      shift
      ;;
    --rough)
      TASK="Isaac-Velocity-Rough-Yuil-Dog-Play-v0"
      TELEOP_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/yuil_dog_rough_leg_major"
      ROBOT_TYPE="yuil_dog"
      shift
      ;;
    --flat)
      TASK="Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0"
      TELEOP_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
      ROBOT_TYPE="go2"
      shift
      ;;
    --task)
      [[ $# -ge 2 ]] || { echo "Missing value for --task" >&2; exit 2; }
      TASK="$2"
      shift 2
      ;;
    --task=*)
      TASK="${1#*=}"
      shift
      ;;
    --gui)
      VISUALIZER="kit"
      shift
      ;;
    --headless)
      VISUALIZER="none"
      shift
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
    --lidar-vis)
      LIDAR_VIS=true
      shift
      ;;
    --camera-vis|--cam-vis)
      CAMERA_VIS=true
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
  "[Yuil Dog Teleop] task=${TASK}, num_envs=${NUM_ENVS}, visualizer=${VISUALIZER}, vx=${VX}m/s, vy=${VY}m/s, wz=${WZ}rad/s," \
  "lidar_vis=${LIDAR_VIS}, camera_vis=${CAMERA_VIS}, checkpoint=${CHECKPOINT_PATH}"

export ROBOT_TYPE
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

"${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/teleop_policy.py" \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num_envs "${NUM_ENVS}" \
  --visualizer "${VISUALIZER}" \
  --vx "${VX}" \
  --vx-fast "${VX_FAST}" \
  --vx-hold-time "${VX_HOLD_TIME}" \
  --vy "${VY}" \
  --wz "${WZ}" \
  "${LIDAR_ARG[@]}" \
  "${CAMERA_ARG[@]}" \
  "${FORWARDED_ARGS[@]}"

stty sane 2>/dev/null || true
