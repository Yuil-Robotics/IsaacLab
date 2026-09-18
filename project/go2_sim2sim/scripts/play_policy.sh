#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NUM_ENVS="${GO2_PLAY_NUM_ENVS:-16}"
VISUALIZER="${GO2_PLAY_VISUALIZER:-kit}"
PHYSICS="physx"
CHECKPOINT_PATH=""
PRINT_CHECKPOINT=false
FORWARDED_ARGS=()

show_help() {
  cat <<'EOF'
Usage: play_policy.sh [options] [Isaac Lab options]

  --physics NAME       physx or newton_mjwarp (default: physx)
  --num_envs N         Evaluation environments (default: 16)
  --checkpoint PATH    Explicit checkpoint (default: highest iteration)
  --print-checkpoint   Print the resolved checkpoint and exit
  --gui                Enable the Kit visualizer (default)
  --headless           Disable visualization
  --lidar-vis          Enable LiDAR point cloud visualization
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --physics)
      [[ $# -ge 2 ]] || { echo "Missing value for --physics" >&2; exit 2; }
      PHYSICS="$2"
      shift 2
      ;;
    --physics=*)
      PHYSICS="${1#*=}"
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
    --print-checkpoint)
      PRINT_CHECKPOINT=true
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
    --lidar-vis)
      FORWARDED_ARGS+=("env.scene.lidar.debug_vis=true")
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

[[ "${PHYSICS}" == "physx" || "${PHYSICS}" == "newton_mjwarp" ]] || {
  echo "--physics must be physx or newton_mjwarp." >&2
  exit 2
}
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num_envs must be a positive integer." >&2; exit 2; }
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(select_highest_iteration_checkpoint)"
fi
[[ -f "${CHECKPOINT_PATH}" ]] || { echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2; exit 2; }

if [[ "${PRINT_CHECKPOINT}" == true ]]; then
  echo "${CHECKPOINT_PATH}"
  exit 0
fi

echo \
  "[Go2] play backend=${PHYSICS}, num_envs=${NUM_ENVS}," \
  "visualizer=${VISUALIZER}, checkpoint=${CHECKPOINT_PATH}"

ensure_go2_asset
prepare_project_environment
exec "${ISAACLAB_ROOT}/isaaclab.sh" play \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0 \
  --external_callback go2_sim2sim.isaaclab_cfg.register_tasks \
  --num_envs "${NUM_ENVS}" \
  --viz "${VISUALIZER}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  "physics=${PHYSICS}" \
  "env.commands.base_velocity.debug_vis=true" \
  "${FORWARDED_ARGS[@]}"
