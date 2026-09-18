#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

NUM_ENVS="${YUIL_PLAY_NUM_ENVS:-16}"
VISUALIZER="${YUIL_PLAY_VISUALIZER:-kit}"
CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/reach_yuil_3kg_gravity"
CHECKPOINT_PATH=""
PRINT_CHECKPOINT=false
ISAAC_ARGS=()

select_highest_iteration_checkpoint() {
  local candidate
  local candidate_mtime
  local checkpoint_name
  local iteration
  local best_path=""
  local best_iteration=-1
  local best_mtime=-1

  [[ -d "${CHECKPOINT_ROOT}" ]] || {
    echo "Checkpoint directory does not exist: ${CHECKPOINT_ROOT}" >&2
    return 1
  }

  while IFS= read -r -d '' candidate; do
    checkpoint_name="${candidate##*/}"
    [[ "${checkpoint_name}" =~ ^model_([0-9]+)\.pt$ ]] || continue
    iteration=$((10#${BASH_REMATCH[1]}))
    candidate_mtime="$(stat -c '%Y' -- "${candidate}")"
    if (( iteration > best_iteration || (iteration == best_iteration && candidate_mtime > best_mtime) )); then
      best_path="${candidate}"
      best_iteration="${iteration}"
      best_mtime="${candidate_mtime}"
    fi
  done < <(find "${CHECKPOINT_ROOT}" -type f -name 'model_*.pt' -print0)

  [[ -n "${best_path}" ]] || {
    echo "No model_<iteration>.pt checkpoint found under: ${CHECKPOINT_ROOT}" >&2
    return 1
  }
  realpath -- "${best_path}"
}

show_project_help() {
  cat <<'EOF'
Usage: play_isaac.sh [project options] [Isaac Lab options]

Project options:
  --num_envs N     Evaluation environments (default: 16)
  --checkpoint P   Explicit checkpoint path (default: highest model iteration)
  --print-checkpoint
                    Print the resolved checkpoint and exit
  --gui            Enable the Kit visualizer (default)
  --headless       Disable visualization and frame markers
  -h, --help       Show this help

The automatic selector only considers model_<iteration>.pt files under the
project's gravity-on experiment directory. PhysX and Newton/MJWarp therefore
load the same gravity-trained policy when no explicit checkpoint is given.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    -h|--help)
      show_project_help
      exit 0
      ;;
    *)
      ISAAC_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num_envs must be a positive integer." >&2; exit 2; }
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(select_highest_iteration_checkpoint)"
fi
[[ -f "${CHECKPOINT_PATH}" ]] || { echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2; exit 2; }

if [[ "${PRINT_CHECKPOINT}" == true ]]; then
  echo "${CHECKPOINT_PATH}"
  exit 0
fi

if [[ "${VISUALIZER}" == "none" ]]; then
  DEBUG_VIS="false"
else
  DEBUG_VIS="true"
fi

echo \
  "[Yuil 3kg] play num_envs=${NUM_ENVS}, visualizer=${VISUALIZER}," \
  "debug_vis=${DEBUG_VIS}, checkpoint=${CHECKPOINT_PATH}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${ISAACLAB_ROOT}/isaaclab.sh" play \
  --rl_library rsl_rl \
  --task Isaac-Reach-Yuil-3kg-Play-v0 \
  --external_callback yuil_3kg_robot_arm.isaaclab_cfg.register_tasks \
  --num_envs "${NUM_ENVS}" \
  --viz "${VISUALIZER}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  "env.commands.ee_pose.debug_vis=${DEBUG_VIS}" \
  "${ISAAC_ARGS[@]}"
