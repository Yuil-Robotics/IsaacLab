#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

NUM_ENVS="${YUIL_NUM_ENVS:-4096}"
MAX_ITERATIONS="${YUIL_MAX_ITERATIONS:-1500}"
VISUALIZER="${YUIL_VISUALIZER:-none}"
CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/reach_yuil_3kg_gravity"
RESUME=false
PRINT_RESUME_CHECKPOINT=false
ISAAC_ARGS=()
RESUME_ARGS=()

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
Usage: train_isaac.sh [project options] [Isaac Lab options]

Project options:
  --num_envs N          Parallel environments (default: 4096)
  --max_iterations N    PPO iterations (default: 1500)
  --resume              Resume the highest gravity-trained model iteration
  --print-resume-checkpoint
                        Print the checkpoint selected by --resume and exit
  --headless             Disable visualization (default)
  --gui                  Enable the Kit visualizer
  --project-help         Show this help

Environment variables:
  YUIL_NUM_ENVS
  YUIL_MAX_ITERATIONS
  YUIL_VISUALIZER        Usually "none" or "kit"

All unrecognized arguments are forwarded to Isaac Lab.
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
    --max_iterations)
      [[ $# -ge 2 ]] || { echo "Missing value for --max_iterations" >&2; exit 2; }
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --max_iterations=*)
      MAX_ITERATIONS="${1#*=}"
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --print-resume-checkpoint)
      RESUME=true
      PRINT_RESUME_CHECKPOINT=true
      shift
      ;;
    --headless)
      VISUALIZER="none"
      shift
      ;;
    --gui)
      VISUALIZER="kit"
      shift
      ;;
    -h|--help|--project-help)
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
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max_iterations must be a positive integer." >&2
  exit 2
}

if [[ "${RESUME}" == true ]]; then
  RESUME_CHECKPOINT="$(select_highest_iteration_checkpoint)"
  RESUME_RUN="$(basename -- "$(dirname -- "${RESUME_CHECKPOINT}")")"
  RESUME_MODEL="$(basename -- "${RESUME_CHECKPOINT}")"
  RESUME_ARGS=(--resume --load_run "^${RESUME_RUN}$" --checkpoint "^${RESUME_MODEL}$")
  if [[ "${PRINT_RESUME_CHECKPOINT}" == true ]]; then
    echo "${RESUME_CHECKPOINT}"
    exit 0
  fi
fi

if [[ "${VISUALIZER}" == "none" ]]; then
  DEBUG_VIS="false"
else
  DEBUG_VIS="true"
fi

echo \
  "[Yuil 3kg] num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}," \
  "visualizer=${VISUALIZER}, debug_vis=${DEBUG_VIS}"
echo "[Yuil 3kg] rollout=64, physics=120 Hz, policy=60 Hz"
echo "[Yuil 3kg] logs=${PROJECT_ROOT}/logs/rsl_rl/reach_yuil_3kg_gravity"
if [[ "${RESUME}" == true ]]; then
  echo "[Yuil 3kg] resume=${RESUME_CHECKPOINT}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${ISAACLAB_ROOT}/isaaclab.sh" train \
  --rl_library rsl_rl \
  --task Isaac-Reach-Yuil-3kg-v0 \
  --external_callback yuil_3kg_robot_arm.isaaclab_cfg.register_tasks \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --viz "${VISUALIZER}" \
  "${RESUME_ARGS[@]}" \
  "env.commands.ee_pose.debug_vis=${DEBUG_VIS}" \
  "${ISAAC_ARGS[@]}"
