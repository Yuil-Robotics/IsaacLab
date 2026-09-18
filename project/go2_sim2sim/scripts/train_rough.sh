#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NUM_ENVS="${GO2_ROUGH_NUM_ENVS:-4096}"
MAX_ITERATIONS="${GO2_ROUGH_MAX_ITERATIONS:-3000}"
VISUALIZER="${GO2_ROUGH_VISUALIZER:-none}"
RESUME=false
PRINT_RESUME_CHECKPOINT=false
FORWARDED_ARGS=()
RESUME_ARGS=()
TRAIN_TASK_ID="${GO2_ROUGH_TRAIN_TASK_ID:-Isaac-Velocity-Rough-Go2-Sim2Sim-v0}"
TRAIN_CHECKPOINT_ROOT="${GO2_ROUGH_CHECKPOINT_ROOT:-${ROUGH_CHECKPOINT_ROOT}}"
TRAIN_LABEL="${GO2_ROUGH_TRAIN_LABEL:-Go2-Rough}"

show_help() {
  cat <<'EOF'
Usage: train_rough.sh [options] [Isaac Lab options]

  --num_envs N          Parallel environments (default: 4096)
  --max_iterations N    Additional PPO iterations (default: 3000)
  --resume              Continue the highest model iteration
  --print-resume-checkpoint
                        Print the checkpoint selected by --resume and exit
  --load_run RUN_DIR    Resume from a specific run directory name (under logs/rsl_rl/)
  --checkpoint FILE     Resume from a specific checkpoint filename (e.g. model_1499.pt)
  --gui                 Enable the Kit visualizer
  --headless            Disable visualization (default)
EOF
}

LOAD_RUN=""
CHECKPOINT=""

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
    --load_run)
      [[ $# -ge 2 ]] || { echo "Missing value for --load_run" >&2; exit 2; }
      LOAD_RUN="$2"
      RESUME=true
      shift 2
      ;;
    --load_run=*)
      LOAD_RUN="${1#*=}"
      RESUME=true
      shift
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; exit 2; }
      CHECKPOINT="$2"
      RESUME=true
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT="${1#*=}"
      RESUME=true
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
    --gui)
      VISUALIZER="kit"
      shift
      ;;
    --headless)
      VISUALIZER="none"
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
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max_iterations must be a positive integer." >&2
  exit 2
}
DEBUG_VIS=false
if [[ "${VISUALIZER}" != "none" ]]; then
  DEBUG_VIS=true
fi

if [[ "${RESUME}" == true ]]; then
  if [[ -n "${LOAD_RUN}" && -n "${CHECKPOINT}" ]]; then
    RESUME_CHECKPOINT="${TRAIN_CHECKPOINT_ROOT}/${LOAD_RUN}/${CHECKPOINT}"
    [[ -f "${RESUME_CHECKPOINT}" ]] || { echo "Checkpoint not found: ${RESUME_CHECKPOINT}" >&2; exit 1; }
    RESUME_RUN="${LOAD_RUN}"
    RESUME_MODEL="${CHECKPOINT}"
  elif [[ -n "${LOAD_RUN}" ]]; then
    RESUME_CHECKPOINT="$(select_highest_iteration_checkpoint "${TRAIN_CHECKPOINT_ROOT}/${LOAD_RUN}")"
    RESUME_RUN="${LOAD_RUN}"
    RESUME_MODEL="$(basename -- "${RESUME_CHECKPOINT}")"
  else
    RESUME_CHECKPOINT="$(select_highest_iteration_checkpoint "${TRAIN_CHECKPOINT_ROOT}")"
    RESUME_RUN="$(basename -- "$(dirname -- "${RESUME_CHECKPOINT}")")"
    RESUME_MODEL="$(basename -- "${RESUME_CHECKPOINT}")"
  fi
  if [[ "${PRINT_RESUME_CHECKPOINT}" == true ]]; then
    echo "${RESUME_CHECKPOINT}"
    exit 0
  fi
  RESUME_ARGS=(--resume --load_run "^${RESUME_RUN}$" --checkpoint "^${RESUME_MODEL}$")
fi

echo \
  "[${TRAIN_LABEL}] train backend=physx, num_envs=${NUM_ENVS}," \
  "max_iterations=${MAX_ITERATIONS}, visualizer=${VISUALIZER}, debug_vis=${DEBUG_VIS}"
if [[ "${RESUME}" == true ]]; then
  echo "[${TRAIN_LABEL}] resume=${RESUME_CHECKPOINT}"
fi
echo "[${TRAIN_LABEL}] logs=${TRAIN_CHECKPOINT_ROOT}"

ensure_go2_asset
prepare_project_environment
exec "${ISAACLAB_ROOT}/isaaclab.sh" train \
  --rl_library rsl_rl \
  --task "${TRAIN_TASK_ID}" \
  --external_callback go2_sim2sim.isaaclab_cfg.register_tasks \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --viz "${VISUALIZER}" \
  "${RESUME_ARGS[@]}" \
  physics=physx \
  "env.commands.base_velocity.debug_vis=${DEBUG_VIS}" \
  "${FORWARDED_ARGS[@]}"
