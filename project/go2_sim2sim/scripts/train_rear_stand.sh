#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

EXPERIMENT_NAME="go2_rear_bent_balance"
CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"
NUM_ENVS="${GO2_REAR_STAND_NUM_ENVS:-4096}"
MAX_ITERATIONS="${GO2_REAR_STAND_MAX_ITERATIONS:-1500}"
VISUALIZER="${GO2_REAR_STAND_VISUALIZER:-none}"
RESUME=false
FORWARDED_ARGS=()
RESUME_ARGS=()

show_help() {
  cat <<'EOF'
Usage: train_rear_stand.sh [options] [Isaac Lab options]

  --num_envs N          Parallel environments (default: 4096)
  --max_iterations N    Additional PPO iterations (default: 1500)
  --resume              Continue the highest rear-stand model iteration
  --print-resume-checkpoint
                        Print the checkpoint selected by --resume and exit
  --load_run RUN_DIR    Resume from a specific run directory name (under logs/rsl_rl/)
  --checkpoint FILE     Resume from a specific checkpoint filename (e.g. model_1499.pt)
  --gui                  Enable the Kit visualizer
  --headless             Disable visualization (default)
EOF
}

LOAD_RUN=""
CHECKPOINT=""
PRINT_RESUME_CHECKPOINT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_envs)
      [[ $# -ge 2 ]] || { echo "Missing value for --num_envs" >&2; exit 2; }
      NUM_ENVS="$2"
      shift 2
      ;;
    --num_envs=*) NUM_ENVS="${1#*=}"; shift ;;
    --max_iterations)
      [[ $# -ge 2 ]] || { echo "Missing value for --max_iterations" >&2; exit 2; }
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --max_iterations=*) MAX_ITERATIONS="${1#*=}"; shift ;;
    --load_run)
      [[ $# -ge 2 ]] || { echo "Missing value for --load_run" >&2; exit 2; }
      LOAD_RUN="$2"
      RESUME=true
      shift 2
      ;;
    --load_run=*) LOAD_RUN="${1#*=}"; RESUME=true; shift ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; exit 2; }
      CHECKPOINT="$2"
      RESUME=true
      shift 2
      ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; RESUME=true; shift ;;
    --resume) RESUME=true; shift ;;
    --print-resume-checkpoint) RESUME=true; PRINT_RESUME_CHECKPOINT=true; shift ;;
    --gui) VISUALIZER="kit"; shift ;;
    --headless) VISUALIZER="none"; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) FORWARDED_ARGS+=("$1"); shift ;;
  esac
done

[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num_envs must be a positive integer." >&2; exit 2; }
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max_iterations must be a positive integer." >&2
  exit 2
}

if [[ "${RESUME}" == true ]]; then
  if [[ -n "${LOAD_RUN}" && -n "${CHECKPOINT}" ]]; then
    RESUME_CHECKPOINT="${CHECKPOINT_ROOT}/${LOAD_RUN}/${CHECKPOINT}"
    [[ -f "${RESUME_CHECKPOINT}" ]] || { echo "Checkpoint not found: ${RESUME_CHECKPOINT}" >&2; exit 1; }
    RESUME_RUN="${LOAD_RUN}"
    RESUME_MODEL="${CHECKPOINT}"
  elif [[ -n "${LOAD_RUN}" ]]; then
    RESUME_CHECKPOINT="$(select_highest_iteration_checkpoint "${CHECKPOINT_ROOT}/${LOAD_RUN}")"
    RESUME_RUN="${LOAD_RUN}"
    RESUME_MODEL="$(basename -- "${RESUME_CHECKPOINT}")"
  else
    RESUME_CHECKPOINT="$(select_highest_iteration_checkpoint "${CHECKPOINT_ROOT}")"
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
  "[Go2 rear stand] train backend=physx, num_envs=${NUM_ENVS}," \
  "max_iterations=${MAX_ITERATIONS}, visualizer=${VISUALIZER}"
echo "[Go2 rear stand] logs=${CHECKPOINT_ROOT}"

ensure_go2_asset
prepare_project_environment
exec "${ISAACLAB_ROOT}/isaaclab.sh" train \
  --rl_library rsl_rl \
  --task Isaac-Rear-Stand-Go2-Sim2Sim-v0 \
  --external_callback go2_sim2sim.isaaclab_cfg.register_tasks \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --viz "${VISUALIZER}" \
  "${RESUME_ARGS[@]}" \
  physics=physx \
  "${FORWARDED_ARGS[@]}"
