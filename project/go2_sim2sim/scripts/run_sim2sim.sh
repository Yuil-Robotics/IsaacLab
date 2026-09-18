#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export ROBOT_TYPE="${ROBOT_TYPE:-yuil_dog}"
SIM2SIM_CHECKPOINT_ROOT="${GO2_SIM2SIM_CHECKPOINT_ROOT:-${PROJECT_ROOT}/logs/rsl_rl/yuil_dog_flat_robotlab}"
SIM2SIM_TASK="${GO2_SIM2SIM_TASK:-Isaac-Velocity-Flat-Yuil-Dog-RobotLab-Eval-v0}"
SIM2SIM_LABEL="${GO2_SIM2SIM_LABEL:-Yuil-Dog-Flat-RobotLab Sim2Sim}"
RESULT_SUFFIX="${GO2_SIM2SIM_RESULT_SUFFIX:-_yuil_dog_flat_robotlab}"
CHECKPOINT_PATH="${GO2_SIM2SIM_CHECKPOINT:-}"
EVALUATION_RUN_NAME="${GO2_SIM2SIM_EVALUATION_RUN_NAME:-}"
NUM_ENVS="${GO2_EVAL_NUM_ENVS:-256}"
DURATION="${GO2_EVAL_DURATION:-10.0}"
SEED="${GO2_EVAL_SEED:-42}"
DEVICE="${GO2_SIM2SIM_DEVICE:-cuda:0}"

show_help() {
  cat <<'EOF'
Usage: run_sim2sim.sh [options]

  --checkpoint PATH       Checkpoint file or run/experiment directory
  --checkpoint_root PATH  Experiment directory used for automatic selection
  --task ID               Evaluation task ID (default: flat Yuil Dog RobotLab)
  --result_suffix TEXT    Suffix appended to the checkpoint result directory
  --evaluation_run NAME   Evaluation result subdirectory (default: timestamp)
  --num_envs N            Parallel evaluation environments (default: 256)
  --duration S             Evaluation duration in seconds (default: 10)
  --seed N                 Fixed evaluation seed (default: 42)
  --device NAME            Simulation device (default: cuda:0)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; exit 2; }
      CHECKPOINT_PATH="$(realpath -- "$2")"
      shift 2
      ;;
    --checkpoint_root)
      [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint_root" >&2; exit 2; }
      SIM2SIM_CHECKPOINT_ROOT="$(realpath -- "$2")"
      shift 2
      ;;
    --task)
      [[ $# -ge 2 ]] || { echo "Missing value for --task" >&2; exit 2; }
      SIM2SIM_TASK="$2"
      shift 2
      ;;
    --result_suffix)
      [[ $# -ge 2 ]] || { echo "Missing value for --result_suffix" >&2; exit 2; }
      RESULT_SUFFIX="$2"
      shift 2
      ;;
    --evaluation_run)
      [[ $# -ge 2 ]] || { echo "Missing value for --evaluation_run" >&2; exit 2; }
      EVALUATION_RUN_NAME="$2"
      shift 2
      ;;
    --num_envs)
      [[ $# -ge 2 ]] || { echo "Missing value for --num_envs" >&2; exit 2; }
      NUM_ENVS="$2"
      shift 2
      ;;
    --duration)
      [[ $# -ge 2 ]] || { echo "Missing value for --duration" >&2; exit 2; }
      DURATION="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || { echo "Missing value for --device" >&2; exit 2; }
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num_envs must be a positive integer." >&2; exit 2; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer." >&2; exit 2; }
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(select_highest_iteration_checkpoint "${SIM2SIM_CHECKPOINT_ROOT}")"
elif [[ -d "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(select_highest_iteration_checkpoint "${CHECKPOINT_PATH}")"
fi
[[ -f "${CHECKPOINT_PATH}" ]] || { echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2; exit 2; }

RUN_NAME="$(basename -- "$(dirname -- "${CHECKPOINT_PATH}")")"
MODEL_NAME="$(basename -- "${CHECKPOINT_PATH}" .pt)"
RESULT_ROOT="${PROJECT_ROOT}/logs/sim2sim/${RUN_NAME}_${MODEL_NAME}${RESULT_SUFFIX}"
EVALUATION_RUN_NAME="${EVALUATION_RUN_NAME:-$(date +%Y-%m-%d_%H-%M-%S)}"
RESULT_DIR="${RESULT_ROOT}/${EVALUATION_RUN_NAME}"
TASK_ARGS=()
if [[ -n "${SIM2SIM_TASK}" ]]; then
  TASK_ARGS=(--task "${SIM2SIM_TASK}")
fi

mkdir -p "${RESULT_DIR}"
export WARP_CACHE_PATH="${GO2_WARP_CACHE_PATH:-${RESULT_ROOT}/warp_cache}"
export MPLCONFIGDIR="${GO2_MPLCONFIGDIR:-${RESULT_ROOT}/matplotlib}"

echo "[${SIM2SIM_LABEL}] task=${SIM2SIM_TASK}"
echo "[${SIM2SIM_LABEL}] checkpoint=${CHECKPOINT_PATH}"
echo "[${SIM2SIM_LABEL}] num_envs=${NUM_ENVS}, duration=${DURATION}s, seed=${SEED}, device=${DEVICE}"
echo "[${SIM2SIM_LABEL}] results=${RESULT_DIR}"

"${SCRIPT_DIR}/evaluate_policy.sh" \
  --physics physx \
  "${TASK_ARGS[@]}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${RESULT_DIR}/physx.json" \
  --num_envs "${NUM_ENVS}" \
  --duration "${DURATION}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

"${SCRIPT_DIR}/evaluate_policy.sh" \
  --physics newton_mjwarp \
  "${TASK_ARGS[@]}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${RESULT_DIR}/newton_mjwarp.json" \
  --num_envs "${NUM_ENVS}" \
  --duration "${DURATION}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

"${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/compare_policy_metrics.py" \
  "${RESULT_DIR}/physx.json" \
  "${RESULT_DIR}/newton_mjwarp.json" \
  --output-json "${RESULT_DIR}/comparison.json" \
  --output-markdown "${RESULT_DIR}/report.md"
