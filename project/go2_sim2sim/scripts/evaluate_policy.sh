#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PHYSICS="physx"
NUM_ENVS="${GO2_EVAL_NUM_ENVS:-256}"
DURATION="${GO2_EVAL_DURATION:-10.0}"
SEED="${GO2_EVAL_SEED:-42}"
DEVICE="${GO2_SIM2SIM_DEVICE:-cuda:0}"
TASK=""
CHECKPOINT_PATH=""
OUTPUT_PATH=""

show_help() {
  cat <<'EOF'
Usage: evaluate_policy.sh [options]

  --physics NAME       physx or newton_mjwarp (default: physx)
  --task ID            Evaluation task ID (default: evaluator default)
  --checkpoint PATH    Checkpoint file or log folder (default: highest PhysX checkpoint)
  --output PATH        Output JSON path
  --num_envs N         Parallel evaluation environments (default: 256)
  --duration S         Evaluation duration in seconds (default: 10)
  --seed N             Fixed evaluation seed (default: 42)
  --device NAME        Simulation device (default: cuda:0)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --physics)
      PHYSICS="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$(realpath -- "$2")"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$(realpath -m -- "$2")"
      shift 2
      ;;
    --num_envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --device)
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

[[ "${PHYSICS}" == "physx" || "${PHYSICS}" == "newton_mjwarp" ]] || {
  echo "--physics must be physx or newton_mjwarp." >&2
  exit 2
}
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num_envs must be a positive integer." >&2; exit 2; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer." >&2; exit 2; }
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(select_highest_iteration_checkpoint)"
elif [[ -d "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(select_highest_iteration_checkpoint "${CHECKPOINT_PATH}")"
fi
if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${PROJECT_ROOT}/logs/sim2sim/manual/${PHYSICS}.json"
fi

echo \
  "[Go2 evaluation] backend=${PHYSICS}, num_envs=${NUM_ENVS}, duration=${DURATION}s," \
  "seed=${SEED}, checkpoint=${CHECKPOINT_PATH}"

ensure_go2_asset
prepare_project_environment
TASK_ARGS=()
if [[ -n "${TASK}" ]]; then
  TASK_ARGS=(--task "${TASK}")
fi
exec "${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/evaluate_policy.py" \
  "${TASK_ARGS[@]}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --num_envs "${NUM_ENVS}" \
  --duration "${DURATION}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --viz none \
  "physics=${PHYSICS}"
