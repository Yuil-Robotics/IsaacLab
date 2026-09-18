#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Script to train either the deterministic benchmark or Sim2Real robust task.
#
# Usage:
#   ./run_robust_train.sh                          # headless (default)
#   ./run_robust_train.sh --gui                    # GUI 표시
#   ./run_robust_train.sh --baseline               # benchmark reproduction
#   ./run_robust_train.sh --robust                 # Sim2Real randomization
#   ./run_robust_train.sh --single_goal            # single-goal Sim2Real task
#   ./run_robust_train.sh --single_goal_nominal    # single-goal curriculum stage 1
#   ./run_robust_train.sh --task Isaac-Repose-Cube-Robotis-Robust-Train-v0
#   ./run_robust_train.sh --num_envs 4096
#   NUM_ENVS=2048 ./run_robust_train.sh --gui

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_LAUNCHER="${ISAACLAB_ROOT}/isaaclab.sh"

if [[ ! -x "${ISAACLAB_LAUNCHER}" ]]; then
    echo "[ERROR] Isaac Lab launcher not found or not executable: ${ISAACLAB_LAUNCHER}" >&2
    exit 1
fi

# Install the sim2sim_newton package if not already installed
SIM2SIM_PKG="${SCRIPT_DIR}/sim2sim_newton"
if [[ -d "${SIM2SIM_PKG}" ]]; then
    "${ISAACLAB_LAUNCHER}" -p -m pip install --editable "${SIM2SIM_PKG}" --quiet
fi

##
# Defaults (override via env vars or CLI flags below)
##
TASK="${TASK:-Isaac-Repose-Cube-Robotis-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-8192}"
VISUALIZER="none"       # headless by default
FORWARDED_ARGS=()

##
# Argument parsing
##
show_help() {
    cat <<'EOF'
Usage: run_robust_train.sh [options] [Isaac Lab options]

Options:
  --gui                  Enable the Isaac Sim viewer (slower, useful for debugging)
  --headless             Disable viewer (default)
  --baseline             Use the deterministic benchmark task (default)
  --robust               Use the Sim2Real randomized task
  --single_goal          Use the single-goal Sim2Real randomized task
  --single_goal_nominal  Use single-goal curriculum stage 1 without Sim2Real randomization or latency
  --task TASK_ID         Gym task ID (default: Isaac-Repose-Cube-Robotis-Direct-v0)
  --num_envs N           Parallel environments (default: 8192)
  --resume               Resume the most recently modified model_*.pt checkpoint
  -h, --help             Show this help

Environment variables:
  TASK                   Override default task ID
  NUM_ENVS               Override default number of environments

All unrecognized arguments are forwarded to the Isaac Lab training command.

Examples:
  ./run_robust_train.sh --gui --num_envs 1024
  ./run_robust_train.sh --robust --headless --num_envs 8192
  ./run_robust_train.sh --single_goal --headless --num_envs 8192
  ./run_robust_train.sh --single_goal_nominal --headless --num_envs 8192
  ./run_robust_train.sh --headless --num_envs 8192 --resume
  ./run_robust_train.sh --task Isaac-Repose-Cube-Robotis-Robust-Train-v0
  NUM_ENVS=4096 ./run_robust_train.sh --headless
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
        --baseline)
            TASK="Isaac-Repose-Cube-Robotis-Direct-v0"
            shift
            ;;
        --robust)
            TASK="Isaac-Repose-Cube-Robotis-Robust-Train-v0"
            shift
            ;;
        --single_goal)
            TASK="Isaac-Repose-Cube-Robotis-Single-Goal-Train-v0"
            shift
            ;;
        --single_goal_nominal)
            TASK="Isaac-Repose-Cube-Robotis-Single-Goal-Nominal-Train-v0"
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
        --num_envs)
            [[ $# -ge 2 ]] || { echo "Missing value for --num_envs" >&2; exit 2; }
            NUM_ENVS="$2"
            shift 2
            ;;
        --num_envs=*)
            NUM_ENVS="${1#*=}"
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

# RSL-RL can resolve a checkpoint when load_run/checkpoint are supplied. Make
# the common `--resume`-only case deterministic by selecting the most recently
# modified checkpoint across all runs in this experiment.
RESUME_REQUESTED=false
RESUME_TARGET_SPECIFIED=false
for arg in "${FORWARDED_ARGS[@]}"; do
    case "${arg}" in
        --resume)
            RESUME_REQUESTED=true
            ;;
        --load_run|--load_run=*|--checkpoint|--checkpoint=*)
            RESUME_TARGET_SPECIFIED=true
            ;;
    esac
done

if [[ "${RESUME_REQUESTED}" == true && "${RESUME_TARGET_SPECIFIED}" == false ]]; then
    LOG_ROOT="${SCRIPT_DIR}/logs/rsl_rl/robotis_hand_state_based_small_cube"
    [[ -d "${LOG_ROOT}" ]] || {
        echo "[ERROR] Cannot resume: log directory not found: ${LOG_ROOT}" >&2
        exit 1
    }

    LATEST_CHECKPOINT="$({
        find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' -printf '%T@ %p\n'
    } | sort -nr | sed -n '1{s/^[^ ]* //;p;}')"
    [[ -n "${LATEST_CHECKPOINT}" ]] || {
        echo "[ERROR] Cannot resume: no model_*.pt checkpoint found under: ${LOG_ROOT}" >&2
        exit 1
    }

    LATEST_RUN="$(basename -- "$(dirname -- "${LATEST_CHECKPOINT}")")"
    LATEST_MODEL="$(basename -- "${LATEST_CHECKPOINT}")"
    FORWARDED_ARGS+=(--load_run "${LATEST_RUN}" --checkpoint "${LATEST_MODEL}")
    echo "[INFO] Auto-resume checkpoint: ${LATEST_RUN}/${LATEST_MODEL}"
fi

echo "=========================================================="
echo " Robotis Hand Cube — July 2026 Benchmark Reproduction"
echo " Task       : ${TASK}"
echo " Num Envs   : ${NUM_ENVS}"
echo " Visualizer : ${VISUALIZER}"
echo "=========================================================="

cd "${SCRIPT_DIR}"

exec "${ISAACLAB_LAUNCHER}" train \
    --rl_library rsl_rl \
    --task "${TASK}" \
    --seed 42 \
    --num_envs "${NUM_ENVS}" \
    --viz "${VISUALIZER}" \
    "${FORWARDED_ARGS[@]}"
