#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Play the most recently modified Robotis HX5 RSL-RL checkpoint in PhysX.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_LAUNCHER="${ISAACLAB_ROOT}/isaaclab.sh"
LOG_ROOT="${SCRIPT_DIR}/logs/rsl_rl/robotis_hand_state_based_small_cube"

TASK="${TASK:-Isaac-Repose-Cube-Robotis-Robust-Train-v0}"
NUM_ENVS="${NUM_ENVS:-16}"
VISUALIZER="${VISUALIZER:-kit}"
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single_goal)
            TASK="Isaac-Repose-Cube-Robotis-Single-Goal-Play-v0"
            shift
            ;;
        --single_goal_nominal)
            TASK="Isaac-Repose-Cube-Robotis-Single-Goal-Nominal-Play-v0"
            shift
            ;;
        --single_goal_hold15)
            TASK="Isaac-Repose-Cube-Robotis-Single-Goal-Hold15-Play-v0"
            shift
            ;;
        --continuous_goal)
            TASK="Isaac-Repose-Cube-Robotis-Robust-Train-v0"
            shift
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ -x "${ISAACLAB_LAUNCHER}" ]] || {
    echo "[ERROR] Isaac Lab launcher not found or not executable: ${ISAACLAB_LAUNCHER}" >&2
    exit 1
}
[[ -d "${LOG_ROOT}" ]] || {
    echo "[ERROR] RSL-RL log directory not found: ${LOG_ROOT}" >&2
    exit 1
}
[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] NUM_ENVS must be a positive integer: ${NUM_ENVS}" >&2
    exit 2
}

LATEST_CHECKPOINT="$({
    find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' -printf '%T@ %p\n'
} | sort -nr | sed -n '1{s/^[^ ]* //;p;}')"
[[ -n "${LATEST_CHECKPOINT}" ]] || {
    echo "[ERROR] No model_*.pt checkpoint found under: ${LOG_ROOT}" >&2
    exit 1
}

echo "=========================================================="
echo " Robotis Hand Cube — latest PhysX play"
echo " Task       : ${TASK}"
echo " Checkpoint : ${LATEST_CHECKPOINT}"
echo " Num Envs   : ${NUM_ENVS}"
echo " Visualizer : ${VISUALIZER}"
echo "=========================================================="

cd "${SCRIPT_DIR}"
exec "${ISAACLAB_LAUNCHER}" play \
    --rl_library rsl_rl \
    --task "${TASK}" \
    --checkpoint "${LATEST_CHECKPOINT}" \
    --num_envs "${NUM_ENVS}" \
    --viz "${VISUALIZER}" \
    --real-time \
    "${FORWARDED_ARGS[@]}"
