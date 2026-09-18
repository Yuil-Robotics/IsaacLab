#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Play the TorchScript policy exported from the latest RSL-RL checkpoint in
# Newton MJWarp (MuJoCo model and solver pipeline).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${SCRIPT_DIR}/logs/rsl_rl/robotis_hand_state_based_small_cube"
NEWTON_LAUNCHER="${SCRIPT_DIR}/sim2sim_newton/run_sim2sim.sh"

[[ -x "${NEWTON_LAUNCHER}" ]] || {
    echo "[ERROR] Newton play launcher not found or not executable: ${NEWTON_LAUNCHER}" >&2
    exit 1
}
[[ -d "${LOG_ROOT}" ]] || {
    echo "[ERROR] RSL-RL log directory not found: ${LOG_ROOT}" >&2
    exit 1
}

LATEST_CHECKPOINT="$({
    find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' -printf '%T@ %p\n'
} | sort -nr | sed -n '1{s/^[^ ]* //;p;}')"
[[ -n "${LATEST_CHECKPOINT}" ]] || {
    echo "[ERROR] No model_*.pt checkpoint found under: ${LOG_ROOT}" >&2
    exit 1
}

EXPORTED_POLICY="$(dirname -- "${LATEST_CHECKPOINT}")/exported/policy.pt"
if [[ ! -f "${EXPORTED_POLICY}" ]]; then
    echo "[ERROR] The latest checkpoint has not been exported to TorchScript:" >&2
    echo "        ${LATEST_CHECKPOINT}" >&2
    echo "        Run ./run_latest_isaac_play.sh once, then stop it with Ctrl+C." >&2
    exit 1
fi
if [[ "${LATEST_CHECKPOINT}" -nt "${EXPORTED_POLICY}" ]]; then
    echo "[ERROR] The exported policy is older than the latest checkpoint:" >&2
    echo "        checkpoint: ${LATEST_CHECKPOINT}" >&2
    echo "        policy    : ${EXPORTED_POLICY}" >&2
    echo "        Run ./run_latest_isaac_play.sh once to refresh the export." >&2
    exit 1
fi

echo "=========================================================="
echo " Robotis Hand Cube — latest Newton-MuJoCo play"
echo " Checkpoint : ${LATEST_CHECKPOINT}"
echo " Policy     : ${EXPORTED_POLICY}"
echo "=========================================================="

cd "${SCRIPT_DIR}"
exec "${NEWTON_LAUNCHER}" \
    --policy "${EXPORTED_POLICY}" \
    --steps 0 \
    --real_time \
    "$@"
