#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RESULT_DIR="${PROJECT_ROOT}/logs/sim2sim"
DEVICE="${YUIL_SIM2SIM_DEVICE:-cuda:0}"
DURATION="${YUIL_SIM2SIM_DURATION:-2.0}"

mkdir -p "${RESULT_DIR}"
export WARP_CACHE_PATH="${YUIL_WARP_CACHE_PATH:-${RESULT_DIR}/warp_cache}"
export MPLCONFIGDIR="${YUIL_MPLCONFIGDIR:-${RESULT_DIR}/matplotlib}"

run_probe() {
  local preset="$1"
  local output="$2"
  "${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/sim2sim_probe.py" \
    --device "${DEVICE}" \
    --viz none \
    --duration "${DURATION}" \
    --output "${output}" \
    "physics=${preset}"
}

echo "[Yuil sim2sim] USD model, device=${DEVICE}, duration=${DURATION}s"
run_probe physx "${RESULT_DIR}/physx.json"
run_probe newton_mjwarp "${RESULT_DIR}/newton_mjwarp.json"
"${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/compare_sim2sim.py" \
  "${RESULT_DIR}/physx.json" \
  "${RESULT_DIR}/newton_mjwarp.json"
