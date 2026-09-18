#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Train the deterministic benchmark and continue the resulting checkpoint in
# the Sim2Real robust task.  Use --from-run to skip the baseline stage and
# continue an existing run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_robust_train.sh"
LOG_ROOT="${SCRIPT_DIR}/logs/rsl_rl/robotis_hand_state_based_small_cube"

NUM_ENVS="${NUM_ENVS:-8192}"
BASE_ITERATIONS="${BASE_ITERATIONS:-50000}"
ROBUST_ITERATIONS="${ROBUST_ITERATIONS:-10000}"
VISUALIZER_ARG="--headless"
FROM_RUN=""
FORWARDED_ARGS=()

show_help() {
    cat <<'EOF'
Usage: run_two_stage_train.sh [options] [shared Isaac Lab options]

Options:
  --gui                    Enable the viewer for both stages
  --headless               Disable the viewer (default)
  --num_envs N             Parallel environments (default: 8192)
  --from-run RUN           Skip baseline and resume the latest checkpoint in RUN
  -h, --help               Show this help

Environment variables:
  BASE_ITERATIONS          Baseline iterations (default: 50000)
  ROBUST_ITERATIONS        Additional robust iterations (default: 10000)
  NUM_ENVS                 Parallel environments (default: 8192)

Examples:
  BASE_ITERATIONS=15000 ROBUST_ITERATIONS=10000 ./run_two_stage_train.sh --headless
  ROBUST_ITERATIONS=10000 ./run_two_stage_train.sh --from-run 2026-08-10_08-23-39 --headless
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gui)
            VISUALIZER_ARG="--gui"
            shift
            ;;
        --headless)
            VISUALIZER_ARG="--headless"
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
        --from-run)
            [[ $# -ge 2 ]] || { echo "Missing value for --from-run" >&2; exit 2; }
            FROM_RUN="$2"
            shift 2
            ;;
        --from-run=*)
            FROM_RUN="${1#*=}"
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

[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ENVS must be a positive integer." >&2; exit 2; }
[[ "${BASE_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || { echo "BASE_ITERATIONS must be a positive integer." >&2; exit 2; }
[[ "${ROBUST_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || { echo "ROBUST_ITERATIONS must be a positive integer." >&2; exit 2; }

cd "${SCRIPT_DIR}"

if [[ -z "${FROM_RUN}" ]]; then
    "${TRAIN_SCRIPT}" "${VISUALIZER_ARG}" --num_envs "${NUM_ENVS}" --baseline \
        --max_iterations "${BASE_ITERATIONS}" --run_name baseline "${FORWARDED_ARGS[@]}"

    FROM_RUN="$(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
fi

[[ "${FROM_RUN}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run directory name: ${FROM_RUN}" >&2; exit 2; }
RUN_DIR="${LOG_ROOT}/${FROM_RUN}"
[[ -d "${RUN_DIR}" ]] || { echo "Run directory not found: ${RUN_DIR}" >&2; exit 1; }

CHECKPOINT="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)"
[[ -n "${CHECKPOINT}" ]] || { echo "No model_*.pt checkpoint found in: ${RUN_DIR}" >&2; exit 1; }

echo "Continuing Sim2Real robust training from ${FROM_RUN}/${CHECKPOINT}"
"${TRAIN_SCRIPT}" "${VISUALIZER_ARG}" --num_envs "${NUM_ENVS}" --robust \
    --resume --load_run "${FROM_RUN}" --checkpoint "${CHECKPOINT}" \
    --max_iterations "${ROBUST_ITERATIONS}" --run_name sim2real_robust "${FORWARDED_ARGS[@]}"
