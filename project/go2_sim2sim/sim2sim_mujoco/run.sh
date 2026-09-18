#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_POLICY="${MUJOCO_POLICY:-policy.pt}"
POLICY_NAME="${DEFAULT_POLICY}"
SERVER_ADDRESS="127.0.0.1:7000"
SERVER_HOST="127.0.0.1"
SERVER_PORT="7000"
SERVER_PID=""
POLICY_ARGS=()

show_help() {
  cat <<'EOF'
Usage: run.sh [options] [policy-client options]

Starts the MuJoCo GUI/server at command (0, 0, 0), waits for it to become
ready, and then runs the selected policy continuously.

  --policy NAME_OR_PATH  Policy in policies/ or a direct TorchScript path
                         (default: policy.pt; .pt may be omitted)
  --list                 List policies and exit without starting the server
  --duration SECONDS     Stop after this simulated duration (default: unlimited)
  --rate_hz HZ           Policy wall-clock rate (default: 50)
  -h, --help             Show this help

Examples:
  ./sim2sim_mujoco/run.sh
  ./sim2sim_mujoco/run.sh --policy policy_v5
  ./sim2sim_mujoco/run.sh --policy policy_v5.pt --duration 20
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      [[ $# -ge 2 ]] || { echo "Missing value for --policy" >&2; exit 2; }
      POLICY_NAME="$2"
      shift 2
      ;;
    --policy=*)
      POLICY_NAME="${1#*=}"
      shift
      ;;
    --list)
      exec "${TASK_DIR}/run_policy.sh" --list
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      POLICY_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -f "${POLICY_NAME}" ]]; then
  POLICY_NAME="$(realpath -- "${POLICY_NAME}")"
fi

cleanup() {
  if [[ -z "${SERVER_PID}" ]] || ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    return
  fi
  echo "[Sim2Sim] Stopping MuJoCo server..."
  kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
  for _ in {1..50}; do
    kill -0 "${SERVER_PID}" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
}

wait_for_server() {
  for _ in {1..600}; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "MuJoCo server exited before becoming ready." >&2
      wait "${SERVER_PID}" || true
      return 1
    fi
    if (exec 3<>"/dev/tcp/${SERVER_HOST}/${SERVER_PORT}") 2>/dev/null; then
      exec 3>&-
      exec 3<&-
      return 0
    fi
    sleep 0.1
  done
  echo "Timed out waiting for MuJoCo server at ${SERVER_ADDRESS}." >&2
  return 1
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[Sim2Sim] Starting server: command=(0.0, 0.0, 0.0)"
setsid "${TASK_DIR}/run_server.sh" \
  --address "${SERVER_ADDRESS}" \
  --command 0.0 0.0 0.0 &
SERVER_PID=$!

wait_for_server

echo "[Sim2Sim] Starting policy: ${POLICY_NAME}"
"${TASK_DIR}/run_policy.sh" \
  --server "${SERVER_ADDRESS}" \
  --policy "${POLICY_NAME}" \
  "${POLICY_ARGS[@]}"
