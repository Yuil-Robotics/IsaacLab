#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

NUM_ENVS="${UR10_NUM_ENVS:-4096}"
MAX_ITERATIONS="${UR10_MAX_ITERATIONS:-1500}"
VISUALIZER="${UR10_VISUALIZER:-none}"
ISAAC_ARGS=()

show_project_help() {
  cat <<'EOF'
Usage: train_isaac.sh [project options] [Isaac Lab options]

Project options:
  --num_envs N          Parallel training environments (default: 4096)
  --max_iterations N    PPO iterations (default: 1500)
  --headless             Disable the visualizer (default)
  --gui                  Enable the Kit visualizer
  --project-help         Show this help

Environment variables:
  UR10_NUM_ENVS
  UR10_MAX_ITERATIONS
  UR10_VISUALIZER        Usually "none" or "kit"

All unrecognized arguments are forwarded to the Isaac Lab training command.
EOF
}

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
    --headless)
      VISUALIZER="none"
      shift
      ;;
    --gui)
      VISUALIZER="kit"
      shift
      ;;
    -h|--help|--project-help)
      show_project_help
      exit 0
      ;;
    *)
      ISAAC_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num_envs must be a positive integer." >&2; exit 2; }
[[ "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max_iterations must be a positive integer." >&2
  exit 2
}

if [[ "${VISUALIZER}" == "none" ]]; then
  DEBUG_VIS="false"
else
  DEBUG_VIS="true"
fi

echo \
  "[UR10 Sim2Sim] num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}," \
  "visualizer=${VISUALIZER}, debug_vis=${DEBUG_VIS}"
echo "[UR10 Sim2Sim] rollout=64"
echo "[UR10 Sim2Sim] logs=${PROJECT_ROOT}/logs/rsl_rl/reach_ur10_sim2sim"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${ISAACLAB_ROOT}/isaaclab.sh" train \
  --rl_library rsl_rl \
  --task Isaac-Reach-UR10e-Sim2Sim-v0 \
  --external_callback ur10_sim2sim.isaaclab_cfg.register_tasks \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --viz "${VISUALIZER}" \
  "env.commands.ee_pose.debug_vis=${DEBUG_VIS}" \
  "${ISAAC_ARGS[@]}"
