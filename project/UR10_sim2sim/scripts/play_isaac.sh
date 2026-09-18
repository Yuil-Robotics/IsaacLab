#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

NUM_ENVS="${UR10_PLAY_NUM_ENVS:-50}"
VISUALIZER="${UR10_PLAY_VISUALIZER:-kit}"
ISAAC_ARGS=()

show_project_help() {
  cat <<'EOF'
Usage: play_isaac.sh [project options] [Isaac Lab options]

Project options:
  --num_envs N     Evaluation environments (default: 50)
  --checkpoint P   Checkpoint path (relative or absolute)
  --gui            Enable the Kit visualizer (default)
  --headless       Disable the visualizer and frame markers
  -h, --help       Show this help
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
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "Missing value for --checkpoint" >&2; exit 2; }
      CHECKPOINT_PATH="$(realpath -- "$2")"
      ISAAC_ARGS+=(--checkpoint "${CHECKPOINT_PATH}")
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT_PATH="$(realpath -- "${1#*=}")"
      ISAAC_ARGS+=(--checkpoint "${CHECKPOINT_PATH}")
      shift
      ;;
    --gui)
      VISUALIZER="kit"
      shift
      ;;
    --headless)
      VISUALIZER="none"
      shift
      ;;
    -h|--help)
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
if [[ "${VISUALIZER}" == "none" ]]; then
  DEBUG_VIS="false"
else
  DEBUG_VIS="true"
fi

echo "[UR10 Sim2Sim] play num_envs=${NUM_ENVS}, visualizer=${VISUALIZER}, debug_vis=${DEBUG_VIS}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${ISAACLAB_ROOT}/isaaclab.sh" play \
  --rl_library rsl_rl \
  --task Isaac-Reach-UR10e-Sim2Sim-Play-v0 \
  --external_callback ur10_sim2sim.isaaclab_cfg.register_tasks \
  --num_envs "${NUM_ENVS}" \
  --viz "${VISUALIZER}" \
  "env.commands.ee_pose.debug_vis=${DEBUG_VIS}" \
  "${ISAAC_ARGS[@]}"
