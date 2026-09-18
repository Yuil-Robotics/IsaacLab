#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NUM_ENVS="1"
VISUALIZER="${GO2_PLAY_VISUALIZER:-kit}"
CHECKPOINT_PATH=""
MAP_YAML="${PROJECT_ROOT}/logs/maps/go2_map.yaml"
FORWARDED_ARGS=()

show_help() {
  cat <<'EOF'
Usage: autonomous_nav.sh [options]

Autonomous Navigation Controls:
  [1] ~ [8]       : Select Preset Goal Destinations (Rooms A-D / Corridors)
  [+] / [-]       : Increase / Decrease simulation playback speed (+/- 0.5x)
  [0]             : Max Uncapped Simulation Speed (fastest GPU framerate)
  [SPACE] / [K]   : Emergency Stop / Cancel Navigation

Options:
  --checkpoint <path> Path to trained policy checkpoint
  --map <path>        Path to 2D costmap YAML (default: logs/maps/go2_map.yaml)
  --num_envs <int>    Number of environments (default: 1)
  --speed <float>     Simulation speed multiplier (default: 1.5, 0 for max)
  --fast              Fast 2.5x simulation speed
  --max               Uncapped maximum simulation speed
  --headless          Run in headless mode
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --map)
      MAP_YAML="$2"
      shift 2
      ;;
    --num_envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --speed)
      FORWARDED_ARGS+=("--speed" "$2")
      shift 2
      ;;
    --fast)
      FORWARDED_ARGS+=("--speed" "2.5")
      shift
      ;;
    --max)
      FORWARDED_ARGS+=("--speed" "0.0")
      shift
      ;;
    --headless)
      VISUALIZER="none"
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

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(select_highest_iteration_checkpoint)"
fi
[[ -f "${CHECKPOINT_PATH}" ]] || { echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2; exit 2; }

echo \
  "[Go2 Auto-Nav] num_envs=${NUM_ENVS}, visualizer=${VISUALIZER}, map=${MAP_YAML}," \
  "checkpoint=${CHECKPOINT_PATH}"

ensure_go2_asset
prepare_project_environment

trap 'stty sane 2>/dev/null || true' EXIT INT TERM

"${ISAACLAB_ROOT}/isaaclab.sh" -p "${SCRIPT_DIR}/autonomous_nav.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --map_yaml "${MAP_YAML}" \
  --num_envs "${NUM_ENVS}" \
  --visualizer "${VISUALIZER}" \
  "${FORWARDED_ARGS[@]}"

stty sane 2>/dev/null || true
