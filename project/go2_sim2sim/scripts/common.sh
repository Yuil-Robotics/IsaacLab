#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

EXPERIMENT_NAME="go2_rough_spot_rewards_sim2sim_dr"
ROUGH_EXPERIMENT_NAME="go2_rough_spot_rewards_sim2sim_dr"
HISTORY_EXPERIMENT_NAME="go2_rough_history_5step_sim2real"

CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"
ROUGH_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/${ROUGH_EXPERIMENT_NAME}"
HISTORY_CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/${HISTORY_EXPERIMENT_NAME}"

LOCAL_GO2_USD_PATH="${PROJECT_ROOT}/assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
LOCAL_GROUND_USD_PATH="${PROJECT_ROOT}/assets/Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd"
CUSTOM_ROBOT_DEFAULT_USD="${PROJECT_ROOT}/assets/custom_robot/robot.usd"
YUIL_DOG_DEFAULT_USD="${PROJECT_ROOT}/assets/yuil_dog/yuil_dog.usda"

select_highest_iteration_checkpoint() {
  local target_root="${1:-${CHECKPOINT_ROOT}}"
  local latest_run_dir=""
  local candidate
  local candidate_mtime
  local checkpoint_name
  local iteration
  local best_path=""
  local best_iteration=-1
  local best_mtime=-1

  [[ -d "${target_root}" ]] || {
    echo "Checkpoint directory does not exist: ${target_root}" >&2
    return 1
  }

  # If target_root itself directly contains model_*.pt (e.g. specific run directory passed), use it directly.
  if compgen -G "${target_root}/model_*.pt" > /dev/null; then
    latest_run_dir="${target_root}"
  else
    # Otherwise find the most recently created run directory containing model_*.pt checkpoints.
    latest_run_dir="$(find "${target_root}" -mindepth 1 -maxdepth 2 -type f -name 'model_*.pt' -exec dirname {} + | sort -u -r | head -n 1)"
  fi

  [[ -n "${latest_run_dir}" && -d "${latest_run_dir}" ]] || {
    echo "No model_<iteration>.pt checkpoint found under: ${target_root}" >&2
    return 1
  }

  while IFS= read -r -d '' candidate; do
    checkpoint_name="${candidate##*/}"
    [[ "${checkpoint_name}" =~ ^model_([0-9]+)\.pt$ ]] || continue
    iteration=$((10#${BASH_REMATCH[1]}))
    candidate_mtime="$(stat -c '%Y' -- "${candidate}")"
    if (( iteration > best_iteration || (iteration == best_iteration && candidate_mtime > best_mtime) )); then
      best_path="${candidate}"
      best_iteration="${iteration}"
      best_mtime="${candidate_mtime}"
    fi
  done < <(find "${latest_run_dir}" -maxdepth 1 -type f -name 'model_*.pt' -print0)

  [[ -n "${best_path}" ]] || {
    echo "No model_<iteration>.pt checkpoint found under: ${latest_run_dir}" >&2
    return 1
  }
  realpath -- "${best_path}"
}

prepare_project_environment() {
  cd "${PROJECT_ROOT}"
  mkdir -p "${PROJECT_ROOT}/logs/cache/warp" "${PROJECT_ROOT}/logs/cache/matplotlib"
  export WARP_CACHE_PATH="${GO2_WARP_CACHE_PATH:-${WARP_CACHE_PATH:-${PROJECT_ROOT}/logs/cache/warp}}"
  export MPLCONFIGDIR="${GO2_MPLCONFIGDIR:-${MPLCONFIGDIR:-${PROJECT_ROOT}/logs/cache/matplotlib}}"
  export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

ensure_go2_asset() {
  local robot_type="${ROBOT_TYPE:-go2}"
  if [[ "${robot_type,,}" == "yuil_dog" || "${robot_type,,}" == "yuil" ]]; then
    local yuil_dog_usd="${YUIL_DOG_USD_PATH:-${YUIL_DOG_DEFAULT_USD}}"
    [[ -f "${yuil_dog_usd}" ]] || {
      echo "Yuil Dog USD is missing at: ${yuil_dog_usd}" >&2
      echo "Run: ${SCRIPT_DIR}/convert_yuil_dog_urdf.sh --strict" >&2
      return 1
    }
    return 0
  fi
  if [[ "${robot_type,,}" == "custom" ]]; then
    local custom_usd="${CUSTOM_ROBOT_USD_PATH:-${ROBOT_USD_PATH:-${CUSTOM_ROBOT_DEFAULT_USD}}}"
    [[ -f "${custom_usd}" ]] || {
      echo "Custom robot USD is missing at: ${custom_usd}" >&2
      echo "Convert your URDF to USD using: ${SCRIPT_DIR}/convert_urdf_to_usd.sh --urdf <PATH_TO_URDF>" >&2
      return 1
    }
    return 0
  fi

  [[ -f "${LOCAL_GO2_USD_PATH}" && -f "${LOCAL_GROUND_USD_PATH}" ]] || {
    echo "Local Go2 or ground USD is missing." >&2
    echo "Run: ${SCRIPT_DIR}/setup_assets.sh" >&2
    return 1
  }
}
