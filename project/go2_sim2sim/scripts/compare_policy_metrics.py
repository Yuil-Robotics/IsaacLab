#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare deterministic PhysX and Newton policy-evaluation results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CONTRACT_ARRAY_FIELDS = (
    "body_mass_kg",
    "body_com_m",
    "body_inertia_kg_m2",
    "joint_armature_kg_m2",
    "joint_stiffness_nm_rad",
    "joint_damping_nm_s_rad",
    "joint_friction",
)
METRIC_FIELDS = (
    "success_rate",
    "fall_rate",
    "falls_per_env",
    "lin_vel_xy_rmse_m_s",
    "yaw_rate_rmse_rad_s",
    "mean_reward_per_step",
    "mean_base_height_m",
    "base_height_std_m",
    "base_tilt_rms",
    "action_rms",
    "action_saturation_rate",
    "action_any_saturation_rate",
    "action_delta_rms_per_policy_step",
    "joint_velocity_rms_rad_s",
    "joint_torque_rms_nm",
    "mean_abs_mechanical_power_w",
    "mean_contact_foot_slip_m_s",
    "foot_contact_ratio",
)
THRESHOLD_METRIC_FIELDS = (
    "mean_contact_foot_slip_m_s",
    "foot_contact_ratio",
)
SUSTAINED_CONTACT_METRIC_FIELDS = (
    "mean_foot_horizontal_displacement_m_per_policy_step",
    "mean_foot_horizontal_speed_m_s",
    "eligible_foot_sample_ratio",
)


def _flatten(value) -> list[float]:
    """Flatten nested numeric lists."""
    if isinstance(value, list):
        return [item for nested in value for item in _flatten(nested)]
    return [float(value)]


def _max_abs_difference(first, second) -> float:
    """Return the largest element-wise absolute difference."""
    first_flat = _flatten(first)
    second_flat = _flatten(second)
    if len(first_flat) != len(second_flat):
        raise ValueError(f"Array lengths differ: {len(first_flat)} != {len(second_flat)}")
    return max((abs(a - b) for a, b in zip(first_flat, second_flat)), default=0.0)


def _metric_delta(physx_value: float, newton_value: float) -> dict[str, float]:
    """Return absolute and symmetric relative differences."""
    absolute = abs(physx_value - newton_value)
    denominator = abs(physx_value) + abs(newton_value)
    symmetric_percent = 0.0 if denominator == 0.0 else 200.0 * absolute / denominator
    return {
        "physx": physx_value,
        "newton_mjwarp": newton_value,
        "absolute_difference": absolute,
        "symmetric_relative_difference_percent": symmetric_percent,
    }


def _validate_contract(physx: dict, newton: dict) -> None:
    """Check that both files describe the same evaluation protocol."""
    exact_fields = (
        "checkpoint",
        "task",
        "seed",
        "num_envs",
        "duration_s",
        "num_policy_steps",
        "commands",
        "contact_force_thresholds_n",
        "sustained_contact_frames",
        "action_clip_limit",
    )
    for field in exact_fields:
        if physx[field] != newton[field]:
            raise ValueError(f"Evaluation field differs for {field!r}: {physx[field]!r} != {newton[field]!r}")
    for field in ("joint_names", "body_names"):
        if physx["model_contract"][field] != newton["model_contract"][field]:
            raise ValueError(f"Model contract differs for {field!r}.")
    for field in ("physics_dt_s", "policy_dt_s"):
        if not math.isclose(physx[field], newton[field], rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Simulation timing differs for {field!r}.")


def _build_comparison(physx: dict, newton: dict) -> dict:
    """Build a JSON-compatible comparison report."""
    _validate_contract(physx, newton)
    contract_differences = {
        field: _max_abs_difference(
            physx["model_contract"][field],
            newton["model_contract"][field],
        )
        for field in CONTRACT_ARRAY_FIELDS
    }
    for field in ("ground_static_friction", "ground_dynamic_friction", "ground_restitution"):
        contract_differences[field] = abs(physx["model_contract"][field] - newton["model_contract"][field])

    overall = {
        field: _metric_delta(
            physx["metrics"]["overall"][field],
            newton["metrics"]["overall"][field],
        )
        for field in METRIC_FIELDS
    }
    by_command = {}
    for command_name in physx["metrics"]["by_command"]:
        if command_name not in newton["metrics"]["by_command"]:
            raise ValueError(f"Newton result is missing command scenario: {command_name}")
        by_command[command_name] = {
            field: _metric_delta(
                physx["metrics"]["by_command"][command_name][field],
                newton["metrics"]["by_command"][command_name][field],
            )
            for field in METRIC_FIELDS
        }

    threshold_sweep = {
        "overall": _compare_threshold_sweep(
            physx["metrics"]["overall"]["contact_force_threshold_sweep"],
            newton["metrics"]["overall"]["contact_force_threshold_sweep"],
        ),
        "by_command": {
            command_name: _compare_threshold_sweep(
                physx["metrics"]["by_command"][command_name]["contact_force_threshold_sweep"],
                newton["metrics"]["by_command"][command_name]["contact_force_threshold_sweep"],
            )
            for command_name in physx["metrics"]["by_command"]
        },
    }
    sustained_contact_sweep = {
        "overall": _compare_sweep(
            physx["metrics"]["overall"]["sustained_contact_sweep"],
            newton["metrics"]["overall"]["sustained_contact_sweep"],
            SUSTAINED_CONTACT_METRIC_FIELDS,
            "Sustained-contact sweep",
        ),
        "by_command": {
            command_name: _compare_sweep(
                physx["metrics"]["by_command"][command_name]["sustained_contact_sweep"],
                newton["metrics"]["by_command"][command_name]["sustained_contact_sweep"],
                SUSTAINED_CONTACT_METRIC_FIELDS,
                "Sustained-contact sweep",
            )
            for command_name in physx["metrics"]["by_command"]
        },
    }
    action_saturation_by_joint = {
        "overall": _compare_joint_saturation(
            physx["metrics"]["overall"]["action_saturation_rate_by_joint"],
            newton["metrics"]["overall"]["action_saturation_rate_by_joint"],
        ),
        "by_command": {
            command_name: _compare_joint_saturation(
                physx["metrics"]["by_command"][command_name]["action_saturation_rate_by_joint"],
                newton["metrics"]["by_command"][command_name]["action_saturation_rate_by_joint"],
            )
            for command_name in physx["metrics"]["by_command"]
        },
    }
    directional_saturation_overall = _compare_optional_directional_joint_saturation(
        physx["metrics"]["overall"].get("directional_action_saturation_rate_by_joint"),
        newton["metrics"]["overall"].get("directional_action_saturation_rate_by_joint"),
    )
    directional_saturation_by_command = {
        command_name: _compare_optional_directional_joint_saturation(
            physx["metrics"]["by_command"][command_name].get("directional_action_saturation_rate_by_joint"),
            newton["metrics"]["by_command"][command_name].get("directional_action_saturation_rate_by_joint"),
        )
        for command_name in physx["metrics"]["by_command"]
    }
    if directional_saturation_overall is None:
        if any(value is not None for value in directional_saturation_by_command.values()):
            raise ValueError("Directional action saturation is missing from the overall metrics.")
        directional_action_saturation_by_joint = None
    else:
        if any(value is None for value in directional_saturation_by_command.values()):
            raise ValueError("Directional action saturation is missing from a command scenario.")
        directional_action_saturation_by_joint = {
            "overall": directional_saturation_overall,
            "by_command": directional_saturation_by_command,
        }

    metrics = {
        "overall": overall,
        "by_command": by_command,
        "contact_force_threshold_sweep": threshold_sweep,
        "sustained_contact_sweep": sustained_contact_sweep,
        "action_saturation_by_joint": action_saturation_by_joint,
    }
    if directional_action_saturation_by_joint is not None:
        metrics["directional_action_saturation_by_joint"] = directional_action_saturation_by_joint

    return {
        "schema_version": 5,
        "checkpoint": physx["checkpoint"],
        "seed": physx["seed"],
        "num_envs": physx["num_envs"],
        "duration_s": physx["duration_s"],
        "physics_dt_s": physx["physics_dt_s"],
        "policy_dt_s": physx["policy_dt_s"],
        "action_clip_limit": physx["action_clip_limit"],
        "contact_force_thresholds_n": physx["contact_force_thresholds_n"],
        "sustained_contact_frames": physx["sustained_contact_frames"],
        "model_contract_max_abs_differences": contract_differences,
        "metrics": metrics,
    }


def _compare_threshold_sweep(physx: dict, newton: dict) -> dict:
    """Compare slip and contact ratio at each contact-force threshold."""
    return _compare_sweep(physx, newton, THRESHOLD_METRIC_FIELDS, "Contact-force threshold sweep")


def _compare_joint_saturation(physx: dict, newton: dict) -> dict:
    """Compare action saturation rates for matching policy joints."""
    if physx.keys() != newton.keys():
        raise ValueError("Action-saturation joint names differ between backends.")
    return {joint_name: _metric_delta(physx[joint_name], newton[joint_name]) for joint_name in physx}


def _compare_optional_directional_joint_saturation(physx: dict | None, newton: dict | None) -> dict | None:
    """Compare positive and negative saturation rates when both inputs provide them."""
    if physx is None and newton is None:
        return None
    if physx is None or newton is None:
        raise ValueError("Directional action saturation is missing from one backend.")
    if physx.keys() != newton.keys():
        raise ValueError("Directional action-saturation joint names differ between backends.")
    result = {}
    for joint_name in physx:
        if physx[joint_name].keys() != newton[joint_name].keys():
            raise ValueError(f"Directional action-saturation keys differ for joint {joint_name!r}.")
        result[joint_name] = {
            direction: _metric_delta(physx[joint_name][direction], newton[joint_name][direction])
            for direction in ("positive", "negative")
        }
    return result


def _compare_sweep(physx: dict, newton: dict, metric_fields: tuple[str, ...], label: str) -> dict:
    """Compare matching metric groups in two parameter sweeps."""
    if physx.keys() != newton.keys():
        raise ValueError(f"{label} keys differ between backends.")
    return {
        key: {field: _metric_delta(physx[key][field], newton[key][field]) for field in metric_fields} for key in physx
    }


def _write_markdown(comparison: dict, path: Path) -> None:
    """Write a concise Korean-language comparison table."""
    lines = [
        "# PhysX–Newton/MJWarp 정책 비교 보고서",
        "",
        f"- 체크포인트: `{comparison['checkpoint']}`",
        f"- 환경 수: {comparison['num_envs']}",
        f"- 평가 시간: {comparison['duration_s']} s",
        f"- 물리 시뮬레이션 간격(dt): {comparison['physics_dt_s']} s",
        f"- 정책 실행 간격(dt): {comparison['policy_dt_s']} s",
        "",
        "## 모델 설정 일치도",
        "",
        "| 설정 항목 | 최대 절대 차이 |",
        "|---|---:|",
    ]
    for field, difference in comparison["model_contract_max_abs_differences"].items():
        lines.append(f"| {field} | {difference:.9g} |")
    lines.extend(
        [
            "",
            "## 전체 정책 지표",
            "",
            "| 지표 | PhysX | Newton/MJWarp | 절대 차이 | 대칭 상대 차이 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for field, values in comparison["metrics"]["overall"].items():
        lines.append(
            f"| {field} | {values['physx']:.6g} | {values['newton_mjwarp']:.6g} | "
            f"{values['absolute_difference']:.6g} | "
            f"{values['symmetric_relative_difference_percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## 관절별 Action 포화율",
            "",
            f"정책 출력이 `abs(action) >= {comparison['action_clip_limit']:.6g}`이면 포화로 계산합니다. "
            "시뮬레이터 wrapper가 clamp하기 전의 원본 정책 출력을 사용합니다.",
            "",
            "### 전체",
            "",
            "| 관절 | PhysX 포화율 | Newton/MJWarp 포화율 | 절대 차이 |",
            "|---|---:|---:|---:|",
        ]
    )
    saturation_overall = comparison["metrics"]["action_saturation_by_joint"]["overall"]
    for joint_name, values in saturation_overall.items():
        lines.append(
            f"| {joint_name} | {values['physx']:.6g} | {values['newton_mjwarp']:.6g} | "
            f"{values['absolute_difference']:.6g} |"
        )
    lines.extend(
        [
            "",
            "### 명령별",
            "",
            "| 명령 | 관절 | PhysX 포화율 | Newton/MJWarp 포화율 | 절대 차이 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    saturation_commands = comparison["metrics"]["action_saturation_by_joint"]["by_command"]
    for command_name, joint_metrics in saturation_commands.items():
        for joint_name, values in joint_metrics.items():
            lines.append(
                f"| {command_name} | {joint_name} | {values['physx']:.6g} | "
                f"{values['newton_mjwarp']:.6g} | {values['absolute_difference']:.6g} |"
            )
    directional_saturation = comparison["metrics"].get("directional_action_saturation_by_joint")
    if directional_saturation is not None:
        lines.extend(
            [
                "",
                "### 전체 방향별",
                "",
                f"`+`는 원본 action이 `+{comparison['action_clip_limit']:.6g}` 이상인 경우이고, "
                f"`-`는 `-{comparison['action_clip_limit']:.6g}` 이하인 경우입니다. "
                "양수 action은 관절 목표각을 증가시키고 음수 action은 감소시킵니다.",
                "",
                "| 관절 | 방향 | PhysX 포화율 | Newton/MJWarp 포화율 | 절대 차이 |",
                "|---|:---:|---:|---:|---:|",
            ]
        )
        for joint_name, direction_metrics in directional_saturation["overall"].items():
            for direction, symbol in (("positive", "+"), ("negative", "-")):
                values = direction_metrics[direction]
                lines.append(
                    f"| {joint_name} | {symbol} | {values['physx']:.6g} | "
                    f"{values['newton_mjwarp']:.6g} | {values['absolute_difference']:.6g} |"
                )
        lines.extend(
            [
                "",
                "### 명령별 방향",
                "",
                "| 명령 | 관절 | 방향 | PhysX 포화율 | Newton/MJWarp 포화율 | 절대 차이 |",
                "|---|---|:---:|---:|---:|---:|",
            ]
        )
        for command_name, joint_metrics in directional_saturation["by_command"].items():
            for joint_name, direction_metrics in joint_metrics.items():
                for direction, symbol in (("positive", "+"), ("negative", "-")):
                    values = direction_metrics[direction]
                    lines.append(
                        f"| {command_name} | {joint_name} | {symbol} | {values['physx']:.6g} | "
                        f"{values['newton_mjwarp']:.6g} | {values['absolute_difference']:.6g} |"
                    )
    lines.extend(
        [
            "",
            "## 접촉력 임계값별 결과",
            "",
            "기존 `mean_contact_foot_slip_m_s` 지표는 1 N 행의 값을 사용합니다.",
            "",
            "### 전체",
            "",
            "| 임계값 [N] | PhysX 미끄러짐 [m/s] | Newton/MJWarp 미끄러짐 [m/s] | 미끄러짐 차이 [m/s] | "
            "PhysX 접촉 비율 | Newton/MJWarp 접촉 비율 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    overall_sweep = comparison["metrics"]["contact_force_threshold_sweep"]["overall"]
    for threshold, threshold_metrics in overall_sweep.items():
        slip = threshold_metrics["mean_contact_foot_slip_m_s"]
        contact_ratio = threshold_metrics["foot_contact_ratio"]
        lines.append(
            f"| {threshold} | {slip['physx']:.6g} | {slip['newton_mjwarp']:.6g} | "
            f"{slip['absolute_difference']:.6g} | {contact_ratio['physx']:.6g} | "
            f"{contact_ratio['newton_mjwarp']:.6g} |"
        )
    lines.extend(
        [
            "",
            "### 명령별",
            "",
            "| 명령 | 임계값 [N] | PhysX 미끄러짐 [m/s] | Newton/MJWarp 미끄러짐 [m/s] | 미끄러짐 차이 [m/s] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    command_sweeps = comparison["metrics"]["contact_force_threshold_sweep"]["by_command"]
    for command_name, command_sweep in command_sweeps.items():
        for threshold, threshold_metrics in command_sweep.items():
            slip = threshold_metrics["mean_contact_foot_slip_m_s"]
            lines.append(
                f"| {command_name} | {threshold} | {slip['physx']:.6g} | {slip['newton_mjwarp']:.6g} | "
                f"{slip['absolute_difference']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## 지속 접촉 중 발의 수평 움직임",
            "",
            "같은 발의 접촉력이 지정된 정책 프레임 수만큼 연속해서 1 N을 넘은 경우만 집계합니다. "
            "환경이 초기화되는 전환 구간은 제외합니다.",
            "",
            "### 전체",
            "",
            "| 최소 연속 프레임 | PhysX 변위 [m/step] | Newton/MJWarp 변위 [m/step] | "
            "PhysX 속도 [m/s] | Newton/MJWarp 속도 [m/s] | 속도 차이 [m/s] | "
            "PhysX 유효 표본 비율 | Newton/MJWarp 유효 표본 비율 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    sustained_overall = comparison["metrics"]["sustained_contact_sweep"]["overall"]
    for minimum_frames, metrics in sustained_overall.items():
        displacement = metrics["mean_foot_horizontal_displacement_m_per_policy_step"]
        speed = metrics["mean_foot_horizontal_speed_m_s"]
        eligible_ratio = metrics["eligible_foot_sample_ratio"]
        lines.append(
            f"| {minimum_frames} | {displacement['physx']:.6g} | {displacement['newton_mjwarp']:.6g} | "
            f"{speed['physx']:.6g} | {speed['newton_mjwarp']:.6g} | {speed['absolute_difference']:.6g} | "
            f"{eligible_ratio['physx']:.6g} | {eligible_ratio['newton_mjwarp']:.6g} |"
        )
    lines.extend(
        [
            "",
            "### 명령별",
            "",
            "| 명령 | 최소 연속 프레임 | PhysX 속도 [m/s] | Newton/MJWarp 속도 [m/s] | 속도 차이 [m/s] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    sustained_commands = comparison["metrics"]["sustained_contact_sweep"]["by_command"]
    for command_name, command_sweep in sustained_commands.items():
        for minimum_frames, metrics in command_sweep.items():
            speed = metrics["mean_foot_horizontal_speed_m_s"]
            lines.append(
                f"| {command_name} | {minimum_frames} | {speed['physx']:.6g} | "
                f"{speed['newton_mjwarp']:.6g} | {speed['absolute_difference']:.6g} |"
            )
    lines.extend(["", "## 명령별 핵심 지표", ""])
    for command_name, metrics in comparison["metrics"]["by_command"].items():
        lines.extend(
            [
                f"### {command_name}",
                "",
                "| 지표 | PhysX | Newton/MJWarp | 절대 차이 |",
                "|---|---:|---:|---:|",
            ]
        )
        for field in (
            "success_rate",
            "fall_rate",
            "lin_vel_xy_rmse_m_s",
            "yaw_rate_rmse_rad_s",
            "mean_contact_foot_slip_m_s",
        ):
            values = metrics[field]
            lines.append(
                f"| {field} | {values['physx']:.6g} | {values['newton_mjwarp']:.6g} | "
                f"{values['absolute_difference']:.6g} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Load, validate, compare, and report two evaluation files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physx", type=Path)
    parser.add_argument("newton_mjwarp", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    physx = json.loads(args.physx.read_text(encoding="utf-8"))
    newton = json.loads(args.newton_mjwarp.read_text(encoding="utf-8"))
    comparison = _build_comparison(physx, newton)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    _write_markdown(comparison, args.output_markdown)
    print(args.output_markdown.read_text(encoding="utf-8"))
    print(f"[Sim2Sim comparison] wrote {args.output_json.resolve()}")
    print(f"[Sim2Sim comparison] wrote {args.output_markdown.resolve()}")


if __name__ == "__main__":
    main()
