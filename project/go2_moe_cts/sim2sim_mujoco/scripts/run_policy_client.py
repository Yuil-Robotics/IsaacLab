#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the exported 450-input/12-output policy continuously against the MuJoCo server."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT_DIR.parent
ACTION_DIM = 12
POLICY_OBSERVATION_DIM = 450
PACKET_HEADER = struct.Struct("<I")

POLICIES_DIR = ROOT_DIR / "policies"
DEFAULT_POLICY = POLICIES_DIR / "policy.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="TorchScript policy file (.pt). Can be a direct path or filename inside policies/.",
    )
    parser.add_argument("--list", action="store_true", help="List available policies in policies/ directory and exit.")
    parser.add_argument(
        "--server", "--address", dest="address", default="127.0.0.1:7001", help="Server address as HOST:PORT."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Evaluation duration [s]; zero runs continuously until interrupted (default: 0).",
    )
    parser.add_argument("--rate_hz", type=float, default=50.0, help="Wall-clock policy rate [Hz]; zero runs unpaced.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "logs" / "sim2sim_mujoco" / time.strftime("%Y-%m-%d_%H-%M-%S") / "policy_result.json",
        help="Summary JSON path.",
    )
    parser.add_argument("--log_every", type=int, default=50, help="Status print interval in policy steps.")
    return parser.parse_args()


def resolve_policy_path(policy_arg: str | None) -> Path:
    """Resolve policy path from direct path, filename in policies/, or default."""
    if policy_arg is not None:
        candidate = Path(policy_arg).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        in_dir = POLICIES_DIR / policy_arg
        if in_dir.is_file():
            return in_dir.resolve()
        with_ext = POLICIES_DIR / f"{policy_arg}.pt"
        if with_ext.is_file():
            return with_ext.resolve()
        raise FileNotFoundError(f"Policy not found: '{policy_arg}'. Looked in current directory and '{POLICIES_DIR}'.")
    if DEFAULT_POLICY.is_file():
        return DEFAULT_POLICY.resolve()
    pt_files = sorted(POLICIES_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pt_files:
        return pt_files[0].resolve()
    raise FileNotFoundError(f"No policy (.pt) file found in {POLICIES_DIR}.")


def split_address(address: str) -> tuple[str, int]:
    """Split a HOST:PORT endpoint."""
    host, separator, port = address.rpartition(":")
    if not separator or not host:
        raise ValueError(f"Address must use HOST:PORT syntax: {address}")
    return host, int(port)


def infer(policy: torch.jit.ScriptModule, observation: np.ndarray) -> np.ndarray:
    """Run deterministic policy inference and validate the output shape."""
    if observation.shape != (POLICY_OBSERVATION_DIM,):
        raise ValueError(f"Expected a 450-D observation, got {observation.shape}.")
    if not np.all(np.isfinite(observation)):
        raise ValueError("Non-finite policy observation.")
    tensor = torch.from_numpy(observation.astype(np.float32, copy=False)).unsqueeze(0)
    with torch.inference_mode():
        output = policy(tensor)
    action = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
        raise ValueError(f"Policy must return one finite 12-D action, got {action.shape}.")
    return action


def receive_exact(connection: socket.socket, size: int) -> bytes:
    """Receive exactly ``size`` bytes or raise on early disconnect."""
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Rust MuJoCo server closed the connection.")
        data.extend(chunk)
    return bytes(data)


def receive_vector(connection: socket.socket, expected_size: int) -> np.ndarray:
    """Receive one length-prefixed little-endian float64 vector."""
    payload_size = PACKET_HEADER.unpack(receive_exact(connection, PACKET_HEADER.size))[0]
    expected_bytes = expected_size * np.dtype("<f8").itemsize
    if payload_size != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} payload bytes, received {payload_size}.")
    return np.frombuffer(receive_exact(connection, payload_size), dtype="<f8").astype(np.float64, copy=True)


def send_vector(connection: socket.socket, values: np.ndarray) -> None:
    """Send one length-prefixed little-endian float64 vector."""
    payload = np.asarray(values, dtype="<f8").tobytes()
    connection.sendall(PACKET_HEADER.pack(len(payload)) + payload)


def main() -> None:
    """Evaluate the exported policy through the lock-step TCP contract."""
    args = parse_args()
    if args.list:
        print(f"Available policies in {POLICIES_DIR}:")
        if not POLICIES_DIR.is_dir():
            print("  (policies directory does not exist)")
            return
        files = sorted(POLICIES_DIR.glob("*.pt"))
        if not files:
            print("  (no .pt policy files found)")
            return
        for f in files:
            marker = " [DEFAULT]" if f.name == "policy.pt" else ""
            print(f"  - {f.name}{marker}")
        return

    policy_path = resolve_policy_path(args.policy)
    print(f"[Sim2Sim] Policy: {policy_path.name} ({policy_path})")
    contract_path = policy_path.with_suffix(".json")
    if not contract_path.is_file():
        contract_path = policy_path.parent / "policy_contract.json"
    contract = json.loads(contract_path.read_text())
    expected = {
        "algorithm": "MoE-CTS student",
        "input_dim": 450,
        "action_dim": 12,
        "history_layout": "term-major, oldest-to-newest",
        "observation_scales": [0.25, 1.0, 1.0, 1.0, 0.05, 1.0],
        "control_dt": 0.02,
        "action_scale": 0.25,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Incompatible policy contract {key}: {contract.get(key)!r}; expected {value!r}")
    torch.set_num_threads(1)
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    if args.duration < 0.0 or args.rate_hz < 0.0:
        raise ValueError("--duration and --rate_hz must be non-negative.")
    host, port = split_address(args.address)
    policy_dt_s = 0.02
    requested_steps = None if args.duration == 0.0 else max(1, round(args.duration / policy_dt_s))
    completed_steps = 0
    action_rms_sum = 0.0
    action_peak = np.zeros(ACTION_DIM, dtype=np.float64)
    start_time = time.monotonic()
    try:
        with socket.create_connection((host, port)) as connection:
            send_vector(connection, np.zeros(ACTION_DIM, dtype=np.float64))
            observation = receive_vector(connection, POLICY_OBSERVATION_DIM)
            next_deadline = time.monotonic()
            while requested_steps is None or completed_steps < requested_steps:
                if args.rate_hz > 0.0:
                    next_deadline += 1.0 / args.rate_hz
                action = infer(policy, observation)
                action_rms = float(np.sqrt(np.mean(np.square(action))))
                action_rms_sum += action_rms
                action_peak = np.maximum(action_peak, np.abs(action))
                send_vector(connection, action)
                observation = receive_vector(connection, POLICY_OBSERVATION_DIM)
                completed_steps += 1
                if args.log_every > 0 and completed_steps % args.log_every == 0:
                    print(
                        f"POLICY step={completed_steps} sim_time_s={completed_steps * policy_dt_s:.3f} "
                        f"action_rms={action_rms:.4f} action_abs_max={np.max(np.abs(action)):.4f}",
                        flush=True,
                    )
                if args.rate_hz > 0.0:
                    remaining = next_deadline - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[Sim2Sim] Interrupted by user; closing the policy connection.")
    result = {
        "policy": str(policy_path),
        "server": args.address,
        "input_dimension": POLICY_OBSERVATION_DIM,
        "output_dimension": ACTION_DIM,
        "num_policy_steps": completed_steps,
        "simulated_duration_s": completed_steps * policy_dt_s,
        "wall_time_s": time.monotonic() - start_time,
        "action_rms_mean": action_rms_sum / completed_steps if completed_steps else 0.0,
        "action_abs_max_by_policy_index": action_peak.tolist(),
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"result={output_path}")


if __name__ == "__main__":
    main()
