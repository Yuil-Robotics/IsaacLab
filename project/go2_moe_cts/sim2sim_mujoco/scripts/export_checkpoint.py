#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a local Yuil Dog MoE-CTS checkpoint without starting a simulator."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from go2_moe_cts.learning.runner import OnPolicyRunnerCTS
from go2_moe_cts.tasks.yuil_dog.env_cfg import YuilDogMoECTSEnvCfg
from tensordict import TensorDict

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT_DIR.parent


def main() -> None:
    """Export the student and its deployment contract into the local policy folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path relative to this project, or absolute.")
    parser.add_argument("--name", default="policy", help="Output policy basename, without .pt.")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        raise ValueError("--name must be a basename.")
    checkpoint_path = args.checkpoint.expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    checkpoint_path = checkpoint_path.resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "go2_moe_cts_v1":
        raise ValueError("Expected a MoE-CTS training checkpoint, not a flat PPO model or TorchScript file.")
    obs = TensorDict(
        {key: torch.zeros(1, size) for key, size in (("policy", 450), ("single_obs", 45), ("critic", 263))},
        batch_size=[1],
    )
    env = SimpleNamespace(
        num_actions=12, get_observations=lambda: obs, unwrapped=SimpleNamespace(cfg=YuilDogMoECTSEnvCfg())
    )
    runner = OnPolicyRunnerCTS(env, checkpoint["config"], inference_only=True)
    runner.load(str(checkpoint_path), load_optimizer=False)
    destination = ROOT_DIR / "policies"
    runner.export_policy_to_jit(str(destination), filename=f"{args.name}.pt")
    # Verify that TorchScript keeps the original student's inference behavior.
    scripted = torch.jit.load(str(destination / f"{args.name}.pt")).eval()
    history = torch.randn(4, 450, generator=torch.Generator().manual_seed(42))
    single = torch.cat(
        [history[:, end - width : end] for end, width in ((30, 3), (60, 3), (90, 3), (210, 12), (330, 12), (450, 12))],
        dim=-1,
    )
    sample = TensorDict({"policy": history, "single_obs": single, "critic": torch.zeros(4, 263)}, batch_size=[4])
    runner.alg.policy.eval()
    with torch.inference_mode():
        torch.testing.assert_close(scripted(history), runner.alg.policy.act_inference(sample))
    contract = json.loads((destination / "policy_contract.json").read_text())
    contract.update(
        checkpoint=str(checkpoint_path.relative_to(PROJECT_DIR))
        if checkpoint_path.is_relative_to(PROJECT_DIR)
        else str(checkpoint_path),
        iteration=checkpoint["iter"],
        action_clip=100.0,
    )
    (destination / f"{args.name}.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"Exported student: {destination / (args.name + '.pt')} (iteration={checkpoint['iter']})")


if __name__ == "__main__":
    main()
