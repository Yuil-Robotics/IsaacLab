# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check CTS student inference, observation history, contacts, and reset semantics in PhysX."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="CTS checkpoint to validate.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app
try:
    import gymnasium as gym
    import torch
    from go2_moe_cts.agents.moe_cts_cfg import MoECTSRunnerCfg
    from go2_moe_cts.learning.runner import OnPolicyRunnerCTS
    from go2_moe_cts.tasks.yuil_dog.env_cfg import YuilDogMoECTSEvalEnvCfg

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    cfg = YuilDogMoECTSEvalEnvCfg()
    cfg.scene.num_envs = 8
    env = gym.make("Isaac-Velocity-Rough-Yuil-Dog-MoECTS-Eval-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)
    try:
        runner = OnPolicyRunnerCTS(env, MoECTSRunnerCfg().to_dict(), device=env.device, inference_only=True)
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
        checkpoint = str(checkpoint_path.resolve())
        runner.load(checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy()
        runner.export_policy_to_jit(str(Path(checkpoint).parent / "exported"))
        jit = torch.jit.load(str(Path(checkpoint).parent / "exported/policy.pt"), map_location=env.device)
        obs = env.get_observations()
        max_contact = 0.0
        max_pose_error = 0.0
        max_export_error = 0.0
        resets = 0
        scan_valid_min = 1.0
        with torch.no_grad():
            for i in range(200):
                if i == 100:
                    raw = env.unwrapped
                    raw.reset(env_ids=torch.tensor([0], device=env.device))
                    obs = env.get_observations()
                    term = raw.action_manager.get_term("joint_pos")
                    assert torch.count_nonzero(term.previous[0]) == 0
                    assert torch.count_nonzero(term.previous_previous[0]) == 0
                    offset = 0
                    for width in (3, 3, 3, 12, 12, 12):
                        block = obs["policy"][0, offset : offset + 10 * width].reshape(10, width)
                        torch.testing.assert_close(block, block[-1:].expand_as(block))
                        offset += 10 * width
                if i == 130:
                    env.unwrapped.episode_length_buf[1] = env.unwrapped.max_episode_length - 1
                assert all(torch.isfinite(x).all() for x in obs.values())
                assert tuple(obs["policy"].shape) == (8, 450)
                assert tuple(obs["critic"].shape) == (8, 263)
                latest = torch.cat(
                    [
                        obs["policy"][:, e - w : e]
                        for e, w in [(30, 3), (60, 3), (90, 3), (210, 12), (330, 12), (450, 12)]
                    ],
                    -1,
                )
                torch.testing.assert_close(latest, obs["single_obs"])
                actions = policy(obs)
                expected = jit(obs["policy"])
                max_export_error = max(max_export_error, (actions - expected).abs().max().item())
                torch.testing.assert_close(actions, expected)
                obs, reward, dones, _ = env.step(actions)
                assert torch.isfinite(reward).all()
                resets += int(dones.sum())
                if i == 130:
                    assert dones[1], "Forced timeout did not reset environment 1"
                raw = env.unwrapped
                sensor = raw.scene["height_scanner"]
                pose_error = (
                    (sensor.data.pos_w.torch[:, :2] - raw.scene["robot"].data.root_pos_w.torch[:, :2])
                    .abs()
                    .max()
                    .item()
                )
                max_pose_error = max(max_pose_error, pose_error)
                contact = raw.scene["contact_forces"].data.net_forces_w.torch.norm(dim=-1).max().item()
                max_contact = max(max_contact, contact)
                scan_valid_min = min(scan_valid_min, torch.isfinite(sensor.data.ray_hits_w.torch).float().mean().item())
        assert max_contact > 1.0, max_contact
        assert max_pose_error < 1e-3, max_pose_error
        assert scan_valid_min == 1.0, scan_valid_min
        print(
            "VALIDATED",
            {
                "steps": 200,
                "envs": 8,
                "terrain_rows": cfg.scene.terrain.terrain_generator.num_rows,
                "max_contact_N": max_contact,
                "scanner_xy_error_m": max_pose_error,
                "scan_valid_fraction_min": scan_valid_min,
                "jit_error_max": max_export_error,
                "episode_resets": resets,
                "partial_reset_checked": True,
                "timeout_reset_checked": True,
            },
            flush=True,
        )
    finally:
        env.close()
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    app.close()
