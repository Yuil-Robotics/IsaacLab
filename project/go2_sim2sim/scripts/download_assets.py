#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download the official Isaac Lab Go2 USD and all referenced dependencies."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go2_sim2sim.asset_cfg import (  # noqa: E402
    ASSET_ROOT,
    LOCAL_GO2_USD_PATH,
    LOCAL_GROUND_USD_PATH,
    REMOTE_GO2_USD_URL,
    REMOTE_GROUND_USD_URL,
)

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402


def main() -> None:
    """Mirror the official Go2 and ground-plane dependency trees."""
    assets = (
        (REMOTE_GO2_USD_URL, LOCAL_GO2_USD_PATH),
        (REMOTE_GROUND_USD_URL, LOCAL_GROUND_USD_PATH),
    )
    for remote_url, expected_path in assets:
        downloaded_path = Path(
            retrieve_file_path(
                remote_url,
                download_dir=str(ASSET_ROOT),
            )
        )
        if downloaded_path.resolve() != expected_path.resolve():
            raise RuntimeError(f"Unexpected downloaded root: {downloaded_path}; expected: {expected_path}")
        if not expected_path.is_file():
            raise FileNotFoundError(f"Asset was not downloaded: {expected_path}")
    print(f"[Go2 assets] ready: {LOCAL_GO2_USD_PATH}")
    print(f"[Go2 assets] ready: {LOCAL_GROUND_USD_PATH}")


if __name__ == "__main__":
    main()
