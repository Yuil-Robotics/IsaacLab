// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    let download_dir = env::var_os("MUJOCO_DOWNLOAD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join(".mujoco")
        });
    let library_dir = download_dir.join("mujoco-3.9.0").join("lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_dir.display());
}
