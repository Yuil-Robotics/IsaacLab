// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use crate::state::{ACTION_DIM, SharedSimState};

fn read_f64s(stream: &mut TcpStream) -> Option<Vec<f64>> {
    let mut length = [0_u8; 4];
    stream.read_exact(&mut length).ok()?;
    let byte_length = u32::from_le_bytes(length) as usize;
    if !byte_length.is_multiple_of(8) || byte_length > 1024 * 1024 {
        return None;
    }

    let mut payload = vec![0_u8; byte_length];
    stream.read_exact(&mut payload).ok()?;
    Some(
        payload
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
    )
}

fn write_f64s(stream: &mut TcpStream, values: &[f64]) -> bool {
    let byte_length = size_of_val(values) as u32;
    let mut packet = Vec::with_capacity(4 + byte_length as usize);
    packet.extend_from_slice(&byte_length.to_le_bytes());
    for &value in values {
        packet.extend_from_slice(&value.to_le_bytes());
    }
    stream.write_all(&packet).is_ok()
}

fn handle_client(mut stream: TcpStream, shared: SharedSimState) {
    let mut bootstrap_pending = true;
    while let Some(action) = read_f64s(&mut stream) {
        if action.len() != ACTION_DIM {
            eprintln!(
                "invalid action payload: expected {ACTION_DIM} values, got {}",
                action.len()
            );
            break;
        }
        if action.iter().any(|value| !value.is_finite()) {
            eprintln!("closing connection after non-finite action payload");
            break;
        }

        if bootstrap_pending {
            bootstrap_pending = false;
            let observation = {
                let (mutex, _) = &*shared;
                mutex.lock().unwrap().observation
            };
            if !write_f64s(&mut stream, &observation) {
                break;
            }
            continue;
        }

        let requested_version = {
            let (mutex, updated) = &*shared;
            let mut state = mutex.lock().unwrap();
            state.action.copy_from_slice(&action);
            state.action_version += 1;
            updated.notify_all();
            state.action_version
        };

        let observation = {
            let (mutex, updated) = &*shared;
            let mut state = mutex.lock().unwrap();
            while state.completed_action_version < requested_version {
                let wait = updated.wait_timeout(state, Duration::from_secs(2)).unwrap();
                state = wait.0;
                if wait.1.timed_out() {
                    eprintln!("timed out waiting for the MuJoCo physics step");
                    return;
                }
            }
            state.observation
        };

        if !write_f64s(&mut stream, &observation) {
            break;
        }
    }
    let (episode_step, base_position, fall_count) = {
        let (mutex, _) = &*shared;
        let state = mutex.lock().unwrap();
        (state.episode_step, state.base_position, state.fall_count)
    };
    println!(
        "policy client disconnected: episode_step={episode_step} base_position={base_position:?} falls={fall_count}"
    );
}

pub fn run_server(address: &str, shared: SharedSimState) {
    let listener = TcpListener::bind(address).expect("failed to bind TCP policy server");
    println!("TCP policy server listening on {address}");
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let peer = stream
                    .peer_addr()
                    .map_or_else(|_| "unknown".to_owned(), |address| address.to_string());
                println!("policy client connected: {peer}");
                let shared = shared.clone();
                std::thread::spawn(move || handle_client(stream, shared));
            }
            Err(error) => eprintln!("TCP accept error: {error}"),
        }
    }
}
