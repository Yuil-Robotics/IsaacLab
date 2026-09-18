#!/usr/bin/env python3

"""Restart robot_state_publisher whenever its URDF file changes."""

import argparse
import json
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def stop_process(process):
    if process is None or process.poll() is not None:
        return

    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('urdf', type=Path)
    parser.add_argument('robot_state_publisher', type=Path)
    parser.add_argument('--poll-period', type=float, default=0.25)
    parser.add_argument('--settle-time', type=float, default=0.35)
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    publisher_path = args.robot_state_publisher.resolve()
    process = None
    running = True
    parameter_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', prefix='dog_description_', delete=False
    )
    parameter_path = Path(parameter_file.name)
    parameter_file.close()

    def request_shutdown(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    last_signature = None
    pending_signature = None
    changed_at = 0.0

    print(f'[urdf_live_reloader] Watching {urdf_path}', flush=True)

    try:
        while running:
            try:
                stat = urdf_path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                signature = None

            if signature != last_signature and signature != pending_signature:
                pending_signature = signature
                changed_at = time.monotonic()

            if (
                pending_signature is not None
                and time.monotonic() - changed_at >= args.settle_time
            ):
                try:
                    ET.parse(urdf_path)
                except (ET.ParseError, OSError) as error:
                    print(
                        f'[urdf_live_reloader] Waiting for valid XML: {error}',
                        file=sys.stderr,
                        flush=True,
                    )
                    changed_at = time.monotonic()
                else:
                    stop_process(process)
                    robot_description = urdf_path.read_text(encoding='utf-8')
                    parameter_path.write_text(
                        json.dumps({
                            '/**': {
                                'ros__parameters': {
                                    'robot_description': robot_description,
                                }
                            }
                        }),
                        encoding='utf-8',
                    )
                    process = subprocess.Popen(
                        [
                            str(publisher_path),
                            '--ros-args',
                            '--params-file',
                            str(parameter_path),
                        ]
                    )
                    last_signature = pending_signature
                    pending_signature = None
                    print(
                        '[urdf_live_reloader] Reloaded robot description',
                        flush=True,
                    )

            time.sleep(args.poll_period)
    finally:
        stop_process(process)
        parameter_path.unlink(missing_ok=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
