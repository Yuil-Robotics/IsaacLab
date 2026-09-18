# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable keyboard input state for Go2 teleoperation."""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable


class LongitudinalHoldCommand:
    """Increase longitudinal speed after a direction key is held.

    GUI key events expose both press and release events, while terminal input
    only exposes repeated characters. Terminal commands therefore remain
    latched at the base speed and use continuous key-repeat events as evidence
    that the key is still being held.
    """

    def __init__(
        self,
        base_speed: float = 0.5,
        fast_speed: float = 1.0,
        hold_duration: float = 1.0,
        terminal_repeat_timeout: float = 0.75,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the longitudinal command state.

        Args:
            base_speed: Speed before the hold threshold [m/s].
            fast_speed: Speed after the hold threshold [m/s].
            hold_duration: Required continuous hold duration [s].
            terminal_repeat_timeout: Maximum terminal repeat gap considered continuous [s].
            clock: Monotonic clock returning time [s].
        """
        if base_speed < 0.0:
            raise ValueError(f"Expected non-negative base_speed, got {base_speed}.")
        if fast_speed < base_speed:
            raise ValueError(f"Expected fast_speed ({fast_speed}) to be at least base_speed ({base_speed}).")
        if hold_duration < 0.0:
            raise ValueError(f"Expected non-negative hold_duration, got {hold_duration}.")
        if terminal_repeat_timeout <= 0.0:
            raise ValueError(f"Expected positive terminal_repeat_timeout, got {terminal_repeat_timeout}.")

        self.base_speed = base_speed
        self.fast_speed = fast_speed
        self.hold_duration = hold_duration
        self.terminal_repeat_timeout = terminal_repeat_timeout
        self._clock = clock

        self._gui_holds: dict[Hashable, tuple[int, float]] = {}
        self._terminal_direction = 0
        self._terminal_started_at = 0.0
        self._terminal_last_event_at = 0.0
        self._terminal_repeat_confirmed = False

    def press(self, key: Hashable, direction: int) -> None:
        """Register a GUI direction-key press without resetting repeated events.

        Args:
            key: Stable identifier for the pressed key.
            direction: ``1`` for forward or ``-1`` for backward.
        """
        self._validate_direction(direction)
        self._clear_terminal()
        self._gui_holds.setdefault(key, (direction, self._clock()))

    def release(self, key: Hashable) -> None:
        """Register a GUI direction-key release.

        Args:
            key: Stable identifier for the released key.
        """
        self._gui_holds.pop(key, None)

    def terminal_press(self, direction: int) -> None:
        """Register one terminal character or auto-repeat event.

        Args:
            direction: ``1`` for forward or ``-1`` for backward.
        """
        self._validate_direction(direction)
        self._gui_holds.clear()
        now = self._clock()
        repeat_is_continuous = (
            self._terminal_direction == direction and now - self._terminal_last_event_at <= self.terminal_repeat_timeout
        )
        if not repeat_is_continuous:
            self._terminal_started_at = now
            self._terminal_repeat_confirmed = False
        else:
            self._terminal_repeat_confirmed = True
        self._terminal_direction = direction
        self._terminal_last_event_at = now

    def stop(self) -> None:
        """Clear all longitudinal input state and return to zero speed."""
        self._gui_holds.clear()
        self._clear_terminal()

    def velocity(self) -> float:
        """Return the current signed longitudinal velocity command [m/s]."""
        now = self._clock()
        if self._gui_holds:
            forward_starts = [started_at for direction, started_at in self._gui_holds.values() if direction > 0]
            backward_starts = [started_at for direction, started_at in self._gui_holds.values() if direction < 0]
            if forward_starts and backward_starts:
                return 0.0
            direction = 1 if forward_starts else -1
            started_at = min(forward_starts or backward_starts)
            speed = self.fast_speed if now - started_at >= self.hold_duration else self.base_speed
            return direction * speed

        if self._terminal_direction == 0:
            return 0.0
        terminal_hold_active = (
            self._terminal_repeat_confirmed
            and now - self._terminal_last_event_at <= self.terminal_repeat_timeout
            and now - self._terminal_started_at >= self.hold_duration
        )
        speed = self.fast_speed if terminal_hold_active else self.base_speed
        return self._terminal_direction * speed

    def _clear_terminal(self) -> None:
        """Clear latched terminal input state."""
        self._terminal_direction = 0
        self._terminal_started_at = 0.0
        self._terminal_last_event_at = 0.0
        self._terminal_repeat_confirmed = False

    @staticmethod
    def _validate_direction(direction: int) -> None:
        """Validate that a direction is forward or backward."""
        if direction not in (-1, 1):
            raise ValueError(f"Expected direction to be -1 or 1, got {direction}.")


def restore_terminal(settings) -> None:
    """Restore terminal terminal attributes to saved settings."""
    import contextlib
    import os
    import sys
    import termios

    if settings is not None and sys.stdin.isatty():
        with contextlib.suppress(Exception):
            termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)
    if sys.stdin.isatty():
        with contextlib.suppress(Exception):
            os.system("stty sane 2>/dev/null")


class UnifiedKeyboardTeleop:
    """Unified keyboard teleoperator listening to BOTH Isaac Sim Kit GUI window and terminal."""

    def __init__(
        self,
        vx: float = 0.5,
        vx_fast: float = 1.0,
        vx_hold_time: float = 1.0,
        vy: float = 0.5,
        wz: float = 1.0,
        device: str = "cuda:0",
        enable_terminal: bool = True,
    ):
        import select
        import sys
        import numpy as np

        self.vx_step = vx
        self.vx_fast = max(vx_fast, vx)
        self.vx_hold_time = vx_hold_time
        self.vy_step = vy
        self.wz_step = wz
        self.device = device
        self.current_cmd = np.zeros(3, dtype=np.float32)
        self._record_requested = False
        self._push_requested = False
        self._longitudinal_command = LongitudinalHoldCommand(
            base_speed=self.vx_step,
            fast_speed=self.vx_fast,
            hold_duration=self.vx_hold_time,
        )

        # 1. Terminal keyboard setup (termios)
        self._term_settings = None
        if enable_terminal and sys.stdin.isatty():
            try:
                import atexit
                import termios
                import tty

                self._term_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                atexit.register(restore_terminal, self._term_settings)
            except Exception:
                pass

        # 2. Isaac Sim Omniverse Kit GUI Window keyboard setup (carb.input)
        self._input = None
        self._keyboard_sub = None
        try:
            import carb
            import carb.input
            import omni.appwindow

            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()

            self._forward_keys = {carb.input.KeyboardInput.W, carb.input.KeyboardInput.UP}
            self._backward_keys = {carb.input.KeyboardInput.S, carb.input.KeyboardInput.DOWN}
            self._key_mapping = {
                carb.input.KeyboardInput.A: np.array([0.0, self.vy_step, 0.0], dtype=np.float32),
                carb.input.KeyboardInput.LEFT: np.array([0.0, self.vy_step, 0.0], dtype=np.float32),
                carb.input.KeyboardInput.D: np.array([0.0, -self.vy_step, 0.0], dtype=np.float32),
                carb.input.KeyboardInput.RIGHT: np.array([0.0, -self.vy_step, 0.0], dtype=np.float32),
                carb.input.KeyboardInput.Q: np.array([0.0, 0.0, self.wz_step], dtype=np.float32),
                carb.input.KeyboardInput.Z: np.array([0.0, 0.0, self.wz_step], dtype=np.float32),
                carb.input.KeyboardInput.E: np.array([0.0, 0.0, -self.wz_step], dtype=np.float32),
                carb.input.KeyboardInput.X: np.array([0.0, 0.0, -self.wz_step], dtype=np.float32),
            }

            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_carb_keyboard_event,
            )
            print("[INFO] Isaac Sim window keyboard listener successfully registered.")
        except Exception as e:
            print(f"[WARN] Isaac Sim window keyboard listener not active ({e}). Terminal input remains active.")

        self._print_controls()

    def _print_controls(self):
        print("\n" + "=" * 65)
        print("Yuil Dog RL Teleoperation Active (Isaac Window & Terminal)")
        print("=" * 65)
        print(
            f"  [W] / [Up]      : Forward   (+{self.vx_step:.2f}, hold {self.vx_hold_time:.1f}s: "
            f"+{self.vx_fast:.2f} m/s)"
        )
        print(
            f"  [S] / [Down]    : Backward  (-{self.vx_step:.2f}, hold {self.vx_hold_time:.1f}s: "
            f"-{self.vx_fast:.2f} m/s)"
        )
        print(f"  [A] / [Left]    : Left      (+{self.vy_step:.2f} m/s)")
        print(f"  [D] / [Right]   : Right     (-{self.vy_step:.2f} m/s)")
        print(f"  [Q] / [Z]       : Turn Left (+{self.wz_step:.2f} rad/s)")
        print(f"  [E] / [X]       : Turn Right(-{self.wz_step:.2f} rad/s)")
        print("  [SPACE] / [K]   : Stop (0.0 m/s)")
        print("  [R] / [C]       : Record 10s CSV (input 45, output 12, joint rad)")
        print("  [P]             : Push robot (manual external force impulse)")
        print("=" * 65 + "\n")

    def _on_carb_keyboard_event(self, event, *args, **kwargs):
        import carb.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input in (carb.input.KeyboardInput.R, carb.input.KeyboardInput.C):
                self._record_requested = True
            elif event.input == carb.input.KeyboardInput.P:
                self._push_requested = True
            elif event.input in (carb.input.KeyboardInput.SPACE, carb.input.KeyboardInput.K, carb.input.KeyboardInput.L):
                self.current_cmd.fill(0.0)
                self._longitudinal_command.stop()
            elif event.input in self._forward_keys:
                self._longitudinal_command.press(event.input, 1)
            elif event.input in self._backward_keys:
                self._longitudinal_command.press(event.input, -1)
            elif event.input in self._key_mapping:
                self.current_cmd += self._key_mapping[event.input]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in self._forward_keys or event.input in self._backward_keys:
                self._longitudinal_command.release(event.input)
            elif event.input in self._key_mapping:
                self.current_cmd -= self._key_mapping[event.input]

    def _poll_terminal(self):
        """Read pending terminal keys non-blockingly."""
        import select
        import sys

        if not sys.stdin.isatty():
            return

        while select.select([sys.stdin], [], [], 0.0)[0]:
            try:
                ch = sys.stdin.read(1)
            except Exception:
                break

            if ch == "\x1b":
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        if select.select([sys.stdin], [], [], 0.0)[0]:
                            ch3 = sys.stdin.read(1)
                            if ch3 == "A":  # Up Arrow
                                self._longitudinal_command.terminal_press(1)
                            elif ch3 == "B":  # Down Arrow
                                self._longitudinal_command.terminal_press(-1)
                            elif ch3 == "C":  # Right Arrow
                                self.current_cmd[1] = -self.vy_step
                            elif ch3 == "D":  # Left Arrow
                                self.current_cmd[1] = self.vy_step
                continue

            ch_lower = ch.lower()
            if ch_lower in ("r", "c"):
                self._record_requested = True
            elif ch_lower == "p":
                self._push_requested = True
            elif ch_lower == "w":
                self._longitudinal_command.terminal_press(1)
            elif ch_lower == "s":
                self._longitudinal_command.terminal_press(-1)
            elif ch_lower == "a":
                self.current_cmd[1] = self.vy_step
            elif ch_lower == "d":
                self.current_cmd[1] = -self.vy_step
            elif ch_lower == "q":
                self.current_cmd[2] = self.wz_step
            elif ch_lower == "e":
                self.current_cmd[2] = -self.wz_step
            elif ch_lower in (" ", "k", "x"):
                self.current_cmd.fill(0.0)
                self._longitudinal_command.stop()

    def check_record_trigger(self) -> bool:
        """Check if record key was pressed, clearing the trigger state."""
        if self._record_requested:
            self._record_requested = False
            return True
        return False

    def check_push_trigger(self) -> bool:
        """Check if push key (P) was pressed, clearing the trigger state."""
        if self._push_requested:
            self._push_requested = False
            return True
        return False

    def get_command(self):
        import numpy as np
        import torch

        self._poll_terminal()
        self.current_cmd[0] = self._longitudinal_command.velocity()
        cmd = np.clip(self.current_cmd, [-1.0, -1.0, -3.0], [1.0, 1.0, 3.0])
        return torch.tensor(cmd, dtype=torch.float32, device=self.device)

    def close(self):
        import contextlib

        if self._input is not None and self._keyboard_sub is not None:
            with contextlib.suppress(Exception):
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        restore_terminal(self._term_settings)

