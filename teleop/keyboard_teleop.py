from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Mapping
from typing import Protocol


class KeyStateProvider(Protocol):
    def pressed(self) -> Mapping[str, bool]:
        """Return the current key state for supported teleop keys."""


class KeyboardTeleop:
    """Poll keyboard input and return one-step XYZ motion commands.

    Key mapping:
    - Up / Down: +X / -X
    - Right / Left: +Y / -Y
    - PageUp / PageDown: +Z / -Z

    Each step is a normalized direction command. Multiply the returned step by
    an external scale such as ``thetat`` to get the Cartesian displacement for
    the current control frame. Pressing backslash toggles the gripper state.
    """

    KEY_UP = "up"
    KEY_DOWN = "down"
    KEY_LEFT = "left"
    KEY_RIGHT = "right"
    KEY_PAGE_UP = "page_up"
    KEY_PAGE_DOWN = "page_down"
    KEY_BACKSLASH = "backslash"

    def __init__(
        self,
        hz: float = 30.0,
        *,
        gripper_closed: bool = False,
        key_state_provider: KeyStateProvider | None = None,
    ) -> None:
        if hz <= 0.0:
            raise ValueError("hz must be positive")

        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self._last_step = [0.0, 0.0, 0.0]
        self._gripper_closed = bool(gripper_closed)
        self._previous_keys: dict[str, bool] = {}

        self._keys = key_state_provider or _default_key_state_provider()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None



    def stop(self) -> None:
        """Stop the keyboard polling loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if hasattr(self._keys, "stop"):
            self._keys.stop()  # type: ignore[attr-defined]

    def reset(self) -> None:
        """Reset the latest XYZ step to zero."""
        with self._lock:
            self._last_step = [0.0, 0.0, 0.0]
            self._gripper_closed = False
            self._previous_keys = {}

    def update(self) -> list[float]:
        """Poll the keyboard once and return this frame's XYZ direction."""
        key_state = self._keys.pressed()
        x, y, z = self._direction_from_keys(key_state)
        step = [x, y, z]
        toggle_gripper = bool(key_state.get(self.KEY_BACKSLASH)) and not bool(
            self._previous_keys.get(self.KEY_BACKSLASH)
        )

        with self._lock:
            if toggle_gripper:
                self._gripper_closed = not self._gripper_closed
            self._last_step = step
            self._previous_keys = dict(key_state)
            return self._last_step.copy()

    def get_step(self) -> list[float]:
        """Return the latest one-frame ``[x, y, z]`` direction command."""
        with self._lock:
            return self._last_step.copy()

    def get_delta(self) -> list[float]:
        """Alias for ``get_step``."""
        return self.get_step()

    def get_gripper_closed(self) -> bool:
        """Return True when the gripper toggle is in the closed state."""
        with self._lock:
            return self._gripper_closed

    def get_gripper_command(
        self,
        *,
        open_value: float = 0.0,
        closed_value: float = 0.8,
    ) -> float:
        """Return the current gripper command value."""
        with self._lock:
            return closed_value if self._gripper_closed else open_value

    def close(self) -> None:
        self.stop()

    def __call__(self) -> list[float]:
        return self.get_step()



    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _poll_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            self.update()
            next_tick += self.dt
            delay = max(0.0, next_tick - time.perf_counter())
            self._stop_event.wait(delay)

    @classmethod
    def _direction_from_keys(cls, keys: Mapping[str, bool]) -> tuple[int, int, int]:
        x = int(bool(keys.get(cls.KEY_UP))) - int(bool(keys.get(cls.KEY_DOWN)))
        y = int(bool(keys.get(cls.KEY_RIGHT))) - int(bool(keys.get(cls.KEY_LEFT)))
        z = int(bool(keys.get(cls.KEY_PAGE_UP))) - int(
            bool(keys.get(cls.KEY_PAGE_DOWN))
        )
        return x, y, z


def _default_key_state_provider() -> KeyStateProvider:
    if os.name == "nt":
        return _WindowsKeyStateProvider()
    return _TerminalKeyStateProvider()


class _WindowsKeyStateProvider:
    _VK_CODES = {
        KeyboardTeleop.KEY_UP: 0x26,
        KeyboardTeleop.KEY_DOWN: 0x28,
        KeyboardTeleop.KEY_LEFT: 0x25,
        KeyboardTeleop.KEY_RIGHT: 0x27,
        KeyboardTeleop.KEY_PAGE_UP: 0x21,
        KeyboardTeleop.KEY_PAGE_DOWN: 0x22,
        KeyboardTeleop.KEY_BACKSLASH: 0xDC,
    }

    def __init__(self) -> None:
        import ctypes

        self._user32 = ctypes.windll.user32

    def pressed(self) -> Mapping[str, bool]:
        return {
            key: bool(self._user32.GetAsyncKeyState(vk_code) & 0x8000)
            for key, vk_code in self._VK_CODES.items()
        }


class _TerminalKeyStateProvider:
    _ESCAPE_SEQUENCES = {
        "\x1b[A": KeyboardTeleop.KEY_UP,
        "\x1b[B": KeyboardTeleop.KEY_DOWN,
        "\x1b[C": KeyboardTeleop.KEY_RIGHT,
        "\x1b[D": KeyboardTeleop.KEY_LEFT,
        "\x1b[5~": KeyboardTeleop.KEY_PAGE_UP,
        "\x1b[6~": KeyboardTeleop.KEY_PAGE_DOWN,
    }

    def __init__(self) -> None:
        self._old_settings = None
        self._buffer = ""

    def start(self) -> None:
        if not sys.stdin.isatty():
            return

        import termios
        import tty

        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def stop(self) -> None:
        if self._old_settings is None:
            return

        import termios

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        self._old_settings = None

    def pressed(self) -> Mapping[str, bool]:
        keys = {
            KeyboardTeleop.KEY_UP: False,
            KeyboardTeleop.KEY_DOWN: False,
            KeyboardTeleop.KEY_LEFT: False,
            KeyboardTeleop.KEY_RIGHT: False,
            KeyboardTeleop.KEY_PAGE_UP: False,
            KeyboardTeleop.KEY_PAGE_DOWN: False,
            KeyboardTeleop.KEY_BACKSLASH: False,
        }
        if not sys.stdin.isatty():
            return keys

        self._read_available_stdin()
        if "\\" in self._buffer:
            keys[KeyboardTeleop.KEY_BACKSLASH] = True
            self._buffer = self._buffer.replace("\\", "", 1)
        for sequence, key in self._ESCAPE_SEQUENCES.items():
            if sequence in self._buffer:
                keys[key] = True
                self._buffer = self._buffer.replace(sequence, "", 1)
        self._buffer = self._buffer[-8:]
        return keys

    def _read_available_stdin(self) -> None:
        import select

        while select.select([sys.stdin], [], [], 0.0)[0]:
            self._buffer += sys.stdin.read(1)
