"""Keyboard + mouse snapshot for 30Hz recording.

Minecraft uses raw mouse look. On Windows we prefer WM_INPUT; pynput is the
fallback for keys and for other OS.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field

from atomspike.models.actions import (
    KEY_HOLD,
    KEY_IDLE,
    KEY_PRESS,
    KEY_RELEASE,
    ActionSpec,
    ActionStateMachine,
)


@dataclass
class InputSnapshot:
    keys_down: dict[str, bool]
    mouse_dx: float
    mouse_dy: float
    lmb: bool
    rmb: bool
    cursor_sx: float = 0.0
    cursor_sy: float = 0.0
    wheel: float = 0.0
    stop: bool = False
    mark: bool = False


class InputTap:
    def __init__(self, spec: ActionSpec, stop_key: str = "F9", mark_key: str = "F8"):
        self.spec = spec
        self.stop_key = stop_key.upper()
        self.mark_key = mark_key.upper()
        self.machine = ActionStateMachine(spec)
        self._down: set[str] = set()
        self._lmb = False
        self._rmb = False
        self._dx = 0.0
        self._dy = 0.0
        self._cursor_sx = 0.0
        self._cursor_sy = 0.0
        self._have_cursor = False
        self._wheel = 0.0
        self._stop = False
        self._mark = False
        self._lock = threading.Lock()
        self._prev_keys = [False] * spec.n_keys
        self._prev_lmb = False
        self._prev_rmb = False
        self._listener_kb = None
        self._listener_ms = None
        self._raw = None
        self._ok = False
        self._start()

    def _start(self) -> None:
        try:
            from pynput import keyboard, mouse
        except Exception:
            return
        self._ok = True

        def on_press(key):
            name = _key_name(key)
            if name is None:
                return
            with self._lock:
                self._down.add(name)
                if name == self.stop_key:
                    self._stop = True
                if name == self.mark_key:
                    self._mark = True

        def on_release(key):
            name = _key_name(key)
            if name is None:
                return
            with self._lock:
                self._down.discard(name)

        def on_move(x, y):
            with self._lock:
                if self._raw is None and self._have_cursor:
                    self._dx += float(x) - self._cursor_sx
                    self._dy += float(y) - self._cursor_sy
                self._cursor_sx = float(x)
                self._cursor_sy = float(y)
                self._have_cursor = True

        def on_click(x, y, button, pressed):
            with self._lock:
                self._cursor_sx = float(x)
                self._cursor_sy = float(y)
                b = str(button).lower()
                if "left" in b:
                    self._lmb = bool(pressed)
                elif "right" in b:
                    self._rmb = bool(pressed)

        def on_scroll(x, y, dx, dy):
            with self._lock:
                self._cursor_sx = float(x)
                self._cursor_sy = float(y)
                self._wheel += float(dy)

        self._listener_kb = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener_ms = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        self._listener_kb.daemon = True
        self._listener_ms.daemon = True
        self._listener_kb.start()
        self._listener_ms.start()
        if sys.platform == "win32":
            try:
                from atomspike.capture.raw_mouse import WinRawMouse

                self._raw = WinRawMouse()
                self._raw.start()
            except Exception:
                self._raw = None

    def snapshot(self) -> InputSnapshot:
        with self._lock:
            dx, dy = self._dx, self._dy
            self._dx = 0.0
            self._dy = 0.0
            mark = self._mark
            self._mark = False
            stop = self._stop
            down = set(self._down)
            lmb, rmb = self._lmb, self._rmb
            cx, cy = self._cursor_sx, self._cursor_sy
            wheel = self._wheel
            self._wheel = 0.0
        if self._raw is not None:
            rdx, rdy = self._raw.consume()
            dx += rdx
            dy += rdy
        keys = {name: name.upper() in down for name in self.spec.key_names}
        return InputSnapshot(
            keys_down=keys,
            mouse_dx=dx,
            mouse_dy=dy,
            lmb=lmb,
            rmb=rmb,
            cursor_sx=cx,
            cursor_sy=cy,
            wheel=wheel,
            stop=stop,
            mark=mark,
        )

    @property
    def ok(self) -> bool:
        return self._ok

    def to_tokens(
        self,
        snap: InputSnapshot,
        *,
        gui_open: bool = False,
        cursor_norm: tuple[float, float] | None = None,
    ):
        import numpy as np
        import torch

        tokens: list[int] = []
        for i, name in enumerate(self.spec.key_names):
            now = bool(snap.keys_down.get(name, False))
            was = self._prev_keys[i]
            tokens.append(_edge(was, now))
            self._prev_keys[i] = now
        if gui_open and cursor_norm is not None and abs(snap.mouse_dx) + abs(snap.mouse_dy) <= 3.0:
            tokens.append(self.spec.mouse_unit_to_bin(cursor_norm[0]))
            tokens.append(self.spec.mouse_unit_to_bin(cursor_norm[1]))
        else:
            tokens.append(self.spec.mouse_delta_to_bin(snap.mouse_dx))
            tokens.append(self.spec.mouse_delta_to_bin(snap.mouse_dy))
        tokens.append(_edge(self._prev_lmb, snap.lmb))
        tokens.append(_edge(self._prev_rmb, snap.rmb))
        self._prev_lmb, self._prev_rmb = snap.lmb, snap.rmb
        arr = np.asarray(tokens, dtype=np.int64)
        self.machine.apply(torch.from_numpy(arr.copy()))
        return arr

    def close(self) -> None:
        for lis in (self._listener_kb, self._listener_ms):
            if lis is not None:
                try:
                    lis.stop()
                except Exception:
                    pass
        if self._raw is not None:
            self._raw.close()


def _edge(was_down: bool, now_down: bool) -> int:
    if now_down and not was_down:
        return KEY_PRESS
    if now_down and was_down:
        return KEY_HOLD
    if (not now_down) and was_down:
        return KEY_RELEASE
    return KEY_IDLE


def _key_name(key) -> str | None:
    try:
        from pynput.keyboard import Key
    except Exception:
        return None
    if isinstance(key, Key):
        mapping = {
            Key.space: "SPACE",
            Key.shift: "SHIFT",
            Key.shift_r: "SHIFT",
            Key.ctrl: "CTRL",
            Key.ctrl_l: "CTRL",
            Key.ctrl_r: "CTRL",
            Key.alt: "ALT",
            Key.tab: "TAB",
            Key.esc: "ESC",
            Key.f8: "F8",
            Key.f9: "F9",
            Key.f10: "F10",
        }
        if key in mapping:
            return mapping[key]
        return key.name.upper() if key.name else None
    ch = getattr(key, "char", None)
    if ch:
        return ch.upper()
    return None
