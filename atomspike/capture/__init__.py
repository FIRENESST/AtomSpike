"""Screen capture, human demo recording, and research-only local inject.

Compliance: Minecraft singleplayer / LAN research capture only. Do not point
the injector at online servers or competitive clients (ToS / EULA).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from atomspike.capture.mc_collector import run_minecraft_collector
from atomspike.capture.window import WindowGrabber, list_windows
from atomspike.config import CaptureConfig
from atomspike.data.io import EpisodeWriter


@dataclass
class FramePacket:
    t_ns: int
    frame: NDArray[np.uint8]


class ScreenCapture:
    def __init__(self, cfg: CaptureConfig | None = None):
        self.cfg = cfg or CaptureConfig()
        self._mss = None
        try:
            import mss

            self._mss = mss.mss()
        except Exception:
            self._mss = None

    def grab(self) -> FramePacket:
        t_ns = time.time_ns()
        if self._mss is None:
            frame = np.zeros((84, 84, 3), dtype=np.uint8)
            return FramePacket(t_ns=t_ns, frame=frame)
        mon = self._mss.monitors[self.cfg.monitor]
        if self.cfg.region is not None:
            x, y, w, h = self.cfg.region
            mon = {"left": x, "top": y, "width": w, "height": h}
        raw = np.array(self._mss.grab(mon))[:, :, :3][:, :, ::-1]
        return FramePacket(t_ns=t_ns, frame=raw)

    def close(self) -> None:
        if self._mss is not None:
            self._mss.close()


class DemoRecorder:
    """Record (frame, state, action) from an env-like source into HDF5."""

    def __init__(self, writer: EpisodeWriter):
        self.writer = writer
        self.n = 0

    def push(self, frame, state, action, reward: float, done: bool) -> None:
        self.writer.add(frame, state, action, reward, done)
        self.n += 1

    def close(self):
        return self.writer.close()


_KEY_ALIASES = {
    "SPACE": "space",
    "SHIFT": "shift",
    "CTRL": "ctrl",
    "ALT": "alt",
    "TAB": "tab",
    "ENTER": "enter",
    "ESC": "esc",
}

MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0


def _sendinput_relative(dx: int, dy: int) -> bool:
    """Relative SendInput so Minecraft's WM_INPUT path sees the look delta."""
    if (dx == 0 and dy == 0) or sys.platform != "win32":
        return False
    try:
        import ctypes

        ulong_ptr = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = (
                ("dx", ctypes.c_int32),
                ("dy", ctypes.c_int32),
                ("mouseData", ctypes.c_uint32),
                ("dwFlags", ctypes.c_uint32),
                ("time", ctypes.c_uint32),
                ("dwExtraInfo", ulong_ptr),
            )

        if ctypes.sizeof(ctypes.c_void_p) == 8:
            class INPUT(ctypes.Structure):
                _fields_ = (
                    ("type", ctypes.c_uint32),
                    ("_pad", ctypes.c_uint32),
                    ("mi", MOUSEINPUT),
                )
        else:
            class INPUT(ctypes.Structure):
                _fields_ = (("type", ctypes.c_uint32), ("mi", MOUSEINPUT))

        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi.dx = int(dx)
        inp.mi.dy = int(dy)
        inp.mi.dwFlags = MOUSEEVENTF_MOVE
        n = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        return int(n) == 1
    except Exception:
        return False


def _focus_hwnd(hwnd: int) -> None:
    if not hwnd or sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        if int(user32.GetForegroundWindow() or 0) == int(hwnd):
            return
        user32.SetForegroundWindow(int(hwnd))
    except Exception:
        return


class InputInjector:
    """OS key/mouse inject for local singleplayer closed loop. Disabled unless explicitly enabled."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._kb = None
        self._mouse = None
        self._keys_down: set[str] = set()
        self._btns_down: set[str] = set()
        self._acc_x = 0.0
        self._acc_y = 0.0
        if enabled:
            try:
                from pynput.keyboard import Controller as KeyCtl
                from pynput.mouse import Controller as MouseCtl

                self._kb = KeyCtl()
                self._mouse = MouseCtl()
            except Exception as exc:
                raise RuntimeError("pynput is required for inject") from exc

    @staticmethod
    def _key(name: str):
        from pynput.keyboard import Key, KeyCode

        alias = _KEY_ALIASES.get(name.upper())
        if alias is not None:
            return getattr(Key, alias, KeyCode.from_char(alias))
        return KeyCode.from_char(name.lower())

    def _set_key(self, name: str, want_down: bool) -> None:
        if want_down == (name in self._keys_down):
            return
        key = self._key(name)
        if want_down:
            self._kb.press(key)
            self._keys_down.add(name)
        else:
            self._kb.release(key)
            self._keys_down.discard(name)

    def _set_btn(self, name: str, want_down: bool) -> None:
        from pynput.mouse import Button

        if want_down == (name in self._btns_down):
            return
        btn = {"LMB": Button.left, "RMB": Button.right}[name]
        if want_down:
            self._mouse.press(btn)
            self._btns_down.add(name)
        else:
            self._mouse.release(btn)
            self._btns_down.discard(name)

    def inject(self, controls: dict, *, win=None, gui_open: bool = False) -> None:
        """Apply one tick of decoded controls.

        gui_open=True: mouse axes are absolute window-normalized cursor coords.
        gui_open=False: mouse axes are relative look deltas (pixels per tick).
        """
        if not self.enabled:
            return
        assert self._kb is not None and self._mouse is not None
        if win is not None:
            _focus_hwnd(int(getattr(win, "hwnd", 0) or 0))
        for name, state in controls["keys"].items():
            self._set_key(name, state in ("press", "hold"))
        mouse = controls["mouse"]
        if gui_open and win is not None and win.width > 0 and win.height > 0:
            x = win.left + float(mouse["nx"]) * win.width
            y = win.top + float(mouse["ny"]) * win.height
            self._mouse.position = (int(x), int(y))
            self._acc_x = 0.0
            self._acc_y = 0.0
        else:
            self._acc_x += float(mouse["dx"])
            self._acc_y += float(mouse["dy"])
            dx = int(self._acc_x)
            dy = int(self._acc_y)
            self._acc_x -= dx
            self._acc_y -= dy
            if dx or dy:
                if not _sendinput_relative(dx, dy):
                    self._mouse.move(dx, dy)
        for name, state in mouse["buttons"].items():
            self._set_btn(name, state in ("press", "hold"))

    def release_all(self) -> None:
        if not self.enabled:
            return
        for name in list(self._keys_down):
            self._set_key(name, False)
        for name in list(self._btns_down):
            self._set_btn(name, False)


def paced_loop(hz: float, fn: Callable[[], None], n_ticks: int | None = None) -> None:
    period = 1.0 / hz
    i = 0
    while n_ticks is None or i < n_ticks:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        extra = period - dt
        if extra > 0:
            time.sleep(extra)
        i += 1
