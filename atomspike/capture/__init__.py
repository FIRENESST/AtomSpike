"""Screen capture, human demo recording, and research-only local inject.

Compliance: only synthetic / ViZDoom / MineRL-style research envs. Do not
point the injector at online competitive games (ToS / EULA).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

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


class InputInjector:
    """OS key/mouse inject. Disabled unless explicitly enabled."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._kb = None
        self._mouse = None
        if enabled:
            try:
                from pynput.keyboard import Controller as KeyCtl
                from pynput.mouse import Controller as MouseCtl

                self._kb = KeyCtl()
                self._mouse = MouseCtl()
            except Exception as exc:
                raise RuntimeError("pynput is required for inject") from exc

    def inject(self, controls: dict) -> None:
        if not self.enabled:
            return
        assert self._kb is not None and self._mouse is not None
        from pynput.keyboard import KeyCode
        from pynput.mouse import Button

        for name, state in controls["keys"].items():
            key = KeyCode.from_char(name.lower())
            if state in ("press", "hold"):
                self._kb.press(key)
            else:
                self._kb.release(key)
        self._mouse.move(controls["mouse"]["dx"], controls["mouse"]["dy"])
        mapping = {"LMB": Button.left, "RMB": Button.right}
        for name, state in controls["mouse"]["buttons"].items():
            btn = mapping[name]
            if state in ("press", "hold"):
                self._mouse.press(btn)
            else:
                self._mouse.release(btn)


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
