"""OS key/mouse inject for local singleplayer closed loop.

Disabled unless explicitly enabled. Minecraft singleplayer / LAN only.
"""

from __future__ import annotations

import sys
import time

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
MOUSEEVENTF_WHEEL = 0x0800
INPUT_MOUSE = 0
WHEEL_DELTA = 120


def _mouse_input(dx: int = 0, dy: int = 0, mouse_data: int = 0, flags: int = 0):
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
    inp.mi.mouseData = int(mouse_data) & 0xFFFFFFFF
    inp.mi.dwFlags = int(flags)
    n = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    return int(n) == 1


def _sendinput_relative(dx: int, dy: int) -> bool:
    """Relative SendInput so Minecraft's WM_INPUT path sees the look delta."""
    if (dx == 0 and dy == 0) or sys.platform != "win32":
        return False
    try:
        return _mouse_input(dx, dy, 0, MOUSEEVENTF_MOVE)
    except Exception:
        return False


def _sendinput_wheel(notches: int) -> bool:
    if notches == 0 or sys.platform != "win32":
        return False
    try:
        return _mouse_input(0, 0, int(notches) * WHEEL_DELTA, MOUSEEVENTF_WHEEL)
    except Exception:
        return False


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
        self._acc_w = 0.0
        self._clipped = False
        self._gui_px: tuple[float, float] | None = None
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

    def inject(self, controls: dict, *, win=None, gui_open: bool = False, allow_buttons: bool = True) -> None:
        """Apply one tick of decoded controls.

        gui_open=True: mouse axes are absolute window-normalized cursor coords.
        gui_open=False: mouse axes are relative look deltas (pixels per tick).
        """
        if not self.enabled:
            return
        assert self._kb is not None and self._mouse is not None
        for name, state in controls["keys"].items():
            self._set_key(name, state in ("press", "hold"))
        mouse = controls["mouse"]
        if gui_open:
            self.release_clip()
        elif win is not None:
            self.confine(win)
        if gui_open and win is not None and win.width > 0 and win.height > 0:
            nx = min(1.0, max(0.0, float(mouse["nx"])))
            ny = min(1.0, max(0.0, float(mouse["ny"])))
            pad = 4.0
            tx = win.left + pad + nx * max(1.0, win.width - 2.0 * pad)
            ty = win.top + pad + ny * max(1.0, win.height - 2.0 * pad)
            try:
                cx, cy = self._mouse.position
                cx, cy = float(cx), float(cy)
            except Exception:
                cx, cy = tx, ty
            if self._gui_px is not None:
                gx, gy = self._gui_px
                cx = 0.55 * gx + 0.45 * cx
                cy = 0.55 * gy + 0.45 * cy
            dx = tx - cx
            dy = ty - cy
            dist = (dx * dx + dy * dy) ** 0.5
            max_step = max(28.0, 0.22 * max(float(win.width), float(win.height)))
            if dist > max_step and dist > 1e-3:
                s = max_step / dist
                dx *= s
                dy *= s
            px, py = cx + dx, cy + dy
            self._mouse.position = (int(round(px)), int(round(py)))
            self._gui_px = (px, py)
            self._acc_x = 0.0
            self._acc_y = 0.0
        else:
            self._gui_px = None
            self._acc_x += float(mouse["dx"])
            self._acc_y += float(mouse["dy"])
            dx = int(self._acc_x)
            dy = int(self._acc_y)
            self._acc_x -= dx
            self._acc_y -= dy
            if dx or dy:
                if not _sendinput_relative(dx, dy):
                    self._mouse.move(dx, dy)
        wheel = float(mouse.get("wheel", 0.0) or 0.0)
        self._acc_w += wheel
        notches = int(self._acc_w)
        self._acc_w -= notches
        if notches:
            if not _sendinput_wheel(notches):
                try:
                    self._mouse.scroll(0, int(notches))
                except Exception:
                    pass
        for name, state in mouse["buttons"].items():
            want = bool(allow_buttons) and state in ("press", "hold")
            self._set_btn(name, want)

    def confine(self, win) -> None:
        hwnd = int(win) if isinstance(win, int) else int(getattr(win, "hwnd", 0) or 0)
        if hwnd <= 0:
            return
        try:
            from atomspike.capture.window import clip_cursor_to_hwnd

            self._clipped = bool(clip_cursor_to_hwnd(hwnd))
        except Exception:
            self._clipped = False

    def release_clip(self) -> None:
        if not self._clipped:
            return
        try:
            from atomspike.capture.window import unclip_cursor

            unclip_cursor()
        except Exception:
            pass
        self._clipped = False

    def tap_key(self, name: str) -> None:
        if not self.enabled or self._kb is None:
            return
        key = self._key(name)
        try:
            self._kb.press(key)
            time.sleep(0.03)
            self._kb.release(key)
        except Exception:
            pass

    def release_all(self) -> None:
        if not self.enabled:
            return
        for name in list(self._keys_down):
            self._set_key(name, False)
        for name in list(self._btns_down):
            self._set_btn(name, False)
        self._gui_px = None
