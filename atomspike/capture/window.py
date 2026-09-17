"""Find and grab a game window (Windows client area; mss fallback)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_MAX_W = 7680
_MAX_H = 4320


def _set_dpi_aware() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_set_dpi_aware()


def _sanitize_box(box: dict) -> dict:
    w = max(8, min(int(box.get("width", 0)), _MAX_W))
    h = max(8, min(int(box.get("height", 0)), _MAX_H))
    return {"left": int(box.get("left", 0)), "top": int(box.get("top", 0)), "width": w, "height": h}


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def list_windows(title_substr: str = "") -> list[WindowInfo]:
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lp):  # type: ignore[misc]
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        if title_substr and title_substr.lower() not in title.lower():
            return True
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        w = int(rect.right - rect.left)
        h = int(rect.bottom - rect.top)
        if w < 64 or h < 64:
            return True
        found.append(WindowInfo(int(hwnd), title, int(pt.x), int(pt.y), w, h))
        return True

    user32.EnumWindows(_cb, 0)
    found.sort(key=lambda w: (-w.width * w.height, w.title))
    return found


def pick_minecraft_window(title_substr: str = "Minecraft") -> WindowInfo | None:
    hits = list_windows(title_substr)
    if hits:
        return hits[0]
    for alt in ("Lunar Client", "Badlion", "Feather", "Labymod", "Prism Launcher", "MultiMC"):
        hits = list_windows(alt)
        if hits:
            return hits[0]
    return None


class WindowGrabber:
    """30fps-capable BGR→RGB grab of a client rectangle."""

    def __init__(
        self,
        title_substr: str = "Minecraft",
        monitor: int = 1,
        region: tuple[int, int, int, int] | None = None,
        desktop: bool = False,
        dest_size: tuple[int, int] | None = None,
    ):
        self.title_substr = title_substr
        self.monitor = monitor
        self.region = region
        self.desktop = desktop
        self.dest_size = None if dest_size is None else (max(8, int(dest_size[0])), max(8, int(dest_size[1])))
        self.window: WindowInfo | None = None
        self._mss = None
        self._gdi: _WinStretchGrab | None = None
        self._geom_i = 0
        try:
            import mss

            self._mss = mss.mss()
        except Exception as exc:
            if sys.platform != "win32":
                raise RuntimeError(
                    'Window capture needs mss. Install with: pip install "atomspike[capture]"'
                ) from exc
        if sys.platform == "win32":
            self._gdi = _WinStretchGrab()
        self._refresh_target()

    def _refresh_target(self) -> None:
        if self.desktop or self.region is not None:
            self.window = None
            return
        self.window = pick_minecraft_window(self.title_substr)

    def grab(self) -> tuple[NDArray[np.uint8], int]:
        t_ns = time.time_ns()
        self._geom_i += 1
        box = _sanitize_box(self._box(refresh=self._geom_i % 8 == 1))
        dw, dh = (self.dest_size or (int(box["width"]), int(box["height"])))
        if self._gdi is not None and self.window is not None and self.region is None and not self.desktop:
            rgb = self._gdi.grab(self.window.hwnd, int(box["width"]), int(box["height"]), dw, dh)
            if rgb is not None:
                return rgb, t_ns
        if self._mss is None:
            raise RuntimeError("window grab failed (no GDI bitmap and no mss)")
        try:
            raw = np.asarray(self._mss.grab(box))[:, :, :3][:, :, ::-1]
        except Exception:
            self._refresh_target()
            box = _sanitize_box(self._box(refresh=True))
            raw = np.asarray(self._mss.grab(box))[:, :, :3][:, :, ::-1]
        raw = np.ascontiguousarray(raw)
        if raw.shape[1] != dw or raw.shape[0] != dh:
            raw = _resize_rgb(raw, dw, dh)
        return raw, t_ns

    def _box(self, refresh: bool = True) -> dict:
        if self.region is not None:
            x, y, w, h = self.region
            return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
        if self.window is not None:
            if refresh:
                refreshed = _hwnd_geometry(self.window.hwnd)
                if refreshed is not None:
                    self.window = refreshed
            w = self.window
            return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}
        mon = self._mss.monitors[self.monitor] if self._mss is not None else {"left": 0, "top": 0, "width": 160, "height": 90}
        return {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}

    def close(self) -> None:
        if self._gdi is not None:
            self._gdi.close()
            self._gdi = None
        if self._mss is not None:
            self._mss.close()
            self._mss = None


def _resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(frame)
    return np.asarray(img.resize((int(width), int(height)), Image.BILINEAR), dtype=np.uint8)


class _WinStretchGrab:
    """StretchBlt the window client into dest_size. Avoids full-res mss copies."""

    SRCCOPY = 0x00CC0020
    COLORONCOLOR = 3
    DIB_RGB_COLORS = 0

    def __init__(self) -> None:
        self._hwnd = 0
        self._dw = 0
        self._dh = 0
        self._hdc_win = None
        self._hdc_mem = None
        self._hbmp = None
        self._old = None
        self._bits = None
        self._buf: np.ndarray | None = None

    def grab(self, hwnd: int, src_w: int, src_h: int, dest_w: int, dest_h: int) -> np.ndarray | None:
        if hwnd <= 0 or src_w < 8 or src_h < 8 or dest_w < 8 or dest_h < 8:
            return None
        try:
            self._ensure(hwnd, dest_w, dest_h)
            gdi32 = self._gdi32
            if not gdi32.StretchBlt(
                self._hdc_mem,
                0,
                0,
                dest_w,
                dest_h,
                self._hdc_win,
                0,
                0,
                src_w,
                src_h,
                self.SRCCOPY,
            ):
                return None
            assert self._buf is not None
            # 32-bit DIB is BGRA, top-down.
            return np.ascontiguousarray(self._buf[:, :, :3][:, :, ::-1])
        except Exception:
            self.close()
            return None

    def _ensure(self, hwnd: int, dest_w: int, dest_h: int) -> None:
        if self._hwnd == hwnd and self._dw == dest_w and self._dh == dest_h and self._buf is not None:
            return
        self.close()
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        self._user32 = user32
        self._gdi32 = gdi32
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        hdc_win = user32.GetDC(hwnd)
        if not hdc_win:
            raise OSError("GetDC failed")
        hdc_mem = gdi32.CreateCompatibleDC(hdc_win)
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = int(dest_w)
        bmi.bmiHeader.biHeight = -int(dest_h)
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0
        ptr = ctypes.c_void_p()
        hbmp = gdi32.CreateDIBSection(
            hdc_mem,
            ctypes.byref(bmi),
            self.DIB_RGB_COLORS,
            ctypes.byref(ptr),
            None,
            0,
        )
        if not hbmp or not ptr.value:
            user32.ReleaseDC(hwnd, hdc_win)
            if hdc_mem:
                gdi32.DeleteDC(hdc_mem)
            raise OSError("CreateDIBSection failed")
        old = gdi32.SelectObject(hdc_mem, hbmp)
        gdi32.SetStretchBltMode(hdc_mem, self.COLORONCOLOR)
        nbytes = int(dest_w) * int(dest_h) * 4
        buf = (ctypes.c_char * nbytes).from_address(int(ptr.value))
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(int(dest_h), int(dest_w), 4)
        self._hwnd = int(hwnd)
        self._dw = int(dest_w)
        self._dh = int(dest_h)
        self._hdc_win = hdc_win
        self._hdc_mem = hdc_mem
        self._hbmp = hbmp
        self._old = old
        self._bits = ptr
        self._buf = arr

    def close(self) -> None:
        try:
            import ctypes  # noqa: F401

            gdi32 = getattr(self, "_gdi32", None)
            user32 = getattr(self, "_user32", None)
            if gdi32 is not None and self._hdc_mem and self._old:
                gdi32.SelectObject(self._hdc_mem, self._old)
            if gdi32 is not None and self._hbmp:
                gdi32.DeleteObject(self._hbmp)
            if gdi32 is not None and self._hdc_mem:
                gdi32.DeleteDC(self._hdc_mem)
            if user32 is not None and self._hdc_win and self._hwnd:
                user32.ReleaseDC(self._hwnd, self._hdc_win)
        except Exception:
            pass
        self._hwnd = 0
        self._dw = 0
        self._dh = 0
        self._hdc_win = None
        self._hdc_mem = None
        self._hbmp = None
        self._old = None
        self._bits = None
        self._buf = None


def _hwnd_geometry(hwnd: int) -> WindowInfo | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd):
        return None
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    w = int(rect.right - rect.left)
    h = int(rect.bottom - rect.top)
    if w < 8 or h < 8:
        return None
    return WindowInfo(hwnd, buf.value, int(pt.x), int(pt.y), w, h)


def client_cursor_norm(win: WindowInfo | None) -> tuple[float, float]:
    """OS cursor in the game client rectangle, each axis in [0, 1]."""
    if win is None or win.width <= 0 or win.height <= 0:
        return 0.5, 0.5
    if sys.platform != "win32":
        return 0.5, 0.5
    import ctypes
    from ctypes import wintypes

    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    nx = (float(pt.x) - float(win.left)) / float(win.width)
    ny = (float(pt.y) - float(win.top)) / float(win.height)
    return float(max(0.0, min(1.0, nx))), float(max(0.0, min(1.0, ny)))
