"""Win32 overlay that never becomes the foreground window.

pygame/SDL ShowWindow + flip() steal focus from Minecraft. This path creates a
WS_EX_NOACTIVATE | WS_EX_TOPMOST tool window and blits RGB with GDI.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

if sys.platform != "win32":
    raise ImportError("Win32Overlay is Windows only")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
gdi32 = ctypes.windll.gdi32

LRESULT = ctypes.c_ssize_t
UINT_PTR = ctypes.c_size_t
LONG_PTR = ctypes.c_ssize_t
HANDLE_PTR = ctypes.c_void_p

WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_QUIT = 0x0012
WM_ERASEBKGND = 0x0014
WM_ACTIVATE = 0x0006
WM_PAINT = 0x000F
WM_MOUSEACTIVATE = 0x0021
WM_NCACTIVATE = 0x0086
WM_LBUTTONDOWN = 0x0201
MA_NOACTIVATE = 3
PM_REMOVE = 1
SW_SHOWNOACTIVATE = 4
SW_RESTORE = 9
GA_ROOT = 2
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

WS_POPUP = 0x80000000
WS_CAPTION = 0x00C00000
WS_SYSMENU = 0x00080000
WS_CLIPCHILDREN = 0x02000000
WS_CLIPSIBLINGS = 0x04000000

WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

CS_HREDRAW = 0x0002
CS_VREDRAW = 0x0001

IDC_ARROW = 32512
SM_CXSCREEN = 0

WNDPROCTYPE = ctypes.WINFUNCTYPE(LRESULT, HANDLE_PTR, wintypes.UINT, UINT_PTR, LONG_PTR)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROCTYPE),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", HANDLE_PTR),
        ("hCursor", HANDLE_PTR),
        ("hbrBackground", HANDLE_PTR),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


user32.DefWindowProcW.argtypes = [HANDLE_PTR, wintypes.UINT, UINT_PTR, LONG_PTR]
user32.DefWindowProcW.restype = LRESULT
user32.PeekMessageW.argtypes = [
    ctypes.POINTER(wintypes.MSG),
    HANDLE_PTR,
    wintypes.UINT,
    wintypes.UINT,
    wintypes.UINT,
]
user32.PeekMessageW.restype = wintypes.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.TranslateMessage.restype = wintypes.BOOL
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.restype = LRESULT
user32.PostMessageW.argtypes = [HANDLE_PTR, wintypes.UINT, UINT_PTR, LONG_PTR]
user32.PostMessageW.restype = wintypes.BOOL
user32.PostQuitMessage.argtypes = [ctypes.c_int]
user32.PostQuitMessage.restype = None
user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.RegisterClassW.restype = wintypes.ATOM
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    HANDLE_PTR,
    HANDLE_PTR,
    wintypes.HINSTANCE,
    ctypes.c_void_p,
]
user32.CreateWindowExW.restype = HANDLE_PTR
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.DestroyWindow.argtypes = [HANDLE_PTR]
user32.DestroyWindow.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [HANDLE_PTR, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [
    HANDLE_PTR,
    HANDLE_PTR,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowPos.restype = wintypes.BOOL
user32.GetDC.argtypes = [HANDLE_PTR]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [HANDLE_PTR, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.BeginPaint.argtypes = [HANDLE_PTR, ctypes.c_void_p]
user32.BeginPaint.restype = wintypes.HDC
user32.EndPaint.argtypes = [HANDLE_PTR, ctypes.c_void_p]
user32.EndPaint.restype = wintypes.BOOL
user32.GetClientRect.argtypes = [HANDLE_PTR, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.AdjustWindowRectEx.argtypes = [
    ctypes.POINTER(wintypes.RECT),
    wintypes.DWORD,
    wintypes.BOOL,
    wintypes.DWORD,
]
user32.AdjustWindowRectEx.restype = wintypes.BOOL
user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
user32.LoadCursorW.restype = HANDLE_PTR
user32.GetForegroundWindow.restype = HANDLE_PTR
user32.SetForegroundWindow.argtypes = [HANDLE_PTR]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.IsWindow.argtypes = [HANDLE_PTR]
user32.IsWindow.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [HANDLE_PTR, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.AttachThreadInput.restype = wintypes.BOOL
user32.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]
user32.AllowSetForegroundWindow.restype = wintypes.BOOL
kernel32.GetCurrentThreadId.restype = wintypes.DWORD
kernel32.GetConsoleWindow.restype = HANDLE_PTR
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
user32.ClientToScreen.argtypes = [HANDLE_PTR, ctypes.POINTER(wintypes.POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.GetAncestor.argtypes = [HANDLE_PTR, wintypes.UINT]
user32.GetAncestor.restype = HANDLE_PTR
user32.IsIconic.argtypes = [HANDLE_PTR]
user32.IsIconic.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = [HANDLE_PTR]
user32.BringWindowToTop.restype = wintypes.BOOL
user32.SendInput.argtypes = [wintypes.UINT, ctypes.c_void_p, ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
gdi32.StretchDIBits.argtypes = [
    wintypes.HDC,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.UINT,
    wintypes.DWORD,
]
gdi32.StretchDIBits.restype = ctypes.c_int


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
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 1)]


class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [
        ("hdc", wintypes.HDC),
        ("fErase", wintypes.BOOL),
        ("rcPaint", wintypes.RECT),
        ("fRestore", wintypes.BOOL),
        ("fIncUpdate", wintypes.BOOL),
        ("rgbReserved", wintypes.BYTE * 32),
    ]


def _hwnd_int(v) -> int:
    if v is None:
        return 0
    if isinstance(v, ctypes.c_void_p):
        return int(v.value or 0)
    return int(v or 0)


def _u64(v) -> int:
    if v is None:
        return 0
    if isinstance(v, ctypes.c_void_p):
        return int(v.value or 0) & 0xFFFFFFFFFFFFFFFF
    return int(v) & 0xFFFFFFFFFFFFFFFF


def _signed_ptr(v) -> int:
    x = _u64(v)
    if x >= 0x8000000000000000:
        x -= 0x10000000000000000
    return x


def _pid_of(hwnd) -> int:
    hwnd = _hwnd_int(hwnd)
    if not hwnd:
        return 0
    dummy = wintypes.DWORD()
    user32.GetWindowThreadProcessId(HANDLE_PTR(hwnd), ctypes.byref(dummy))
    return int(dummy.value)


def _shell_pids() -> set[int]:
    pids = {int(os.getpid())}
    try:
        pids.add(int(os.getppid()))
    except Exception:
        pass
    try:
        con = _hwnd_int(kernel32.GetConsoleWindow())
        if con:
            pids.add(_pid_of(con))
            root = _hwnd_int(user32.GetAncestor(HANDLE_PTR(con), GA_ROOT))
            if root:
                pids.add(_pid_of(root))
    except Exception:
        pass
        return {p for p in pids if p}


def detach_console_keep_stdio() -> bool:
    """Detach this process from cmd so later windows cannot pull the console forward."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    out_fd = err_fd = None
    try:
        out_fd = os.dup(1)
        err_fd = os.dup(2)
    except OSError:
        pass
    try:
        ok = bool(kernel32.FreeConsole())
    except Exception:
        ok = False
    if out_fd is not None:
        try:
            sys.stdout = open(out_fd, "w", encoding="utf-8", errors="replace", closefd=True, buffering=1)
            sys.stderr = open(err_fd, "w", encoding="utf-8", errors="replace", closefd=True, buffering=1)
        except Exception:
            pass
    return ok


def _pythonw_executable() -> str:
    cand = Path(sys.executable).with_name("pythonw.exe")
    if cand.is_file():
        return str(cand)
    return sys.executable


def foreground_is_shell() -> bool:
    fg = _hwnd_int(user32.GetForegroundWindow())
    if not fg:
        return False
    return _pid_of(fg) in _shell_pids()


def _nudge_foreground_lock() -> None:
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
            _fields_ = (("type", ctypes.c_uint32), ("_pad", ctypes.c_uint32), ("mi", MOUSEINPUT))
    else:
        class INPUT(ctypes.Structure):
            _fields_ = (("type", ctypes.c_uint32), ("mi", MOUSEINPUT))

    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dwFlags = MOUSEEVENTF_MOVE
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def _park_cursor_in_game(hwnd: int) -> None:
    rc = wintypes.RECT()
    if not user32.GetClientRect(HANDLE_PTR(hwnd), ctypes.byref(rc)):
        return
    pt = wintypes.POINT(int((rc.right - rc.left) // 2), int((rc.bottom - rc.top) // 2))
    if not user32.ClientToScreen(HANDLE_PTR(hwnd), ctypes.byref(pt)):
        return
    user32.SetCursorPos(int(pt.x), int(pt.y))


def restore_game_focus(hwnd, *, park_cursor: bool = False) -> None:
    """Pull foreground (and optionally the cursor) back from cmd/Windows Terminal."""
    hwnd = _hwnd_int(hwnd)
    if not hwnd:
        return
    try:
        if not user32.IsWindow(HANDLE_PTR(hwnd)):
            return
        fg = _hwnd_int(user32.GetForegroundWindow())
        already = fg == hwnd
        if already and not park_cursor:
            return
        try:
            user32.AllowSetForegroundWindow(0xFFFFFFFF)
        except Exception:
            pass
        _nudge_foreground_lock()
        our = int(kernel32.GetCurrentThreadId())
        dummy = wintypes.DWORD()
        fg_tid = int(user32.GetWindowThreadProcessId(HANDLE_PTR(fg), ctypes.byref(dummy))) if fg else 0
        game_tid = int(user32.GetWindowThreadProcessId(HANDLE_PTR(hwnd), ctypes.byref(dummy)))
        attached_fg = False
        attached_game = False
        if fg_tid and fg_tid != our:
            attached_fg = bool(user32.AttachThreadInput(our, fg_tid, True))
        if game_tid and game_tid != our and game_tid != fg_tid:
            attached_game = bool(user32.AttachThreadInput(our, game_tid, True))
        if user32.IsIconic(HANDLE_PTR(hwnd)):
            user32.ShowWindow(HANDLE_PTR(hwnd), SW_RESTORE)
        user32.BringWindowToTop(HANDLE_PTR(hwnd))
        user32.SetForegroundWindow(HANDLE_PTR(hwnd))
        if attached_game:
            user32.AttachThreadInput(our, game_tid, False)
        if attached_fg:
            user32.AttachThreadInput(our, fg_tid, False)
        if park_cursor:
            _park_cursor_in_game(hwnd)
    except Exception:
        return


class Win32Overlay:
    """Owned by a daemon thread that both pumps messages and presents frames."""

    def __init__(
        self,
        *,
        title: str,
        width: int,
        height: int,
        x: int | None,
        y: int,
        hz: float,
        game_hwnd: int = 0,
    ):
        self.title = str(title)
        self.width = max(8, int(width))
        self.height = max(8, int(height))
        self.x = x
        self.y = int(y)
        self.hz = max(10.0, float(hz))
        self.game_hwnd = int(game_hwnd or 0)
        self.alive = False
        self._ok = False
        self._hwnd = 0
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._composer = None
        self._bgra: np.ndarray | None = None
        self._last: np.ndarray | None = None
        self._wndproc = WNDPROCTYPE(self._wnd_proc)
        self._wc = None
        self._class_name = f"AtomSpikeOverlay{id(self)}"

    def start(self, composer) -> bool:
        self._composer = composer
        self._ready.clear()
        self._thread = threading.Thread(target=self._loop, name="atomspike-overlay", daemon=True)
        self._thread.start()
        self._ready.wait(2.5)
        if self._ok:
            for i in range(15):
                restore_game_focus(self.game_hwnd, park_cursor=(i == 0))
                if _hwnd_int(user32.GetForegroundWindow()) == int(self.game_hwnd):
                    break
                time.sleep(0.04)
            return True
        self.close()
        return False

    def close(self) -> None:
        self.alive = False
        hwnd = self._hwnd
        if hwnd:
            try:
                user32.PostMessageW(HANDLE_PTR(hwnd), WM_CLOSE, 0, 0)
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._hwnd = 0

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        try:
            msg = int(msg)
            if msg == WM_MOUSEACTIVATE:
                return MA_NOACTIVATE
            if msg == WM_LBUTTONDOWN:
                x = int(_u64(lparam) & 0xFFFF)
                y = int((_u64(lparam) >> 16) & 0xFFFF)
                if x >= max(8, self.width - 40) and y <= 40:
                    self.alive = False
                    user32.DestroyWindow(hwnd)
                    return 0
            if msg == WM_NCACTIVATE:
                return user32.DefWindowProcW(hwnd, msg, 0, _signed_ptr(lparam))
            if msg == WM_ACTIVATE:
                return 0
            if msg == WM_ERASEBKGND:
                return 1
            if msg == WM_PAINT:
                ps = PAINTSTRUCT()
                hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
                self._blit(hdc)
                user32.EndPaint(hwnd, ctypes.byref(ps))
                return 0
            if msg == WM_CLOSE:
                self.alive = False
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                self.alive = False
                self._hwnd = 0
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, _u64(wparam), _signed_ptr(lparam))
        except Exception:
            return 0

    def _create(self) -> int:
        inst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSW()
        wc.style = CS_HREDRAW | CS_VREDRAW
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = inst
        wc.hCursor = user32.LoadCursorW(None, ctypes.cast(IDC_ARROW, wintypes.LPCWSTR))
        wc.lpszClassName = self._class_name
        self._wc = wc
        user32.RegisterClassW(ctypes.byref(wc))
        style = WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS
        ex = WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
        outer_w = int(self.width)
        outer_h = int(self.height)
        if self.x is None:
            screen_w = int(user32.GetSystemMetrics(SM_CXSCREEN) or 1280)
            pos_x = max(16, screen_w - outer_w - 24)
        else:
            pos_x = int(self.x)
        pos_y = int(self.y)
        hwnd_p = user32.CreateWindowExW(
            ex,
            self._class_name,
            self.title,
            style,
            pos_x,
            pos_y,
            outer_w,
            outer_h,
            None,
            None,
            inst,
            None,
        )
        hwnd = _hwnd_int(hwnd_p)
        if not hwnd:
            err = int(ctypes.GetLastError() or 0)
            print(f"[overlay] CreateWindowExW failed winerr={err}", flush=True)
            return 0
        user32.SetWindowPos(
            HANDLE_PTR(hwnd),
            HANDLE_PTR(HWND_TOPMOST),
            pos_x,
            pos_y,
            outer_w,
            outer_h,
            SWP_NOACTIVATE,
        )
        user32.ShowWindow(HANDLE_PTR(hwnd), SW_SHOWNOACTIVATE)
        return hwnd

    def _to_bgra(self, rgb: np.ndarray) -> np.ndarray:
        src = np.ascontiguousarray(rgb, dtype=np.uint8)
        if src.ndim != 3 or src.shape[2] < 3:
            raise ValueError("overlay frame must be HxWx3")
        h, w = int(src.shape[0]), int(src.shape[1])
        if self._bgra is None or self._bgra.shape[0] != h or self._bgra.shape[1] != w:
            self._bgra = np.empty((h, w, 4), dtype=np.uint8)
            self._bgra[:, :, 3] = 255
        self._bgra[:, :, 0] = src[:, :, 2]
        self._bgra[:, :, 1] = src[:, :, 1]
        self._bgra[:, :, 2] = src[:, :, 0]
        return self._bgra

    def _blit(self, hdc) -> None:
        frame = self._last
        if frame is None or not hdc:
            return
        h, w = int(frame.shape[0]), int(frame.shape[1])
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = w
        info.bmiHeader.biHeight = -h
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        bits = frame.ctypes.data_as(ctypes.c_void_p)
        gdi32.StretchDIBits(
            hdc,
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            w,
            h,
            bits,
            ctypes.byref(info),
            DIB_RGB_COLORS,
            SRCCOPY,
        )

    def _present(self, rgb: np.ndarray) -> None:
        hwnd = self._hwnd
        if not hwnd:
            return
        self._last = self._to_bgra(rgb)
        hdc = user32.GetDC(HANDLE_PTR(hwnd))
        try:
            self._blit(hdc)
        finally:
            user32.ReleaseDC(HANDLE_PTR(hwnd), hdc)

    def _blit_shm(self) -> None:
        buf = self._shm_buf
        n = int(self.width) * int(self.height) * 4
        lock = getattr(self, "_shm_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            src = np.frombuffer(buf.get_obj(), dtype=np.uint8, count=n).reshape(self.height, self.width, 4)
            if self._last is None or self._last.shape != src.shape:
                self._last = np.empty_like(src)
            np.copyto(self._last, src)
        finally:
            if lock is not None:
                lock.release()
        hwnd = self._hwnd
        if not hwnd:
            return
        hdc = user32.GetDC(HANDLE_PTR(hwnd))
        try:
            self._blit(hdc)
        finally:
            user32.ReleaseDC(HANDLE_PTR(hwnd), hdc)

    def _drain(self) -> None:
        msg = wintypes.MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            if int(msg.message) == WM_QUIT:
                self.alive = False
                return
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _loop(self) -> None:
        try:
            hwnd = self._create()
            if not hwnd:
                self._ok = False
                self.alive = False
                if getattr(self, "_alive_v", None) is not None:
                    self._alive_v.value = 0
                if getattr(self, "_ready_evt", None) is not None:
                    self._ready_evt.set()
                else:
                    self._ready.set()
                return
            self._hwnd = hwnd
            self._ok = True
            self.alive = True
            if getattr(self, "_alive_v", None) is not None:
                self._alive_v.value = 1
            if getattr(self, "_ready_evt", None) is not None:
                self._ready_evt.set()
            else:
                self._ready.set()
            period = 1.0 / self.hz
            next_t = time.perf_counter()
            while self.alive:
                stop = getattr(self, "_stop_evt", None)
                if stop is not None and stop.is_set():
                    break
                self._drain()
                if not self.alive:
                    break
                if getattr(self, "_shm_buf", None) is not None:
                    try:
                        self._blit_shm()
                    except Exception:
                        pass
                elif self._composer is not None:
                    try:
                        rgb = self._composer()
                        if rgb is not None:
                            self._present(rgb)
                    except Exception:
                        pass
                next_t += period
                leftover = next_t - time.perf_counter()
                if leftover > 0:
                    time.sleep(leftover)
                else:
                    next_t = time.perf_counter()
        except Exception:
            self._ok = False
            self.alive = False
            if getattr(self, "_alive_v", None) is not None:
                self._alive_v.value = 0
            if getattr(self, "_ready_evt", None) is not None:
                self._ready_evt.set()
            else:
                self._ready.set()
        finally:
            if getattr(self, "_alive_v", None) is not None:
                self._alive_v.value = 0
            hwnd = self._hwnd
            self._hwnd = 0
            self.alive = False
            if hwnd:
                try:
                    if user32.IsWindow(HANDLE_PTR(hwnd)):
                        user32.DestroyWindow(HANDLE_PTR(hwnd))
                except Exception:
                    pass


def _overlay_child_main(title, width, height, x, y, hz, buf, seq, alive_v, ready, stop, lock):
    """Runs in a pythonw/detached process. Must not import torch."""
    try:
        try:
            kernel32.FreeConsole()
        except Exception:
            pass
        try:
            con = _hwnd_int(kernel32.GetConsoleWindow())
            if con:
                user32.ShowWindow(HANDLE_PTR(con), 0)
        except Exception:
            pass
        ov = Win32Overlay(
            title=title,
            width=int(width),
            height=int(height),
            x=x,
            y=int(y),
            hz=float(hz),
            game_hwnd=0,
        )
        ov._shm_buf = buf
        ov._shm_seq = seq
        ov._shm_lock = lock
        ov._alive_v = alive_v
        ov._ready_evt = ready
        ov._stop_evt = stop
        ov._loop()
    except Exception:
        try:
            alive_v.value = 0
            ready.set()
        except Exception:
            pass


class DetachedOverlay:
    """Parent-side overlay: the HWND lives in another process so cmd cannot steal focus."""

    def __init__(
        self,
        *,
        title: str,
        width: int,
        height: int,
        x: int | None,
        y: int,
        hz: float,
        game_hwnd: int = 0,
    ):
        self.title = str(title)
        self.width = max(8, int(width))
        self.height = max(8, int(height))
        self.x = x
        self.y = int(y)
        self.hz = max(10.0, float(hz))
        self.game_hwnd = int(game_hwnd or 0)
        self._alive_flag = False
        self._composer = None
        self._proc = None
        self._feed = None
        self._stop_feed = threading.Event()
        self._ctx = mp.get_context("spawn")
        n = self.width * self.height * 4
        self._buf = self._ctx.Array("B", n)
        self._seq = self._ctx.Value("Q", 0)
        self._alive_v = self._ctx.Value("i", 0)
        self._ready = self._ctx.Event()
        self._stop = self._ctx.Event()
        self._lock = self._ctx.Lock()
        self._bgra = np.empty((self.height, self.width, 4), dtype=np.uint8)
        self._bgra[:, :, 3] = 255

    @property
    def alive(self) -> bool:
        try:
            child = int(self._alive_v.value) == 1
        except Exception:
            child = False
        return bool(self._alive_flag) and child

    def start(self, composer) -> bool:
        self._composer = composer
        saved_exe = sys.executable
        try:
            mp.set_executable(_pythonw_executable())
            self._proc = self._ctx.Process(
                target=_overlay_child_main,
                args=(
                    self.title,
                    self.width,
                    self.height,
                    self.x,
                    self.y,
                    self.hz,
                    self._buf,
                    self._seq,
                    self._alive_v,
                    self._ready,
                    self._stop,
                    self._lock,
                ),
                daemon=True,
                name="atomspike-overlay",
            )
            self._proc.start()
        finally:
            try:
                mp.set_executable(saved_exe)
            except Exception:
                pass
        if not self._ready.wait(5.0) or int(self._alive_v.value) != 1:
            self.close()
            return False
        self._alive_flag = True
        self._stop_feed.clear()
        self._feed = threading.Thread(target=self._feed_loop, name="overlay-feed", daemon=True)
        self._feed.start()
        return True

    def _feed_loop(self) -> None:
        period = 1.0 / self.hz
        next_t = time.perf_counter()
        n = self.width * self.height * 4
        while not self._stop_feed.is_set() and not self._stop.is_set():
            if self._composer is not None:
                try:
                    rgb = self._composer()
                    if rgb is not None:
                        src = np.ascontiguousarray(rgb, dtype=np.uint8)
                        if src.shape[0] == self.height and src.shape[1] == self.width and src.shape[2] >= 3:
                            self._bgra[:, :, 0] = src[:, :, 2]
                            self._bgra[:, :, 1] = src[:, :, 1]
                            self._bgra[:, :, 2] = src[:, :, 0]
                            dest = np.frombuffer(self._buf.get_obj(), dtype=np.uint8, count=n).reshape(
                                self.height, self.width, 4
                            )
                            with self._lock:
                                np.copyto(dest, self._bgra)
                                self._seq.value = int(self._seq.value) + 1
                except Exception:
                    pass
            if int(self._alive_v.value) != 1:
                break
            next_t += period
            leftover = next_t - time.perf_counter()
            if leftover > 0:
                time.sleep(leftover)
            else:
                next_t = time.perf_counter()
        self._alive_flag = False

    def close(self) -> None:
        self._alive_flag = False
        self._stop_feed.set()
        try:
            self._stop.set()
        except Exception:
            pass
        if self._feed is not None and self._feed.is_alive():
            self._feed.join(timeout=0.8)
        self._feed = None
        proc = self._proc
        self._proc = None
        if proc is not None and proc.is_alive():
            proc.join(timeout=1.2)
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass

