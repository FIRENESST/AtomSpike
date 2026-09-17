"""Windows raw mouse (WM_INPUT) so Minecraft look deltas are recorded.

64-bit Python must prototype DefWindowProcW / WNDPROC with pointer-sized
WPARAM/LPARAM. Unprototyped windll calls treat them as 32-bit int and overflow.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes

if sys.platform != "win32":
    raise ImportError("WinRawMouse is Windows only")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
WM_INPUT = 0x00FF
WM_DESTROY = 0x0002
HWND_MESSAGE = ctypes.c_void_p(-3)
WS_POPUP = 0x80000000

# Pointer-sized Win32 types. wintypes.LPARAM can still be 32-bit on some builds.
LRESULT = ctypes.c_ssize_t
UINT_PTR = ctypes.c_size_t
LONG_PTR = ctypes.c_ssize_t
HANDLE_PTR = ctypes.c_void_p

user32.DefWindowProcW.argtypes = [HANDLE_PTR, wintypes.UINT, UINT_PTR, LONG_PTR]
user32.DefWindowProcW.restype = LRESULT
user32.GetRawInputData.argtypes = [
    HANDLE_PTR,
    wintypes.UINT,
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.UINT),
    wintypes.UINT,
]
user32.GetRawInputData.restype = wintypes.UINT
user32.PostQuitMessage.argtypes = [ctypes.c_int]
user32.PostQuitMessage.restype = None
user32.PostMessageW.argtypes = [HANDLE_PTR, wintypes.UINT, UINT_PTR, LONG_PTR]
user32.PostMessageW.restype = wintypes.BOOL
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), HANDLE_PTR, wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype = wintypes.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.TranslateMessage.restype = wintypes.BOOL
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.restype = LRESULT
user32.RegisterClassW.restype = wintypes.ATOM
user32.CreateWindowExW.restype = HANDLE_PTR
user32.RegisterRawInputDevices.restype = wintypes.BOOL


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


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", HANDLE_PTR),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", HANDLE_PTR),
        ("wParam", UINT_PTR),
    ]


class _RawMouseButtons(ctypes.Structure):
    _fields_ = [("usButtonFlags", wintypes.USHORT), ("usButtonData", wintypes.USHORT)]


class _RawMouseUnion(ctypes.Union):
    _fields_ = [("ulButtons", wintypes.ULONG), ("buttons", _RawMouseButtons)]


class RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("u", _RawMouseUnion),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]


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


class WinRawMouse:
    def __init__(self):
        self._dx = 0
        self._dy = 0
        self._lock = threading.Lock()
        self._hwnd = None
        self._alive = True
        self._thread: threading.Thread | None = None
        self._wndproc = WNDPROCTYPE(self._wnd_proc)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._pump, name="mc-raw-mouse", daemon=True)
        self._thread.start()

    def consume(self) -> tuple[float, float]:
        with self._lock:
            d = (float(self._dx), float(self._dy))
            self._dx = 0
            self._dy = 0
            return d

    def close(self) -> None:
        self._alive = False
        if self._hwnd:
            try:
                user32.PostMessageW(self._hwnd, WM_DESTROY, 0, 0)
            except Exception:
                pass

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == WM_INPUT:
                self._on_input(lparam)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, int(msg), _u64(wparam), _signed_ptr(lparam))
        except Exception:
            return 0

    def _on_input(self, lparam) -> None:
        handle = HANDLE_PTR(_u64(lparam))
        size = wintypes.UINT(0)
        user32.GetRawInputData(handle, RID_INPUT, None, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
        if size.value < ctypes.sizeof(RAWINPUTHEADER):
            return
        buf = ctypes.create_string_buffer(size.value)
        got = user32.GetRawInputData(handle, RID_INPUT, buf, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
        if got == 0 or got == 0xFFFFFFFF:
            return
        ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
        if ri.header.dwType != RIM_TYPEMOUSE:
            return
        with self._lock:
            self._dx += int(ri.mouse.lLastX)
            self._dy += int(ri.mouse.lLastY)

    def _pump(self) -> None:
        try:
            class_name = "AtomSpikeRawMouse"
            inst = kernel32.GetModuleHandleW(None)
            wc = WNDCLASSW()
            wc.lpfnWndProc = self._wndproc
            wc.hInstance = inst
            wc.lpszClassName = class_name
            user32.RegisterClassW(ctypes.byref(wc))
            hwnd = user32.CreateWindowExW(
                0, class_name, "atomspike-raw", WS_POPUP, 0, 0, 0, 0, HWND_MESSAGE, None, inst, None
            )
            hwnd_val = hwnd.value if isinstance(hwnd, ctypes.c_void_p) else int(hwnd or 0)
            if not hwnd_val:
                return
            self._hwnd = hwnd
            rid = RAWINPUTDEVICE()
            rid.usUsagePage = 0x01
            rid.usUsage = 0x02
            rid.dwFlags = RIDEV_INPUTSINK
            rid.hwndTarget = hwnd
            if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                return
            msg = wintypes.MSG()
            while self._alive:
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret == 0 or ret == -1:
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception:
            return
