"""Win32 overlay that never becomes the foreground window.

pygame/SDL ShowWindow + flip() steal focus from Minecraft. This path creates a
WS_EX_NOACTIVATE | WS_EX_TOPMOST tool window and blits RGB with GDI.

Do not spawn a child process for the overlay: starting pythonw from a console
app pulls cmd / Windows Terminal back to the foreground.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes

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
WM_SETFOCUS = 0x0007
WM_MOUSEACTIVATE = 0x0021
WM_NCACTIVATE = 0x0086
WM_LBUTTONDOWN = 0x0201
MA_NOACTIVATE = 3
PM_REMOVE = 1
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
SW_SHOW = 5
SW_SHOWMINNOACTIVE = 7
SW_FORCEMINIMIZE = 11
SW_RESTORE = 9
SW_SHOWNA = 8
ATTACH_PARENT_PROCESS = 0xFFFFFFFF
GA_ROOT = 2
GA_ROOTOWNER = 3
HWND_NOTOPMOST = -2
SWP_NOZORDER = 0x0004
VK_MENU = 0x12
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
ASFW_ANY = 0xFFFFFFFF
SPI_GETFOREGROUNDLOCKTIMEOUT = 0x2000
SPI_SETFOREGROUNDLOCKTIMEOUT = 0x2001
TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
HWND_TOPMOST = -1
HWND_TOP = 0
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040

WS_CHILD = 0x40000000
WS_POPUP = 0x80000000
WS_CAPTION = 0x00C00000
WS_SYSMENU = 0x00080000
WS_CLIPCHILDREN = 0x02000000
WS_CLIPSIBLINGS = 0x04000000

WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
GWL_STYLE = -16
GWL_EXSTYLE = -20
LWA_COLORKEY = 0x00000001
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
WDA_EXCLUDEFROMCAPTURE = 0x00000011

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


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.UINT),
        ("flags", wintypes.UINT),
        ("showCmd", wintypes.UINT),
        ("ptMinPosition", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("rcNormalPosition", wintypes.RECT),
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
user32.IsWindowVisible.argtypes = [HANDLE_PTR]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
kernel32.FreeConsole.argtypes = []
kernel32.FreeConsole.restype = wintypes.BOOL
kernel32.AttachConsole.argtypes = [wintypes.DWORD]
kernel32.AttachConsole.restype = wintypes.BOOL
user32.GetWindowPlacement.argtypes = [HANDLE_PTR, ctypes.c_void_p]
user32.GetWindowPlacement.restype = wintypes.BOOL
user32.SetWindowPlacement.argtypes = [HANDLE_PTR, ctypes.c_void_p]
user32.SetWindowPlacement.restype = wintypes.BOOL
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
user32.GetClassNameW.argtypes = [HANDLE_PTR, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.SetActiveWindow.argtypes = [HANDLE_PTR]
user32.SetActiveWindow.restype = HANDLE_PTR
user32.SwitchToThisWindow.argtypes = [HANDLE_PTR, wintypes.BOOL]
user32.SwitchToThisWindow.restype = None
user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, wintypes.DWORD, ctypes.c_size_t]
user32.keybd_event.restype = None
user32.SystemParametersInfoW.argtypes = [wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
user32.SystemParametersInfoW.restype = wintypes.BOOL
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = HANDLE_PTR
kernel32.CloseHandle.argtypes = [HANDLE_PTR]
kernel32.CloseHandle.restype = wintypes.BOOL
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
user32.SetLayeredWindowAttributes.argtypes = [HANDLE_PTR, wintypes.DWORD, wintypes.BYTE, wintypes.DWORD]
user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
user32.SetParent.argtypes = [HANDLE_PTR, HANDLE_PTR]
user32.SetParent.restype = HANDLE_PTR
user32.GetParent.argtypes = [HANDLE_PTR]
user32.GetParent.restype = HANDLE_PTR
user32.SetWindowDisplayAffinity.argtypes = [HANDLE_PTR, wintypes.DWORD]
user32.SetWindowDisplayAffinity.restype = wintypes.BOOL


class SIZE(ctypes.Structure):
    _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", wintypes.BYTE),
        ("BlendFlags", wintypes.BYTE),
        ("SourceConstantAlpha", wintypes.BYTE),
        ("AlphaFormat", wintypes.BYTE),
    ]


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


user32.GetWindowRect.argtypes = [HANDLE_PTR, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.UpdateLayeredWindow.argtypes = [
    HANDLE_PTR,
    wintypes.HDC,
    ctypes.POINTER(wintypes.POINT),
    ctypes.POINTER(SIZE),
    wintypes.HDC,
    ctypes.POINTER(wintypes.POINT),
    wintypes.DWORD,
    ctypes.POINTER(BLENDFUNCTION),
    wintypes.DWORD,
]
user32.UpdateLayeredWindow.restype = wintypes.BOOL
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC,
    ctypes.c_void_p,
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    HANDLE_PTR,
    wintypes.DWORD,
]
gdi32.CreateDIBSection.restype = HANDLE_PTR
gdi32.SelectObject.argtypes = [HANDLE_PTR, HANDLE_PTR]
gdi32.SelectObject.restype = HANDLE_PTR
gdi32.DeleteObject.argtypes = [HANDLE_PTR]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


kernel32.Process32FirstW.argtypes = [HANDLE_PTR, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32FirstW.restype = wintypes.BOOL
kernel32.Process32NextW.argtypes = [HANDLE_PTR, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.restype = wintypes.BOOL


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


_SHELL_EXES = {
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "conhost.exe",
    "windowsterminal.exe",
    "openconsole.exe",
    "windowsterminalpreview.exe",
}
_SHELL_CLASSES = {
    "ConsoleWindowClass",
    "CASCADIA_HOSTING_WINDOW_CLASS",
    "CASCADIA_MAIN_WINDOW",
    "PseudoConsoleWindow",
    "WinUIDesktopWin32WindowClass",
}


def _class_name(hwnd) -> str:
    hwnd = _hwnd_int(hwnd)
    if not hwnd:
        return ""
    buf = ctypes.create_unicode_buffer(256)
    n = int(user32.GetClassNameW(HANDLE_PTR(hwnd), buf, 256) or 0)
    return str(buf.value) if n else ""


def _process_rows() -> list[tuple[int, int, str]]:
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or _hwnd_int(snap) == (INVALID_HANDLE_VALUE or 0):
        return []
    rows: list[tuple[int, int, str]] = []
    try:
        pe = PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if not kernel32.Process32FirstW(snap, ctypes.byref(pe)):
            return []
        while True:
            rows.append((int(pe.th32ProcessID), int(pe.th32ParentProcessID), str(pe.szExeFile).lower()))
            if not kernel32.Process32NextW(snap, ctypes.byref(pe)):
                break
    finally:
        kernel32.CloseHandle(snap)
    return rows


def _shell_pids() -> set[int]:
    """Pids that belong to *this* console tree, not the whole desktop (e.g. Cursor)."""
    pids = {int(os.getpid())}
    try:
        rows = _process_rows()
    except Exception:
        rows = []
    by_pid = {pid: (parent, exe) for pid, parent, exe in rows}
    try:
        cur = int(os.getppid())
    except Exception:
        cur = 0
    while cur and cur not in pids:
        parent, exe = by_pid.get(cur, (0, ""))
        name = exe.rsplit("\\", 1)[-1]
        if name not in _SHELL_EXES:
            break
        pids.add(cur)
        cur = int(parent or 0)
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


WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, HANDLE_PTR, wintypes.LPARAM)


def _minimize_hwnd(hwnd: int) -> None:
    hwnd = _hwnd_int(hwnd)
    if not hwnd:
        return
    try:
        user32.ShowWindow(HANDLE_PTR(hwnd), SW_SHOWMINNOACTIVE)
        if not user32.IsIconic(HANDLE_PTR(hwnd)):
            user32.ShowWindow(HANDLE_PTR(hwnd), SW_FORCEMINIMIZE)
    except Exception:
        return


def _windows_for_pids(pids: set[int]) -> list[int]:
    hits: list[int] = []
    if not pids:
        return hits

    @WNDENUMPROC
    def _cb(hwnd, _lp):  # type: ignore[misc]
        h = _hwnd_int(hwnd)
        if h and user32.IsWindowVisible(HANDLE_PTR(h)) and _pid_of(h) in pids:
            hits.append(h)
        return True

    try:
        user32.EnumWindows(_cb, 0)
    except Exception:
        return hits
    return hits


def _visible_shell_hwnd() -> int:
    """The window the user actually sees (Windows Terminal host, not a nested conhost)."""
    con = _hwnd_int(kernel32.GetConsoleWindow())
    chain: list[int] = []
    if con:
        chain.append(con)
        for flag in (GA_ROOTOWNER, GA_ROOT):
            h = _hwnd_int(user32.GetAncestor(HANDLE_PTR(con), flag))
            if h and h not in chain:
                chain.append(h)
    pids = _shell_pids()
    for h in chain:
        pid = _pid_of(h)
        if pid:
            pids.add(pid)
    ranked: list[tuple[str, int]] = []
    seen: set[int] = set()
    for h in chain + _windows_for_pids(pids):
        if not h or h in seen:
            continue
        seen.add(h)
        ranked.append((_class_name(h), h))
    prefer = (
        "CASCADIA_HOSTING_WINDOW_CLASS",
        "CASCADIA_MAIN_WINDOW",
        "WinUIDesktopWin32WindowClass",
        "ConsoleWindowClass",
        "PseudoConsoleWindow",
    )
    for want in prefer:
        for cls, h in ranked:
            if cls == want:
                return h
    for cls, h in ranked:
        if cls in _SHELL_CLASSES:
            return h
    return chain[0] if chain else 0


def _minimize_shell_windows(pids: set[int], extra: list[int], *, game_hwnd: int = 0) -> None:
    our = int(os.getpid())
    seen: set[int] = set()
    for h in list(extra) + _windows_for_pids(pids):
        h = _hwnd_int(h)
        if not h or h in seen:
            continue
        seen.add(h)
        if _pid_of(h) == our:
            continue
        if game_hwnd and _hwnds_related(h, game_hwnd):
            continue
        if _class_name(h) in _SHELL_CLASSES or _pid_of(h) in pids:
            _minimize_hwnd(h)


def detach_console_keep_stdio() -> bool:
    """Detach from cmd/Windows Terminal so CreateWindow cannot pull that console forward."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        return bool(kernel32.FreeConsole())
    except Exception:
        return False


_CTRL_HANDLER_REF = None


def ignore_console_ctrl_c() -> None:
    """Sprint is CTRL. A console-attached Python dies on CTRL+C without this."""
    global _CTRL_HANDLER_REF
    try:
        Handler = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _handled(_etype):
            return True

        if _CTRL_HANDLER_REF is None:
            _CTRL_HANDLER_REF = Handler(_handled)
        kernel32.SetConsoleCtrlHandler(_CTRL_HANDLER_REF, True)
        kernel32.SetConsoleCtrlHandler(None, True)
    except Exception:
        pass


def foreground_is_shell(thief_pids: set[int] | None = None) -> bool:
    fg = _hwnd_int(user32.GetForegroundWindow())
    if not fg:
        return False
    if _class_name(fg) in _SHELL_CLASSES:
        return True
    pids = thief_pids if thief_pids is not None else _shell_pids()
    return _pid_of(fg) in pids


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


def _hwnds_related(a, b) -> bool:
    try:
        from atomspike.capture.window import hwnds_related

        return bool(hwnds_related(_hwnd_int(a), _hwnd_int(b)))
    except Exception:
        a, b = _hwnd_int(a), _hwnd_int(b)
        if not a or not b:
            return False
        if a == b:
            return True
        ra = _hwnd_int(user32.GetAncestor(HANDLE_PTR(a), GA_ROOTOWNER)) or _hwnd_int(
            user32.GetAncestor(HANDLE_PTR(a), GA_ROOT)
        )
        rb = _hwnd_int(user32.GetAncestor(HANDLE_PTR(b), GA_ROOTOWNER)) or _hwnd_int(
            user32.GetAncestor(HANDLE_PTR(b), GA_ROOT)
        )
        return bool(ra and rb and ra == rb)


def restore_game_focus(hwnd, *, park_cursor: bool = False, allow_alt: bool = True) -> bool:
    """Pull foreground (and optionally the cursor) back from cmd/Windows Terminal."""
    hwnd = _hwnd_int(hwnd)
    if not hwnd:
        return False
    try:
        if not user32.IsWindow(HANDLE_PTR(hwnd)):
            return False
        fg = _hwnd_int(user32.GetForegroundWindow())
        if _hwnds_related(fg, hwnd):
            if park_cursor:
                _park_cursor_in_game(hwnd)
            return True
        try:
            user32.AllowSetForegroundWindow(ASFW_ANY)
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
        try:
            if user32.IsIconic(HANDLE_PTR(hwnd)):
                user32.ShowWindow(HANDLE_PTR(hwnd), SW_RESTORE)
            user32.SetForegroundWindow(HANDLE_PTR(hwnd))
            try:
                user32.SetActiveWindow(HANDLE_PTR(hwnd))
            except Exception:
                pass
            if allow_alt and not _hwnds_related(_hwnd_int(user32.GetForegroundWindow()), hwnd):
                user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY, 0)
                try:
                    user32.SetForegroundWindow(HANDLE_PTR(hwnd))
                    user32.SwitchToThisWindow(HANDLE_PTR(hwnd), True)
                finally:
                    user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
        finally:
            if attached_game:
                user32.AttachThreadInput(our, game_tid, False)
            if attached_fg:
                user32.AttachThreadInput(our, fg_tid, False)
        if park_cursor:
            _park_cursor_in_game(hwnd)
        return _hwnds_related(_hwnd_int(user32.GetForegroundWindow()), hwnd)
    except Exception:
        return False


def reveal_window_inactive(hwnd: int) -> None:
    """Un-minimize a window without making it the foreground (cmd must not steal MC)."""
    hwnd = _hwnd_int(hwnd)
    if not hwnd:
        return
    try:
        wp = WINDOWPLACEMENT()
        wp.length = ctypes.sizeof(WINDOWPLACEMENT)
        if user32.GetWindowPlacement(HANDLE_PTR(hwnd), ctypes.byref(wp)):
            wp.flags = 0
            wp.showCmd = SW_SHOWNOACTIVATE
            user32.SetWindowPlacement(HANDLE_PTR(hwnd), ctypes.byref(wp))
        else:
            user32.ShowWindow(HANDLE_PTR(hwnd), SW_SHOWNOACTIVATE)
        user32.SetWindowPos(
            HANDLE_PTR(hwnd),
            HANDLE_PTR(HWND_NOTOPMOST),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER | SWP_SHOWWINDOW,
        )
    except Exception:
        try:
            user32.ShowWindow(HANDLE_PTR(hwnd), SW_SHOWNOACTIVATE)
        except Exception:
            pass


def handoff_game_focus(hwnd: int) -> None:
    """After python exits, cmd often takes focus. A detached helper gives it back."""
    hwnd = _hwnd_int(hwnd)
    if not hwnd:
        return
    try:
        import subprocess

        create_no_window = 0x08000000
        detached = 0x00000008
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys,time,ctypes; time.sleep(0.4); ctypes.windll.user32.SetForegroundWindow(int(sys.argv[1]))",
                str(hwnd),
            ],
            creationflags=detached | create_no_window,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        try:
            user32.AllowSetForegroundWindow(int(proc.pid))
        except Exception:
            pass
    except Exception:
        pass


class GameFocusSession:
    """Hide cmd, detach the console, and yank focus back whenever the shell steals it.

    Does not steal from other apps (browser, Cursor, etc.) — only our console tree
    and this process's overlay. Console prints during the session go to `log_path`
    so Windows Terminal is not woken by `[learn]` / `[predict]` lines.
    """

    def __init__(self, game_hwnd: int = 0, log_path: str | os.PathLike | None = None):
        self.game_hwnd = int(game_hwnd or 0)
        self._console_hwnd = _visible_shell_hwnd() or _hwnd_int(kernel32.GetConsoleWindow())
        self._thief_pids = _shell_pids()
        host_pid = _pid_of(self._console_hwnd)
        if host_pid:
            self._thief_pids.add(host_pid)
        self._hidden = False
        self._detached = False
        self._lock_timeout = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._parked = False
        self._log_path = os.fspath(log_path) if log_path else None
        self._old_out = None
        self._old_err = None
        self._log_f = None
        self._saved_out_fd = None
        self._saved_err_fd = None

    def is_thief(self, hwnd: int | None = None) -> bool:
        hwnd = _hwnd_int(hwnd if hwnd is not None else user32.GetForegroundWindow())
        if not hwnd:
            return False
        if _hwnds_related(hwnd, self.game_hwnd):
            return False
        if _pid_of(hwnd) == int(os.getpid()):
            return True
        if _pid_of(hwnd) in self._thief_pids:
            return True
        if self._console_hwnd and hwnd == self._console_hwnd:
            return True
        return _class_name(hwnd) in _SHELL_CLASSES

    def nudge(self, *, park: bool = False) -> bool:
        if not self.game_hwnd:
            return False
        fg = _hwnd_int(user32.GetForegroundWindow())
        if _hwnds_related(fg, self.game_hwnd):
            if park and not self._parked:
                _park_cursor_in_game(self.game_hwnd)
                self._parked = True
            return True
        if not self.is_thief(fg) and fg:
            return False
        ok = restore_game_focus(self.game_hwnd, park_cursor=bool(park and not self._parked), allow_alt=False)
        if park:
            self._parked = True
        return ok

    def _silence_console(self) -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            self._saved_out_fd = os.dup(1)
            self._saved_err_fd = os.dup(2)
        except OSError:
            self._saved_out_fd = self._saved_err_fd = None
        if self._log_path:
            try:
                parent = os.path.dirname(self._log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self._old_out, self._old_err = sys.stdout, sys.stderr
                self._log_f = open(self._log_path, "a", encoding="utf-8", errors="replace", buffering=1)
                sys.stdout = self._log_f
                sys.stderr = self._log_f
                try:
                    os.dup2(self._log_f.fileno(), 1)
                    os.dup2(self._log_f.fileno(), 2)
                except OSError:
                    pass
            except Exception:
                self._log_f = None
        self._detached = detach_console_keep_stdio()

    def _restore_stdio(self) -> None:
        if self._detached:
            try:
                kernel32.AttachConsole(ATTACH_PARENT_PROCESS)
            except Exception:
                pass
            self._detached = False
        if self._saved_out_fd is not None:
            try:
                os.dup2(self._saved_out_fd, 1)
                os.close(self._saved_out_fd)
            except OSError:
                pass
            self._saved_out_fd = None
        if self._saved_err_fd is not None:
            try:
                os.dup2(self._saved_err_fd, 2)
                os.close(self._saved_err_fd)
            except OSError:
                pass
            self._saved_err_fd = None
        if self._old_out is not None:
            sys.stdout = self._old_out
            self._old_out = None
        if self._old_err is not None:
            sys.stderr = self._old_err
            self._old_err = None
        if self._log_f is not None:
            try:
                self._log_f.flush()
                self._log_f.close()
            except Exception:
                pass
            self._log_f = None

    def begin(self, *, hide_console: bool = True, park: bool = False) -> None:
        if hide_console:
            _minimize_shell_windows(self._thief_pids, [self._console_hwnd], game_hwnd=self.game_hwnd)
            self._hidden = bool(self._console_hwnd)
            self._silence_console()
        ignore_console_ctrl_c()
        try:
            user32.AllowSetForegroundWindow(ASFW_ANY)
        except Exception:
            pass
        timeout = wintypes.UINT(0)
        try:
            if user32.SystemParametersInfoW(
                SPI_GETFOREGROUNDLOCKTIMEOUT, 0, ctypes.byref(timeout), 0
            ):
                self._lock_timeout = int(timeout.value)
                user32.SystemParametersInfoW(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, ctypes.c_void_p(0), 0)
        except Exception:
            self._lock_timeout = None
        self.nudge(park=park)
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="atomspike-focus", daemon=True)
            self._thread.start()

    def _confine_cursor(self) -> None:
        if not self.game_hwnd:
            return
        try:
            from atomspike.capture.window import clip_cursor_to_hwnd

            clip_cursor_to_hwnd(self.game_hwnd)
        except Exception:
            pass

    def _release_cursor(self) -> None:
        try:
            from atomspike.capture.window import unclip_cursor

            unclip_cursor()
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.wait(0.05):
            try:
                fg = _hwnd_int(user32.GetForegroundWindow())
                if _hwnds_related(fg, self.game_hwnd):
                    continue
                if self.is_thief(fg) or not fg:
                    self.nudge(park=False)
                    continue
                self._release_cursor()
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        self._release_cursor()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.6)
        self._thread = None
        if self._lock_timeout is not None:
            try:
                user32.SystemParametersInfoW(
                    SPI_SETFOREGROUNDLOCKTIMEOUT,
                    0,
                    ctypes.c_void_p(self._lock_timeout),
                    0,
                )
            except Exception:
                pass
            self._lock_timeout = None
        self._restore_stdio()
        if self._hidden and self._console_hwnd:
            try:
                if user32.IsWindow(HANDLE_PTR(self._console_hwnd)):
                    reveal_window_inactive(self._console_hwnd)
            except Exception:
                pass
            self._hidden = False
        if self.game_hwnd:
            restore_game_focus(self.game_hwnd, park_cursor=False, allow_alt=False)


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
        layered: bool = False,
        click_through: bool = False,
        chroma_rgb: tuple[int, int, int] | None = None,
        follow: bool = False,
    ):
        self.title = str(title)
        self.width = max(8, int(width))
        self.height = max(8, int(height))
        self.x = x
        self.y = int(y)
        self.hz = max(8.0, float(hz))
        self.game_hwnd = int(game_hwnd or 0)
        self.layered = bool(layered or chroma_rgb is not None or follow)
        self.click_through = bool(click_through)
        self.chroma_rgb = None if chroma_rgb is None else (int(chroma_rgb[0]), int(chroma_rgb[1]), int(chroma_rgb[2]))
        self.follow = bool(follow)
        self._embedded = False
        self.alive = False
        self._ok = False
        self._hwnd = 0
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._composer = None
        self._bgra: np.ndarray | None = None
        self._last: np.ndarray | None = None
        self._last_pos: tuple[int, int, int, int] | None = None
        self._last_z = 0.0
        self._visible = True
        self._hdc_mem = None
        self._hbm = None
        self._hbm_old = None
        self._dib_bits = None
        self._dib_wh = (0, 0)
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
            restore_game_focus(self.game_hwnd, park_cursor=False, allow_alt=False)
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
        self._release_dib()

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        try:
            msg = int(msg)
            if msg == WM_MOUSEACTIVATE:
                return MA_NOACTIVATE
            if msg == WM_ACTIVATE:
                if (int(_u64(wparam)) & 0xFFFF) != 0 and self.game_hwnd:
                    try:
                        user32.SetForegroundWindow(HANDLE_PTR(self.game_hwnd))
                    except Exception:
                        pass
                return 0
            if msg == WM_SETFOCUS:
                if self.game_hwnd:
                    try:
                        user32.SetForegroundWindow(HANDLE_PTR(self.game_hwnd))
                    except Exception:
                        pass
                return 0
            if msg == WM_LBUTTONDOWN:
                if self.click_through:
                    return 0
                x = int(_u64(lparam) & 0xFFFF)
                y = int((_u64(lparam) >> 16) & 0xFFFF)
                if x >= max(8, self.width - 40) and y <= 40:
                    self.alive = False
                    user32.DestroyWindow(hwnd)
                    return 0
            if msg == WM_NCACTIVATE:
                return user32.DefWindowProcW(hwnd, msg, 0, _signed_ptr(lparam))
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
        if self.follow and self.game_hwnd:
            hwnd = self._create_owned_follow(inst)
            if hwnd:
                return hwnd
            print("[overlay] could not attach HUD to the Minecraft window", flush=True)
            return 0
        style = WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS
        ex = WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
        if self.layered:
            ex |= WS_EX_LAYERED
        if self.click_through:
            ex |= WS_EX_TRANSPARENT
        outer_w = int(self.width)
        outer_h = int(self.height)
        if self.x is None:
            screen_w = int(user32.GetSystemMetrics(SM_CXSCREEN) or 1280)
            pos_x = max(16, screen_w - outer_w - 24)
            pos_y = int(self.y)
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
        self._decorate(hwnd)
        user32.SetWindowPos(
            HANDLE_PTR(hwnd),
            HANDLE_PTR(HWND_TOPMOST),
            pos_x,
            pos_y,
            outer_w,
            outer_h,
            SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_FRAMECHANGED,
        )
        user32.ShowWindow(HANDLE_PTR(hwnd), SW_SHOWNA)
        return hwnd

    def _create_owned_follow(self, inst) -> int:
        """Layered popup owned by Minecraft, locked to the client rect.

        A WS_CHILD of the GLFW/OpenGL window is painted over every swap, so
        the HUD must be a sibling overlay in screen space, not a child.
        """
        game = self.game_hwnd
        screen = self._game_client_screen(game)
        if screen is None:
            return 0
        x, y, cw, ch = screen
        self.width, self.height = cw, ch
        self._embedded = False
        style = WS_POPUP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN
        ex = (
            WS_EX_NOACTIVATE
            | WS_EX_TOOLWINDOW
            | WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOPMOST
        )
        hwnd_p = user32.CreateWindowExW(
            ex,
            self._class_name,
            self.title,
            style,
            x,
            y,
            cw,
            ch,
            HANDLE_PTR(game),
            None,
            inst,
            None,
        )
        hwnd = _hwnd_int(hwnd_p)
        if not hwnd:
            return 0
        self._decorate(hwnd)
        user32.SetWindowPos(
            HANDLE_PTR(hwnd),
            HANDLE_PTR(HWND_TOPMOST),
            x,
            y,
            cw,
            ch,
            SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_FRAMECHANGED,
        )
        user32.ShowWindow(HANDLE_PTR(hwnd), SW_SHOWNA)
        return hwnd

    def _decorate(self, hwnd: int) -> None:
        try:
            getter = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            setter = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            style_ex = int(getter(HANDLE_PTR(hwnd), GWL_EXSTYLE) or 0)
            extra = WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            if self.layered:
                extra |= WS_EX_LAYERED
            if self.click_through:
                extra |= WS_EX_TRANSPARENT
            extra |= WS_EX_TOPMOST
            setter(HANDLE_PTR(hwnd), GWL_EXSTYLE, style_ex | extra)
        except Exception:
            pass
        try:
            user32.SetWindowDisplayAffinity(HANDLE_PTR(hwnd), WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

    def _game_client_size(self, game: int) -> tuple[int, int] | None:
        if not game or not user32.IsWindow(HANDLE_PTR(game)):
            return None
        rc = wintypes.RECT()
        if not user32.GetClientRect(HANDLE_PTR(game), ctypes.byref(rc)):
            return None
        w = int(rc.right - rc.left)
        h = int(rc.bottom - rc.top)
        if w < 8 or h < 8:
            return None
        return w, h

    def _game_client_screen(self, game: int) -> tuple[int, int, int, int] | None:
        size = self._game_client_size(game)
        if size is None:
            return None
        pt = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(HANDLE_PTR(game), ctypes.byref(pt)):
            return None
        return int(pt.x), int(pt.y), int(size[0]), int(size[1])

    def _to_bgra(self, rgb: np.ndarray, *, keyed: bool = False) -> np.ndarray:
        src = np.ascontiguousarray(rgb, dtype=np.uint8)
        if src.ndim != 3 or src.shape[2] < 3:
            raise ValueError("overlay frame must be HxWx3")
        h, w = int(src.shape[0]), int(src.shape[1])
        if self._bgra is None or self._bgra.shape[0] != h or self._bgra.shape[1] != w:
            self._bgra = np.empty((h, w, 4), dtype=np.uint8)
        self._bgra[:, :, 0] = src[:, :, 2]
        self._bgra[:, :, 1] = src[:, :, 1]
        self._bgra[:, :, 2] = src[:, :, 0]
        if keyed and self.chroma_rgb is not None:
            r, g, b = self.chroma_rgb
            mask = (src[:, :, 0] == r) & (src[:, :, 1] == g) & (src[:, :, 2] == b)
            self._bgra[:, :, 3] = np.where(mask, 0, 255).astype(np.uint8)
            vis = ~mask
            self._bgra[:, :, 0] = np.where(vis, self._bgra[:, :, 0], 0)
            self._bgra[:, :, 1] = np.where(vis, self._bgra[:, :, 1], 0)
            self._bgra[:, :, 2] = np.where(vis, self._bgra[:, :, 2], 0)
        else:
            self._bgra[:, :, 3] = 255
        return self._bgra

    def _release_dib(self) -> None:
        try:
            if self._hdc_mem and self._hbm_old:
                gdi32.SelectObject(HANDLE_PTR(self._hdc_mem), HANDLE_PTR(self._hbm_old))
        except Exception:
            pass
        try:
            if self._hbm:
                gdi32.DeleteObject(HANDLE_PTR(self._hbm))
        except Exception:
            pass
        try:
            if self._hdc_mem:
                gdi32.DeleteDC(self._hdc_mem)
        except Exception:
            pass
        self._hdc_mem = None
        self._hbm = None
        self._hbm_old = None
        self._dib_bits = None
        self._dib_wh = (0, 0)

    def _ensure_dib(self, w: int, h: int) -> ctypes.c_void_p | None:
        w, h = int(w), int(h)
        if w < 1 or h < 1:
            return None
        if self._hdc_mem and self._dib_bits and self._dib_wh == (w, h):
            return self._dib_bits
        self._release_dib()
        hdc_screen = user32.GetDC(None)
        if not hdc_screen:
            return None
        try:
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        finally:
            user32.ReleaseDC(None, hdc_screen)
        if not hdc_mem:
            return None
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = w
        info.bmiHeader.biHeight = -h
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        bits = ctypes.c_void_p()
        hbm = gdi32.CreateDIBSection(
            hdc_mem,
            ctypes.byref(info),
            DIB_RGB_COLORS,
            ctypes.byref(bits),
            None,
            0,
        )
        if not hbm or not bits.value:
            gdi32.DeleteDC(hdc_mem)
            return None
        old = gdi32.SelectObject(HANDLE_PTR(hdc_mem), HANDLE_PTR(hbm))
        self._hdc_mem = hdc_mem
        self._hbm = hbm
        self._hbm_old = old
        self._dib_bits = bits
        self._dib_wh = (w, h)
        return bits

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

    def _present_layered(self, rgb: np.ndarray) -> None:
        hwnd = self._hwnd
        if not hwnd:
            return
        bgra = np.ascontiguousarray(self._to_bgra(rgb, keyed=True))
        h, w = int(bgra.shape[0]), int(bgra.shape[1])
        bits = self._ensure_dib(w, h)
        if bits is None or self._hdc_mem is None:
            return
        ctypes.memmove(bits, bgra.ctypes.data, int(bgra.nbytes))
        rc = wintypes.RECT()
        if not user32.GetWindowRect(HANDLE_PTR(hwnd), ctypes.byref(rc)):
            return
        pt_dst = wintypes.POINT(int(rc.left), int(rc.top))
        pt_src = wintypes.POINT(0, 0)
        size = SIZE(w, h)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        user32.UpdateLayeredWindow(
            HANDLE_PTR(hwnd),
            None,
            ctypes.byref(pt_dst),
            ctypes.byref(size),
            self._hdc_mem,
            ctypes.byref(pt_src),
            0,
            ctypes.byref(blend),
            ULW_ALPHA,
        )
        self.width, self.height = w, h

    def _present(self, rgb: np.ndarray) -> None:
        hwnd = self._hwnd
        if not hwnd:
            return
        if self.layered and self.chroma_rgb is not None:
            self._present_layered(rgb)
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

    def _sync_follow(self) -> bool:
        """Pin the HUD to the Minecraft client in screen space (never a GL child)."""
        hwnd = self._hwnd
        game = self.game_hwnd
        if not hwnd or not game:
            return False
        try:
            if not user32.IsWindow(HANDLE_PTR(game)) or user32.IsIconic(HANDLE_PTR(game)):
                user32.ShowWindow(HANDLE_PTR(hwnd), SW_HIDE)
                self._visible = False
                return False
            from atomspike.capture.window import is_window_foreground

            if not is_window_foreground(game):
                user32.ShowWindow(HANDLE_PTR(hwnd), SW_HIDE)
                self._visible = False
                return False
            screen = self._game_client_screen(game)
            if screen is None:
                user32.ShowWindow(HANDLE_PTR(hwnd), SW_HIDE)
                self._visible = False
                return False
            self._embedded = False
            pos = screen
            self.width, self.height = pos[2], pos[3]
            now = time.perf_counter()
            moved = pos != self._last_pos
            if moved or (now - self._last_z) > 0.25 or not self._visible:
                user32.SetWindowPos(
                    HANDLE_PTR(hwnd),
                    HANDLE_PTR(HWND_TOPMOST),
                    pos[0],
                    pos[1],
                    pos[2],
                    pos[3],
                    SWP_NOACTIVATE | SWP_SHOWWINDOW,
                )
                user32.ShowWindow(HANDLE_PTR(hwnd), SW_SHOWNA)
                self._last_pos = pos
                self._last_z = now
                self._visible = True
            return True
        except Exception:
            return False

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
                if self.follow and not self._sync_follow():
                    next_t += period
                    leftover = next_t - time.perf_counter()
                    if leftover > 0:
                        time.sleep(leftover)
                    else:
                        next_t = time.perf_counter()
                    continue
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


class DetachedOverlay:
    """In-process overlay. A child pythonw process was pulling cmd back to the front."""

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
        self._inner = Win32Overlay(
            title=title,
            width=width,
            height=height,
            x=x,
            y=y,
            hz=hz,
            game_hwnd=game_hwnd,
        )
        self.game_hwnd = int(game_hwnd or 0)

    @property
    def alive(self) -> bool:
        return bool(self._inner.alive)

    def start(self, composer) -> bool:
        return self._inner.start(composer)

    def close(self) -> None:
        self._inner.close()

