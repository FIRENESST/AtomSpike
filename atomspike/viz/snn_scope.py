"""Small always-on-top SNN monitor for predict-mc.

Policy ticks stay at 30Hz. This window redraws around 120Hz so membrane
glow and spike flashes decay smoothly instead of looking like a log dump.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIN_W = 512
WIN_H = 784
BG = (8, 11, 16)
PANEL = (14, 19, 27)
LINE = (32, 44, 58)
MUTED = (122, 138, 158)
TEXT = (226, 234, 244)
CYAN = (90, 228, 255)
AMBER = (255, 186, 74)
GREEN = (86, 232, 164)
MAGENTA = (255, 92, 176)


def dummy_payload(n: int = 64, layers: int = 1) -> dict:
    rng = np.random.default_rng(0)
    mem = rng.random(n).astype(np.float32) * 0.9
    spk = (rng.random(n) > 0.82).astype(np.float32)
    layer = {
        "mem": rng.random(n).astype(np.float32) * 0.8,
        "spike": (rng.random(n) > 0.85).astype(np.float32),
        "q_s": (rng.random(n) > 0.9).astype(np.float32),
        "k_s": (rng.random(n) > 0.9).astype(np.float32),
        "q_mem": rng.random(n).astype(np.float32) * 0.7,
        "k_mem": rng.random(n).astype(np.float32) * 0.7,
    }
    mix = np.zeros(8, dtype=np.float32)
    mix[2] = 0.55
    mix[5] = 0.25
    mix[0] = 0.2
    return {
        "seq": 1,
        "kind": "spike",
        "in_spike": spk,
        "in_mem": mem,
        "layers": [layer] * layers,
        "sparsity": float(spk.mean()),
        "v_th": 1.0,
        "plan_mix": mix,
        "plan_z": rng.normal(0, 0.4, 32).astype(np.float32),
        "slot_names": ["W", "A", "S", "D", "MX", "MY", "LMB", "RMB"],
        "slot_kinds": ["key", "key", "key", "key", "mouse_axis", "mouse_axis", "mouse_btn", "mouse_btn"],
        "tokens": [2, 0, 0, 0, 20, 20, 0, 0],
        "keys": {"W": "hold", "A": "idle", "S": "idle", "D": "idle"},
        "mouse": {"dx": 4.0, "dy": -2.0, "nx": 0.5, "ny": 0.5, "buttons": {"LMB": "idle", "RMB": "idle"}},
        "latency_ms": 4.2,
        "perception": True,
        "policy_hz": 30.0,
        "perception_hz": 5.0,
        "viz_hz": 120.0,
    }


class ScopeBuffers:
    def __init__(self, raster_cols: int = 168, raster_rows: int = 72):
        self.raster_cols = int(raster_cols)
        self.raster_rows = int(raster_rows)
        self.raster = np.zeros((self.raster_rows, self.raster_cols), dtype=np.float32)
        self.rate = np.zeros(self.raster_cols, dtype=np.float32)
        self.glow_in = np.zeros(1, dtype=np.float32)
        self.glow_layers: list[np.ndarray] = []
        self.last_seq = -1
        self.frames = 0
        self._fps_t = time.perf_counter()
        self.fps = 0.0

    def tick_fps(self) -> None:
        self.frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 0.25:
            self.fps = self.frames / max(1e-6, now - self._fps_t)
            self.frames = 0
            self._fps_t = now


def _font(size: int) -> ImageFont.ImageFont:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    names = (
        "msyh.ttc",
        "msyhl.ttc",
        "msyhbd.ttc",
        "simhei.ttf",
        "Deng.ttf",
        "NotoSansSC-VF.ttf",
        "simsun.ttc",
        "segoeui.ttf",
        "arial.ttf",
    )
    for name in names:
        path = windir / name
        if not path.is_file():
            continue
        try:
            return ImageFont.truetype(str(path), size, index=0)
        except OSError:
            continue
    return ImageFont.load_default()


_FONT_SM = None
_FONT_MD = None
_FONT_LG = None


def _fonts():
    global _FONT_SM, _FONT_MD, _FONT_LG
    if _FONT_SM is None:
        _FONT_SM = _font(13)
        _FONT_MD = _font(15)
        _FONT_LG = _font(18)
    return _FONT_SM, _FONT_MD, _FONT_LG


def _f32(x, fallback: int = 1) -> np.ndarray:
    if x is None:
        return np.zeros(max(1, fallback), dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(max(1, fallback), dtype=np.float32)
    return arr


def _shape_for(n: int) -> tuple[int, int]:
    n = max(1, int(n))
    cols = int(np.ceil(np.sqrt(n * 1.7)))
    cols = max(8, min(24, cols))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _mem_rgb(v: np.ndarray, v_th: float) -> np.ndarray:
    t = np.clip(v / max(1e-6, float(v_th)), 0.0, 1.0)
    r = np.clip(18 + 210 * np.power(t, 0.85) + 40 * np.maximum(t - 0.72, 0) * 4, 0, 255)
    g = np.clip(14 + 70 * t + 190 * np.maximum(t - 0.55, 0) ** 1.2, 0, 255)
    b = np.clip(38 + 160 * (1.0 - t) * t * 3.2 + 20 * t, 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _spike_rgb(v: np.ndarray) -> np.ndarray:
    t = np.clip(v, 0.0, 1.0)
    r = (40 + 70 * t).astype(np.float32)
    g = (80 + 160 * t).astype(np.float32)
    b = (90 + 165 * t).astype(np.float32)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _fill(canvas: np.ndarray, x: int, y: int, w: int, h: int, color) -> None:
    canvas[y : y + h, x : x + w] = color


def _border(canvas: np.ndarray, x: int, y: int, w: int, h: int, color=LINE) -> None:
    canvas[y : y + 1, x : x + w] = color
    canvas[y + h - 1 : y + h, x : x + w] = color
    canvas[y : y + h, x : x + 1] = color
    canvas[y : y + h, x + w - 1 : x + w] = color


def _panel(canvas: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    _fill(canvas, x, y, w, h, PANEL)
    _border(canvas, x, y, w, h)


def _paint_grid(canvas, values, spikes, x, y, w, h, v_th: float) -> None:
    values = _f32(values)
    spikes = _f32(spikes, values.size)
    n = int(values.size)
    rows, cols = _shape_for(n)
    pad = np.zeros(rows * cols, dtype=np.float32)
    spad = np.zeros(rows * cols, dtype=np.float32)
    pad[:n] = values[:n]
    spad[: min(n, spikes.size)] = spikes[: min(n, spikes.size)]
    gap = 1
    cell_w = max(2, (w - (cols + 1) * gap) // cols)
    cell_h = max(2, (h - (rows + 1) * gap) // rows)
    mem = _mem_rgb(pad, v_th).reshape(rows, cols, 3)
    spk = spad.reshape(rows, cols)
    ox = x + (w - (cell_w * cols + gap * (cols + 1))) // 2
    oy = y + (h - (cell_h * rows + gap * (rows + 1))) // 2
    _fill(canvas, x, y, w, h, (10, 14, 20))
    for r in range(rows):
        for c in range(cols):
            px = ox + gap + c * (cell_w + gap)
            py = oy + gap + r * (cell_h + gap)
            col = mem[r, c]
            if spk[r, c] > 0.15:
                t = float(np.clip(spk[r, c], 0, 1))
                col = (col.astype(np.float32) * (1 - 0.65 * t) + np.array(CYAN, np.float32) * (0.65 * t)).astype(
                    np.uint8
                )
            canvas[py : py + cell_h, px : px + cell_w] = col


def _paint_raster(canvas, raster: np.ndarray, x, y, w, h) -> None:
    img = _spike_rgb(np.clip(raster, 0, 1))
    tile = Image.fromarray(img, "RGB").resize((max(1, w), max(1, h)), Image.NEAREST)
    canvas[y : y + h, x : x + w] = np.asarray(tile)


def _paint_bars(canvas, values, x, y, w, h, color) -> None:
    values = _f32(values)
    if values.size == 0:
        return
    n = int(values.size)
    gap = 2
    bw = max(2, (w - gap * (n + 1)) // n)
    mx = float(np.max(np.abs(values))) + 1e-6
    _fill(canvas, x, y, w, h, (10, 14, 20))
    for i, v in enumerate(values):
        t = float(np.clip(abs(v) / mx, 0, 1))
        bh = max(1, int(t * (h - 4)))
        px = x + gap + i * (bw + gap)
        py = y + h - 2 - bh
        canvas[py : py + bh, px : px + bw] = color


def _downsample(vec: np.ndarray, rows: int) -> np.ndarray:
    vec = _f32(vec)
    if vec.size == rows:
        return vec
    if vec.size < rows:
        out = np.zeros(rows, dtype=np.float32)
        out[: vec.size] = vec
        return out
    idx = np.linspace(0, vec.size, rows + 1).astype(np.int64)
    out = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        a, b = int(idx[i]), int(idx[i + 1])
        out[i] = float(vec[a:b].mean()) if b > a else float(vec[min(a, vec.size - 1)])
    return out


def compose_frame(payload: dict | None, buf: ScopeBuffers, dt: float = 1.0 / 120.0, viz_hz: float = 120.0) -> np.ndarray:
    if payload is None:
        payload = {
            "seq": 0,
            "kind": "spike",
            "in_mem": np.zeros(32, dtype=np.float32),
            "in_spike": np.zeros(32, dtype=np.float32),
            "layers": [],
            "sparsity": 0.0,
            "v_th": 1.0,
            "plan_mix": np.zeros(8, dtype=np.float32),
            "slot_names": [],
            "slot_kinds": [],
            "tokens": [],
            "keys": {},
            "mouse": {},
            "latency_ms": 0.0,
            "perception": False,
        }
    v_th = float(payload.get("v_th") or 1.0)
    in_mem = _f32(payload.get("in_mem"), 16)
    in_spk = _f32(payload.get("in_spike"), in_mem.size)
    layers = list(payload.get("layers") or [])
    seq = int(payload.get("seq") or 0)
    decay = float(np.exp(-max(1e-4, dt) / 0.10))

    if buf.glow_in.size != in_spk.size:
        buf.glow_in = np.zeros_like(in_spk)
    buf.glow_in *= decay
    while len(buf.glow_layers) < len(layers):
        buf.glow_layers.append(np.zeros(1, dtype=np.float32))
    for i, layer in enumerate(layers):
        sp = _f32(layer.get("spike"))
        if buf.glow_layers[i].size != sp.size:
            buf.glow_layers[i] = np.zeros_like(sp)
        buf.glow_layers[i] *= decay

    new_tick = seq != buf.last_seq and seq > 0
    if new_tick:
        buf.glow_in = np.maximum(buf.glow_in, in_spk)
        for i, layer in enumerate(layers):
            buf.glow_layers[i] = np.maximum(buf.glow_layers[i], _f32(layer.get("spike")))
        parts = [buf.glow_in]
        parts.extend(buf.glow_layers[: len(layers)])
        col = _downsample(np.concatenate(parts, axis=0), buf.raster_rows)
        buf.raster[:, :-1] = buf.raster[:, 1:]
        buf.raster[:, -1] = col
        buf.rate[:-1] = buf.rate[1:]
        buf.rate[-1] = float(payload.get("sparsity") or 0.0)
        buf.last_seq = seq
    else:
        buf.raster *= 0.996

    canvas = np.empty((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:] = BG
    _panel(canvas, 12, 12, WIN_W - 24, 64)
    _panel(canvas, 12, 86, WIN_W - 24, 188)
    _panel(canvas, 12, 284, WIN_W - 24, 118)
    _panel(canvas, 12, 412, WIN_W - 24, 132)
    _panel(canvas, 12, 554, WIN_W - 24, 86)
    _panel(canvas, 12, 650, WIN_W - 24, 122)

    _paint_grid(canvas, in_mem, buf.glow_in, 22, 118, WIN_W - 44, 146, v_th)

    strip_y = 316
    n_lay = max(1, len(layers))
    sw = (WIN_W - 48 - 8 * (n_lay - 1)) // n_lay
    if layers:
        for i, layer in enumerate(layers):
            sx = 22 + i * (sw + 8)
            _paint_grid(canvas, layer.get("mem"), buf.glow_layers[i], sx, strip_y, sw, 76, v_th)
    else:
        _paint_grid(canvas, in_mem, buf.glow_in, 22, strip_y, WIN_W - 44, 76, v_th)

    _paint_raster(canvas, buf.raster, 22, 444, WIN_W - 44, 64)
    _paint_bars(canvas, buf.rate, 22, 512, WIN_W - 44, 22, GREEN)

    mix = payload.get("plan_mix")
    z = payload.get("plan_z")
    if mix is None:
        mix = np.zeros(8, np.float32)
    _paint_bars(canvas, mix, 22, 586, WIN_W - 44, 44, MAGENTA)
    if z is not None:
        _paint_bars(canvas, np.abs(_f32(z))[:48], 22, 586, WIN_W - 44, 12, (90, 110, 160))

    names = list(payload.get("slot_names") or [])
    kinds = list(payload.get("slot_kinds") or [])
    tokens = payload.get("tokens") or []
    keys = payload.get("keys") or {}
    n_slots = max(len(names), len(tokens), 1)
    gap = 6
    bw = max(28, (WIN_W - 44 - gap * (n_slots - 1)) // n_slots)
    for i in range(n_slots):
        sx = 22 + i * (bw + gap)
        kind = kinds[i] if i < len(kinds) else "key"
        on = False
        if kind == "key" and i < len(names):
            on = str(keys.get(names[i], "idle")) in {"press", "hold"}
        elif kind == "mouse_btn":
            btns = (payload.get("mouse") or {}).get("buttons") or {}
            label = names[i] if i < len(names) else ""
            on = str(btns.get(label, "idle")) in {"press", "hold"}
        elif kind == "mouse_axis" and i < len(tokens) and tokens:
            center = 20
            on = abs(int(tokens[i]) - center) > 1
        col = GREEN if on else (28, 36, 48)
        _fill(canvas, sx, 682, bw, 54, col if on else (22, 28, 38))
        _border(canvas, sx, 682, bw, 54, CYAN if on else LINE)

    img = Image.fromarray(canvas, "RGB")
    draw = ImageDraw.Draw(img)
    sm, md, lg = _fonts()
    kind = str(payload.get("kind") or "spike").upper()
    title = "AtomSpike  ·  脉冲网络"
    draw.text((24, 22), title, fill=TEXT, font=lg)
    hz = f"{float(viz_hz):.0f} Hz 窗口"
    live = f"{buf.fps:.0f} fps" if buf.fps else "—"
    sp = 100.0 * float(payload.get("sparsity") or 0.0)
    sub = f"{kind}   发放 {sp:.1f}%   可视化 {live} / {hz}   推理 {float(payload.get('latency_ms') or 0):.1f} ms"
    draw.text((24, 46), sub, fill=MUTED, font=sm)
    perc = "感知 tick" if payload.get("perception") else "复用语义"
    draw.text(
        (24, 94),
        f"输入 LIF 膜电位  ·  亮点是正在发放的神经元  ·  {perc}",
        fill=CYAN,
        font=sm,
    )
    draw.text((24, 292), "策略层  ·  每层 MLP-LIF（膜 + 脉冲余辉）", fill=AMBER, font=sm)
    draw.text((24, 420), "脉冲光栅  ·  时间向右  ·  下行是发放率", fill=GREEN, font=sm)
    draw.text((24, 562), "自己安排的任务槽  ·  高柱 = 当前更想做的事", fill=MAGENTA, font=sm)
    draw.text((24, 658), "键鼠输出  ·  亮着 = 这一拍按下 / 转头", fill=TEXT, font=sm)
    for i, name in enumerate(names[:n_slots]):
        sx = 22 + i * (bw + gap)
        label = str(name)[:4]
        draw.text((sx + 6, 690), label, fill=TEXT, font=sm)
    mouse = payload.get("mouse") or {}
    dx, dy = float(mouse.get("dx") or 0.0), float(mouse.get("dy") or 0.0)
    draw.text((24, 748), f"关闭本窗口即停止预测    鼠标 Δ {dx:+.1f}, {dy:+.1f}", fill=MUTED, font=sm)
    return np.asarray(img, dtype=np.uint8)


def _noactivate(hwnd) -> None:
    if not hwnd:
        return
    try:
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        SWP_NOZORDER = 0x0004
        SWP_NOACTIVATE = 0x0010
        SWP_FRAMECHANGED = 0x0020
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(int(hwnd), GWL_EXSTYLE)
        user32.SetWindowLongW(int(hwnd), GWL_EXSTYLE, int(style) | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        user32.SetWindowPos(
            int(hwnd),
            0,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
    except Exception:
        return


def _show_noactivate(hwnd) -> None:
    if not hwnd:
        return
    try:
        import ctypes

        ctypes.windll.user32.ShowWindow(int(hwnd), 4)  # SW_SHOWNOACTIVATE
    except Exception:
        return


def _tk_root_hwnd(root) -> int:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        wid = int(root.winfo_id())
        return int(user32.GetAncestor(wid, 2) or wid)  # GA_ROOT
    except Exception:
        return 0


def _restore_focus(hwnd) -> None:
    """Give keyboard/mouse grab back to Minecraft after an overlay is created."""
    if not hwnd:
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = int(hwnd)
        if not user32.IsWindow(hwnd):
            return
        if int(user32.GetForegroundWindow() or 0) == hwnd:
            return
        try:
            user32.AllowSetForegroundWindow(-1)
        except Exception:
            pass
        fg = int(user32.GetForegroundWindow() or 0)
        our = int(kernel32.GetCurrentThreadId())
        fg_tid = 0
        attached = False
        if fg:
            dummy = wintypes.DWORD()
            fg_tid = int(user32.GetWindowThreadProcessId(fg, ctypes.byref(dummy)))
            if fg_tid and fg_tid != our:
                attached = bool(user32.AttachThreadInput(our, fg_tid, True))
        user32.SetForegroundWindow(hwnd)
        if attached:
            user32.AttachThreadInput(our, fg_tid, False)
    except Exception:
        return


def _topmost(hwnd, x: int | None = None, y: int | None = None) -> None:
    try:
        import ctypes

        HWND_TOPMOST = -1
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        SWP_NOACTIVATE = 0x0010
        flags = SWP_NOSIZE | SWP_NOACTIVATE
        px = int(x or 0)
        py = int(y or 0)
        if x is None or y is None:
            flags |= SWP_NOMOVE
        ctypes.windll.user32.SetWindowPos(int(hwnd), HWND_TOPMOST, px, py, 0, 0, flags)
    except Exception:
        return


def _decorate_overlay(hwnd, x: int | None = None, y: int | None = None) -> None:
    """Pin overlay on top without taking foreground from the game."""
    _noactivate(hwnd)
    _topmost(hwnd, x, y)
    _show_noactivate(hwnd)


class SNNScope:
    """120Hz overlay. Prefers pygame on a side thread; falls back to Tk."""

    def __init__(self, hz: float = 120.0):
        self.hz = max(30.0, float(hz))
        self.backend = "none"
        self.alive = False
        self._lock = threading.Lock()
        self._payload: dict | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._ok = False
        self._tk = None
        self._photo = None
        self._label = None
        self.buffers = ScopeBuffers()
        self._game_hwnd = 0
        self._overlay = None

    def start(self, *, game_hwnd: int = 0) -> bool:
        self._game_hwnd = int(game_hwnd or 0)
        if sys.platform == "win32":
            return self._start_win32()
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._pygame_loop, name="snn-scope", daemon=True)
        self._thread.start()
        self._ready.wait(2.5)
        if self._ok:
            self.backend = "pygame"
            self.alive = True
            _restore_focus(self._game_hwnd)
            return True
        self._stop.set()
        ok = self._start_tk()
        if ok:
            _restore_focus(self._game_hwnd)
        return ok

    def _compose_rgb(self):
        rgb = compose_frame(self._snapshot(), self.buffers, dt=1.0 / self.hz, viz_hz=self.hz)
        self.buffers.tick_fps()
        return rgb

    def _start_win32(self) -> bool:
        try:
            from atomspike.viz.overlay_win import DetachedOverlay
        except Exception:
            return False
        overlay = DetachedOverlay(
            title="AtomSpike · 脉冲网络",
            width=WIN_W,
            height=WIN_H,
            x=None,
            y=48,
            hz=self.hz,
            game_hwnd=self._game_hwnd,
        )
        if not overlay.start(self._compose_rgb):
            overlay.close()
            return False
        self._overlay = overlay
        self.backend = "win32"
        self.alive = True
        return True

    def push(self, payload: dict) -> None:
        self._seq += 1
        packed = dict(payload)
        packed["seq"] = self._seq
        with self._lock:
            self._payload = packed

    def pump(self) -> bool:
        if self._overlay is not None:
            self.alive = bool(self._overlay.alive)
            return self.alive
        if self.backend == "tk":
            return self._tk_pump()
        return self.alive

    def idle(self, remaining: float) -> None:
        deadline = time.perf_counter() + max(0.0, remaining)
        frame = 1.0 / self.hz
        while time.perf_counter() < deadline and self.alive:
            t0 = time.perf_counter()
            if not self.pump():
                return
            leftover = frame - (time.perf_counter() - t0)
            sleep_for = min(max(0.0, leftover), deadline - time.perf_counter())
            if sleep_for > 0:
                time.sleep(sleep_for)

    def close(self) -> None:
        self.alive = False
        self._stop.set()
        if self._overlay is not None:
            self._overlay.close()
            self._overlay = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._tk is not None:
            try:
                self._tk.destroy()
            except Exception:
                pass
            self._tk = None
        self.backend = "none"

    def _snapshot(self) -> dict | None:
        with self._lock:
            return self._payload

    def _pygame_loop(self) -> None:
        try:
            os.environ.setdefault("SDL_VIDEO_ALLOW_SCREENSAVER", "1")
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "40,48")
            os.environ.setdefault("SDL_HINT_FORCE_RAISEWINDOW", "0")
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame

            pygame.init()
            pygame.display.set_caption("AtomSpike · 脉冲网络")
            flags = pygame.HIDDEN if hasattr(pygame, "HIDDEN") else 0
            screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
            try:
                info = pygame.display.Info()
                x = max(16, int(getattr(info, "current_w", 1280) or 1280) - WIN_W - 24)
            except Exception:
                x = 40
            wm = pygame.display.get_wm_info() if hasattr(pygame.display, "get_wm_info") else {}
            hwnd = wm.get("window")
            _decorate_overlay(hwnd, x, 48)
            _restore_focus(self._game_hwnd)
            clock = pygame.time.Clock()
            self._ok = True
            self.alive = True
            self._ready.set()
            while not self._stop.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.alive = False
                        self._stop.set()
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.alive = False
                        self._stop.set()
                        break
                rgb = compose_frame(self._snapshot(), self.buffers, dt=1.0 / self.hz, viz_hz=self.hz)
                self.buffers.tick_fps()
                view = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
                try:
                    pygame.surfarray.blit_array(screen, view)
                except ValueError:
                    screen.blit(pygame.surfarray.make_surface(view), (0, 0))
                pygame.display.flip()
                clock.tick(self.hz)
            pygame.display.quit()
            pygame.quit()
        except Exception:
            self._ok = False
            self.alive = False
            self._ready.set()

    def _start_tk(self) -> bool:
        try:
            import tkinter as tk
            from PIL import ImageTk
        except Exception:
            self.backend = "none"
            self.alive = False
            return False
        root = tk.Tk()
        root.withdraw()
        root.title("AtomSpike · 脉冲网络")
        root.configure(bg="#080b10")
        root.geometry(f"{WIN_W}x{WIN_H}+40+48")
        root.resizable(False, False)
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        label = tk.Label(root, bd=0, bg="#080b10")
        label.pack()
        root.protocol("WM_DELETE_WINDOW", self._tk_close)
        self._tk = root
        self._label = label
        self._ImageTk = ImageTk
        self.backend = "tk"
        self.alive = True
        root.update_idletasks()
        hwnd = _tk_root_hwnd(root)
        _decorate_overlay(hwnd, 40, 48)
        _restore_focus(self._game_hwnd)
        self._tk_pump()
        return True

    def _tk_close(self) -> None:
        self.alive = False
        self._stop.set()
        if self._tk is not None:
            try:
                self._tk.destroy()
            except Exception:
                pass
            self._tk = None

    def _tk_pump(self) -> bool:
        if self._tk is None or not self.alive:
            return False
        try:
            rgb = compose_frame(self._snapshot(), self.buffers, dt=1.0 / self.hz, viz_hz=self.hz)
            self.buffers.tick_fps()
            img = Image.fromarray(rgb, "RGB")
            photo = self._ImageTk.PhotoImage(img)
            self._photo = photo
            self._label.configure(image=photo)
            self._tk.update_idletasks()
            self._tk.update()
        except Exception:
            self.alive = False
            return False
        return True
