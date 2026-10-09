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

WIN_W = 768
WIN_H = 980
BG = (8, 11, 16)
PANEL = (14, 19, 27)
LINE = (32, 44, 58)
MUTED = (122, 138, 158)
TEXT = (226, 234, 244)
CYAN = (90, 228, 255)
AMBER = (255, 186, 74)
GREEN = (86, 232, 164)
MAGENTA = (255, 92, 176)
ITER_AUX_LABELS = ("进度", "GUI", "暂停", "背包", "视野", "死亡", "看空", "成就")
_PAD = 12
_INSET = 22


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
        "slot_names": ["W", "A", "S", "D", "SPACE", "SHIFT", "CTRL", "E", "MX", "MY", "LMB", "RMB", "WHL"],
        "slot_kinds": ["key"] * 8 + ["mouse_axis", "mouse_axis", "mouse_btn", "mouse_btn", "wheel"],
        "tokens": [2, 0, 0, 0, 0, 0, 0, 0, 20, 20, 0, 0, 5],
        "keys": {"W": "hold", "A": "idle", "S": "idle", "D": "idle"},
        "mouse": {"dx": 4.0, "dy": -2.0, "nx": 0.5, "ny": 0.5, "buttons": {"LMB": "idle", "RMB": "idle"}},
        "latency_ms": 4.2,
        "perception": True,
        "policy_hz": 30.0,
        "perception_hz": 10.0,
        "viz_hz": 120.0,
        "iter_task": "story/mine_stone",
        "iter_task_id": 2,
        "iter_aux": np.array([0.4, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.1], dtype=np.float32),
        "iter_change": 0.42,
        "iter_grid": rng.random(24).astype(np.float32),
        "iter_h": rng.normal(0, 0.3, 32).astype(np.float32),
        "iter_bias": 0.08,
        "iter_hw": 4,
        "iter_gh": 4,
        "iter_gw": 6,
        "iter_hold": False,
        "curl_energy": 0.0,
        "group_energy": rng.random(8).astype(np.float32),
        "group_alive": 0,
        "group_n": 8,
        "sea_mask": (rng.random(64) > 0.8).astype(np.float32),
        "sea_k": 8,
        "sea_n": 64,
        "sea_energy": 0.22,
        "event_order": rng.random(8).astype(np.float32),
        "event_n": 8,
        "iter_frac": 0.4,
        "inv_hand": "stone_pickaxe",
        "inv_hover": "cobblestone",
        "inv_carry": "-",
        "inv_n": 6.0,
        "inv_spike": 0.12,
        "inv_bias": 0.03,
    }


class ScopeBuffers:
    def __init__(self, raster_cols: int = 220, raster_rows: int = 72):
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
        _FONT_SM = _font(14)
        _FONT_MD = _font(16)
        _FONT_LG = _font(20)
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


def _paint_meter(canvas, t: float, x, y, w, h, color) -> None:
    t = float(np.clip(t, 0.0, 1.0))
    _fill(canvas, x, y, w, h, (10, 14, 20))
    _border(canvas, x, y, w, h, LINE)
    inner = max(1, int(t * max(1, w - 4)))
    canvas[y + 2 : y + h - 2, x + 2 : x + 2 + inner] = color


def _paint_spatial_map(canvas, values, x, y, w, h, shape: tuple[int, int] | None = None) -> None:
    values = _f32(values)
    n = max(1, int(values.size))
    if shape is not None and int(shape[0]) >= 2 and int(shape[1]) >= 2:
        rows, cols = int(shape[0]), int(shape[1])
    else:
        cols = max(2, int(round(n ** 0.5)))
        rows = max(2, int(round(n / max(1, cols))))
        if rows * cols < n:
            cols = max(cols, (n + rows - 1) // rows)
    pad = np.zeros(rows * cols, dtype=np.float32)
    pad[: min(n, pad.size)] = values[: min(n, pad.size)]
    floor = float(np.median(pad))
    pad = np.clip(pad - floor, 0.0, None)
    mx = float(pad.max())
    if mx < 1e-5:
        t = np.zeros((rows, cols), dtype=np.float32)
    else:
        t = np.clip(pad / mx, 0.0, 1.0).reshape(rows, cols)
    gap = 2
    cell = max(4, min((w - (cols + 1) * gap) // cols, (h - (rows + 1) * gap) // rows))
    ox = x + (w - (cell * cols + gap * (cols + 1))) // 2
    oy = y + (h - (cell * rows + gap * (rows + 1))) // 2
    _fill(canvas, x, y, w, h, (10, 14, 20))
    _border(canvas, x, y, w, h, LINE)
    for r in range(rows):
        for c in range(cols):
            v = float(t[r, c])
            col = (
                int(18 + (AMBER[0] - 18) * v),
                int(16 + (AMBER[1] - 16) * v),
                int(22 + (AMBER[2] - 22) * v),
            )
            px = ox + gap + c * (cell + gap)
            py = oy + gap + r * (cell + gap)
            canvas[py : py + cell, px : px + cell] = col


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

    pad, inset = _PAD, _INSET
    panel_w = WIN_W - 2 * pad
    content_w = WIN_W - 2 * inset
    map_w, map_h = 240, 118
    map_x, map_y = inset, 678
    rest_x = map_x + map_w + 12
    rest_w = max(120, WIN_W - rest_x - inset)
    half = max(48, (rest_w - 8) // 2)

    canvas = np.empty((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:] = BG
    _panel(canvas, pad, 12, panel_w, 64)
    _panel(canvas, pad, 86, panel_w, 188)
    _panel(canvas, pad, 284, panel_w, 118)
    _panel(canvas, pad, 412, panel_w, 132)
    _panel(canvas, pad, 554, panel_w, 86)
    _panel(canvas, pad, 650, panel_w, 150)
    _panel(canvas, pad, 810, panel_w, 154)

    _paint_grid(canvas, in_mem, buf.glow_in, inset, 118, content_w, 146, v_th)

    strip_y = 316
    n_lay = max(1, len(layers))
    sw = (content_w - 8 * (n_lay - 1)) // n_lay
    if layers:
        for i, layer in enumerate(layers):
            sx = inset + i * (sw + 8)
            _paint_grid(canvas, layer.get("mem"), buf.glow_layers[i], sx, strip_y, sw, 76, v_th)
    else:
        _paint_grid(canvas, in_mem, buf.glow_in, inset, strip_y, content_w, 76, v_th)

    _paint_raster(canvas, buf.raster, inset, 444, content_w, 64)
    _paint_bars(canvas, buf.rate, inset, 512, content_w, 22, GREEN)

    mix = payload.get("plan_mix")
    z = payload.get("plan_z")
    if mix is None:
        mix = np.zeros(8, np.float32)
    _paint_bars(canvas, mix, inset, 586, content_w, 44, MAGENTA)
    if z is not None:
        _paint_bars(canvas, np.abs(_f32(z))[:48], inset, 586, content_w, 12, (90, 110, 160))
    ge = payload.get("group_energy")
    if ge is not None:
        _paint_bars(canvas, ge, inset, 638, content_w, 10, AMBER)

    grid = payload.get("iter_grid")
    gh = max(2, int(payload.get("iter_gh") or 4))
    gw = max(2, int(payload.get("iter_gw") or 6))
    if grid is None:
        grid = np.zeros(gh * gw, dtype=np.float32)
    _paint_spatial_map(canvas, grid, map_x, map_y, map_w, map_h, shape=(gh, gw))
    change = float(payload.get("iter_change") or 0.0)
    frac = float(payload.get("iter_frac") or 0.0)
    bias_t = float(np.clip(float(payload.get("iter_bias") or 0.0) / 0.25, 0.0, 1.0))
    _paint_meter(canvas, change, rest_x, map_y, rest_w, 12, AMBER)
    _paint_bars(canvas, payload.get("iter_h"), rest_x, map_y + 18, rest_w, 34, CYAN)
    _paint_meter(canvas, max(change, frac), rest_x, map_y + 58, half, 10, GREEN)
    _paint_meter(canvas, bias_t, rest_x + half + 8, map_y + 58, half, 10, MAGENTA)

    names = list(payload.get("slot_names") or [])
    kinds = list(payload.get("slot_kinds") or [])
    tokens = payload.get("tokens") or []
    keys = payload.get("keys") or {}
    n_slots = max(len(names), len(tokens), 1)
    gap = 8 if n_slots <= 10 else 6
    bw = max(36, (content_w - gap * (n_slots - 1)) // n_slots)
    key_y = 842
    for i in range(n_slots):
        sx = inset + i * (bw + gap)
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
        elif kind == "wheel" and i < len(tokens) and tokens:
            center = 5
            on = abs(int(tokens[i]) - center) > 0
        _fill(canvas, sx, key_y, bw, 54, GREEN if on else (22, 28, 38))
        _border(canvas, sx, key_y, bw, 54, CYAN if on else LINE)

    img = Image.fromarray(canvas, "RGB")
    draw = ImageDraw.Draw(img)
    sm, _, lg = _fonts()
    kind = str(payload.get("kind") or "spike").upper()
    draw.text((inset + 2, 22), "AtomSpike  ·  脉冲网络", fill=TEXT, font=lg)
    hz = f"{float(viz_hz):.0f} Hz 窗口"
    live = f"{buf.fps:.0f} fps" if buf.fps else "—"
    sp = 100.0 * float(payload.get("sparsity") or 0.0)
    sub = f"{kind}   发放 {sp:.1f}%   可视化 {live} / {hz}   推理 {float(payload.get('latency_ms') or 0):.1f} ms"
    draw.text((inset + 2, 46), sub, fill=MUTED, font=sm)
    perc = "感知 tick" if payload.get("perception") else "复用语义"
    draw.text(
        (inset + 2, 94),
        f"输入 LIF 膜电位  ·  亮点是正在发放的神经元  ·  {perc}",
        fill=CYAN,
        font=sm,
    )
    draw.text((inset + 2, 292), "策略层  ·  每层 MLP-LIF（膜 + 脉冲余辉）", fill=AMBER, font=sm)
    draw.text((inset + 2, 420), "脉冲光栅  ·  时间向右  ·  下行是发放率", fill=GREEN, font=sm)
    draw.text((inset + 2, 562), "自己安排的任务槽  ·  高柱 = 当前更想做的事", fill=MAGENTA, font=sm)
    task = str(payload.get("iter_task") or "空任务")
    if "/" in task:
        task = task.split("/")[-1]
    hold = bool(payload.get("iter_hold"))
    hold_txt = "GUI冻结" if hold else "世界视角"
    curl_e = float(payload.get("curl_energy") or 0.0)
    g_n = int(payload.get("group_n") or 0)
    g_alive = int(payload.get("group_alive") or 0)
    g_txt = f"  ·  集团 {g_alive}/{g_n}" if g_n > 0 else ""
    sea_n = int(payload.get("sea_n") or 0)
    sea_k = int(payload.get("sea_k") or 0)
    sea_txt = f"  ·  符合 {sea_k}/{sea_n}" if sea_n > 0 else ""
    det_n = int(payload.get("detect_n") or 0)
    det_k = int(payload.get("detect_k") or 0)
    kinds = payload.get("detect_kinds") or []
    kind_s = ",".join(str(k) for k in kinds[:4]) if kinds else ""
    det_txt = f"  ·  网格 {det_k}/{det_n}{(' ' + kind_s) if kind_s else ''}" if det_n > 0 else ""
    ev_n = int(payload.get("event_n") or 0)
    ev_txt = f"  ·  事件序 {ev_n}" if ev_n > 0 else ""
    draw.text(
        (inset + 2, 656),
        f"10Hz 迭代器  ·  {gh}×{gw} 空间差  ·  {task}",
        fill=AMBER,
        font=sm,
    )
    draw.text(
        (rest_x, 656),
        f"变化 {change:.2f}  ·  旋度 {curl_e:.2f}  ·  {hold_txt}{g_txt}{sea_txt}{det_txt}{ev_txt}",
        fill=AMBER,
        font=sm,
    )
    aux = _f32(payload.get("iter_aux"), 8)
    step = max(48, rest_w // max(1, len(ITER_AUX_LABELS)))
    ax = rest_x
    for i, name in enumerate(ITER_AUX_LABELS):
        on = float(aux[i]) > 0.35 if i < aux.size else False
        draw.text((ax, 752), name, fill=AMBER if on else MUTED, font=sm)
        ax += step
    hand = str(payload.get("inv_hand") or "-")
    hover = str(payload.get("inv_hover") or "-")
    inv_n = float(payload.get("inv_n") or 0.0)
    draw.text(
        (rest_x, 772),
        f"物品 LIF  手:{hand[:16]}  悬:{hover[:16]}  n={inv_n:.0f}",
        fill=CYAN,
        font=sm,
    )
    draw.text((inset + 2, 818), "键鼠输出  ·  亮着 = 这一拍按下 / 转头 / 滚轮", fill=TEXT, font=sm)
    max_ch = 6 if bw >= 52 else (5 if bw >= 42 else 4)
    for i, name in enumerate(names[:n_slots]):
        sx = inset + i * (bw + gap)
        draw.text((sx + 4, 850), str(name)[:max_ch], fill=TEXT, font=sm)
    mouse = payload.get("mouse") or {}
    dx, dy = float(mouse.get("dx") or 0.0), float(mouse.get("dy") or 0.0)
    wheel = float(mouse.get("wheel") or 0.0)
    draw.text(
        (inset + 2, 932),
        f"关掉本窗口不停，F9 才停    鼠标 Δ {dx:+.1f}, {dy:+.1f}    轮 {wheel:+.1f}",
        fill=MUTED,
        font=sm,
    )
    return np.asarray(img, dtype=np.uint8)


def _overlay_origin() -> tuple[int, int]:
    """Park the monitor on the right so it does not sit on the Minecraft window."""
    y = 40
    x = 24
    try:
        if sys.platform == "win32":
            import ctypes

            sw = int(ctypes.windll.user32.GetSystemMetrics(0) or 1280)
            x = max(16, sw - WIN_W - 24)
    except Exception:
        x = 24
    return x, y


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
        from atomspike.viz.overlay_win import restore_game_focus

        restore_game_focus(int(hwnd), park_cursor=False, allow_alt=False)
        return
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = int(hwnd)
        if not user32.IsWindow(hwnd):
            return
        try:
            from atomspike.capture.window import hwnds_related

            if hwnds_related(int(user32.GetForegroundWindow() or 0), hwnd):
                return
        except Exception:
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

    def __init__(self, hz: float = 60.0):
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
            y=_overlay_origin()[1],
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
            ox, oy = _overlay_origin()
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", f"{ox},{oy}")
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
                x = ox
            wm = pygame.display.get_wm_info() if hasattr(pygame.display, "get_wm_info") else {}
            hwnd = wm.get("window")
            _decorate_overlay(hwnd, x, oy)
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
        ox, oy = _overlay_origin()
        root.geometry(f"{WIN_W}x{WIN_H}+{ox}+{oy}")
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
        _decorate_overlay(hwnd, ox, oy)
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
