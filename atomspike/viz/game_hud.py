"""Click-through HUD on the Minecraft client: latent grid, influence, vectors.

Composes a chroma-keyed RGB frame on a side thread. Magenta pixels are
transparent; boxes, arrows, and a thin side panel sit over the game.
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHROMA = (255, 0, 255)
INK = (236, 240, 244)
MUTED = (186, 196, 208)
BAR = (18, 22, 28)
CYAN = (80, 214, 255)
AMBER = (255, 186, 74)
OBJ_MIN = 0.08
INFL_MIN = 0.04
VEC_MIN = 0.08
SEA_MIN = 0.04
CHG_KEEP = 3
COMPASS = ("右", "右上", "上", "左上", "左", "左下", "下", "右下")

# The classifier currently has no semantic ground truth. Keep its colors
# useful for debugging without presenting arbitrary latent indices as objects.
LATENT_RGB = (
    (130, 198, 255), (196, 154, 92), (255, 196, 64), (255, 86, 86),
    (92, 232, 150), (64, 156, 255), (96, 214, 96), (255, 176, 88),
)


def _kind_name(idx: int) -> str:
    return f"潜类{int(idx)}"


def _kind_rgb(idx: int) -> tuple[int, int, int]:
    return LATENT_RGB[int(idx) % len(LATENT_RGB)]


def _short(text: str, n: int = 18) -> str:
    s = str(text or "").replace("\\", "/").split("/")[-1].strip()
    if s.startswith("minecraft:"):
        s = s[10:]
    return s if len(s) <= n else (s[: n - 1] + "…")


def _arr(v, dtype) -> np.ndarray:
    if v is None:
        return np.zeros(0, dtype=dtype)
    return np.asarray(v, dtype=dtype).reshape(-1)


def _hw2(v) -> tuple[int, int]:
    if v is None:
        return (0, 0)
    if isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 2:
        return int(v[0] or 0), int(v[1] or 0)
    return (0, 0)


def _cell_box(row: int, col: int, gh: int, gw: int, w: int, h: int, pad: int = 1) -> tuple[int, int, int, int]:
    x0 = int(col * w / gw) + pad
    y0 = int(row * h / gh) + pad
    x1 = max(x0 + 6, int((col + 1) * w / gw) - pad)
    y1 = max(y0 + 6, int((row + 1) * h / gh) - pad)
    return x0, y0, x1, y1


def compass_label(vx: float, vy: float) -> str:
    """Image space: +x right, +y down. Screen-up is negative vy."""
    if abs(vx) + abs(vy) < 1e-8:
        return "静"
    ang = math.atan2(-float(vy), float(vx))
    i = int(((ang + math.pi / 8) % (2 * math.pi)) / (math.pi / 4)) % 8
    return COMPASS[i]


def merge_detect_boxes(
    idx,
    cls,
    obj,
    infl=None,
    conf=None,
    hw: tuple[int, int] = (0, 0),
    obj_min: float = OBJ_MIN,
) -> list[dict]:
    """4-connected same-latent components among firing grid cells."""
    gh, gw = int(hw[0] or 0), int(hw[1] or 0)
    if gh < 2 or gw < 2:
        return []
    ids = _arr(idx, np.int32)
    kinds = _arr(cls, np.int32)
    scores = _arr(obj, np.float32)
    weights = _arr(infl, np.float32) if infl is not None else scores
    confs = _arr(conf, np.float32) if conf is not None else np.ones_like(scores)
    n = min(ids.size, kinds.size, scores.size, weights.size, confs.size)
    from atomspike.models.curl_scene import chrome_keep_map

    world = chrome_keep_map(gh, gw)
    grid_c = np.full(gh * gw, -1, dtype=np.int32)
    grid_o = np.zeros(gh * gw, dtype=np.float32)
    grid_w = np.zeros(gh * gw, dtype=np.float32)
    grid_p = np.zeros(gh * gw, dtype=np.float32)
    for i in range(n):
        if float(scores[i]) < obj_min:
            continue
        cell = int(ids[i])
        if cell < 0 or cell >= gh * gw:
            continue
        row, col = divmod(cell, gw)
        if not bool(world[row, col]):
            continue
        grid_c[cell] = int(kinds[i])
        grid_o[cell] = max(grid_o[cell], float(scores[i]))
        grid_w[cell] = max(grid_w[cell], float(weights[i]))
        grid_p[cell] = max(grid_p[cell], float(confs[i]))
    seen = np.zeros(gh * gw, dtype=bool)
    boxes: list[dict] = []
    for start in range(gh * gw):
        if seen[start] or int(grid_c[start]) < 0:
            continue
        kind = int(grid_c[start])
        stack = [start]
        seen[start] = True
        cells: list[int] = []
        while stack:
            u = stack.pop()
            cells.append(u)
            r, c = divmod(u, gw)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < gh and 0 <= cc < gw:
                    v = rr * gw + cc
                    if not seen[v] and int(grid_c[v]) == kind:
                        seen[v] = True
                        stack.append(v)
        rs = [divmod(u, gw)[0] for u in cells]
        cs = [divmod(u, gw)[1] for u in cells]
        boxes.append(
            {
                "cls": kind,
                "obj": float(max(grid_o[u] for u in cells)),
                "infl": float(sum(grid_w[u] for u in cells)),
                "conf": float(max(grid_p[u] for u in cells)),
                "r0": int(min(rs)),
                "r1": int(max(rs) + 1),
                "c0": int(min(cs)),
                "c1": int(max(cs) + 1),
                "n": len(cells),
                "cx": (min(cs) + max(cs) + 1) / (2.0 * gw),
                "cy": (min(rs) + max(rs) + 1) / (2.0 * gh),
            }
        )
    total = sum(float(b["infl"]) for b in boxes) or 1.0
    for b in boxes:
        b["share"] = float(b["infl"]) / total
    boxes.sort(key=lambda b: -float(b["infl"]))
    return boxes


@lru_cache(maxsize=8)
def _font(size: int):
    size = max(10, int(size))
    for name in (
        "msyh.ttc",
        "msyh.ttf",
        "Microsoft YaHei.ttf",
        "segoeui.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def dummy_overlay_snap() -> dict:
    gh, gw = 4, 6
    vx = np.zeros((gh, gw), dtype=np.float16)
    vy = np.zeros((gh, gw), dtype=np.float16)
    vx[1, 3] = 0.4
    vy[1, 3] = 0.2
    chg = np.zeros((gh, gw), dtype=np.float16)
    chg[2, 2] = 0.9
    return {
        "enabled": True,
        "hw": (gh, gw),
        "idx": np.asarray([6, 7, 14], dtype=np.int16),
        "cls": np.asarray([2, 2, 3], dtype=np.int16),
        "obj": np.asarray([1.0, 0.8, 0.7], dtype=np.float32),
        "infl": np.asarray([0.45, 0.30, 0.25], dtype=np.float32),
        "conf": np.asarray([0.82, 0.70, 0.61], dtype=np.float32),
        "n": 24,
        "k": 3,
        "sea_idx": np.asarray([7], dtype=np.int16),
        "sea_w": np.asarray([0.55], dtype=np.float32),
        "sea_hw": (gh, gw),
        "vx": vx,
        "vy": vy,
        "vec_hw": (gh, gw),
        "chg": chg,
        "iter_hw": (gh, gw),
        "latency_ms": 8.0,
        "perception": True,
        "keys": ["W", "LMB"],
        "task": "mine_stone",
        "hand": "stone_pickaxe",
        "gui": False,
        "look_miss": False,
    }


def _draw_arrow(draw: ImageDraw.ImageDraw, x: float, y: float, dx: float, dy: float, color, width: int = 2) -> None:
    x2 = x + dx
    y2 = y + dy
    draw.line((x, y, x2, y2), fill=color, width=width)
    mag = math.hypot(dx, dy)
    if mag < 6:
        return
    ux, uy = dx / mag, dy / mag
    hx, hy = -uy, ux
    draw.line((x2, y2, x2 - 6 * ux + 4 * hx, y2 - 6 * uy + 4 * hy), fill=color, width=width)
    draw.line((x2, y2, x2 - 6 * ux - 4 * hx, y2 - 6 * uy - 4 * hy), fill=color, width=width)


def _vec_stats(vx: np.ndarray | None, vy: np.ndarray | None) -> tuple[float, float, float, float]:
    if vx is None or vy is None:
        return 0.0, 0.0, 0.0, 0.0
    ax = np.asarray(vx, dtype=np.float32)
    ay = np.asarray(vy, dtype=np.float32)
    if ax.size == 0 or ax.shape != ay.shape:
        return 0.0, 0.0, 0.0, 0.0
    if ax.ndim == 2:
        from atomspike.models.curl_scene import chrome_keep_map

        keep = chrome_keep_map(ax.shape[0], ax.shape[1])
        if bool(keep.any()):
            ax = np.where(keep, ax, 0.0)
            ay = np.where(keep, ay, 0.0)
            n = float(keep.sum()) or 1.0
            mx = float(ax.sum() / n)
            my = float(ay.sum() / n)
        else:
            mx = float(ax.mean())
            my = float(ay.mean())
    else:
        mx = float(ax.mean())
        my = float(ay.mean())
    mag = math.hypot(mx, my)
    curl = 0.0
    if ax.ndim == 2 and ax.shape[0] > 1 and ax.shape[1] > 1:
        dvy_dx = np.diff(ay, axis=1)[:-1, :]
        dvx_dy = np.diff(ax, axis=0)[:, :-1]
        curl = float(np.mean(np.abs(dvy_dx - dvx_dy)))
    return mx, my, mag, curl


def compose_hud(snap: dict | None, width: int, height: int, standalone: bool = False) -> np.ndarray:
    """RGB frame filled with chroma; boxes/labels are the only opaque pixels."""
    w = max(64, int(width))
    h = max(48, int(height))
    img = Image.new("RGB", (w, h), (12, 14, 18) if standalone else CHROMA)
    draw = ImageDraw.Draw(img)
    sm = _font(13 if w >= 800 else 11)
    lg = _font(15 if w >= 800 else 12)
    snap = snap or {}
    hw = _hw2(snap.get("hw"))
    gh, gw = hw
    boxes = merge_detect_boxes(
        snap.get("idx"),
        snap.get("cls"),
        snap.get("obj"),
        infl=snap.get("infl"),
        conf=snap.get("conf"),
        hw=hw,
    )
    view_w = w
    panel_w = 188 if w >= 720 else 0

    bar_h = 42
    ranks = boxes[:5]
    for i, box in enumerate(boxes):
        x0 = int(box["c0"] * view_w / gw) if gw else 0
        y0 = int(box["r0"] * h / gh) if gh else 0
        x1 = max(x0 + 8, int(box["c1"] * view_w / gw) - 1) if gw else x0
        y1 = max(y0 + 8, int(box["r1"] * h / gh) - 1) if gh else y0
        y0 = max(bar_h, y0)
        y1 = max(y0 + 8, y1)
        x1 = max(x0 + 8, x1)
        if y1 <= y0 or x1 <= x0:
            continue
        color = _kind_rgb(int(box["cls"]))
        thick = 3 if i == 0 else 2
        draw.rectangle((x0, y0, x1, y1), outline=color, width=thick)
        draw.rectangle((x0, y0, x0 + 8, y0 + 8), fill=color)
        draw.rectangle((x1 - 8, y0, x1, y0 + 8), fill=color)
        draw.rectangle((x0, y1 - 8, x0 + 8, y1), fill=color)
        draw.rectangle((x1 - 8, y1 - 8, x1, y1), fill=color)
        label = f"{_kind_name(int(box['cls']))} 分值{float(box['conf']):.0%} 权{100.0 * float(box['share']):.0f}%"
        ty = y0 + 4 if y0 < bar_h + 8 else y0 - 17
        tw = min(view_w - 4, x0 + 8 + 7 * len(label))
        draw.rectangle((x0, ty, tw, ty + 16), fill=BAR)
        draw.text((x0 + 4, ty), label, fill=color, font=sm)
        if i < 5:
            badge = f"{i + 1}"
            draw.ellipse((x0 + 2, y0 + 2, x0 + 16, y0 + 16), fill=BAR, outline=color)
            draw.text((x0 + 5, y0 + 2), badge, fill=INK, font=sm)

    sea_idx = _arr(snap.get("sea_idx"), np.int32)
    sea_w = _arr(snap.get("sea_w"), np.float32)
    sh, sw = _hw2(snap.get("sea_hw"))
    if sh >= 2 and sw >= 2:
        from atomspike.models.curl_scene import chrome_keep_map

        sea_world = chrome_keep_map(sh, sw)
        nsea = min(sea_idx.size, sea_w.size)
        for i in range(nsea):
            wt = float(sea_w[i])
            if wt < SEA_MIN:
                continue
            cell = int(sea_idx[i])
            if cell < 0:
                continue
            row, col = divmod(cell, sw)
            if row >= sh or col >= sw or not bool(sea_world[row, col]):
                continue
            x0, y0, x1, y1 = _cell_box(row, col, sh, sw, view_w, h, pad=3)
            y0 = max(bar_h, y0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = max(5, min(12, int(6 + 10 * wt)))
            draw.polygon(
                [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                outline=CYAN,
            )

    vx = snap.get("vx")
    vy = snap.get("vy")
    vh, vw = _hw2(snap.get("vec_hw"))
    mx, my, mag, curl_e = _vec_stats(
        None if vx is None else np.asarray(vx),
        None if vy is None else np.asarray(vy),
    )
    if vx is not None and vy is not None and vh >= 2 and vw >= 2:
        from atomspike.models.curl_scene import chrome_keep_map

        try:
            ax = np.asarray(vx, dtype=np.float32).reshape(-1)
            ay = np.asarray(vy, dtype=np.float32).reshape(-1)
            if int(ax.size) != vh * vw or int(ay.size) != vh * vw:
                ax = ay = None
            else:
                ax = ax.reshape(vh, vw)
                ay = ay.reshape(vh, vw)
        except Exception:
            ax = ay = None
        if ax is not None and ay is not None:
            world = chrome_keep_map(vh, vw)
            mag_map = np.hypot(np.where(world, ax, 0.0), np.where(world, ay, 0.0))
            peak = float(np.max(mag_map))
            span = min(view_w / vw, (h - bar_h) / vh) * 0.55
            if peak > 1e-4:
                for r in range(vh):
                    for c in range(vw):
                        if not bool(world[r, c]):
                            continue
                        dx, dy = float(ax[r, c]), float(ay[r, c])
                        m = math.hypot(dx, dy)
                        if m < 0.45 * peak:
                            continue
                        cx = (c + 0.5) * view_w / vw
                        cy = max(bar_h + 8, (r + 0.5) * h / vh)
                        _draw_arrow(draw, cx, cy, (dx / peak) * span, (dy / peak) * span, AMBER, 2)

    chg = snap.get("chg")
    ih, iw = _hw2(snap.get("iter_hw"))
    if chg is not None and ih >= 2 and iw >= 2:
        from atomspike.models.curl_scene import chrome_keep_map

        cmap = np.asarray(chg, dtype=np.float32).reshape(ih, iw)
        world = chrome_keep_map(ih, iw)
        cmap = np.where(world, cmap, 0.0)
        flat = cmap.reshape(-1)
        if flat.size:
            keep = min(CHG_KEEP, int(flat.size))
            top = np.argpartition(flat, -keep)[-keep:]
            peak = float(flat.max()) or 1.0
            for cell in top:
                val = float(flat[int(cell)])
                if val < 0.12 * peak:
                    continue
                row, col = divmod(int(cell), iw)
                if not bool(world[row, col]):
                    continue
                x0, y0, x1, y1 = _cell_box(row, col, ih, iw, view_w, h, pad=6)
                y0 = max(bar_h, y0)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                s = 4
                draw.line((cx - s, cy, cx + s, cy), fill=AMBER, width=2)
                draw.line((cx, cy - s, cx, cy + s), fill=AMBER, width=2)

    status = "感知" if snap.get("perception") else "复用"
    if snap.get("gui"):
        status = "GUI冻结"
    keys = snap.get("keys") or []
    key_s = " ".join(str(k) for k in keys[:5]) if keys else "—"
    hand = _short(str(snap.get("hand") or ""), 14)
    task = _short(str(snap.get("task") or ""), 16)
    look = "看空" if snap.get("look_miss") else ""
    lat = float(snap.get("latency_ms") or 0.0)
    grid = f"{gh}×{gw}" if gh and gw else "关"
    top1 = ""
    if ranks:
        top1 = f"  最大权 {_kind_name(int(ranks[0]['cls']))} {100.0 * float(ranks[0]['share']):.0f}%"
    line1 = f"识别 {grid}  {len(boxes)}框  {status}  {lat:.0f}ms{top1}"
    if task:
        line1 += f"  {task}"
    heading = compass_label(mx, my)
    line2 = f"矢量 {heading}  ({mx:+.2f},{my:+.2f})  |v|={mag:.2f}  旋度 {curl_e:.2f}"
    if hand:
        line2 += f"  手:{hand}"
    if look:
        line2 += f"  {look}"
    line2 += f"  {key_s}"
    draw.rectangle((0, 0, w, bar_h), fill=BAR)
    draw.text((8, 2), line1, fill=INK, font=lg)
    draw.text((8, 22), line2, fill=MUTED, font=sm)

    if panel_w:
        x = w - panel_w
        draw.rectangle((x, bar_h, w, h), fill=BAR)
        draw.line((x, bar_h, x, h), fill=(48, 56, 66), width=1)
        y = bar_h + 8
        draw.text((x + 10, y), "对象 / 权重", fill=INK, font=lg)
        y += 20
        if not ranks:
            draw.text((x + 10, y), "无发放对象", fill=MUTED, font=sm)
            y += 18
        for i, box in enumerate(ranks):
            name = _kind_name(int(box["cls"]))
            color = _kind_rgb(int(box["cls"]))
            draw.rectangle((x + 10, y + 3, x + 20, y + 13), fill=color)
            draw.text(
                (x + 26, y),
                f"{i + 1} {name}  权{100.0 * float(box['share']):.0f}%  {float(box['conf']):.0%}",
                fill=INK,
                font=sm,
            )
            y += 16
            draw.text(
                (x + 26, y),
                f"格{int(box['n'])}  ({float(box['cx']):.2f},{float(box['cy']):.2f})",
                fill=MUTED,
                font=sm,
            )
            y += 18
        y += 6
        draw.text((x + 10, y), "空间矢量", fill=INK, font=lg)
        y += 20
        draw.text((x + 10, y), f"方向  {heading}", fill=AMBER, font=sm)
        y += 16
        draw.text((x + 10, y), f"mean  ({mx:+.2f}, {my:+.2f})", fill=MUTED, font=sm)
        y += 16
        draw.text((x + 10, y), f"|v| {mag:.2f}   旋度 {curl_e:.2f}", fill=MUTED, font=sm)
        y += 20
        draw.text((x + 10, y), "符合注意", fill=CYAN, font=lg)
        y += 18
        nsea = min(sea_idx.size, sea_w.size, 4)
        if nsea <= 0:
            draw.text((x + 10, y), "无事件格", fill=MUTED, font=sm)
        else:
            order = np.argsort(-sea_w[:nsea])
            for j in order:
                wt = float(sea_w[int(j)])
                cell = int(sea_idx[int(j)])
                draw.text((x + 10, y), f"格 {cell}  注意 {100.0 * wt:.0f}%", fill=CYAN, font=sm)
                y += 16
    return np.asarray(img, dtype=np.uint8)


class GameHud:
    """Separate recognition window. Closing it does not stop prediction."""

    def __init__(self, game_hwnd: int, hz: float = 15.0):
        self.game_hwnd = int(game_hwnd or 0)
        self.hz = max(8.0, float(hz))
        self._lock = threading.Lock()
        self._snap: dict = {}
        self._overlay = None
        self.backend = "off"

    def start(self) -> bool:
        try:
            from atomspike.viz.overlay_win import DetachedOverlay
        except Exception:
            return False
        overlay = DetachedOverlay(
            title="AtomSpike · 识别",
            width=1100,
            height=680,
            x=16,
            y=32,
            hz=self.hz,
            game_hwnd=self.game_hwnd,
        )
        self._overlay = overlay
        if not overlay.start(self._compose):
            self._overlay = None
            overlay.close()
            return False
        self.backend = "win32"
        return True

    def push(self, snap: dict | None) -> None:
        if not snap:
            return
        with self._lock:
            self._snap = dict(snap)

    def pump(self) -> bool:
        ov = self._overlay
        if ov is None:
            return False
        return bool(ov.alive)

    def close(self) -> None:
        ov = self._overlay
        self._overlay = None
        self.backend = "off"
        if ov is not None:
            try:
                ov.close()
            except Exception:
                pass

    def _compose(self):
        ov = self._overlay
        inner = getattr(ov, "_inner", ov)
        w = int(getattr(inner, "width", 1100) or 1100)
        h = int(getattr(inner, "height", 680) or 680)
        with self._lock:
            snap = self._snap
        try:
            return compose_hud(snap, w, h, standalone=True)
        except Exception:
            try:
                return compose_hud({}, w, h, standalone=True)
            except Exception:
                return None


def compose_status(snap: dict | None, width: int, height: int) -> np.ndarray:
    """Text board: controls, task, vectors, detections, learn fields."""
    w = max(320, int(width))
    h = max(240, int(height))
    img = Image.new("RGB", (w, h), (12, 14, 18))
    draw = ImageDraw.Draw(img)
    font = _font(15 if w >= 480 else 13)
    small = _font(13)
    snap = snap or {}
    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append((f"延迟 {float(snap.get('latency_ms') or 0):.1f} ms   {'感知' if snap.get('perception') else '策略'}", INK))
    lines.append((f"任务 {_short(str(snap.get('task') or '—'), 28)}", AMBER))
    lines.append((f"手持 {_short(str(snap.get('hand') or '—'), 24)}   GUI {'开' if snap.get('gui') else '关'}", INK))
    keys = snap.get("keys") or []
    lines.append(("按键 " + (" ".join(str(k) for k in keys) if keys else "—"), CYAN))
    vx, vy = snap.get("vx"), snap.get("vy")
    mx, my, mag, curl_e = _vec_stats(None if vx is None else np.asarray(vx), None if vy is None else np.asarray(vy))
    lines.append((f"矢量 {compass_label(mx, my)}  ({mx:+.3f},{my:+.3f})  |v| {mag:.3f}", AMBER))
    lines.append((f"旋度 {curl_e:.3f}   看空 {'是' if snap.get('look_miss') else '否'}", MUTED))
    learn = snap.get("learn") if isinstance(snap.get("learn"), dict) else {}
    if learn:
        lines.append((
            f"学习 {learn.get('learn_mode', '')}  步 {learn.get('learn_steps', 0)}  "
            f"loss {float(learn.get('learn_loss') or 0):.3f}",
            INK,
        ))
        lines.append((
            f"奖励 {float(learn.get('reward') or 0):+.3f}  意外 {float(learn.get('surprise') or 0):.3f}  "
            f"回放 {float(learn.get('replay_rms') or 0):.3f}",
            MUTED,
        ))
    lines.append(("识别框", INK))
    boxes = merge_detect_boxes(
        snap.get("idx"), snap.get("cls"), snap.get("obj"),
        infl=snap.get("infl"), conf=snap.get("conf"), hw=_hw2(snap.get("hw")),
    )
    if not boxes:
        lines.append(("  无", MUTED))
    for i, box in enumerate(boxes[:8]):
        lines.append((
            f"  {i + 1} {_kind_name(int(box['cls']))}  分值 {float(box['conf']):.0%}  "
            f"权 {100 * float(box['share']):.0f}%  格 {int(box['n'])}",
            _kind_rgb(int(box["cls"])),
        ))
    sea_idx = _arr(snap.get("sea_idx"), np.int32)
    sea_w = _arr(snap.get("sea_w"), np.float32)
    lines.append(("屏幕事件", CYAN))
    if sea_idx.size == 0:
        lines.append(("  无", MUTED))
    else:
        order = np.argsort(-sea_w[: min(6, sea_w.size)])
        for j in order:
            lines.append((f"  格 {int(sea_idx[int(j)])}  注意 {100 * float(sea_w[int(j)]):.0f}%", CYAN))
    y = 12
    for text, color in lines:
        if y > h - 22:
            break
        draw.text((12, y), text, fill=color, font=font if y < 120 else small)
        y += 22
    return np.asarray(img, dtype=np.uint8)


class InfoBoard:
    """Separate status window. Closing it does not stop prediction."""

    def __init__(self, game_hwnd: int, hz: float = 8.0):
        self.game_hwnd = int(game_hwnd or 0)
        self.hz = max(4.0, float(hz))
        self._lock = threading.Lock()
        self._snap: dict = {}
        self._overlay = None
        self.backend = "off"

    def start(self) -> bool:
        try:
            from atomspike.viz.overlay_win import DetachedOverlay
        except Exception:
            return False
        overlay = DetachedOverlay(
            title="AtomSpike · 状态",
            width=520,
            height=760,
            x=1140,
            y=32,
            hz=self.hz,
            game_hwnd=self.game_hwnd,
        )
        self._overlay = overlay
        if not overlay.start(self._compose):
            self._overlay = None
            overlay.close()
            return False
        self.backend = "win32"
        return True

    def push(self, snap: dict | None) -> None:
        if not snap:
            return
        with self._lock:
            self._snap = dict(snap)

    def pump(self) -> bool:
        ov = self._overlay
        if ov is None:
            return False
        return bool(ov.alive)

    def close(self) -> None:
        ov = self._overlay
        self._overlay = None
        self.backend = "off"
        if ov is not None:
            try:
                ov.close()
            except Exception:
                pass

    def _compose(self):
        inner = getattr(self._overlay, "_inner", self._overlay)
        w = int(getattr(inner, "width", 520) or 520)
        h = int(getattr(inner, "height", 760) or 760)
        with self._lock:
            snap = self._snap
        try:
            return compose_status(snap, w, h)
        except Exception:
            return compose_status({}, w, h)
