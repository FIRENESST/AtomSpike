"""Live window showing the RGB frames collect-mc is writing to HDF5."""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

from atomspike.viz.snn_scope import _decorate_overlay, _font, _restore_focus, _tk_root_hwnd

BAR_H = 36
BG = (8, 11, 16)
TEXT = (226, 234, 244)
MUTED = (122, 138, 158)
RED = (255, 72, 88)
CYAN = (90, 228, 255)
GREEN = (86, 232, 164)


def view_size(frame_w: int, frame_h: int, *, max_w: int = 960, min_w: int = 320) -> tuple[int, int]:
    w = max(8, int(frame_w))
    h = max(8, int(frame_h))
    if w > max_w:
        h = max(8, int(round(h * max_w / w)))
        w = max_w
    if w < min_w:
        h = max(8, int(round(h * min_w / w)))
        w = min_w
    return w, h


def dummy_capture_frame(width: int = 160, height: int = 90) -> np.ndarray:
    w, h = int(width), int(height)
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = (40 + xx * 180 // max(1, w)).astype(np.uint8)
    frame[:, :, 1] = (30 + yy * 140 // max(1, h)).astype(np.uint8)
    frame[:, :, 2] = 48
    return frame


def compose_preview(
    frame: np.ndarray | None,
    info: dict | None = None,
    *,
    view_w: int = 854,
    view_h: int = 480,
) -> np.ndarray:
    """RGB image for the preview window (scaled store frame + status bar)."""
    vw, vh = int(view_w), int(view_h)
    out = np.empty((vh + BAR_H, vw, 3), dtype=np.uint8)
    out[:] = BG
    if frame is not None:
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        if rgb.shape[0] != vh or rgb.shape[1] != vw:
            img = Image.fromarray(rgb, "RGB").resize((vw, vh), Image.BILINEAR)
            rgb = np.asarray(img, dtype=np.uint8)
        out[:vh] = rgb
    meta = info or {}
    cursor = meta.get("cursor")
    if cursor is not None and str(meta.get("mode") or "") == "gui":
        cx = int(round(float(cursor[0]) * (vw - 1)))
        cy = int(round(float(cursor[1]) * (vh - 1)))
        cx = max(0, min(vw - 1, cx))
        cy = max(0, min(vh - 1, cy))
        x0, x1 = max(0, cx - 8), min(vw, cx + 9)
        y0, y1 = max(0, cy - 8), min(vh, cy + 9)
        out[cy, x0:x1] = CYAN
        out[y0:y1, cx] = CYAN
    img = Image.fromarray(out, "RGB")
    draw = ImageDraw.Draw(img)
    font = _font(15)
    n = int(meta.get("n") or 0)
    fps = float(meta.get("fps") or 0.0)
    keys = str(meta.get("keys") or "-")
    mouse = str(meta.get("mouse") or "")
    mode = str(meta.get("mode") or "look")
    grab = meta.get("grab_ms")
    rec = "REC" if meta.get("rec", True) else "—"
    grab_bit = f"  grab {float(grab):.0f}ms" if grab is not None else ""
    line = f"{rec}  {n} 帧  {fps:.1f}fps  {keys}  {mouse}  {mode}{grab_bit}"
    draw.ellipse((10, vh + 10, 22, vh + 22), fill=RED if meta.get("rec", True) else MUTED)
    draw.text((30, vh + 8), line, fill=TEXT, font=font)
    if vw >= 520:
        draw.text((vw - 168, vh + 8), "关窗口不停采集", fill=MUTED, font=font)
    return np.asarray(img, dtype=np.uint8)


class CapturePreview:
    """30Hz overlay of the stored capture. Prefers pygame; falls back to Tk."""

    def __init__(self, width: int = 854, height: int = 480, hz: float = 30.0):
        self.view_w, self.view_h = view_size(width, height)
        self.win_w = self.view_w
        self.win_h = self.view_h + BAR_H
        self.hz = max(10.0, float(hz))
        self.backend = "none"
        self.alive = False
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._info: dict = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._ok = False
        self._tk = None
        self._photo = None
        self._label = None
        self._ImageTk = None
        self._game_hwnd = 0
        self._overlay = None

    def start(self, *, game_hwnd: int = 0) -> bool:
        self._game_hwnd = int(game_hwnd or 0)
        if sys.platform == "win32":
            return self._start_win32()
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._pygame_loop, name="capture-preview", daemon=True)
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
        frame, info = self._snapshot()
        return compose_preview(frame, info, view_w=self.view_w, view_h=self.view_h)

    def _start_win32(self) -> bool:
        try:
            from atomspike.viz.overlay_win import DetachedOverlay
        except Exception:
            return False
        overlay = DetachedOverlay(
            title="AtomSpike · 采集画面",
            width=self.win_w,
            height=self.win_h,
            x=16,
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

    def push(self, frame: np.ndarray, info: dict | None = None) -> None:
        rgb = np.array(frame, copy=True, dtype=np.uint8)
        packed = dict(info or {})
        packed.setdefault("rec", True)
        with self._lock:
            self._frame = rgb
            self._info = packed

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
                leftover = deadline - time.perf_counter()
                if leftover > 0:
                    time.sleep(leftover)
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

    def _snapshot(self) -> tuple[np.ndarray | None, dict]:
        with self._lock:
            return self._frame, dict(self._info)

    def _pygame_loop(self) -> None:
        try:
            os.environ.setdefault("SDL_VIDEO_ALLOW_SCREENSAVER", "1")
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "16,48")
            os.environ.setdefault("SDL_HINT_FORCE_RAISEWINDOW", "0")
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame

            pygame.init()
            pygame.display.set_caption("AtomSpike · 采集画面")
            flags = pygame.HIDDEN if hasattr(pygame, "HIDDEN") else 0
            screen = pygame.display.set_mode((self.win_w, self.win_h), flags)
            wm = pygame.display.get_wm_info() if hasattr(pygame.display, "get_wm_info") else {}
            hwnd = wm.get("window")
            _decorate_overlay(hwnd, 16, 48)
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
                frame, info = self._snapshot()
                rgb = compose_preview(frame, info, view_w=self.view_w, view_h=self.view_h)
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
        root.title("AtomSpike · 采集画面")
        root.configure(bg="#080b10")
        root.geometry(f"{self.win_w}x{self.win_h}+16+48")
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
        _decorate_overlay(hwnd, 16, 48)
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
            frame, info = self._snapshot()
            rgb = compose_preview(frame, info, view_w=self.view_w, view_h=self.view_h)
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
