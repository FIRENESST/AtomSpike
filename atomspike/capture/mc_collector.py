"""30Hz Minecraft human demo collector: window + keys/mouse + fused game state."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from atomspike.config import AtomSpikeConfig
from atomspike.capture.input_tap import InputTap
from atomspike.capture.mc_state import MC_FACT_FIELDS, MC_STATE_FIELDS, MinecraftStateHub
from atomspike.capture.window import WindowGrabber, client_cursor_norm, list_windows, pick_minecraft_window
from atomspike.data.stream import StreamingDemoWriter
from atomspike.models.actions import ActionSpec


def resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    img = Image.fromarray(frame)
    return np.asarray(img.resize((int(width), int(height)), Image.BILINEAR), dtype=np.uint8)


def _log(msg: str, *, replace: bool = False) -> None:
    if replace:
        sys.stderr.write("\r" + msg.ljust(200)[:200])
    else:
        sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def run_minecraft_collector(
    cfg: AtomSpikeConfig,
    out_path: str | Path,
    *,
    seconds: float | None = None,
    dry_run: bool = False,
    desktop: bool = False,
    region: tuple[int, int, int, int] | None = None,
    log_every_s: float = 1.0,
    append: bool = False,
    preview: bool | None = None,
) -> dict:
    spec = ActionSpec.from_config(cfg.action)
    fps = max(1, int(cfg.capture.fps))
    period = 1.0 / fps
    store_w = int(cfg.capture.store_width)
    store_h = int(cfg.capture.store_height)
    mc = cfg.minecraft

    if not dry_run and not desktop and region is None:
        win = pick_minecraft_window(mc.window_title)
        if win is None:
            titles = [w.title for w in list_windows()][:12]
            raise RuntimeError(
                "No Minecraft window found. Open a singleplayer/LAN world, or pass --desktop / --dry-run. "
                f"Visible windows: {titles}"
            )

    grabber = None if dry_run else WindowGrabber(
        title_substr=mc.window_title,
        monitor=cfg.capture.monitor,
        region=region or cfg.capture.region,
        desktop=desktop,
        dest_size=(store_w, store_h),
    )
    tap = None
    if mc.record_input and not dry_run:
        tap = InputTap(spec, stop_key=mc.stop_key, mark_key=mc.mark_key)
        if not tap.ok:
            tap.close()
            tap = None
            if seconds is None:
                raise RuntimeError(
                    'Keyboard tap needs pynput (pip install "atomspike[capture]"). '
                    "Pass --seconds to record frames without a stop key."
                )
    if dry_run and seconds is None:
        seconds = 2.0
    hub = MinecraftStateHub(
        saves_dir=mc.saves_dir or None,
        sidecar_path=mc.sidecar_path or None,
        udp_host=mc.sidecar_host,
        udp_port=0 if dry_run else mc.sidecar_udp_port,
        poll_saves_hz=mc.poll_saves_hz,
        hud_probe_enabled=mc.hud_probe and not dry_run,
        state_dim=cfg.reasoner.game_state_dim,
    )
    if not dry_run:
        if hub.saves_dirs:
            _log("[collect] saves " + " | ".join(str(p) for p in hub.saves_dirs[:4]))
        else:
            _log("[collect] no saves dir found; pass --saves-dir so hp/xyz can be read from NBT")
    writer = StreamingDemoWriter(
        out_path,
        (store_h, store_w),
        cfg.reasoner.game_state_dim,
        spec.n_slots,
        key_names=spec.key_names,
        state_fields=MC_STATE_FIELDS,
        append=append and not dry_run,
        extra_attrs={
            "fps_target": fps,
            "window_title": mc.window_title,
            "dry_run": int(dry_run),
            "fact_fields": ",".join(MC_FACT_FIELDS),
            "cursor_xy": "window-normalized cursor when GUI; OS client fallback",
        },
    )
    if writer.n > 0:
        _log(f"[collect] appending to existing {out_path} ({writer.n} frames already inside)")
    if getattr(writer, "slow_write", False):
        _log(
            "[collect] this h5 still uses old gzip/32-frame chunks; capture will stay slow. "
            "Finish this session, then --overwrite a new file (or a new --out) for ~30fps."
        )

    t_end = None if seconds is None else (time.perf_counter() + max(0.05, float(seconds)))
    latencies: list[float] = []
    n = 0
    n_marks = 0
    wall0 = time.perf_counter()
    target_next = wall0
    last_log = wall0
    win = None if grabber is None else grabber.window
    if win is not None:
        _log(f"[collect] window {win.width}x{win.height} at {win.left},{win.top}  title={win.title!r}")
    elif dry_run:
        _log("[collect] dry-run (dummy frames)")
    elif desktop:
        _log("[collect] capturing whole monitor")
    _log(f"[collect] writing {out_path}  stop={mc.stop_key}  mark={mc.mark_key}  live log every {log_every_s:.1f}s")
    if preview is None:
        preview = not dry_run
    scope = None
    preview_backend = "off"
    if preview:
        from atomspike.viz.capture_preview import CapturePreview

        try:
            scope = CapturePreview(width=store_w, height=store_h, hz=float(fps))
            game_hwnd = int(getattr(win, "hwnd", 0) or 0)
            if not scope.start(game_hwnd=game_hwnd):
                scope.close()
                scope = None
                preview_backend = "unavailable"
                _log("[collect] capture preview window unavailable")
            else:
                preview_backend = scope.backend
                _log(f"[collect] preview window ({preview_backend}) shows the {store_w}x{store_h} store frames")
        except Exception as exc:
            if scope is not None:
                try:
                    scope.close()
                except Exception:
                    pass
            scope = None
            preview_backend = "unavailable"
            _log(f"[collect] capture preview failed: {exc}")
    try:
        while True:
            if t_end is not None and time.perf_counter() >= t_end:
                break
            tick0 = time.perf_counter()
            if dry_run:
                frame = np.zeros((store_h, store_w, 3), dtype=np.uint8)
                frame[:, :, 1] = 40
                t_ns = time.time_ns()
                grab_ms = 0.0
            else:
                assert grabber is not None
                raw, t_ns = grabber.grab()
                grab_ms = (time.perf_counter() - tick0) * 1000.0
                if raw.shape[0] == store_h and raw.shape[1] == store_w:
                    frame = raw
                else:
                    frame = resize_rgb(raw, store_w, store_h)
            snap = tap.snapshot() if tap is not None else None
            if snap is not None and snap.stop:
                break
            win = None if grabber is None else grabber.window
            os_cx, os_cy = client_cursor_norm(win)
            extra = {"cursor_x": os_cx, "cursor_y": os_cy}
            if snap is not None:
                extra["wheel"] = float(snap.wheel)
            kd: dict[str, bool] = snap.keys_down if snap is not None else {}
            state = hub.snapshot(frame if not dry_run else None, kd, extra=extra)
            raw = hub.last_raw
            gui_open = bool(raw.get("gui_open"))
            cx = float(np.clip(float(raw.get("mouse_nx", raw.get("cursor_x", os_cx))), 0.0, 1.0))
            cy = float(np.clip(float(raw.get("mouse_ny", raw.get("cursor_y", os_cy))), 0.0, 1.0))
            tokens = (
                tap.to_tokens(snap, gui_open=gui_open, cursor_norm=(cx, cy))
                if (tap is not None and snap is not None)
                else np.asarray(spec.idle_tokens(), dtype=np.int64)
            )
            keys_held = None
            mouse_xy = (0.0, 0.0)
            if snap is not None:
                mouse_xy = (float(snap.mouse_dx), float(snap.mouse_dy))
                keys_held = np.asarray([1 if kd.get(n, False) else 0 for n in spec.key_names], dtype=np.uint8)
            done = bool(snap.mark) if snap is not None else False
            writer.add(
                frame,
                state,
                tokens,
                0.0,
                done,
                t_ns,
                mouse_xy=mouse_xy,
                keys_held=keys_held,
                facts=hub.last_facts(),
                cursor_xy=(cx, cy),
                wheel=float(raw.get("wheel", extra.get("wheel", 0.0))),
            )
            n += 1
            if done:
                n_marks += 1
                sys.stderr.write("\n")
                _log(f"[collect] episode mark #{n_marks} at frame {n}")
            now = time.perf_counter()
            dt = now - tick0
            latencies.append(dt * 1000.0)
            if scope is not None:
                elapsed_now = max(1e-6, now - wall0)
                held_now = ",".join(name for name, on in kd.items() if on) or "-"
                if gui_open:
                    mouse_bit = f"cursor={cx:.2f},{cy:.2f}"
                    mode = "gui"
                else:
                    mouse_bit = f"mouse={mouse_xy[0]:+.0f},{mouse_xy[1]:+.0f}"
                    mode = "look"
                scope.push(
                    frame,
                    {
                        "n": n,
                        "fps": n / elapsed_now,
                        "keys": held_now,
                        "mouse": mouse_bit,
                        "mode": mode,
                        "grab_ms": grab_ms,
                        "cursor": (cx, cy),
                        "rec": True,
                    },
                )
            if log_every_s > 0 and (now - last_log) >= log_every_s:
                elapsed = max(1e-6, now - wall0)
                held = ",".join(name for name, on in kd.items() if on) or "-"
                facts = hub.last_facts()
                src = hub.source_name()
                lmb = "L" if (snap is not None and snap.lmb) else ""
                rmb = "R" if (snap is not None and snap.rmb) else ""
                click = (lmb + rmb) or "-"
                gui = "gui" if gui_open else "look"
                hover = raw.get("hovered_slot", -1)
                mouse_bit = (
                    f"cursor={cx:.2f},{cy:.2f} hover={hover}"
                    if gui_open
                    else f"mouse={mouse_xy[0]:+.0f},{mouse_xy[1]:+.0f}"
                )
                _log(
                    f"[collect] {elapsed:6.1f}s  n={n:<6} {n / elapsed:5.1f}fps  "
                    f"grab={grab_ms:5.1f}ms  tick={dt * 1000:5.1f}ms  keys={held}  "
                    f"{mouse_bit} click={click} {gui}  "
                    f"hp={facts[5]:.0f} hung={facts[6]:.0f}  "
                    f"xyz={facts[0]:.1f},{facts[1]:.1f},{facts[2]:.1f}  "
                    f"look={facts[3]:.0f},{facts[4]:.0f}  src={src}",
                    replace=True,
                )
                last_log = now
            target_next += period
            sleep_for = target_next - time.perf_counter()
            if sleep_for > 0:
                if scope is not None and scope.backend == "tk":
                    scope.idle(sleep_for)
                else:
                    time.sleep(sleep_for)
            else:
                target_next = time.perf_counter()
    finally:
        path = writer.close()
        hub.close()
        if tap is not None:
            tap.close()
        if grabber is not None:
            grabber.close()
        if scope is not None:
            scope.close()

    elapsed = max(1e-6, time.perf_counter() - wall0)
    _log("")
    _log(
        f"[collect] done  frames={n}  {n / elapsed:.1f}fps  "
        f"p95={float(np.percentile(latencies, 95)) if latencies else 0.0:.1f}ms  "
        f"marks={n_marks}  path={path}"
    )
    return {
        "path": str(path),
        "n_frames": n,
        "fps_target": fps,
        "actual_fps": n / elapsed,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "store": [store_h, store_w],
        "state_dim": cfg.reasoner.game_state_dim,
        "state_fields": list(MC_STATE_FIELDS),
        "fact_fields": list(MC_FACT_FIELDS),
        "n_slots": spec.n_slots,
        "dry_run": dry_run,
        "elapsed_s": elapsed,
        "preview": preview_backend not in {"off", "unavailable"},
        "preview_backend": preview_backend,
    }
