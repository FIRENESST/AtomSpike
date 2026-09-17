"""Live 30Hz Minecraft prediction: window + fused state -> action tokens.

Research-only, singleplayer / LAN. By default it only logs tokens.
With inject=True it closes the loop through the OS (F9 emergency stop).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from atomspike.capture.input_tap import InputTap
from atomspike.capture.mc_collector import resize_rgb
from atomspike.capture.mc_state import MC_FACT_FIELDS, MC_STATE_FIELDS, MinecraftStateHub
from atomspike.capture.window import WindowGrabber, client_cursor_norm, list_windows, pick_minecraft_window
from atomspike.config import AtomSpikeConfig
from atomspike.models.agent import AtomSpikeAgent
from atomspike.models.actions import ActionSpec
from atomspike.runtime.loop import DualRateLoop
from atomspike.train.common import load_agent, resolve_device


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_minecraft_predict(
    cfg: AtomSpikeConfig,
    ckpt: str | Path | None,
    *,
    seconds: float | None = None,
    dry_run: bool = False,
    desktop: bool = False,
    region: tuple[int, int, int, int] | None = None,
    dump_path: str | Path | None = None,
    agent: AtomSpikeAgent | None = None,
    inject: bool = False,
    sample: bool = False,
    greedy: bool = False,
    explore: float | None = None,
    viz: bool | None = None,
    viz_hz: float = 120.0,
) -> dict:
    spec = ActionSpec.from_config(cfg.action)
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

    device = resolve_device(cfg.train.device)
    if agent is None:
        if ckpt is None:
            agent = AtomSpikeAgent(cfg).to(device)
        else:
            agent = load_agent(ckpt, cfg, device)
    agent.eval()
    if explore is not None:
        cfg.planner.explore = max(0.0, min(1.0, float(explore)))
    if greedy:
        cfg.planner.creative = False
        cfg.planner.explore = 0.0
        sample = False
    loop = DualRateLoop(agent, cfg, device, mode="realtime", pace=True)

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
    dump_f = None
    if dump_path is not None:
        dump_p = Path(dump_path)
        dump_p.parent.mkdir(parents=True, exist_ok=True)
        dump_f = dump_p.open("w", encoding="utf-8")
    injector = None
    if inject and not dry_run:
        from atomspike.capture import InputInjector

        injector = InputInjector(enabled=True)

    if not dry_run:
        print(
            "[predict] click into Minecraft so the mouse locks in the game. starting in 3s...",
            flush=True,
        )
        time.sleep(3.0)
        if sys.platform == "win32":
            try:
                from atomspike.viz.overlay_win import detach_console_keep_stdio

                detach_console_keep_stdio()
            except Exception:
                pass

    if viz is None:
        viz = not dry_run
    scope = None
    viz_backend = "off"
    game_hwnd = int(getattr(getattr(grabber, "window", None), "hwnd", 0) or 0)
    if viz:
        from atomspike.viz.snn_scope import SNNScope

        agent.set_activity_trace(True)
        scope = SNNScope(hz=viz_hz)
        if not scope.start(game_hwnd=game_hwnd):
            scope = None
            viz_backend = "unavailable"
        else:
            viz_backend = scope.backend
            if scope.backend == "tk":
                loop.clock.idle_fn = scope.idle

    t_end = None if seconds is None else (time.perf_counter() + max(0.05, float(seconds)))
    latencies: list[float] = []
    n = 0
    last_controls = None
    last_tokens = None
    wall0 = time.perf_counter()
    try:
        while True:
            if t_end is not None and time.perf_counter() >= t_end:
                break
            if dry_run:
                frame = np.zeros((store_h, store_w, 3), dtype=np.uint8)
                frame[:, :, 2] = 40
            else:
                assert grabber is not None
                raw, _t_ns = grabber.grab()
                if raw.shape[0] == store_h and raw.shape[1] == store_w:
                    frame = raw
                else:
                    frame = resize_rgb(raw, store_w, store_h)
            snap = tap.snapshot() if tap is not None else None
            if snap is not None and snap.stop:
                break
            kd = snap.keys_down if snap is not None else {}
            win = None if grabber is None else grabber.window
            os_cx, os_cy = client_cursor_norm(win)
            extra = {"cursor_x": os_cx, "cursor_y": os_cy}
            if snap is not None:
                extra["wheel"] = float(snap.wheel)
            state = hub.snapshot(frame if not dry_run else None, kd, extra=extra)
            tokens, info = loop.tick(frame, state, deterministic=not sample)
            last_controls = info["controls"]
            last_tokens = tokens.tolist()
            if scope is not None:
                scope.push(agent.activity_payload(info))
                if not scope.pump():
                    break
            if injector is not None:
                gui_open = bool(hub.last_raw.get("gui_open"))
                win = None if grabber is None else grabber.window
                injector.inject(last_controls, win=win, gui_open=gui_open)
            n += 1
            latencies.append(float(info["latency_ms"]))
            if dump_f is not None:
                dump_f.write(
                    json.dumps(
                        _json_ready(
                            {
                                "i": n,
                                "tokens": last_tokens,
                                "controls": last_controls,
                                "latency_ms": info["latency_ms"],
                                "run_perception": info["run_perception"],
                                "facts": hub.last_facts().tolist(),
                            }
                        )
                    )
                    + "\n"
                )
    finally:
        if injector is not None:
            injector.release_all()
        if scope is not None:
            scope.close()
            agent.set_activity_trace(False)
        hub.close()
        if tap is not None:
            tap.close()
        if grabber is not None:
            grabber.close()
        if dump_f is not None:
            dump_f.close()

    elapsed = max(1e-6, time.perf_counter() - wall0)
    hz = loop.clock.scheduled_hz()
    return {
        "n_ticks": n,
        "fps_target": int(cfg.capture.fps),
        "actual_fps": n / elapsed,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "n_slots": spec.n_slots,
        "state_dim": cfg.reasoner.game_state_dim,
        "state_fields": list(MC_STATE_FIELDS),
        "fact_fields": list(MC_FACT_FIELDS),
        "policy_hz": hz["policy_hz"],
        "perception_hz": hz["perception_hz"],
        "n_reason": loop.stats.n_reason,
        "last_controls": last_controls,
        "last_tokens": last_tokens,
        "dump": None if dump_path is None else str(dump_path),
        "dry_run": dry_run,
        "elapsed_s": elapsed,
        "injected": injector is not None,
        "sampled": bool(sample),
        "plan_creative": bool(cfg.planner.creative),
        "plan_explore": float(cfg.planner.explore),
        "viz": viz_backend not in {"off", "unavailable"},
        "viz_backend": viz_backend,
        "viz_hz": float(viz_hz),
    }
