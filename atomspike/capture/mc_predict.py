"""Live 30Hz Minecraft prediction: window + fused state -> action tokens.

Research-only, singleplayer / LAN. By default it only logs tokens.
With inject=True it closes the loop through the OS (F9 emergency stop).
"""

from __future__ import annotations

import json
import math
import os
import secrets
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from atomspike.capture.input_tap import InputTap
from atomspike.capture.mc_collector import resize_rgb
from atomspike.capture.mc_state import MC_FACT_FIELDS, MC_STATE_FIELDS, MinecraftStateHub
from atomspike.capture.window import (
    WindowGrabber,
    client_cursor_norm,
    is_window_usable,
    list_windows,
    pick_minecraft_window,
)
from atomspike.config import AtomSpikeConfig
from atomspike.models.agent import AtomSpikeAgent
from atomspike.models.actions import ActionSpec
from atomspike.runtime.loop import DualRateLoop
from atomspike.train.common import load_agent, resolve_device

# Pause/title/death (and friends). A left-click here is "Save and Quit".
_MENU_BLOCK = (
    "pause",
    "title",
    "disconnect",
    "death",
    "gameover",
    "shareto",
    "options",
    "controls",
    "confirm",
    "chat",
    "credits",
    "progress",
    "receiving",
    "genericmessage",
    "multiplayer",
    "selectworld",
    "createworld",
    "winscreen",
    "accessibility",
    "language",
    "warning",
    "notice",
)


def inject_gate(
    raw: dict | None,
    *,
    hwnd: int = 0,
    sidecar_age_s: float = 0.0,
    expected_nonce: str | None = None,
) -> dict:
    """Fail-closed authorization for a single input-injection tick.

    The sidecar must freshly attest to an integrated singleplayer server.  A
    title match, desktop capture, or a stale state packet is never sufficient
    evidence to send OS input.
    """
    raw = raw or {}
    gui = str(raw.get("gui") or "")
    gui_l = gui.lower()
    if int(hwnd or 0) <= 0:
        return {"ok": False, "unpause": False, "buttons": False, "reason": "no_window"}
    try:
        age = float(sidecar_age_s)
    except (TypeError, ValueError):
        age = float("nan")
    if not math.isfinite(age) or age > 1.0:
        return {"ok": False, "unpause": False, "buttons": False, "reason": "stale_sidecar"}
    if expected_nonce is not None and not _nonce_matches(raw.get("session_nonce"), expected_nonce):
        return {"ok": False, "unpause": False, "buttons": False, "reason": "unauthenticated_sidecar"}
    if raw.get("singleplayer") is not True:
        return {"ok": False, "unpause": False, "buttons": False, "reason": "unverified_world"}
    if bool(raw.get("dead")):
        return {"ok": False, "unpause": False, "buttons": False, "reason": "dead"}
    if not is_window_usable(hwnd):
        return {"ok": False, "unpause": False, "buttons": False, "reason": "unusable_window"}
    if not _hwnd_is_foreground(hwnd):
        return {"ok": False, "unpause": False, "buttons": False, "reason": "unfocused"}
    if raw.get("paused") is True or "pause" in gui_l:
        return {
            "ok": False,
            "unpause": True,
            "buttons": False,
            "reason": f"menu:{gui or 'pause'}",
        }
    for token in _MENU_BLOCK:
        if token in gui_l:
            return {
                "ok": False,
                "unpause": "pause" in gui_l,
                "buttons": False,
                "reason": f"menu:{gui or token}",
            }
    grabbed = raw.get("mouse_grabbed")
    if grabbed is False and not bool(raw.get("gui_open")):
        return {"ok": False, "unpause": False, "buttons": False, "reason": "ungrabbed"}
    return {
        "ok": True,
        "unpause": False,
        "buttons": True,
        "reason": "ok",
    }


def _nonce_matches(got: object, expected: str) -> bool:
    """Constant-time check that does not throw when the sidecar omits the nonce."""
    left = str(got or "")
    right = str(expected or "")
    if not right or len(left) != len(right):
        return False
    return secrets.compare_digest(left, right)


def _hwnd_is_foreground(hwnd: int) -> bool:
    if not hwnd or sys.platform != "win32":
        return True
    try:
        from atomspike.capture.window import is_window_foreground

        return bool(is_window_foreground(hwnd))
    except Exception:
        try:
            import ctypes

            return int(ctypes.windll.user32.GetForegroundWindow() or 0) == int(hwnd)
        except Exception:
            return True


def _fg_note() -> str:
    try:
        from atomspike.capture.window import describe_foreground

        return describe_foreground()
    except Exception:
        return ""


def _append_predict_log(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
    except Exception:
        pass


def _ignore_inject_ctrl_c() -> None:
    try:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            from atomspike.viz.overlay_win import ignore_console_ctrl_c

            ignore_console_ctrl_c()
        except Exception:
            pass


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
    viz_hz: float = 60.0,
    learn: bool | None = None,
    resume_plastic: bool = False,
    hud: bool | None = None,
) -> dict:
    spec = ActionSpec.from_config(cfg.action)
    store_w = int(cfg.capture.store_width)
    store_h = int(cfg.capture.store_height)
    mc = cfg.minecraft
    # CLI overrides win over the profile, but injection must reject either
    # source of region capture.  Checking only the CLI flag would let a saved
    # ``capture.region`` profile redirect OS input away from Minecraft.
    effective_region = region if region is not None else cfg.capture.region

    want_learn = bool(cfg.learn.enabled) if learn is None else bool(learn)
    sidecar_nonce = secrets.token_hex(16) if inject else None
    if dry_run and learn is None:
        want_learn = False
    if inject:
        if sys.platform != "win32":
            raise RuntimeError("--inject is supported only on Windows and is disabled on this platform")
        if dry_run:
            raise RuntimeError("--inject cannot run in --dry-run")
        if desktop or effective_region is not None:
            raise RuntimeError("--inject requires capture of a named Minecraft window; --desktop/--region are unsafe")

    if not dry_run and not desktop and effective_region is None:
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
    if inject and want_learn:
        if greedy:
            raise RuntimeError(
                "self-play learning requires sampled exploration; "
                "--greedy cannot be combined with --inject --learn"
            )
        # Policy-gradient updates require actions drawn from the policy.  Never
        # recycle a deterministic argmax as a pretend supervised target.
        sample = True
    loop = DualRateLoop(agent, cfg, device, mode="realtime", pace=True)
    learner = None
    if want_learn:
        from atomspike.runtime.self_learn import OnlineLearner

        plastic = None if ckpt is None else Path(ckpt).with_suffix(".plastic.pt")
        learner = OnlineLearner(agent, cfg, device, save_path=plastic, inject=bool(inject and not dry_run))
        if resume_plastic and plastic is not None:
            learner.load_plastic()
        loop.learner = learner
        agent.eval()

    grabber = None if dry_run else WindowGrabber(
        title_substr=mc.window_title,
        monitor=cfg.capture.monitor,
        region=effective_region,
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
    if viz is None:
        viz = not dry_run
    scope = None
    viz_backend = "off"
    hud_win = None
    info_win = None
    hud_backend = "off"
    game_hwnd = int(getattr(getattr(grabber, "window", None), "hwnd", 0) or 0)
    injector = None
    if inject:
        if game_hwnd <= 0:
            raise RuntimeError("--inject requires a live Minecraft window handle; refusing to send input to the foreground app")
        from atomspike.capture.inject import InputInjector

        injector = InputInjector(enabled=True)
    focus = None
    t_end = None if seconds is None else (time.perf_counter() + max(0.05, float(seconds)))
    latencies: list[float] = []
    n = 0
    last_controls = None
    last_tokens = None
    wall0 = time.perf_counter()
    stop_reason = "ok"
    grab_miss = 0
    log_path = Path(ckpt).with_name("predict.log") if ckpt is not None else Path("runs/predict.log")
    last_esc = 0.0
    last_beat = 0.0
    last_hello = -1e9
    last_gate = ""
    if injector is not None:
        _ignore_inject_ctrl_c()
    _append_predict_log(
        log_path,
        f"start seconds={seconds} inject={injector is not None} pid={os.getpid()}",
    )
    if not dry_run:
        print(
            "[predict] click into Minecraft so the mouse locks. "
            "console minimizes to the taskbar in 3s so it cannot steal focus...",
            flush=True,
        )
        time.sleep(3.0)
        if grabber is not None and not game_hwnd:
            try:
                win = pick_minecraft_window(mc.window_title)
                if win is not None:
                    grabber.window = win
                    game_hwnd = int(win.hwnd)
            except Exception:
                pass
        if sys.platform == "win32":
            try:
                from atomspike.viz.overlay_win import GameFocusSession

                focus = GameFocusSession(game_hwnd, log_path=log_path)
                focus.begin(hide_console=True, park=False)
            except Exception:
                focus = None
    try:
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
            if focus is not None:
                focus.nudge(park=False)
        want_hud = (not dry_run) if hud is None else bool(hud)
        info_win = None
        if want_hud and sys.platform == "win32":
            try:
                from atomspike.viz.game_hud import GameHud, InfoBoard

                hud_win = GameHud(game_hwnd, hz=12.0)
                if not hud_win.start():
                    hud_win.close()
                    hud_win = None
                    hud_backend = "unavailable"
                else:
                    hud_backend = hud_win.backend
                info_win = InfoBoard(game_hwnd, hz=8.0)
                if not info_win.start():
                    info_win.close()
                    info_win = None
                if focus is not None:
                    focus.nudge(park=False)
            except Exception:
                if hud_win is not None:
                    try:
                        hud_win.close()
                    except Exception:
                        pass
                hud_win = None
                hud_backend = "unavailable"
                info_win = None
        try:
            agent.set_overlay_trace(hud_win is not None or info_win is not None)
        except Exception:
            pass
        while True:
            try:
                if t_end is not None and time.perf_counter() >= t_end:
                    stop_reason = "time"
                    break
                if sidecar_nonce is not None and (time.perf_counter() - last_hello) >= 1.0:
                    try:
                        from atomspike.capture.sidecar import send_sidecar_cmd

                        send_sidecar_cmd(
                            "hello",
                            host=mc.sidecar_host,
                            port=int(mc.sidecar_cmd_port or 21338),
                            nonce=sidecar_nonce,
                        )
                    except Exception:
                        pass
                    last_hello = time.perf_counter()
                if dry_run:
                    frame = np.zeros((store_h, store_w, 3), dtype=np.uint8)
                    frame[:, :, 2] = 40
                else:
                    assert grabber is not None
                    try:
                        raw, _t_ns = grabber.grab()
                    except Exception as exc:
                        grab_miss += 1
                        if grab_miss >= 120:
                            stop_reason = f"grab:{exc}"
                            print(f"[predict] stop: {stop_reason}", flush=True)
                            break
                        time.sleep(0.05)
                        continue
                    grab_miss = 0
                    if raw.shape[0] == store_h and raw.shape[1] == store_w:
                        frame = raw
                    else:
                        frame = resize_rgb(raw, store_w, store_h)
                snap = tap.snapshot() if tap is not None else None
                if snap is not None and snap.stop:
                    stop_reason = "f9"
                    print("[predict] stop: F9", flush=True)
                    break
                kd = snap.keys_down if snap is not None else {}
                win = None if grabber is None else grabber.window
                os_cx, os_cy = client_cursor_norm(win)
                extra = {"cursor_x": os_cx, "cursor_y": os_cy}
                if snap is not None:
                    extra["wheel"] = float(snap.wheel)
                state = hub.snapshot(frame if not dry_run else None, kd, extra=extra)
                gate = None
                inject_win = None
                action_executed = False
                if injector is not None:
                    inject_win = None if grabber is None else grabber.window
                    hwnd = int(getattr(inject_win, "hwnd", 0) or 0) or game_hwnd
                    try:
                        age = float(hub.sidecar_age_s())
                    except Exception:
                        age = 0.0
                    gate = inject_gate(
                        hub.last_raw,
                        hwnd=hwnd,
                        sidecar_age_s=age,
                        expected_nonce=sidecar_nonce,
                    )
                    if not gate["ok"] and gate["reason"] == "unfocused" and focus is not None:
                        focus.nudge(park=False)
                        gate = inject_gate(
                            hub.last_raw,
                            hwnd=hwnd,
                            sidecar_age_s=age,
                            expected_nonce=sidecar_nonce,
                        )
                    action_executed = bool(gate["ok"])
                human = None
                if learner is not None and injector is None and snap is not None and tap is not None:
                    human = tap.to_tokens(snap, gui_open=bool(hub.last_raw.get("gui_open")), cursor_norm=(os_cx, os_cy))
                tokens, info = loop.tick(
                    frame,
                    state,
                    deterministic=not sample,
                    human_tokens=human,
                    injected=action_executed,
                    signal=hub.last_raw,
                )
                last_controls = info["controls"]
                last_tokens = tokens.tolist()
                if hud_win is not None or info_win is not None:
                    snap = agent.overlay_snapshot(info)
                    raw = hub.last_raw or {}
                    snap["hand"] = raw.get("hand_item") or ""
                    snap["gui"] = bool(raw.get("gui_open") or raw.get("inv_open") or raw.get("paused"))
                    snap["look_miss"] = bool(raw.get("look_miss"))
                    if learner is not None:
                        try:
                            snap["learn"] = learner.status()
                        except Exception:
                            snap["learn"] = {}
                    if hud_win is not None:
                        hud_win.push(snap)
                        if not hud_win.pump():
                            print("[predict] recognition window closed; prediction continues (F9 to stop)", flush=True)
                            try:
                                hud_win.close()
                            except Exception:
                                pass
                            hud_win = None
                            hud_backend = "closed"
                    if info_win is not None:
                        info_win.push(snap)
                        if not info_win.pump():
                            print("[predict] status window closed; prediction continues (F9 to stop)", flush=True)
                            try:
                                info_win.close()
                            except Exception:
                                pass
                            info_win = None
                    if hud_win is None and info_win is None:
                        try:
                            agent.set_overlay_trace(False)
                        except Exception:
                            pass
                if scope is not None:
                    scope.push(agent.activity_payload(info))
                    if not scope.pump():
                        print("[predict] viz window closed; prediction continues (F9 to stop)", flush=True)
                        try:
                            scope.close()
                        except Exception:
                            pass
                        scope = None
                        viz_backend = "closed"
                        loop.clock.idle_fn = None
                        agent.set_activity_trace(False)
                if injector is not None:
                    assert gate is not None
                    if gate["reason"] == "unfocused":
                        injector.release_clip()
                    if gate["ok"]:
                        last_gate = "ok"
                        injector.inject(
                            last_controls,
                            win=inject_win,
                            gui_open=bool(hub.last_raw.get("gui_open")),
                            allow_buttons=bool(gate["buttons"]),
                        )
                    else:
                        injector.release_all()
                        if gate["unpause"] and (time.perf_counter() - last_esc) > 0.45:
                            injector.tap_key("ESC")
                            last_esc = time.perf_counter()
                        if gate["reason"] != last_gate:
                            note = _fg_note()
                            print(f"[predict] inject paused ({gate['reason']}) {note}", flush=True)
                            if gate["reason"] == "unauthenticated_sidecar":
                                print(
                                    "[predict] sidecar did not echo this session's nonce. "
                                    "Install tools/mc-sidecar 0.1.6 and fully restart Minecraft.",
                                    flush=True,
                                )
                            _append_predict_log(log_path, f"inject paused ({gate['reason']}) {note}")
                            last_gate = str(gate["reason"])
                now = time.perf_counter()
                if now - last_beat >= 2.0:
                    last_beat = now
                    gate_s = last_gate or ("on" if injector is not None else "off")
                    _append_predict_log(
                        log_path,
                        f"alive ticks={n} elapsed={now - wall0:.1f}s inject={gate_s} {_fg_note()}",
                    )
                if (
                    injector is not None
                    and learner is not None
                    and learner.take_kill()
                ):
                    injector.release_all()
                    try:
                        from atomspike.capture.sidecar import send_sidecar_cmd

                        send_sidecar_cmd(
                            "kill",
                            host=mc.sidecar_host,
                            port=int(mc.sidecar_cmd_port or 21338),
                            nonce=sidecar_nonce,
                        )
                    except Exception:
                        pass
                    print(
                        f"[predict] no advancement for {float(cfg.learn.stall_s) / 60.0:.0f} min; kill and respawn",
                        flush=True,
                    )
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
            except KeyboardInterrupt:
                if injector is None:
                    stop_reason = "keyboard"
                    print("[predict] stop: Ctrl+C", flush=True)
                    break
                print("[predict] Ctrl+C ignored while injecting (CTRL is sprint). Press F9 to stop.", flush=True)
                _append_predict_log(log_path, "ctrl+c ignored")
                continue
    except Exception as exc:
        stop_reason = f"error:{type(exc).__name__}:{exc}"
        traceback.print_exc()
        print(f"[predict] stop: {stop_reason}", flush=True)
        _append_predict_log(log_path, stop_reason)
        try:
            with log_path.open("a", encoding="utf-8") as lf:
                traceback.print_exc(file=lf)
        except Exception:
            pass
    finally:
        _append_predict_log(log_path, f"stop {stop_reason} ticks={n}")
        if injector is not None:
            injector.release_all()
            injector.release_clip()
        if hud_win is not None:
            try:
                hud_win.close()
            except Exception:
                pass
            hud_win = None
        if info_win is not None:
            try:
                info_win.close()
            except Exception:
                pass
            info_win = None
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
        if learner is not None:
            learner.close()
        if focus is not None:
            try:
                focus.close()
            except Exception:
                pass

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
        "hud": hud_backend not in {"off", "unavailable", "closed"},
        "hud_backend": hud_backend,
        "learn": learner is not None,
        "learn_mode": None if learner is None else learner.mode,
        "resumed_plastic": bool(resume_plastic and learner is not None),
        "learn_steps": 0 if learner is None else learner.n_steps,
        "learn_loss": 0.0 if learner is None else learner.last_loss,
        "learn_dropped": 0 if learner is None else learner.n_dropped,
        "born_events": 0 if learner is None else int(agent.planner.n_born_events()) if agent.planner is not None else 0,
        "born_layers": 0 if learner is None else int(agent.grow.n_alive()),
        "stop_reason": stop_reason,
        "game_hwnd": int(game_hwnd or 0),
    }
