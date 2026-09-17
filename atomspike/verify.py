"""Regression checks for LoRA reload, PMSM persistence, dual-rate time, Minecraft collector."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from atomspike.config import load_config
from atomspike.convert.pmsm import enable_pmsm, pmsm_status
from atomspike.models.agent import AtomSpikeAgent
from atomspike.models.lora import LoRALinear, apply_lora
from atomspike.runtime.scheduler import DualRateClock
from atomspike.train.common import load_agent, save_checkpoint


def _dummy_batch(agent: AtomSpikeAgent, device: torch.device):
    n = agent.cfg.encoder.frame_size
    frames = torch.rand(2, 3, n, n, device=device)
    state = torch.rand(2, agent.cfg.reasoner.game_state_dim, device=device)
    return frames, state


def _logits(agent: AtomSpikeAgent, frames, state) -> torch.Tensor:
    agent.eval()
    agent.reset_runtime()
    with torch.no_grad():
        out = agent(frames, state, reason_tick=True)
    return torch.cat([s.flatten() for s in out["logits"]])


def check_pmsm_survives_save(cfg_path: Path) -> dict:
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    frames, state = _dummy_batch(agent, device)
    before = _logits(agent, frames, state)
    n = enable_pmsm(agent, threshold=0.05)
    after = _logits(agent, frames, state)
    if torch.allclose(before, after, atol=1e-6):
        raise AssertionError("PMSM conversion did not change network outputs")
    if not pmsm_status(agent)["enabled"]:
        raise AssertionError("PMSM status enabled=False after convert")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "snn.pt"
        save_checkpoint(agent, path, extra={"pmsm": {"enabled": True, "threshold": 0.05}}, verify=False)
        loaded = load_agent(path, cfg, device)
    status = pmsm_status(loaded)
    if not status["enabled"]:
        raise AssertionError("PMSM disabled after load_agent — conversion evaporated")
    reloaded = _logits(loaded, frames, state)
    if not torch.allclose(after, reloaded, atol=1e-5):
        raise AssertionError("PMSM outputs changed after save/load")
    return {"ok": True, "n_acts": n, "status": status}


def check_lora_roundtrip(cfg_path: Path) -> dict:
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    meta = apply_lora(agent, r=4, alpha=8)
    if meta["wrapped"] <= 0:
        raise AssertionError("T4 LoRA wrapped 0 layers")
    if not any(isinstance(m, LoRALinear) for m in agent.reasoner.modules()):
        raise AssertionError("reasoner has no LoRALinear after apply_lora")
    if any(p.requires_grad for p in agent.encoder.parameters()):
        raise AssertionError("encoder was not frozen")
    lora_train = [n for n, p in agent.named_parameters() if p.requires_grad and "lora_" in n]
    if not lora_train:
        raise AssertionError("no trainable lora_* parameters")
    frames, state = _dummy_batch(agent, device)
    y = _logits(agent, frames, state)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "peft.pt"
        save_checkpoint(agent, path, verify=False)
        loaded = load_agent(path, cfg, device)
    if not any(isinstance(m, LoRALinear) for m in loaded.reasoner.modules()):
        raise AssertionError("LoRA wrappers missing after load")
    y2 = _logits(loaded, frames, state)
    if not torch.allclose(y, y2, atol=1e-5):
        raise AssertionError("LoRA outputs changed after save/load")
    return {"ok": True, **meta}


def check_dual_rate_sim() -> dict:
    clock = DualRateClock(perception_hz=5.0, policy_hz=30.0, mode="sim", pace=False)
    perc = 0
    for _ in range(30):
        d = clock.start_tick()
        if not d.run_policy:
            raise AssertionError("sim mode must issue a policy tick every call")
        perc += int(d.run_perception)
        clock.finish_tick(d)
    if clock.n_policy != 30:
        raise AssertionError(f"expected 30 policy ticks, got {clock.n_policy}")
    if perc != 5 or clock.n_perception != 5:
        raise AssertionError(f"expected 5 perception ticks/s, got {clock.n_perception} (flags={perc})")
    hz = clock.scheduled_hz()
    if abs(hz["policy_hz"] - 30.0) > 1e-3:
        raise AssertionError(f"sim policy_hz {hz['policy_hz']} != 30")
    if abs(hz["perception_hz"] - 5.0) > 1e-3:
        raise AssertionError(f"sim perception_hz {hz['perception_hz']} != 5")
    return {"ok": True, **{k: hz[k] for k in ("policy_hz", "perception_hz", "elapsed_s")}}


def check_dual_rate_realtime(seconds: float = 0.35) -> dict:
    import time

    clock = DualRateClock(perception_hz=5.0, policy_hz=30.0, mode="realtime", pace=True)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        d = clock.start_tick()
        clock.finish_tick(d)
    hz = clock.scheduled_hz()
    if not (18.0 <= hz["policy_hz"] <= 42.0):
        raise AssertionError(f"realtime policy_hz {hz['policy_hz']:.1f} not near 30")
    if not (2.0 <= hz["perception_hz"] <= 10.0):
        raise AssertionError(f"realtime perception_hz {hz['perception_hz']:.1f} not near 5")
    return {"ok": True, "policy_hz": hz["policy_hz"], "perception_hz": hz["perception_hz"], "elapsed_s": hz["elapsed_s"]}


def check_mc_collector_dry_run() -> dict:
    import tempfile

    from atomspike.capture.mc_collector import run_minecraft_collector
    from atomspike.config import AtomSpikeConfig

    cfg = AtomSpikeConfig()
    cfg.env.kind = "minecraft"
    cfg.capture.fps = 30
    cfg.capture.store_width = 80
    cfg.capture.store_height = 45
    cfg.action.n_keys = 8
    cfg.action.key_names = ("W", "A", "S", "D", "SPACE", "SHIFT", "CTRL", "E")
    cfg.aligned()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mc.h5"
        result = run_minecraft_collector(cfg, out, seconds=0.45, dry_run=True)
        n = int(result["n_frames"])
        if n < 8:
            raise AssertionError(f"dry-run collector wrote {n} frames, expected ~13 at 30fps")
        import h5py

        with h5py.File(result["path"], "r") as f:
            if "frames" not in f or "game_state" not in f or "actions" not in f:
                raise AssertionError("minecraft h5 missing core datasets")
            if f["game_state"].shape[1] != cfg.reasoner.game_state_dim:
                raise AssertionError(f"minecraft state dim != {cfg.reasoner.game_state_dim}")
            if "facts" not in f or f["facts"].shape[1] != 8:
                raise AssertionError("minecraft h5 missing 8-d facts")
            if "timestamps_ns" not in f or "keys_held" not in f:
                raise AssertionError("minecraft h5 missing timestamps/keys_held")
            if "cursor_xy" not in f or f["cursor_xy"].shape[1] != 2:
                raise AssertionError("minecraft h5 missing cursor_xy")
            if "wheel" not in f:
                raise AssertionError("minecraft h5 missing wheel")
            if f["actions"].shape[1] != cfg.action.n_slots:
                raise AssertionError("action slot count mismatch")
            if f["frames"].shape[1:] != (45, 80, 3):
                raise AssertionError(f"unexpected frame store shape {f['frames'].shape}")
            if int(f["dones"][-1]) != 1:
                raise AssertionError("collector must mark the last frame as the end of a recording")
        latency = float(result["latency_p95_ms"])
    return {"ok": True, "n_frames": n, "latency_p95_ms": latency}


def _mc_cfg() -> "AtomSpikeConfig":
    from atomspike.config import AtomSpikeConfig

    cfg = AtomSpikeConfig()
    cfg.env.kind = "minecraft"
    cfg.capture.fps = 30
    cfg.capture.store_width = 80
    cfg.capture.store_height = 45
    cfg.encoder.frame_size = 64
    cfg.encoder.d_model = 64
    cfg.encoder.stem_channels = (8, 16, 32)
    cfg.reasoner.n_layers = 1
    cfg.reasoner.n_heads = 2
    cfg.policy.n_layers = 1
    cfg.policy.n_heads = 2
    cfg.policy.spike_time_steps = 1
    cfg.action.n_keys = 8
    cfg.action.key_names = ("W", "A", "S", "D", "SPACE", "SHIFT", "CTRL", "E")
    cfg.train.epochs = 1
    cfg.train.batch_size = 8
    cfg.train.device = "cpu"
    cfg.train.precision = "fp32"
    cfg.train.ckpt_verify = False
    cfg.train.lr_schedule = "constant"
    return cfg.aligned()


def check_mc_train_predict() -> dict:
    import tempfile

    from atomspike.capture.mc_collector import run_minecraft_collector
    from atomspike.capture.mc_predict import run_minecraft_predict
    from atomspike.eval.metrics import evaluate_offline
    from atomspike.train.bc import train_bc

    cfg = _mc_cfg()
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "mc.h5"
        ckpt = Path(td) / "bc.pt"
        dump = Path(td) / "pred.jsonl"
        collect = run_minecraft_collector(cfg, data, seconds=0.4, dry_run=True)
        if int(collect["n_frames"]) < 8:
            raise AssertionError("collector too short for train/predict check")
        trained = train_bc(cfg, data, ckpt, epochs=1)
        again = train_bc(cfg, data, ckpt, epochs=1)
        if not again.get("resumed"):
            raise AssertionError("second train-bc should resume existing --out weights")
        wiped = train_bc(cfg, data, ckpt, epochs=1, reset=True)
        if wiped.get("resumed"):
            raise AssertionError("--reset must not resume")
        offline = evaluate_offline(cfg, trained["checkpoint"], data)
        if offline["n"] < 8:
            raise AssertionError("offline eval saw too few steps")
        if "token_acc" not in offline:
            raise AssertionError("offline eval missing token_acc")
        pred = run_minecraft_predict(cfg, trained["checkpoint"], seconds=0.25, dry_run=True, dump_path=dump)
        if int(pred["n_ticks"]) < 4:
            raise AssertionError(f"predict-mc dry-run ticks {pred['n_ticks']}")
        if pred.get("injected"):
            raise AssertionError("predict-mc must not inject")
        if pred.get("viz"):
            raise AssertionError("dry-run predict must not open the SNN window")
        if not dump.is_file() or dump.stat().st_size < 8:
            raise AssertionError("predict-mc dump jsonl missing")
        toks = pred.get("last_tokens") or []
        if len(toks) != cfg.action.n_slots:
            raise AssertionError(f"predicted token width {len(toks)} != {cfg.action.n_slots}")
    return {
        "ok": True,
        "collect_frames": int(collect["n_frames"]),
        "train_loss": trained["loss"][-1] if trained["loss"] else None,
        "token_acc": offline["token_acc"],
        "predict_ticks": int(pred["n_ticks"]),
        "n_slots": cfg.action.n_slots,
        "resumed": bool(again.get("resumed")),
        "reset": bool(wiped.get("reset")),
    }


def check_mouse_asinh() -> dict:
    from atomspike.models.actions import ActionSpec

    spec = ActionSpec(
        n_keys=4,
        n_key_states=4,
        mouse_bins=41,
        n_mouse_buttons=2,
        key_names=("W", "A", "S", "D"),
        n_slots=8,
        mouse_delta_max=240.0,
        mouse_delta_scale=16.0,
    )
    for d in (0.0, 3.0, -18.0, 80.0):
        b = spec.mouse_delta_to_bin(d)
        if not (0 <= b < spec.mouse_bins):
            raise AssertionError(f"look bin {b} out of range for delta {d}")
    if spec.mouse_unit_to_bin(0.0) != 0 or spec.mouse_unit_to_bin(1.0) != spec.mouse_bins - 1:
        raise AssertionError("GUI cursor bins do not span [0, 1]")
    import torch
    from atomspike.models.actions import look_deltas_from_logits

    logits = [
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        torch.zeros(1, spec.mouse_bins),
        torch.zeros(1, spec.mouse_bins),
        torch.zeros(1, 4),
        torch.zeros(1, 4),
    ]
    logits[4][0, spec.mouse_center] = 8.0
    dx, dy = look_deltas_from_logits(logits, spec)
    if abs(float(dx.item())) > 0.5:
        raise AssertionError("peaked idle look should decode near 0")
    logits[4][0, spec.mouse_center + 6] = 12.0
    dx, _ = look_deltas_from_logits(logits, spec)
    if float(dx.item()) <= 1.0:
        raise AssertionError("off-center look mass must decode to a nonzero turn")
    return {"ok": True, "mouse_bins": spec.mouse_bins}


def check_planner(cfg_path: Path) -> dict:
    from atomspike.models.planner import build_episode_timeline

    cfg = load_config(cfg_path)
    if not cfg.planner.enabled:
        return {"ok": True, "skipped": True}
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    agent.eval()
    frames, state = _dummy_batch(agent, device)
    out_a = agent(frames, state, reason_tick=True, plan_sample=False)
    out_b = agent(frames, state, reason_tick=True, plan_sample=False)
    if "plan" not in out_a:
        raise AssertionError("planner enabled but forward has no plan")
    if "mix" not in out_a["plan"]:
        raise AssertionError("planner missing task mix")
    if not torch.allclose(out_a["plan"]["mu"], out_b["plan"]["mu"], atol=1e-5):
        raise AssertionError("deterministic plan mu should be stable")
    n_ph = cfg.planner.n_phases
    dones = [0, 0, 0, 1, 0, 0, 1]
    import numpy as np

    states = np.arange(len(dones) * 4, dtype=np.float32).reshape(len(dones), 4)
    tl = build_episode_timeline(states, dones, n_phases=n_ph, future_steps=2)
    if tl["future_state"].shape[0] != len(dones):
        raise AssertionError("future_state length mismatch")
    agent.train()
    fut = torch.rand_like(state)
    out_tr = agent(frames, state, reason_tick=True, future_state=fut)
    if "plan" not in out_tr or "teacher_mix" not in out_tr["plan"]:
        raise AssertionError("training forward missing self-arranged teacher mix")
    return {
        "ok": True,
        "latent_dim": cfg.planner.latent_dim,
        "n_phases": n_ph,
        "params": agent.parameter_count(),
    }


def check_capture_preview() -> dict:
    from atomspike.viz.capture_preview import BAR_H, compose_preview, dummy_capture_frame, view_size

    src = dummy_capture_frame(80, 45)
    vw, vh = view_size(80, 45, min_w=320, max_w=960)
    img = compose_preview(
        src,
        {"n": 12, "fps": 30.0, "keys": "W", "mouse": "mouse=+3,-1", "mode": "look", "grab_ms": 4.0},
        view_w=vw,
        view_h=vh,
    )
    if img.shape != (vh + BAR_H, vw, 3):
        raise AssertionError(f"preview shape {img.shape} != {(vh + BAR_H, vw, 3)}")
    idle = compose_preview(None, None, view_w=vw, view_h=vh)
    if idle.shape != img.shape:
        raise AssertionError("idle and live preview frames must match window size")
    native_w, native_h = 854, 480
    native = dummy_capture_frame(native_w, native_h)
    shown = compose_preview(native, {"n": 1, "mode": "gui", "cursor": (0.5, 0.5)}, view_w=native_w, view_h=native_h)
    if shown.shape != (native_h + BAR_H, native_w, 3):
        raise AssertionError(f"480p preview shape {shown.shape}")
    return {"ok": True, "shape": list(img.shape), "native": list(shown.shape)}


def check_snn_scope() -> dict:
    from atomspike.viz.snn_scope import ScopeBuffers, compose_frame, dummy_payload

    buf = ScopeBuffers()
    payload = dummy_payload()
    img = compose_frame(payload, buf, dt=1.0 / 120.0, viz_hz=120.0)
    if img.ndim != 3 or img.shape[2] != 3:
        raise AssertionError(f"compose_frame shape {img.shape}")
    if img.shape[0] < 400 or img.shape[1] < 400:
        raise AssertionError("SNN window is unexpectedly small")
    idle = compose_frame(None, ScopeBuffers(), dt=1.0 / 120.0)
    if idle.shape != img.shape:
        raise AssertionError("idle and live frames must match window size")
    cfg = load_config(None)
    cfg.planner.enabled = True
    cfg.aligned()
    agent = AtomSpikeAgent(cfg)
    agent.eval()
    agent.set_activity_trace(True)
    frames, state = _dummy_batch(agent, torch.device("cpu"))
    with torch.no_grad():
        _ = agent.act(frames, state, deterministic=True, reason_tick=True)
    snap = agent.activity_payload({"latency_ms": 1.0, "run_perception": True, "tokens": [0] * agent.spec.n_slots})
    mem = snap.get("in_mem")
    if mem is None or int(np.asarray(mem).reshape(-1).size) < 1:
        raise AssertionError("activity payload missing membrane")
    if snap.get("kind") not in {"spike", "ann"}:
        raise AssertionError(f"unexpected policy kind {snap.get('kind')}")
    return {"ok": True, "shape": list(img.shape), "kind": snap.get("kind")}


def check_session_splits() -> dict:
    from atomspike.data.sessions import CLOCK_DIM, apply_recording_boundaries, build_episode_timeline, clock_vector

    fps = 30.0
    n1, n2 = 12, 9
    t1 = (np.arange(n1) * 1e9 / fps).astype(np.int64)
    t2 = t1[-1] + int(8e9) + (np.arange(n2) * 1e9 / fps).astype(np.int64)
    times = np.concatenate([t1, t2])
    n = int(times.shape[0])
    dones = np.zeros(n, dtype=np.uint8)
    cuts = apply_recording_boundaries(dones, times, gap_s=2.0)
    if int(cuts[n1 - 1]) != 1 or int(cuts[-1]) != 1:
        raise AssertionError("session gap / last frame must close a recording")
    if int(cuts[: n1 - 1].sum()) != 0:
        raise AssertionError("no extra cuts inside the first recording")
    states = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    tl = build_episode_timeline(
        states,
        dones,
        n_phases=8,
        future_steps=3,
        timestamps_ns=times,
        future_near=1,
        future_far=5,
        fps=fps,
    )
    if int(tl["n_sessions"]) != 2:
        raise AssertionError(f"expected 2 recordings, got {tl['n_sessions']}")
    if list(tl["session_lengths"]) != [n1, n2]:
        raise AssertionError(f"session lengths {tl['session_lengths']}")
    sid = tl["session_id"]
    if int(sid[n1 - 1]) == int(sid[n1]):
        raise AssertionError("adjacent recordings share a session id")
    if float(tl["time_frac"][0]) > 1e-6 or float(tl["time_frac"][n1]) > 1e-6:
        raise AssertionError("time_frac must restart at each recording")
    if float(tl["time_frac"][n1 - 1]) < 0.99 or float(tl["time_frac"][-1]) < 0.99:
        raise AssertionError("time_frac must reach 1 at the end of each recording")
    if not np.allclose(tl["future_state"][n1 - 1], states[n1 - 1]):
        raise AssertionError("future_state crossed into the next recording")
    if not np.allclose(tl["future_state_far"][n1 - 1], states[n1 - 1]):
        raise AssertionError("future_state_far crossed into the next recording")
    if np.allclose(tl["future_state"][n1 - 4], states[n1]):
        raise AssertionError("mid-session future should stay inside the first recording")
    vec = clock_vector(0.25, 12.0, 1.0 / 30.0)
    if vec.shape != (CLOCK_DIM,):
        raise AssertionError(f"clock dim {vec.shape} != {CLOCK_DIM}")
    idx = n1
    prev_idx = idx if int(cuts[idx - 1]) != 0 else idx - 1
    if prev_idx != idx:
        raise AssertionError("prev_frame must not use the previous recording's last frame")
    cfg = load_config(None)
    cfg.planner.enabled = True
    cfg.aligned()
    agent = AtomSpikeAgent(cfg)
    agent.eval()
    frames, state = _dummy_batch(agent, torch.device("cpu"))
    clock = torch.zeros(frames.size(0), CLOCK_DIM)
    with torch.no_grad():
        out = agent(frames, state, reason_tick=True, clock=clock)
    if "plan" not in out or "time_pred" not in out["plan"]:
        raise AssertionError("planner missing session time_pred")
    return {
        "ok": True,
        "n_sessions": int(tl["n_sessions"]),
        "session_lengths": list(tl["session_lengths"]),
        "clock_dim": CLOCK_DIM,
    }


def run_verify(cfg_path: Path) -> dict:
    return {
        "pmsm": check_pmsm_survives_save(cfg_path),
        "lora": check_lora_roundtrip(cfg_path),
        "dual_rate_sim": check_dual_rate_sim(),
        "dual_rate_realtime": check_dual_rate_realtime(),
        "mouse_asinh": check_mouse_asinh(),
        "planner": check_planner(cfg_path),
        "session_splits": check_session_splits(),
        "snn_scope": check_snn_scope(),
        "capture_preview": check_capture_preview(),
        "mc_collector": check_mc_collector_dry_run(),
        "mc_train_predict": check_mc_train_predict(),
    }
