"""Regression checks for LoRA reload, PMSM persistence, dual-rate time, Minecraft collector."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from atomspike.config import load_config
from atomspike.convert.pmsm import calibrate_pmsm, enable_pmsm, pmsm_status
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
    b = int(frames.size(0))
    inv_slots = int(agent.inventory.n_slots) if agent.inventory is not None else int(cfg.inventory.n_slots)
    full_batch = {
        "frame": frames,
        "prev_frame": frames.roll(1, dims=0),
        "game_state": state,
        "clock": torch.zeros(b, 8),
        "task_id": torch.zeros(b, 1, dtype=torch.long),
        "task_aux": torch.zeros(b, 8),
        "inv_ids": torch.zeros(b, inv_slots, dtype=torch.long),
        "hand_id": torch.zeros(b, 1, dtype=torch.long),
        "hover_id": torch.zeros(b, 1, dtype=torch.long),
        "carry_id": torch.zeros(b, 1, dtype=torch.long),
        "block_pulse": torch.zeros(b, 2),
    }
    calibration = calibrate_pmsm(agent, [full_batch], percentile=99.0)
    n = enable_pmsm(agent, calibration=calibration)
    after = _logits(agent, frames, state)
    if torch.allclose(before, after, atol=1e-6):
        raise AssertionError("PMSM conversion did not change network outputs")
    if not pmsm_status(agent)["enabled"]:
        raise AssertionError("PMSM status enabled=False after convert")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "snn.pt"
        save_checkpoint(
            agent,
            path,
            extra={"pmsm": {"enabled": True, "method": "calibrated_binary_activation", "layers": calibration}},
            verify=False,
        )
        loaded = load_agent(path, cfg, device)
    status = pmsm_status(loaded)
    if not status["enabled"]:
        raise AssertionError("PMSM disabled after load_agent — conversion evaporated")
    reloaded = _logits(loaded, frames, state)
    if not torch.allclose(after, reloaded, atol=1e-5):
        raise AssertionError("PMSM outputs changed after save/load")
    return {
        "ok": True,
        "n_acts": n,
        "calibrated_layers": len(calibration),
        "non_unit_scales": sum(abs(float(v["scale"]) - 1.0) > 1e-6 for v in calibration.values()),
    }


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
            if "inv_ids" not in f or int(f["inv_ids"].shape[1]) < 36:
                raise AssertionError("minecraft h5 missing inv_ids")
            if "block_pulse" not in f or f["block_pulse"].shape[1] != 2:
                raise AssertionError("minecraft h5 missing break/place pulse")
            if int(f["dones"][-1]) != 1:
                raise AssertionError("collector must mark the last frame as the end of a recording")
        latency = float(result["latency_p95_ms"])
    return {"ok": True, "n_frames": n, "latency_p95_ms": latency}


def _mc_cfg():
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
        # This smoke recording is intentionally only one session.  The normal
        # evaluator refuses to call it a held-out score, so the test opts into
        # the explicit diagnostic mode.
        offline = evaluate_offline(cfg, trained["checkpoint"], data, all_data=True)
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
    from atomspike.models.actions import cursor_from_logits, retokenize_mouse_slots

    logits[4] = torch.zeros(1, spec.mouse_bins)
    logits[4][0, spec.mouse_bins - 1] = 10.0
    nx, ny = cursor_from_logits(logits, spec)
    if float(nx.item()) < 0.8:
        raise AssertionError("GUI cursor softmax should track the peaked bin")
    acts = np.zeros((3, 8), dtype=np.int64)
    mouse = np.array([[20.0, 0.0], [20.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    cursor = np.array([[0.2, 0.7], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    gui = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    tok = retokenize_mouse_slots(acts, spec, mouse_xy=mouse, cursor_xy=cursor, gui_open=gui)
    if abs(spec.bin_to_mouse_unit(int(tok[0, spec.n_keys])) - 0.2) > 0.05:
        raise AssertionError("GUI drag should keep window-normalized bins")
    look_bin = int(spec.mouse_deltas_to_bins(np.array([20.0]))[0])
    if int(tok[1, spec.n_keys]) != look_bin:
        raise AssertionError("parked cursor plus a flick must retokenize as look")
    return {"ok": True, "mouse_bins": spec.mouse_bins}


def check_wheel_and_self_learn(cfg_path: Path) -> dict:
    from atomspike.models.actions import ActionSpec, attach_wheel_slot, tokens_to_controls, wheel_from_logits
    from atomspike.runtime.self_learn import OnlineLearner

    spec = ActionSpec(
        n_keys=8,
        n_key_states=4,
        mouse_bins=41,
        n_mouse_buttons=2,
        key_names=("W", "A", "S", "D", "SPACE", "SHIFT", "CTRL", "E"),
        n_slots=13,
        wheel_bins=11,
        wheel_max=4.0,
    )
    if not spec.has_wheel or spec.wheel_index != 12 or spec.idle_tokens()[-1] != spec.wheel_center:
        raise AssertionError("wheel slot layout is wrong")
    if spec.wheel_to_bin(0.0) != spec.wheel_center:
        raise AssertionError("resting wheel must map to the center bin")
    if spec.bin_to_wheel(spec.wheel_center + 2) <= 0:
        raise AssertionError("positive wheel bins must decode to positive notches")
    old = np.zeros((5, 12), dtype=np.int64)
    wh = np.array([0.0, 1.0, -2.0, 0.0, 4.0], dtype=np.float32)
    padded = attach_wheel_slot(old, spec, wh)
    if padded.shape != (5, 13):
        raise AssertionError(f"attach_wheel_slot shape {padded.shape}")
    if int(padded[0, 12]) != spec.wheel_center:
        raise AssertionError("zero wheel should sit on the center bin")
    logits = [torch.zeros(1, v) for v in spec.slot_vocabs]
    logits[spec.wheel_index][0, spec.wheel_center] = 8.0
    w0 = float(wheel_from_logits(logits, spec).reshape(-1)[0].item())
    if abs(w0) > 0.2:
        raise AssertionError("peaked idle wheel should decode near 0")
    logits[spec.wheel_index][0, spec.wheel_center + 3] = 12.0
    w1 = float(wheel_from_logits(logits, spec).reshape(-1)[0].item())
    if w1 <= 0.2:
        raise AssertionError("off-center wheel mass must decode to a scroll")
    idle = torch.tensor(spec.idle_tokens(), dtype=torch.long)
    ctl = tokens_to_controls(idle, spec)
    if abs(float(ctl["mouse"]["wheel"])) > 1e-6:
        raise AssertionError("idle controls must not scroll")

    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    agent.eval()
    d = cfg.encoder.d_model
    x = torch.randn(2, d)
    if not torch.allclose(agent.grow(x), x, atol=1e-5):
        raise AssertionError("dead residual grow bank must be identity")
    if agent.grow.birth() is None:
        raise AssertionError("grow bank should have a free layer")
    if not torch.allclose(agent.grow(x), x, atol=1e-5):
        raise AssertionError("a newly born residual layer must start as identity")
    from atomspike.models.iterator import TASK_AUX_DIM, task_id_from_name, tokens_to_grid

    if agent.iterator is None:
        raise AssertionError("minecraft iterator must be enabled")
    if float(agent.iterator.bias_out.weight.detach().abs().sum()) != 0:
        raise AssertionError("iterator bias must start at zero so old checkpoints stay put")
    if task_id_from_name("minecraft:story/mine_stone") != task_id_from_name("story/mine_stone"):
        raise AssertionError("advancement ids must normalize")
    if task_id_from_name("story/mine_stone") == 0:
        raise AssertionError("named mine_stone must not collapse to empty")
    from atomspike.data.io import _load_task_arrays

    fields = ["hp", "gui_open"]
    st = np.zeros((3, 2), dtype=np.float32)
    st[1, 1] = 1.0
    tid_old, aux_old = _load_task_arrays({}, 3, st, fields)
    if tid_old.shape != (3,) or aux_old.shape != (3, TASK_AUX_DIM):
        raise AssertionError("old hdf5 task fallback shape")
    if float(aux_old[1, 1]) != 1.0 or float(aux_old[1, 3]) != 1.0:
        raise AssertionError("gui_open should fill task_aux on old files")
    fake_adv = {"adv_next": np.array(["story/mine_stone", "", "minecraft:story/mine_stone"], dtype=object)}
    tid_adv, _ = _load_task_arrays(fake_adv, 3, st, fields)
    if int(tid_adv[0]) != int(tid_adv[2]) or int(tid_adv[0]) == 0:
        raise AssertionError("adv_next strings should map when task_id is missing")
    grid = tokens_to_grid(torch.randn(1, 16, d), (4, 4), 4)
    if tuple(grid.shape) != (1, 16, d):
        raise AssertionError(f"spatial grid {tuple(grid.shape)}")
    wide = tokens_to_grid(torch.randn(1, 12 * 20, d), (12, 20), (4, 6))
    if tuple(wide.shape) != (1, 24, d):
        raise AssertionError(f"wide spatial grid {tuple(wide.shape)}")
    it0 = agent.iterator(
        wide if int(agent.iterator.n_cells) == 24 else grid,
        torch.randn(1, d),
        torch.randn(1, cfg.reasoner.game_state_dim),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, TASK_AUX_DIM),
    )
    if tuple(it0["bias"].shape) != (1, d) or float(it0["bias"].detach().abs().max()) > 1e-5:
        raise AssertionError("fresh iterator must emit a near-zero bias of d_model")
    from atomspike.models.inventory import (
        INV_SLOTS,
        item_code_from_path,
        item_id_from_name,
        inventory_pack_from_signal,
    )
    from atomspike.data.io import _load_inv_arrays
    import zlib

    if agent.inventory is None:
        raise AssertionError("minecraft inventory sense must be enabled")
    if float(agent.inventory.bias_out.weight.detach().abs().sum()) != 0:
        raise AssertionError("inventory bias must start at zero so old checkpoints stay put")
    if item_id_from_name("minecraft:stone") != item_id_from_name("stone"):
        raise AssertionError("item ids must strip the minecraft: prefix")
    if item_id_from_name("stone") == 0:
        raise AssertionError("named stone must not collapse to empty")
    if item_code_from_path("stone") != (zlib.crc32(b"stone") & 0xFFFFFFFF):
        raise AssertionError("item CRC32 must match sidecar / zlib")
    inv_old, hand_old, _, _ = _load_inv_arrays({}, 3)
    if inv_old.shape != (3, INV_SLOTS) or int(hand_old.sum()) != 0:
        raise AssertionError("old hdf5 inventory fallback should be zeros")
    packed = inventory_pack_from_signal({"hand_item": "stone_pickaxe", "inv": [item_code_from_path("stone")] + [0] * 35})
    if int(packed["hand_id"]) != item_id_from_name("stone_pickaxe"):
        raise AssertionError("hand_item path should map to the named catalog")
    if int(packed["inv_ids"][0]) != item_id_from_name("stone"):
        raise AssertionError("inv CRC codes should map to named items")
    st1 = torch.zeros(1, cfg.reasoner.game_state_dim)
    n_inv = int(agent.inventory.n_slots)
    inv0 = agent.inventory(
        torch.zeros(1, n_inv, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        st1,
    )
    ids = torch.zeros(1, n_inv, dtype=torch.long)
    ids[0, 0] = item_id_from_name("stone")
    inv1 = agent.inventory(
        ids,
        torch.tensor([item_id_from_name("stone")]),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        st1,
    )
    if tuple(inv0["bias"].shape) != (1, d) or float(inv1["bias"].detach().abs().max()) > 1e-5:
        raise AssertionError("fresh inventory LIF must emit a near-zero bias of d_model")
    frames, state = _dummy_batch(agent, device)
    with torch.no_grad():
        out = agent(frames, state, reason_tick=True)
    mix = out["plan"]["mix"]
    n_core = cfg.planner.n_phases
    if int(mix.size(-1)) != n_core + cfg.planner.event_pool:
        raise AssertionError("planner mix width should be core events plus the extra pool")
    if float(mix[0, n_core:].sum()) > 1e-4:
        raise AssertionError("dead extra events leaked mix mass")
    if agent.planner.birth_event() is None:
        raise AssertionError("event pool should have a free slot")
    # The two online modes must stay distinct: human input is supervised CE;
    # guarded injected actions use sampled actor-critic updates.  A prediction
    # which was not actually injected must never become experience.
    pooled = torch.randn(1, d)
    fused = torch.randn(1, d)
    st = torch.randn(1, cfg.reasoner.game_state_dim)
    act = torch.zeros(1, agent.spec.n_slots, dtype=torch.long)
    blocked = OnlineLearner(agent, cfg, device, save_path=None, inject=True, background=False)
    blocked.observe(
        pooled=pooled,
        fused_base=fused,
        policy_input=fused,
        state=st,
        human_tokens=None,
        injected=False,
        act=act,
        signal={"hp": 20.0, "adv_done": 0.0, "adv_next": "stone_age"},
    )
    blocked.observe(
        pooled=pooled,
        fused_base=fused,
        policy_input=fused,
        state=st,
        human_tokens=None,
        injected=False,
        act=act,
        signal={"hp": 20.0, "adv_done": 1.0, "adv_next": "iron_tools"},
    )
    if blocked.n_steps != 0:
        raise AssertionError("a gate-blocked action must not become self-play experience")
    neutral = OnlineLearner(agent, cfg, device, save_path=None, inject=True, background=False)
    for _ in range(2):
        neutral.observe(
            pooled=pooled,
            fused_base=fused,
            policy_input=fused,
            state=st,
            human_tokens=None,
            injected=True,
            act=act,
            signal={"hp": 20.0, "adv_done": 0.0, "adv_next": "stone_age"},
        )
    if neutral.n_steps != 1 or not math.isfinite(neutral.last_loss):
        raise AssertionError("neutral executed transitions must train the TD value baseline")
    autonomous = OnlineLearner(agent, cfg, device, save_path=None, inject=True, background=False)
    autonomous.observe(
        pooled=pooled,
        fused_base=fused,
        policy_input=fused,
        state=st,
        human_tokens=None,
        injected=True,
        act=act,
        signal={"hp": 20.0, "adv_done": 0.0, "adv_next": "stone_age"},
    )
    autonomous.observe(
        pooled=pooled,
        fused_base=fused,
        policy_input=fused,
        state=st,
        human_tokens=None,
        injected=True,
        act=act,
        signal={"hp": 20.0, "adv_done": 1.0, "adv_next": "iron_tools"},
    )
    if autonomous.n_steps < 1 or autonomous.mode != "self_play":
        raise AssertionError("an executed sampled action with reward must produce a self-play RL step")
    if not math.isfinite(autonomous.last_loss):
        raise AssertionError("self-play actor-critic loss must remain finite")
    # Foreground verification is intentionally slower than the production CPU
    # worker; do not let the latency budget drop one of the synthetic samples.
    for done in (1.0, 1.0):
        autonomous._skip = False
        autonomous.observe(
            pooled=pooled,
            fused_base=fused,
            policy_input=fused,
            state=st,
            human_tokens=None,
            injected=True,
            act=act,
            signal={"hp": 20.0, "adv_done": done, "adv_next": "iron_tools"},
        )
    autonomous._skip = False
    credit_info = autonomous.observe(
        pooled=pooled,
        fused_base=fused,
        policy_input=fused,
        state=st,
        human_tokens=None,
        injected=True,
        act=act,
        signal={"hp": 20.0, "adv_done": 2.0, "adv_next": "diamonds"},
    )
    if int(credit_info.get("credit_steps", 0)) != int(cfg.learn.credit_steps):
        raise AssertionError("advancement reward must receive the configured discounted backward credit window")
    with tempfile.TemporaryDirectory() as td:
        plastic_path = Path(td) / "plastic.pt"
        autonomous.save_plastic(plastic_path)
        restored = OnlineLearner(agent, cfg, device, save_path=None, inject=True, background=False)
        if not restored.load_plastic(plastic_path):
            raise AssertionError("plastic checkpoint did not reload")
        if restored.n_steps != autonomous.n_steps:
            raise AssertionError("plastic checkpoint did not restore online step count")
        payload = torch.load(plastic_path, map_location="cpu", weights_only=True)
        if "optimizer" not in payload or not payload["optimizer"].get("state"):
            raise AssertionError("plastic checkpoint did not preserve Adam optimizer state")
    background = OnlineLearner(agent, cfg, device, save_path=None, inject=True, background=True)
    live_before = agent.policy.value_head.bias.detach().clone()
    background.observe(
        pooled=pooled,
        fused_base=fused,
        policy_input=fused,
        state=st,
        human_tokens=None,
        injected=True,
        act=act,
        signal={"hp": 20.0, "adv_done": 0.0, "adv_next": "stone_age"},
    )
    background.observe(
        pooled=pooled,
        fused_base=fused,
        policy_input=fused,
        state=st,
        human_tokens=None,
        injected=True,
        act=act,
        signal={"hp": 20.0, "adv_done": 1.0, "adv_next": "iron_tools"},
    )
    background.wait_idle()
    if not torch.equal(agent.policy.value_head.bias.detach(), live_before):
        raise AssertionError("background learner must not mutate live inference parameters from its worker")
    background._publish_cpu()
    if torch.equal(agent.policy.value_head.bias.detach(), live_before):
        raise AssertionError("completed CPU self-play update was not published on the inference thread")
    background.close()
    learner = OnlineLearner(agent, cfg, device, save_path=None, inject=False, background=False)
    learner.observe(
        pooled=pooled, fused_base=fused, state=st, human_tokens=act, injected=False, act=act, signal={}
    )
    learner.observe(
        pooled=pooled, fused_base=fused, state=st, human_tokens=act, injected=False, act=act, signal={}
    )
    if learner.n_steps < 1:
        raise AssertionError("human demonstration should produce an adaptation step")
    from atomspike.runtime.self_learn import goal_hp_reward

    r_goal, _g, penalty = goal_hp_reward(hp0=20, hp1=20, adv0=1, adv1=2, frac0=0.8, frac1=0.0)
    if r_goal <= 0 or penalty != 0:
        raise AssertionError("progress diagnostic must reward advancement without a damage penalty")
    return {
        "ok": True,
        "n_slots": spec.n_slots,
        "learn_steps": learner.n_steps + autonomous.n_steps,
        "mix_dim": int(mix.size(-1)),
        "mode": "human_demo_and_self_play_actor_critic",
    }


def check_planner(cfg_path: Path) -> dict:
    from atomspike.data.sessions import build_episode_timeline

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
    if snap.get("iter_grid") is None:
        raise AssertionError("activity payload missing iterator grid")
    live = compose_frame(snap, ScopeBuffers(), dt=1.0 / 120.0, viz_hz=120.0)
    if live.shape != img.shape:
        raise AssertionError("agent payload frame size must match dummy window")
    return {"ok": True, "shape": list(img.shape), "kind": snap.get("kind")}


def check_session_splits() -> dict:
    from atomspike.data.sessions import (
        CLOCK_DIM,
        apply_recording_boundaries,
        build_episode_timeline,
        clock_vector,
        recording_ids,
        stride_split_segments,
    )

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
    marked = dones.copy()
    marked[3] = 1
    rid = recording_ids(n, times, dones=marked, gap_s=2.0)
    if int(rid[3]) != int(rid[4]):
        raise AssertionError("an in-recording episode mark must not create a validation recording")
    if int(rid[n1 - 1]) == int(rid[n1]):
        raise AssertionError("a timestamp gap must create a validation recording")
    train_idx, val_idx = stride_split_segments(n, 0.25, rid)
    if val_idx.size == 0 or train_idx.size == 0:
        raise AssertionError("two recordings should produce a held-out recording split")
    if set(np.asarray(rid)[train_idx].tolist()) & set(np.asarray(rid)[val_idx].tolist()):
        raise AssertionError("validation may not share a recording id with training")
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
        "n_recordings": int(rid.max()) + 1,
        "session_lengths": list(tl["session_lengths"]),
        "clock_dim": CLOCK_DIM,
    }


def check_depth_identity(cfg_path: Path) -> dict:
    """Extra CNN / reasoner layers and analog SNN skips must start as no-ops."""
    from types import SimpleNamespace

    from atomspike.data.augment import augment_frame_pair
    from atomspike.train.common import expand_state_dict

    def _plain(cfg):
        cfg.reasoner.dropout = 0.0
        cfg.reasoner.drop_path = 0.0
        cfg.reasoner.token_drop = 0.0
        cfg.policy.dropout = 0.0
        cfg.policy.drop_path = 0.0
        return cfg.aligned()

    shallow = _plain(load_config(cfg_path))
    shallow.encoder.n_blocks_per_stage = 1
    shallow.reasoner.n_layers = 2
    deep = _plain(load_config(cfg_path))
    deep.encoder.n_blocks_per_stage = 2
    deep.reasoner.n_layers = 3
    device = torch.device("cpu")
    a = AtomSpikeAgent(shallow).to(device).eval()
    b = AtomSpikeAgent(deep).to(device).eval()
    b.load_state_dict(expand_state_dict(b, a.state_dict()), strict=False)
    frames, state = _dummy_batch(a, device)
    with torch.no_grad():
        t1, p1 = a.encoder(frames)
        t2, p2 = b.encoder(frames)
    if not torch.allclose(t1, t2, atol=1e-5) or not torch.allclose(p1, p2, atol=1e-5):
        raise AssertionError("identity extra CNN blocks changed encoder output")
    y1 = _logits(a, frames, state)
    y2 = _logits(b, frames, state)
    if not torch.allclose(y1, y2, atol=1e-4):
        raise AssertionError("deeper net with identity extras must match the shallow checkpoint")
    if not hasattr(a.policy, "blocks"):
        raise AssertionError("expected a spiking policy")
    for blk in a.policy.blocks:
        if float(blk.analog_scale.detach().abs().sum()) != 0:
            raise AssertionError("spike analog residual must start at zero")
    fake = SimpleNamespace(
        aug_brightness=0.25,
        aug_contrast=0.25,
        aug_saturation=0.2,
        aug_noise=0.02,
        aug_erase=1.0,
        aug_erase_scale=0.1,
    )
    aug, prev = augment_frame_pair(frames, frames, fake, gui=None)
    if tuple(aug.shape) != tuple(frames.shape) or tuple(prev.shape) != tuple(frames.shape):
        raise AssertionError("augmentation must keep BCHW")
    if float(aug.min()) < -1e-5 or float(aug.max()) > 1.0 + 1e-5:
        raise AssertionError("augmentation must stay in [0, 1]")
    return {
        "ok": True,
        "encoder_tokens": int(t1.size(1)),
        "reasoner_layers": int(deep.reasoner.n_layers),
    }


def check_gui_world_hold(cfg_path: Path) -> dict:
    from atomspike.capture.mc_state import MC_STATE_FIELDS
    from atomspike.models.iterator import TASK_AUX_DIM

    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    agent.eval()
    agent.set_activity_trace(True)
    frames_w, state = _dummy_batch(agent, device)
    if "gui_open" not in MC_STATE_FIELDS:
        raise AssertionError("gui_open missing from game_state")
    gi = MC_STATE_FIELDS.index("gui_open")
    state = state.clone()
    if gi < state.size(-1):
        state[:, gi] = 0
    aux_w = torch.zeros(frames_w.size(0), TASK_AUX_DIM)
    with torch.no_grad():
        agent.reset_runtime()
        agent(frames_w, state, reason_tick=True, task_aux=aux_w)
    cells = agent._iter_cells
    if cells is None:
        raise AssertionError("world tick must store iterator cells")
    held = cells.clone()
    frames_g = torch.rand_like(frames_w)
    state_g = state.clone()
    if gi < state_g.size(-1):
        state_g[:, gi] = 1
    aux_g = aux_w.clone()
    aux_g[:, 1] = 1
    with torch.no_grad():
        agent(frames_g, state_g, reason_tick=True, task_aux=aux_g)
    if agent._iter_cells is None or not torch.allclose(agent._iter_cells, held, atol=1e-5):
        raise AssertionError("GUI tick must not overwrite spatial cells")
    delta = None if agent._last_iter is None else agent._last_iter.get("grid_delta")
    if delta is None or float(np.max(np.abs(np.asarray(delta)))) > 1e-5:
        raise AssertionError("GUI spatial delta must be ~0")
    if not bool((agent._last_iter or {}).get("hold")):
        raise AssertionError("GUI tick must set iterator hold")
    ad = agent.adapter
    a = torch.rand(1, 3, frames_w.size(2), frames_w.size(3))
    b = torch.rand_like(a)
    r0 = ad(b, a)
    r1 = ad(b, a, hold_mask=torch.zeros(1, dtype=torch.bool))
    if not torch.allclose(r0, r1, atol=1e-5):
        raise AssertionError("hold_mask=0 must match the old temporal adapter")
    r_hold = ad(b, a, hold_mask=torch.ones(1, dtype=torch.bool))
    r_same = ad(a, a)
    if not torch.allclose(r_hold, r_same, atol=1e-5):
        raise AssertionError("GUI hold must freeze temporal residual to the previous frame")
    return {"ok": True}


def check_lif_curl_scene(cfg_path: Path) -> dict:
    from atomspike.models.curl_scene import discrete_curl
    from atomspike.models.lif import LIFCell

    x = torch.tensor([[0.2, 1.1, 0.9, 2.0]], dtype=torch.float32)
    lif = LIFCell(tau=0.5, v_th=1.0, thresh_shape=(4,))
    if float(lif.v_th.detach().mean()) != 1.0:
        raise AssertionError("learnable thresholds must init at v_th")
    spike, mem = lif(x)
    expect = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    if not torch.equal(spike, expect):
        raise AssertionError("LIF spikes at init must match a hard 1.0 threshold")
    xg = x.clone().requires_grad_(True)
    spk, _ = lif(xg)
    spk.sum().backward()
    if lif.v_th.grad is None or float(lif.v_th.grad.abs().sum()) <= 0:
        raise AssertionError("each neuron threshold must receive a gradient")
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), indexing="ij")
    curl = discrete_curl((-yy).unsqueeze(0), xx.unsqueeze(0))
    if float(curl.mean()) <= 0:
        raise AssertionError("a counter-clockwise field should have positive curl")
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    if agent.scene is None:
        raise AssertionError("curl scene head must be enabled")
    if float(agent.scene.bias_out.weight.detach().abs().sum()) != 0:
        raise AssertionError("curl scene bias must start at zero so old checkpoints stay put")
    n_cells = int(agent.scene.spatial_h * agent.scene.spatial_w)
    probe = torch.zeros(1, n_cells, cfg.encoder.d_model)
    probe[:, ::2, 0] = 2.0
    with torch.no_grad():
        agent.scene.cell_proj.weight.zero_()
        agent.scene.cell_proj.bias.zero_()
        agent.scene.cell_proj.weight[:, 0] = 1.0
        agent.scene.vec.weight.zero_()
        agent.scene.vec.weight[0, 0] = 1.0
        agent.scene.vec.weight[1, 1] = 1.0
        agent.scene.lif.v_th.fill_(1.0)
        field_spiking = agent.scene(probe)["vx"]
        agent.scene.lif.v_th.fill_(4.0)
        field_silent = agent.scene(probe)["vx"]
    if torch.allclose(field_spiking, field_silent):
        raise AssertionError("curl vector field must be driven by LIF spikes, not bypassed analog cells")
    frames, state = _dummy_batch(agent, device)
    agent.eval()
    with torch.no_grad():
        agent.reset_runtime()
        out0 = agent(frames, state, reason_tick=True)
        for name in (
            "_curl_mem",
            "_group_mem",
            "_sea_mem",
            "_detect_mem",
            "_event_slots",
            "_inv_mem",
        ):
            if getattr(agent, name, None) is not None:
                raise AssertionError(f"{name} persisted although offline training is stateless")
        agent.scene.vec.weight.add_(0.35)
        agent.reset_runtime()
        out1 = agent(frames, state, reason_tick=True)
    a = torch.cat([s.flatten() for s in out0["logits"]])
    b = torch.cat([s.flatten() for s in out1["logits"]])
    if not torch.allclose(a, b, atol=1e-5):
        raise AssertionError("zero-init curl bias must ignore the spike vector field")
    return {"ok": True, "v_th": float(lif.v_th.detach().mean())}


def check_group_coalition(cfg_path: Path) -> dict:
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    if agent.groups is None:
        raise AssertionError("spike group coalitions must be enabled")
    g = agent.groups
    if int(g.n_groups) < 1 or int(cfg.encoder.d_model) % int(g.n_groups) != 0:
        raise AssertionError("group count must divide d_model")
    if float(g.resid_scale.detach().abs().sum()) != 0:
        raise AssertionError("group attention residual must start at zero so old checkpoints stay put")
    if int(g.n_alive()) != 0:
        raise AssertionError("group specialists must start dormant")
    frames, state = _dummy_batch(agent, device)
    agent.eval()

    def _flat(out):
        return torch.cat([s.flatten() for s in out["logits"]])

    with torch.no_grad():
        agent.reset_runtime()
        y0 = _flat(agent(frames, state, reason_tick=True))
        g.attn.qkv.weight.add_(0.45)
        agent.reset_runtime()
        y1 = _flat(agent(frames, state, reason_tick=True))
        if not torch.allclose(y0, y1, atol=1e-5):
            raise AssertionError("zero resid_scale must ignore group spike attention")
        x = torch.randn(2, cfg.encoder.d_model)
        if not torch.allclose(g.apply_specialists(x), x, atol=1e-6):
            raise AssertionError("no living specialist must skip the 30Hz loop")
        idx = g.birth_specialist(energy=torch.ones(g.n_groups))
        if idx is None:
            raise AssertionError("birth_specialist must turn on the hottest idle group")
        g.specialists[idx].fc1.weight.add_(0.8)
        agent.reset_runtime()
        y2 = _flat(agent(frames, state, reason_tick=True))
        if not torch.allclose(y0, y2, atol=1e-5):
            raise AssertionError("born specialists stay identity until the last linear learns")
    return {"ok": True, "n_groups": int(g.n_groups), "d_group": int(g.d_group)}


def check_screen_event_attention(cfg_path: Path) -> dict:
    from atomspike.models.sea import coincidence_weights, compact_screen_tokens

    q = torch.zeros(1, 1, 1, 4)
    k = torch.zeros(1, 1, 2, 4)
    q[0, 0, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    k[0, 0, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    k[0, 0, 1] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    w = coincidence_weights(q, k)
    if abs(float(w[0, 0, 0, 0]) - 1.0) > 1e-5 or abs(float(w[0, 0, 0, 1])) > 1e-5:
        raise AssertionError("coincidence must put all mass on co-firing keys and none on silent ones")
    pooled, layout = compact_screen_tokens(torch.randn(2, 100, 8), (10, 10), 64)
    if int(pooled.size(1)) > 64 or layout != (8, 8):
        raise AssertionError("screen tokens must pool to a compact grid before event attention")
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    if agent.sea is None:
        raise AssertionError("SECA must be enabled")
    if float(agent.sea.resid_scale.detach().abs().sum()) != 0:
        raise AssertionError("SECA residual must start at zero so old checkpoints stay put")
    frames, state = _dummy_batch(agent, device)
    agent.eval()

    def _flat(out):
        return torch.cat([s.flatten() for s in out["logits"]])

    with torch.no_grad():
        agent.reset_runtime()
        y0 = _flat(agent(frames, state, reason_tick=True))
        agent.sea.qkv.weight.add_(0.45)
        agent.reset_runtime()
        y1 = _flat(agent(frames, state, reason_tick=True))
        if not torch.allclose(y0, y1, atol=1e-5):
            raise AssertionError("zero resid_scale must ignore screen-event coincidence attention")
    return {"ok": True, "k_events": int(agent.sea.k_events), "max_spatial": int(agent.sea.max_spatial)}


def check_event_seq_and_net(cfg_path: Path) -> dict:
    from atomspike.models.encoder import net_input_hw

    h, w = net_input_hw(480, 854, 160, 12, 20)
    if h > 96 or w > 160:
        raise AssertionError(f"adapter downsample should cap to the token grid, got {h}x{w}")
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    if agent.event_seq is None:
        raise AssertionError("global event sequence must be enabled")
    if float(agent.event_seq.resid_scale.detach().abs().sum()) != 0:
        raise AssertionError("event-seq residual must start at zero")
    if float(agent.event_seq.write_scale.detach().abs()) != 0:
        raise AssertionError("event-seq write must start at zero")
    if agent.inventory is None or float(agent.inventory.pointer_scale.detach().abs()) != 0:
        raise AssertionError("GUI pointer scale must start at zero")
    if not hasattr(agent.adapter, "gui_scale") or float(agent.adapter.gui_scale.detach().abs()) != 0:
        raise AssertionError("GUI adapter scale must start at zero")
    frames, state = _dummy_batch(agent, device)
    agent.eval()

    def _flat(out):
        return torch.cat([s.flatten() for s in out["logits"]])

    with torch.no_grad():
        agent.reset_runtime()
        y0 = _flat(agent(frames, state, reason_tick=True))
        agent.event_seq.qkv.weight.add_(0.4)
        agent.inventory.slot_xy.add_(0.2)
        agent.reset_runtime()
        y1 = _flat(agent(frames, state, reason_tick=True))
        if not torch.allclose(y0, y1, atol=1e-5):
            raise AssertionError("zero event-seq / pointer scales must ignore new heads")
        big = torch.zeros(frames.size(0), 3, 240, 426)
        y2 = _flat(agent(big, state, reason_tick=True))
        if y2.numel() != y0.numel():
            raise AssertionError("native large frames must still produce action logits")
        tok, _ = agent.encoder(big if big.dtype == torch.float32 else big.float())
        gh, gw = agent.encoder.last_hw
        if gh > int(cfg.encoder.max_token_h) or gw > int(cfg.encoder.max_token_w):
            raise AssertionError(f"reasoner tokens should cap at the grid, got {gh}x{gw}")
    return {"ok": True, "n_slots": int(agent.event_seq.n_slots), "n_events": int(agent.event_seq.n_events), "net_hw": [h, w], "token_hw": [gh, gw]}


def check_item_use_semantics(cfg_path: Path) -> dict:
    from atomspike.models.inventory import InventorySense, item_id_from_name, item_use_row

    pick = item_use_row("stone_pickaxe")
    food = item_use_row("bread")
    sword = item_use_row("iron_sword")
    if pick[6] < 0.9 or pick[3] > 0.1:
        raise AssertionError("pickaxe should be an LMB harvest tool, not food")
    if food[3] < 0.9 or food[7] < 0.9:
        raise AssertionError("bread should be edible via RMB")
    if sword[1] < 0.9 or sword[6] < 0.9:
        raise AssertionError("sword should be melee LMB")
    cfg = load_config(cfg_path)
    inv = InventorySense(cfg.inventory, d_model=cfg.encoder.d_model, state_dim=cfg.reasoner.game_state_dim)
    pid = item_id_from_name("stone_pickaxe", inv.n_buckets)
    if float(inv.use_tbl[pid, 6]) < 0.9:
        raise AssertionError("use table missing pickaxe LMB prior")
    b = 2
    ids = torch.zeros(b, inv.n_slots, dtype=torch.long)
    ids[:, 0] = pid
    state = torch.zeros(b, cfg.reasoner.game_state_dim)
    out = inv(ids, torch.full((b,), pid, dtype=torch.long), None, None, state)
    if out["bias"].shape != (b, cfg.encoder.d_model):
        raise AssertionError("inventory bias shape")
    if float(out["bias"].detach().abs().sum()) > 1e-5:
        raise AssertionError("zero use_scale / bias_out must keep old checkpoints silent")
    return {"ok": True, "pickaxe_id": int(pid)}


def check_innate_block_pulse() -> dict:
    from atomspike.models.sea import (
        BlockPulseTracker,
        apply_innate_block_pulse,
        innate_block_patterns,
        look_center_index,
    )

    brk, plc = innate_block_patterns(16, torch.device("cpu"), torch.float32)
    if abs(float((brk * plc).sum())) > 1e-4:
        raise AssertionError("break and place tags must be orthogonal")
    if abs(float(brk.pow(2).mean().sqrt()) - 1.0) > 1e-5:
        raise AssertionError("break tag must be unit RMS")
    if abs(float(plc.pow(2).mean().sqrt()) - 1.0) > 1e-5:
        raise AssertionError("place tag must be unit RMS")
    tok = torch.randn(2, 12, 16)
    hw = (3, 4)
    idx = look_center_index(12, hw)
    zed = apply_innate_block_pulse(tok, torch.zeros(2, 2), hw=hw, amp=0.08)
    if not torch.allclose(tok, zed, atol=1e-6):
        raise AssertionError("zero pulse must leave tokens unchanged")
    broke = apply_innate_block_pulse(tok, torch.tensor([[1.0, 0.0], [1.0, 0.0]]), hw=hw, amp=0.08)
    placed = apply_innate_block_pulse(tok, torch.tensor([[0.0, 1.0], [0.0, 1.0]]), hw=hw, amp=0.08)
    cell = tok[:, idx]
    bcell = broke[:, idx]
    pcell = placed[:, idx]
    cos = (cell * bcell).sum(-1) / (cell.norm(dim=-1) * bcell.norm(dim=-1)).clamp_min(1e-6)
    if float(cos.min()) < 0.99:
        raise AssertionError("break tag drowned look-center appearance")
    if torch.allclose(bcell, pcell, atol=1e-5):
        raise AssertionError("break and place stimuli must differ")
    held = apply_innate_block_pulse(
        tok, torch.ones(2, 2), hw=hw, amp=0.08, hold=torch.ones(2, dtype=torch.bool)
    )
    if not torch.allclose(tok, held, atol=1e-6):
        raise AssertionError("GUI/pause must not emit the block stimulus")
    tr = BlockPulseTracker()
    if float(tr.observe({"broke": 3, "placed": 1}).sum()) != 0:
        raise AssertionError("first sidecar packet is not a pulse")
    one = tr.observe({"broke": 4, "placed": 1})
    if abs(float(one[0]) - 1.0) > 1e-6 or abs(float(one[1])) > 1e-6:
        raise AssertionError("a break increment should pulse once")
    if float(tr.observe({"broke": 4, "placed": 1}).sum()) != 0:
        raise AssertionError("unchanged counters must not restimulate")
    if float(tr.observe({"broke": 0, "placed": 0}).sum()) != 0:
        raise AssertionError("a counter reset is not a pulse")
    return {"ok": True, "center": int(idx)}


def check_game_hud() -> dict:
    from atomspike.viz.game_hud import CHROMA, compose_hud, dummy_overlay_snap, merge_detect_boxes

    idle = compose_hud(None, 320, 180)
    if idle.shape != (180, 320, 3):
        raise AssertionError(f"hud idle shape {idle.shape}")
    chroma = np.array(CHROMA, dtype=np.uint8)
    if not np.all(idle[80, 40] == chroma):
        raise AssertionError("idle hud must be chroma-keyed below the status bar")
    bar0 = idle[4, 20]
    if tuple(int(v) for v in bar0) == CHROMA:
        raise AssertionError("status bar must be opaque even when idle")
    dummy = dummy_overlay_snap()
    boxes = merge_detect_boxes(
        dummy["idx"], dummy["cls"], dummy["obj"], infl=dummy["infl"], conf=dummy["conf"], hw=dummy["hw"]
    )
    if len(boxes) != 2:
        raise AssertionError(f"adjacent same-class cells should merge, got {len(boxes)} boxes")
    from atomspike.models.curl_scene import chrome_keep_map

    keep = chrome_keep_map(4, 6)
    if bool(keep[3, 0]) or bool(keep[3, 5]):
        raise AssertionError("hotbar cells must be dropped from the spatial field")
    if not bool(keep[0, 2]) or not bool(keep[1, 2]):
        raise AssertionError("world cells must stay in the spatial field")
    live = compose_hud(dummy, 800, 360)
    if live.shape != (360, 800, 3):
        raise AssertionError("live hud size must match window")
    painted = int(np.any(live != chroma, axis=-1).sum())
    if painted < 200:
        raise AssertionError("hud must paint boxes, vectors, and the side panel")
    bar = live[4, 20]
    if tuple(int(v) for v in bar) == CHROMA:
        raise AssertionError("status bar must be opaque")
    cfg = load_config(None)
    cfg.aligned()
    agent = AtomSpikeAgent(cfg)
    agent.eval()
    frames, state = _dummy_batch(agent, torch.device("cpu"))
    with torch.no_grad():
        _ = agent.act(frames, state, deterministic=True, reason_tick=True)
    snap = agent.overlay_snapshot({"latency_ms": 2.0, "run_perception": True, "controls": {"keys": {"W": "hold"}}})
    if not snap.get("enabled"):
        raise AssertionError("overlay snapshot missing detect head")
    if snap.get("idx") is None or int(np.asarray(snap["idx"]).size) < 1:
        raise AssertionError("overlay snapshot missing cell indices")
    if snap.get("infl") is None or int(np.asarray(snap["infl"]).size) < 1:
        raise AssertionError("overlay snapshot missing mix influence")
    if snap.get("vx") is None:
        raise AssertionError("overlay snapshot missing spatial vector field")
    drawn = compose_hud(snap, 160, 90)
    if drawn.shape != (90, 160, 3):
        raise AssertionError("agent hud compose size")
    return {"ok": True, "shape": list(live.shape), "painted": painted, "boxes": len(boxes)}


def check_spike_grid_detect(cfg_path: Path) -> dict:
    from atomspike.models.grid_detect import SCREEN_KINDS

    cfg = load_config(cfg_path)
    if len(SCREEN_KINDS) < 4:
        raise AssertionError("screen-kind table is too small")
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    if agent.detect is None:
        raise AssertionError("YOLO-style spike grid detect must be enabled")
    det = agent.detect
    if float(det.resid_scale.detach().abs().sum()) != 0:
        raise AssertionError("detect residual must start at zero so old checkpoints stay put")
    tokens = torch.randn(2, 24, cfg.encoder.d_model)
    out = det.observe(tokens, hw=(4, 6))
    if out["objectness"].shape != (2, det.grid_h * det.grid_w):
        raise AssertionError("objectness must be one score per grid cell")
    if out["class_idx"].size(-1) != int(det.n_keep):
        raise AssertionError("YOLO decode must keep the top-M cells")
    if "idx" not in out or int(out["idx"].size(-1)) != int(det.n_keep):
        raise AssertionError("YOLO decode must return cell indices")
    if out["infl"].shape != out["class_idx"].shape:
        raise AssertionError("per-cell mix influence must match the kept set")
    kept_obj = out["objectness"].gather(1, out["idx"])
    if float(kept_obj[0].sum().item()) > 0:
        if abs(float(out["infl"][0].detach().sum().item()) - 1.0) > 1e-4:
            raise AssertionError("active mix influence must be a simplex over kept cells")
    elif float(out["infl"][0].detach().abs().sum().item()) > 1e-6:
        raise AssertionError("silent top-k bookkeeping cells must have zero influence")
    silent = det.observe(tokens, hw=(4, 6), gui_mask=torch.ones(2, dtype=torch.bool))
    if float(silent["objectness"].sum().item()) != 0:
        raise AssertionError("GUI-held spike-grid should be silent")
    if float(silent["mask"].sum().item()) != 0:
        raise AssertionError("silent spike-grid mask must not claim k detections")
    if out["bias"].shape != (2, cfg.encoder.d_model):
        raise AssertionError("detect bias shape")
    if float(out["bias"].detach().abs().sum()) > 1e-5:
        raise AssertionError("zero resid_scale must keep the presence grid silent")
    frames, state = _dummy_batch(agent, device)
    agent.eval()

    def _flat(out):
        return torch.cat([s.flatten() for s in out["logits"]])

    with torch.no_grad():
        agent.reset_runtime()
        y0 = _flat(agent(frames, state, reason_tick=True))
        det.obj.weight.add_(0.4)
        agent.reset_runtime()
        y1 = _flat(agent(frames, state, reason_tick=True))
        if not torch.allclose(y0, y1, atol=1e-5):
            raise AssertionError("zero resid_scale must ignore spike-grid detections")
    return {
        "ok": True,
        "grid": [int(det.grid_h), int(det.grid_w)],
        "n_keep": int(det.n_keep),
        "n_classes": int(det.n_classes),
    }


def run_verify(cfg_path: Path) -> dict:
    return {
        "pmsm": check_pmsm_survives_save(cfg_path),
        "lora": check_lora_roundtrip(cfg_path),
        "dual_rate_sim": check_dual_rate_sim(),
        "dual_rate_realtime": check_dual_rate_realtime(),
        "mouse_asinh": check_mouse_asinh(),
        "wheel_self_learn": check_wheel_and_self_learn(cfg_path),
        "planner": check_planner(cfg_path),
        "session_splits": check_session_splits(),
        "snn_scope": check_snn_scope(),
        "capture_preview": check_capture_preview(),
        "mc_collector": check_mc_collector_dry_run(),
        "mc_train_predict": check_mc_train_predict(),
        "depth_identity": check_depth_identity(cfg_path),
        "gui_world_hold": check_gui_world_hold(cfg_path),
        "lif_curl_scene": check_lif_curl_scene(cfg_path),
        "group_coalition": check_group_coalition(cfg_path),
        "screen_event_attn": check_screen_event_attention(cfg_path),
        "event_seq_net": check_event_seq_and_net(cfg_path),
        "item_use": check_item_use_semantics(cfg_path),
        "block_pulse": check_innate_block_pulse(),
        "grid_detect": check_spike_grid_detect(cfg_path),
        "game_hud": check_game_hud(),
    }
