"""Opt-in online adaptation while ``predict-mc`` is running.

There are two explicit modes.  Without injection, human actions train a
supervised adapter.  With guarded singleplayer injection, sampled model
actions are updated by an advantage policy gradient with a learned value
baseline, entropy bonus, and categorical KL to a frozen policy teacher.  A
greedy prediction is therefore never recycled as a pretend supervised label.
"""

from __future__ import annotations

import math
import copy
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import AtomSpikeConfig
from atomspike.models.agent import AtomSpikeAgent, action_ce_loss, default_slot_weights
from atomspike.models.policy_spike import SpikePolicy


def goal_hp_reward(
    *,
    hp0: float,
    hp1: float,
    adv0: float,
    adv1: float,
    frac0: float,
    frac1: float,
    goal_w: float = 1.0,
    hp_w: float = 0.2,
) -> tuple[float, float, float]:
    """Return (reward, goal_term, hp_penalty). HP term is only a drop penalty."""
    d_done = max(0.0, float(adv1) - float(adv0))
    d_frac = max(0.0, float(frac1) - float(frac0))
    goal = float(goal_w) * (d_done + 0.35 * d_frac)
    d_hp = float(hp1) - float(hp0)
    penalty = float(hp_w) * max(0.0, -d_hp) if d_hp < -0.25 else 0.0
    return goal - penalty, goal, penalty


def _flag(signal: dict[str, Any] | None, key: str) -> bool | None:
    if not signal or key not in signal:
        return None
    v = signal[key]
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    if v is None:
        return None
    return bool(v)


def open_sky_score(signal: dict[str, Any] | None, *, pitch_up: float = -32.0) -> float:
    """1 if the player is looking up at actually-visible sky, else 0.

    Looking up at a cave ceiling, leaves, or a block is not sky. A horizon
    view (pitch near 0) is not sky even if the upper third of the frame is blue.
    """
    if not signal:
        return 0.0
    if _flag(signal, "gui_open") or _flag(signal, "in_water"):
        return 0.0
    pitch = float(signal.get("pitch") or 0.0)
    if pitch > float(pitch_up):
        return 0.0
    up = min(1.0, (float(pitch_up) - pitch) / 45.0)
    frac = float(signal.get("sky_frac") or 0.0)
    miss = _flag(signal, "look_miss")
    see = _flag(signal, "can_see_sky")
    light_raw = signal.get("sky_light")
    light = None if light_raw is None else float(light_raw)
    if miss is None:
        vis = 0.0 if frac < 0.35 else min(1.0, (frac - 0.20) / 0.50)
        return float(up * vis)
    if not miss:
        return 0.0
    outdoor = see is True or (light is not None and light >= 8.0) or frac >= 0.45
    if not outdoor:
        return 0.0
    return float(up)


def ground_look_score(signal: dict[str, Any] | None, *, pitch_down: float = 40.0) -> float:
    """1 if the player is staring at their feet, else 0.

    A slight downward glance (looking at the block ahead) is free. Pitch in
    Minecraft is positive when looking down.
    """
    if not signal:
        return 0.0
    if _flag(signal, "gui_open") or _flag(signal, "in_water"):
        return 0.0
    pitch = float(signal.get("pitch") or 0.0)
    floor = float(pitch_down)
    if pitch < floor:
        return 0.0
    return float(min(1.0, (pitch - floor) / 40.0))


def sky_stare_penalty(score: float, hold_s: float, *, grace_s: float, weight: float) -> float:
    if score < 0.35 or hold_s <= grace_s or weight <= 0:
        return 0.0
    ramp = min(1.0, (float(hold_s) - float(grace_s)) / 1.5)
    return float(weight) * float(score) * ramp


def yaw_sector(yaw: float, n_bins: int) -> int:
    n = max(2, int(n_bins))
    y = float(yaw) % 360.0
    if y < 0:
        y += 360.0
    return int(y * n / 360.0) % n


def is_inventory(signal: dict[str, Any] | None) -> bool:
    if _flag(signal, "inv_open") is True:
        return True
    gui = str((signal or {}).get("gui") or "").lower()
    return "inventory" in gui


def is_paused(signal: dict[str, Any] | None) -> bool:
    if _flag(signal, "paused") is True:
        return True
    gui = str((signal or {}).get("gui") or "").lower()
    return "pause" in gui


def inventory_open_penalty(was_open: bool, now_open: bool, *, weight: float) -> float:
    """Slight cost only on the open edge. Closing, or leaving it open, is free."""
    if weight <= 0 or now_open is not True or was_open:
        return 0.0
    return float(weight)


def pause_menu_penalty(was_paused: bool, now_paused: bool, *, weight: float) -> float:
    """Pause is not allowed. Rising edge is full weight; staying paused keeps a fraction."""
    if weight <= 0 or not now_paused:
        return 0.0
    return float(weight) if not was_paused else float(weight) * 0.4


def view_block_reward(prev_count: float, now_count: float, *, weight: float, cap: float = 20.0) -> float:
    """Pay for seeing more unique blocks than last perception tick (not for staring)."""
    if weight <= 0:
        return 0.0
    delta = max(0.0, float(now_count) - float(prev_count))
    if delta <= 0:
        return 0.0
    return float(weight) * min(1.0, delta / max(1.0, float(cap)))


def yaw_wrap_delta(prev: float, now: float) -> float:
    d = (float(now) - float(prev) + 180.0) % 360.0 - 180.0
    return d


def circle_spin_penalty(
    trail: list[tuple[float, float, float, float]],
    *,
    window_s: float,
    weight: float,
) -> float:
    """Penalize spinning in place or walking a tight loop (same area, lots of yaw)."""
    if weight <= 0 or len(trail) < 6:
        return 0.0
    t1 = trail[-1][0]
    pts = [p for p in trail if t1 - p[0] <= float(window_s) + 1e-6]
    if len(pts) < 6:
        return 0.0
    if t1 - pts[0][0] < float(window_s) * 0.65:
        return 0.0
    path = 0.0
    yaw_acc = 0.0
    for a, b in zip(pts, pts[1:]):
        path += math.hypot(b[1] - a[1], b[2] - a[2])
        yaw_acc += yaw_wrap_delta(a[3], b[3])
    net = math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
    spun = abs(yaw_acc)
    in_place = path < 0.8 and spun >= 150.0
    looping = path >= 0.8 and spun >= 270.0 and (net / max(path, 1e-3)) < 0.30
    if not in_place and not looping:
        return 0.0
    return float(weight)


def idle_hold_update(
    hold_s: float,
    anchor_x: float | None,
    anchor_z: float | None,
    x: float,
    z: float,
    dt: float,
    *,
    radius: float,
    freeze: bool,
    reset: bool,
) -> tuple[float, float, float]:
    """Accumulate time spent inside `radius` of the last move. Freeze on GUI/pause."""
    if anchor_x is None or anchor_z is None or reset:
        return 0.0, float(x), float(z)
    if freeze:
        return float(hold_s), float(anchor_x), float(anchor_z)
    if math.hypot(float(x) - float(anchor_x), float(z) - float(anchor_z)) >= float(radius):
        return 0.0, float(x), float(z)
    return float(hold_s) + max(0.0, float(dt)), float(anchor_x), float(anchor_z)


def idle_stand_penalty(hold_s: float, *, grace_s: float, weight: float) -> float:
    """Tax standing in a ~2-block patch after `grace_s` (default 30s)."""
    if weight <= 0 or float(grace_s) <= 0 or hold_s <= grace_s:
        return 0.0
    ramp = min(1.0, (float(hold_s) - float(grace_s)) / 3.0)
    return float(weight) * ramp


def wasted_jump_penalty(
    was_ground: bool,
    now_ground: bool,
    dy: float,
    *,
    goal: float,
    weight: float,
) -> float:
    """Tax a jump that is not climbing or falling off a block."""
    if weight <= 0 or not was_ground or now_ground:
        return 0.0
    if float(goal) > 1e-6:
        return 0.0
    if dy > 0.45 or dy < -0.25:
        return 0.0
    return float(weight)


def _num(signal: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not signal or key not in signal:
        return default
    try:
        return float(signal.get(key) if signal.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


@dataclass
class _Snap:
    pooled: Tensor
    state: Tensor
    fused_base: Tensor
    policy_input: Tensor | None
    human: Tensor | None
    act: Tensor | None
    executed: bool
    hp: float
    adv_done: float
    adv_frac: float
    sky: float
    yaw: float
    pitch: float
    gui: bool
    inv: bool
    paused: bool
    dead: bool
    blocks: float
    x: float
    y: float
    z: float
    on_ground: bool
    t: float
    iter_h: Tensor | None = None
    iter_grid: Tensor | None = None
    iter_prev_cells: Tensor | None = None
    task_id: Tensor | None = None
    task_aux: Tensor | None = None
    inv_ids: Tensor | None = None
    hand_id: Tensor | None = None
    hover_id: Tensor | None = None
    carry_id: Tensor | None = None
    inv_mem: Tensor | None = None
    curl_grid: Tensor | None = None
    curl_mem: Tensor | None = None
    group_mem: Tensor | None = None
    group_mem_q: Tensor | None = None
    group_mem_k: Tensor | None = None
    sea_map: Tensor | None = None
    sea_mem: Tensor | None = None
    sea_mem_q: Tensor | None = None
    sea_mem_k: Tensor | None = None
    sea_prev_spike: Tensor | None = None
    detect_mem: Tensor | None = None
    block_pulse: Tensor | None = None
    event_tokens: Tensor | None = None
    event_slots: Tensor | None = None
    event_ages: Tensor | None = None
    event_mem_q: Tensor | None = None
    event_mem_k: Tensor | None = None
    gui_live: Tensor | None = None


def _snap_cpu(snap: "_Snap") -> "_Snap":
    """Detached CPU copy of the learn snapshot. The frame itself is not included."""
    out = {}
    for f in fields(snap):
        v = getattr(snap, f.name)
        if isinstance(v, torch.Tensor):
            v = v.detach().to("cpu")
        out[f.name] = v
    return _Snap(**out)


class _CpuPlastic:
    """Plastic modules on CPU so the backward does not occupy the predict GPU."""

    _names = (
        "planner",
        "grow",
        "iterator",
        "inventory",
        "scene",
        "groups",
        "sea",
        "detect",
        "event_seq",
        "thought",
        "adapter",
        "policy",
    )

    def __init__(self, agent: AtomSpikeAgent, lr: float):
        self.mods: dict[str, nn.Module] = {}
        for name in self._names:
            mod = getattr(agent, name, None)
            if mod is None:
                continue
            self.mods[name] = copy.deepcopy(mod).to("cpu").eval()
        params = [p for m in self.mods.values() for p in m.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=float(lr)) if params else None
        self.teacher_policy = copy.deepcopy(agent.policy).to("cpu").eval()
        for p in self.teacher_policy.parameters():
            p.requires_grad = False

    def __getattr__(self, name: str):
        mods = self.__dict__.get("mods")
        if mods is not None and name in mods:
            return mods[name]
        raise AttributeError(name)

    def push_params(self, agent: AtomSpikeAgent) -> None:
        with torch.no_grad():
            for name, src in self.mods.items():
                dst = getattr(agent, name, None)
                if dst is None:
                    continue
                for ps, pd in zip(src.parameters(), dst.parameters()):
                    if ps.shape != pd.shape:
                        continue
                    pd.copy_(ps.detach().to(device=pd.device, dtype=pd.dtype))

    def pull(self, agent: AtomSpikeAgent) -> None:
        """Copy GPU parameters and buffers onto the CPU replica (after a birth or load)."""
        with torch.no_grad():
            for name, dst in self.mods.items():
                src = getattr(agent, name, None)
                if src is None:
                    continue
                for ps, pd in zip(src.parameters(), dst.parameters()):
                    if ps.shape == pd.shape:
                        pd.copy_(ps.detach().to(device="cpu", dtype=pd.dtype))
                for bs, bd in zip(src.buffers(), dst.buffers()):
                    if bs.shape == bd.shape and bs.dtype == bd.dtype:
                        bd.copy_(bs.detach().cpu())


class OnlineLearner:
    def __init__(
        self,
        agent: AtomSpikeAgent,
        cfg: AtomSpikeConfig,
        device: torch.device,
        save_path: str | Path | None = None,
        *,
        inject: bool = False,
        background: bool = True,
    ):
        self.agent = agent
        self.cfg = cfg
        self.learn = cfg.learn
        self.device = device
        self.inject = bool(inject)
        self.mode = "self_play" if self.inject else "human_demo"
        self.background = bool(background)
        self.save_path = Path(save_path) if save_path is not None else None
        self.delay = 1
        self._buf: deque[_Snap] = deque(maxlen=max(8, int(self.learn.buffer)))
        self._last_save = time.perf_counter()
        self._last_birth = -1e9
        self._sky_hold = 0.0
        self._ground_hold = 0.0
        self._trail: deque[tuple[float, float, float, float]] = deque(maxlen=80)
        self._yaw_seen: dict[int, float] = {}
        self._progress_t: float | None = None
        self._pending_kill = False
        self._stall_lock = False
        self._stall_fired_t = -1e9
        self._ignore_hp_until = 0.0
        self._was_dead = False
        self._idle_hold = 0.0
        self._idle_x: float | None = None
        self._idle_z: float | None = None
        self.n_steps = 0
        self.n_event_births = 0
        self.n_layer_births = 0
        self.n_spec_births = 0
        self.last_loss = 0.0
        self.last_surprise = 0.0
        self.last_reward = 0.0
        self.last_ms = 0.0
        self.last_replay_error = 0.0
        self._skip = False
        self.n_dropped = 0
        self._birth_pending: bool | None = None
        self._cpu: _CpuPlastic | None = None
        self._cpu_stale = False
        self._cpu_dirty = False
        self._cpu_lock = threading.Lock()
        self._work_q: queue.Queue = queue.Queue(maxsize=1)
        self._worker: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._freeze_backbone()
        self._teacher_policy = copy.deepcopy(agent.policy).to(device).eval()
        for p in self._teacher_policy.parameters():
            p.requires_grad = False
        params = list(self._plastic_params())
        self.opt = torch.optim.Adam(params, lr=float(self.learn.lr)) if params else None

    def _freeze_backbone(self) -> None:
        for block in (self.agent.encoder, self.agent.reasoner):
            for p in block.parameters():
                p.requires_grad = False

    def _plastic_params(self):
        seen: set[int] = set()
        modules: list[nn.Module] = []
        if self.agent.planner is not None:
            modules.append(self.agent.planner)
        if self.agent.grow is not None:
            modules.append(self.agent.grow)
        if getattr(self.agent, "iterator", None) is not None:
            modules.append(self.agent.iterator)
        if getattr(self.agent, "inventory", None) is not None:
            modules.append(self.agent.inventory)
        if getattr(self.agent, "scene", None) is not None:
            modules.append(self.agent.scene)
        if getattr(self.agent, "groups", None) is not None:
            modules.append(self.agent.groups)
        if getattr(self.agent, "sea", None) is not None:
            modules.append(self.agent.sea)
        if getattr(self.agent, "detect", None) is not None:
            modules.append(self.agent.detect)
        if getattr(self.agent, "event_seq", None) is not None:
            modules.append(self.agent.event_seq)
        modules.append(self.agent.adapter)
        modules.append(self.agent.policy)
        for mod in modules:
            for p in mod.parameters():
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                yield p

    def _ensure_worker(self) -> None:
        if not self.background:
            return
        w = self._worker
        if w is not None and w.is_alive():
            return
        self._worker_stop.clear()
        self._worker = threading.Thread(
            target=self._worker_main, name="atomspike-learn", daemon=True
        )
        self._worker.start()

    def _worker_main(self) -> None:
        try:
            # Keep the CPU backward off the cores the GPU driver needs.
            torch.set_num_threads(1)
        except Exception:
            pass
        while True:
            try:
                job = self._work_q.get(timeout=0.2)
            except queue.Empty:
                if self._worker_stop.is_set():
                    return
                continue
            try:
                if job is None:
                    return
                kind = job[0]
                if kind == "step":
                    _, past, now, reward, hurt = job
                    t0 = time.perf_counter()
                    self._step(past, now, reward=reward, hurt=hurt)
                    self.last_ms = (time.perf_counter() - t0) * 1000.0
                elif kind == "rollout":
                    _, experiences, _priority = job
                    t0 = time.perf_counter()
                    for past, now, reward, hurt in experiences:
                        self._step(past, now, reward=reward, hurt=hurt)
                    self.last_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                print(f"[learn] background job failed: {type(exc).__name__}: {exc}", flush=True)
            finally:
                self._work_q.task_done()

    def _enqueue(self, job, *, priority: bool = False) -> bool:
        """Enqueue without blocking, preserving sparse reward jobs over neutral ones."""
        if not self.background:
            return False
        self._ensure_worker()
        try:
            self._work_q.put_nowait(job)
            return True
        except queue.Full:
            if not priority:
                self.n_dropped += 1
                return False
            # A slow CPU learner may leave one neutral TD update pending.  A
            # rare advancement/damage event must not be discarded behind it.
            found_pending = False
            try:
                pending = self._work_q.get_nowait()
                found_pending = True
            except queue.Empty:
                pending = None
            if found_pending:
                self._work_q.task_done()
                if pending is None:
                    # Shutdown sentinel wins over newly arriving work.
                    self._work_q.put_nowait(None)
                    self.n_dropped += 1
                    return False
                pending_priority = bool(
                    isinstance(pending, tuple)
                    and len(pending) >= 3
                    and pending[0] == "rollout"
                    and pending[2]
                )
                if pending_priority:
                    try:
                        self._work_q.put_nowait(pending)
                    except queue.Full:
                        pass
                    self.n_dropped += 1
                    return False
                self.n_dropped += 1
            try:
                self._work_q.put_nowait(job)
                return True
            except queue.Full:
                self.n_dropped += 1
                return False

    def wait_idle(self) -> None:
        """Block until every queued background job finished (tests / shutdown)."""
        if self.background:
            self._work_q.join()

    def _publish_cpu(self, *, force: bool = False) -> None:
        """Publish a completed CPU update at a predict-thread safe point.

        The worker never writes live GPU parameters.  ``observe`` runs on the
        inference thread, so the copy is ordered after the current forward and
        before the next one (including on CUDA's current inference stream).
        """
        if self._cpu is None or (not force and not self._cpu_dirty):
            return
        with self._cpu_lock:
            if self._cpu is None or (not force and not self._cpu_dirty):
                return
            self._cpu.push_params(self.agent)
            self._cpu_dirty = False

    def load_plastic(self, path: str | Path | None = None) -> bool:
        p = Path(path) if path is not None else self.save_path
        if p is None or not p.is_file():
            return False
        from atomspike.train.common import load_checkpoint

        loaded, blob = load_checkpoint(p, self.cfg, self.device, restore_rng=False)
        self.agent.load_state_dict(loaded.state_dict(), strict=False)
        extra = blob.get("extra") or {}
        self.n_event_births = int(extra.get("n_event_births") or 0)
        self.n_layer_births = int(extra.get("n_layer_births") or 0)
        self.n_spec_births = int(extra.get("n_spec_births") or 0)
        self.n_steps = int(extra.get("n_steps") or 0)
        # A resumed adapter should be anchored to its explicitly selected
        # starting point, not to the checkpoint that happened to create it.
        self._teacher_policy = copy.deepcopy(self.agent.policy).to(self.device).eval()
        for param in self._teacher_policy.parameters():
            param.requires_grad = False
        target_opt = self.opt
        if self.background:
            self._cpu = _CpuPlastic(self.agent, float(self.learn.lr))
            target_opt = self._cpu.opt
        else:
            self._cpu = None
        if target_opt is not None and "optimizer" in blob:
            try:
                target_opt.load_state_dict(blob["optimizer"])
            except (KeyError, ValueError, RuntimeError):
                print("[learn] plastic optimizer state incompatible; using fresh Adam state", flush=True)
        self._cpu_stale = False
        self._cpu_dirty = False
        return True

    def save_plastic(self, path: str | Path | None = None) -> Path | None:
        p = Path(path) if path is not None else self.save_path
        if p is None:
            return None
        from atomspike.train.common import save_checkpoint

        checkpoint_opt = self._cpu.opt if self.background and self._cpu is not None else self.opt
        return save_checkpoint(
            self.agent,
            p,
            extra={
                "n_event_births": self.n_event_births,
                "n_layer_births": self.n_layer_births,
                "n_spec_births": self.n_spec_births,
                "n_steps": self.n_steps,
                "kind": "self_play_actor_critic" if self.inject else "human_demo_adaptation",
            },
            optimizer=checkpoint_opt,
            # This is an explicit sidecar snapshot, not a rolling training
            # generation.  It gets the same atomic write + SHA verification as
            # a normal model checkpoint.
            keep_k=None,
            verify=True,
        )

    def observe(
        self,
        *,
        pooled: Tensor | None,
        fused_base: Tensor | None,
        policy_input: Tensor | None = None,
        state: Tensor,
        human_tokens: Tensor | None,
        injected: bool,
        act: Tensor | None = None,
        signal: dict[str, Any] | None = None,
    ) -> dict:
        if pooled is None or fused_base is None or self.opt is None:
            return self.status()
        # Apply only a fully completed background update, and only from the
        # inference thread.  This avoids cross-thread CUDA parameter writes.
        self._publish_cpu()
        if self._skip:
            self._skip = False
            return self.status()
        if self._birth_pending is not None:
            trigger_pend = bool(self._birth_pending)
            self._birth_pending = None
            self._maybe_birth(trigger=trigger_pend)
        hp, adv, frac = _signal_progress(signal)
        sky = open_sky_score(signal, pitch_up=float(self.learn.sky_pitch))
        yaw = float((signal or {}).get("yaw") or 0.0)
        pitch = float((signal or {}).get("pitch") or 0.0)
        gui = bool(_flag(signal, "gui_open"))
        inv = is_inventory(signal)
        paused = is_paused(signal)
        grounded = _flag(signal, "on_ground")
        dead_now = bool(_flag(signal, "dead"))
        if signal is not None and "hp" in signal and hp <= 0.0:
            dead_now = True
        if dead_now:
            reset_it = getattr(self.agent, "reset_iterator", None)
            if callable(reset_it):
                reset_it()
            reset_inv = getattr(self.agent, "reset_inventory", None)
            if callable(reset_inv):
                reset_inv()
            reset_sc = getattr(self.agent, "reset_scene", None)
            if callable(reset_sc):
                reset_sc()
            reset_gr = getattr(self.agent, "reset_groups", None)
            if callable(reset_gr):
                reset_gr()
            reset_sea = getattr(self.agent, "reset_sea", None)
            if callable(reset_sea):
                reset_sea()
            reset_det = getattr(self.agent, "reset_detect", None)
            if callable(reset_det):
                reset_det()
            reset_ev = getattr(self.agent, "reset_event_seq", None)
            if callable(reset_ev):
                reset_ev()
        b = int(pooled.size(0))

        def _iter_tensor(name: str) -> Tensor | None:
            t = getattr(self.agent, name, None)
            if t is None or int(t.size(0)) != b:
                return None
            return t.detach().float()

        self._buf.append(
            _Snap(
                pooled=pooled.detach().float(),
                state=state.detach().float(),
                fused_base=fused_base.detach().float(),
                policy_input=None if policy_input is None else policy_input.detach().float(),
                human=None if self.inject or human_tokens is None else human_tokens.detach(),
                act=None if act is None else act.detach(),
                executed=bool(injected),
                hp=hp,
                adv_done=adv,
                adv_frac=frac,
                sky=sky,
                yaw=yaw,
                pitch=pitch,
                gui=gui,
                inv=inv,
                paused=paused,
                dead=dead_now,
                blocks=_num(signal, "blocks_seen"),
                x=_num(signal, "x"),
                y=_num(signal, "y", 64.0),
                z=_num(signal, "z"),
                on_ground=True if grounded is None else bool(grounded),
                t=time.perf_counter(),
                iter_h=_iter_tensor("_last_iter_h_in"),
                iter_grid=_iter_tensor("_last_iter_grid"),
                iter_prev_cells=_iter_tensor("_last_iter_prev_cells"),
                task_id=_iter_tensor("_last_task_id"),
                task_aux=_iter_tensor("_last_task_aux"),
                inv_ids=_iter_tensor("_last_inv_ids"),
                hand_id=_iter_tensor("_last_hand_id"),
                hover_id=_iter_tensor("_last_hover_id"),
                carry_id=_iter_tensor("_last_carry_id"),
                inv_mem=_iter_tensor("_last_inv_mem_in"),
                curl_grid=_iter_tensor("_last_curl_grid"),
                curl_mem=_iter_tensor("_last_curl_mem_in"),
                group_mem=_iter_tensor("_last_group_mem_in"),
                group_mem_q=_iter_tensor("_last_group_mem_q_in"),
                group_mem_k=_iter_tensor("_last_group_mem_k_in"),
                sea_map=_iter_tensor("_last_sea_map"),
                sea_mem=_iter_tensor("_last_sea_mem_in"),
                sea_mem_q=_iter_tensor("_last_sea_mem_q_in"),
                sea_mem_k=_iter_tensor("_last_sea_mem_k_in"),
                sea_prev_spike=_iter_tensor("_last_sea_prev_spike"),
                detect_mem=_iter_tensor("_last_detect_mem_in"),
                block_pulse=_iter_tensor("_last_block_pulse"),
                event_tokens=_iter_tensor("_last_event_tokens"),
                event_slots=_iter_tensor("_last_event_slots"),
                event_ages=_iter_tensor("_last_event_ages"),
                event_mem_q=_iter_tensor("_last_event_mem_q"),
                event_mem_k=_iter_tensor("_last_event_mem_k"),
                gui_live=_iter_tensor("_last_gui_live"),
            )
        )
        if len(self._buf) <= self.delay:
            return self.status()
        past = self._buf[-1 - self.delay]
        now = self._buf[-1]
        # Human mode needs a real input target.  Self-play needs a sampled
        # action which actually passed the singleplayer/focus gate and was sent
        # to the game; paused-gate predictions are never trained as experience.
        if self.inject:
            valid_transition = past.executed and past.act is not None
        else:
            valid_transition = past.human is not None
        if not valid_transition:
            self.last_reward = 0.0
            return self.status()
        dt = max(1e-3, now.t - past.t)
        if not now.gui and not now.paused:
            self._trail.append((now.t, now.x, now.z, now.yaw))
            win = float(self.learn.circle_s)
            while self._trail and now.t - self._trail[0][0] > win + 0.6:
                self._trail.popleft()
        if now.sky >= 0.4:
            self._sky_hold += dt
        else:
            self._sky_hold = max(0.0, self._sky_hold - 2.0 * dt)
        reward, goal, hurt = goal_hp_reward(
            hp0=past.hp,
            hp1=now.hp,
            adv0=past.adv_done,
            adv1=now.adv_done,
            frac0=past.adv_frac,
            frac1=now.adv_frac,
            goal_w=float(self.learn.goal_reward),
            hp_w=float(self.learn.hp_penalty),
        )
        sky_pen = sky_stare_penalty(
            now.sky,
            self._sky_hold,
            grace_s=float(self.learn.sky_grace_s),
            weight=float(self.learn.sky_penalty),
        )
        stall_pen = self._stall_tick(signal, now, goal)
        inv_pen = inventory_open_penalty(past.inv, now.inv, weight=float(self.learn.inv_penalty))
        pause_pen = pause_menu_penalty(past.paused, now.paused, weight=float(self.learn.pause_penalty))
        view = 0.0
        if not now.gui and not now.paused:
            view = view_block_reward(
                past.blocks,
                now.blocks,
                weight=float(self.learn.view_bonus),
            )
        dead = bool(_flag(signal, "dead")) or now.hp <= 0.0
        self._idle_hold, self._idle_x, self._idle_z = idle_hold_update(
            self._idle_hold,
            self._idle_x,
            self._idle_z,
            now.x,
            now.z,
            dt,
            radius=float(self.learn.idle_radius),
            freeze=bool(now.gui or now.paused),
            reset=bool(dead or goal > 1e-6),
        )
        idle_pen = idle_stand_penalty(
            self._idle_hold,
            grace_s=float(self.learn.idle_s),
            weight=float(self.learn.idle_penalty),
        )
        circle_pen = 0.0
        if not now.gui and not now.paused:
            circle_pen = circle_spin_penalty(
                list(self._trail),
                window_s=float(self.learn.circle_s),
                weight=float(self.learn.circle_penalty),
            )
        look = 0.0 if circle_pen > 0 else self._look_bonus(past, now)
        jump_pen = wasted_jump_penalty(
            past.on_ground,
            now.on_ground,
            now.y - past.y,
            goal=goal,
            weight=float(self.learn.jump_penalty),
        )
        ground = ground_look_score(signal, pitch_down=float(self.learn.ground_pitch))
        if ground < 0.35:
            self._ground_hold = max(0.0, self._ground_hold - 2.0 * dt)
        else:
            self._ground_hold += dt
        ground_pen = sky_stare_penalty(
            ground,
            self._ground_hold,
            grace_s=float(self.learn.ground_grace_s),
            weight=float(self.learn.ground_penalty),
        )
        if now.t < self._ignore_hp_until and hurt > 0:
            reward += hurt
            hurt = 0.0
        reward = (
            reward
            - sky_pen
            + look
            - stall_pen
            - inv_pen
            - pause_pen
            - ground_pen
            - jump_pen
            - circle_pen
            - idle_pen
            + view
        )
        self.last_reward = reward
        experiences = [(past, now, reward, hurt > 0)]
        # Achievements and damage are delayed task signals.  Give them a short
        # discounted backward credit window, while keeping momentary shaping
        # (look/jump/menu/etc.) on the immediately preceding action only.
        long_credit = float(goal - hurt)
        if self.inject and abs(long_credit) >= 1e-6:
            horizon = min(int(self.learn.credit_steps), len(self._buf) - 1)
            gamma = float(self.learn.credit_gamma)
            for lag in range(2, horizon + 1):
                older = self._buf[-1 - lag]
                successor = self._buf[-lag]
                if not older.executed or older.act is None:
                    continue
                experiences.append(
                    (older, successor, long_credit * (gamma ** (lag - 1)), hurt > 0)
                )
        if self.background:
            # Copy the tiny snapshot here, after the predict sync, so the worker
            # can train on CPU without taking the CUDA device lock mid-tick.
            cpu_experiences = [
                (_snap_cpu(old), _snap_cpu(new), shaped, was_hurt)
                for old, new, shaped, was_hurt in experiences
            ]
            priority = abs(reward) >= 1e-6
            enqueued = self._enqueue(("rollout", cpu_experiences, priority), priority=priority)
            dt_ms = self.last_ms
            info = {
                "surprise": abs(reward),
                "reward": reward,
                "queued": enqueued,
                "credit_steps": len(experiences),
            }
        else:
            t0 = time.perf_counter()
            info = {}
            for old, new, shaped, was_hurt in experiences:
                info = self._step(old, new, reward=shaped, hurt=was_hurt)
            info["credit_steps"] = len(experiences)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.last_ms = dt_ms
            budget = float(self.learn.budget_ms)
            if budget > 0 and dt_ms > budget * 1.8:
                self._skip = True
        if self.save_path is not None and (time.perf_counter() - self._last_save) >= float(self.learn.save_every_s):
            # Saving reads the live agent, so publish and serialize here on the
            # same thread that owns inference rather than in the CPU worker.
            if self.background:
                self.wait_idle()
            self._publish_cpu()
            self.save_plastic()
            self._last_save = time.perf_counter()
        if abs(reward) >= 0.05:
            extra = ""
            if stall_pen > 0:
                extra = "  stall=1  kill"
            print(
                f"[learn] r={reward:+.3f}  goal={goal:.3f}  hp={hurt:.3f}  "
                f"sky={sky_pen:.3f}  ground={ground_pen:.3f}  look={look:.3f}  inv={inv_pen:.3f}  pause={pause_pen:.3f}  "
                f"jump={jump_pen:.3f}  circle={circle_pen:.3f}  idle={idle_pen:.3f}  "
                f"view={view:.3f}{extra}  next={_signal_next(signal)}",
                flush=True,
            )
        info.update(self.status())
        info["learn_ms"] = dt_ms
        info["reward"] = reward
        info["kill"] = bool(self._pending_kill)
        return info

    def _look_bonus(self, past: _Snap, now: _Snap) -> float:
        """Sparse bonus for turning into a new yaw sector while looking at the world."""
        w = float(self.learn.look_bonus)
        if w <= 0 or now.gui or now.sky >= 0.35:
            return 0.0
        if abs(now.pitch) > 25.0:
            return 0.0
        n = int(self.learn.look_bins)
        if yaw_sector(past.yaw, n) == yaw_sector(now.yaw, n):
            return 0.0
        mem = float(self.learn.look_memory_s)
        sector = yaw_sector(now.yaw, n)
        last = self._yaw_seen.get(sector, -1e9)
        if (now.t - last) < mem:
            return 0.0
        self._yaw_seen[sector] = now.t
        return w

    def take_kill(self) -> bool:
        if not self._pending_kill:
            return False
        self._pending_kill = False
        return True

    def _stall_tick(self, signal: dict[str, Any] | None, now: _Snap, goal: float) -> float:
        """Big penalty + one kill/respawn if no advancement progress for stall_s."""
        if not self.inject or float(self.learn.stall_s) <= 0:
            return 0.0
        if not signal or ("adv_done" not in signal and "adv_next" not in signal):
            return 0.0
        nxt = str(signal.get("adv_next") or "")
        if not nxt:
            return 0.0
        dead = bool(_flag(signal, "dead")) or now.hp <= 0.0
        if goal > 1e-6:
            self._progress_t = now.t
            self._was_dead = dead
            return 0.0
        if dead:
            self._was_dead = True
            return 0.0
        if self._was_dead:
            self._was_dead = False
            self._stall_lock = False
            self._progress_t = now.t
            return 0.0
        if self._stall_lock:
            if (now.t - self._stall_fired_t) > 15.0:
                self._stall_lock = False
                self._progress_t = now.t
            return 0.0
        if self._progress_t is None:
            self._progress_t = now.t
            return 0.0
        if (now.t - self._progress_t) < float(self.learn.stall_s):
            return 0.0
        self._stall_lock = True
        self._stall_fired_t = now.t
        self._pending_kill = True
        self._ignore_hp_until = now.t + 8.0
        self._sky_hold = 0.0
        self._ground_hold = 0.0
        self._idle_hold = 0.0
        self._idle_x = now.x
        self._idle_z = now.z
        self._trail.clear()
        print(
            f"[learn] stall {float(self.learn.stall_s):.0f}s without advancement "
            f"next={nxt} -> penalty {float(self.learn.stall_penalty):.2f} and kill",
            flush=True,
        )
        return float(self.learn.stall_penalty)

    def _step(self, past: _Snap, now: _Snap, *, reward: float, hurt: bool) -> dict:
        if self.background:
            with self._cpu_lock:
                return self._step_impl(past, now, reward=reward, hurt=hurt)
        return self._step_impl(past, now, reward=reward, hurt=hurt)

    def _step_impl(self, past: _Snap, now: _Snap, *, reward: float, hurt: bool) -> dict:
        target = past.act if self.inject else past.human
        if target is None or (self.inject and not past.executed):
            self.last_loss = 0.0
            self.last_surprise = 0.0
            why = "action_not_executed" if self.inject else "no_human_demo"
            return {"surprise": 0.0, "born": None, "reward": 0.0, "skipped": why}
        self.n_steps += 1
        cpu = None
        if self.background:
            if self._cpu is None:
                self._cpu = _CpuPlastic(self.agent, float(self.learn.lr))
            cpu = self._cpu
            if self._cpu_stale:
                cpu.pull(self.agent)
                self._cpu_stale = False
            if past.fused_base.device.type != "cpu":
                past = _snap_cpu(past)
        host = cpu if cpu is not None else self.agent
        opt = cpu.opt if cpu is not None else self.opt
        planner = getattr(host, "planner", None)
        grow = getattr(host, "grow", None)
        iterator = getattr(host, "iterator", None)
        inventory = getattr(host, "inventory", None)
        scene = getattr(host, "scene", None)
        groups = getattr(host, "groups", None)
        sea = getattr(host, "sea", None)
        detect = getattr(host, "detect", None)
        event_seq = getattr(host, "event_seq", None)
        adapter = getattr(host, "adapter", None)
        loss = past.fused_base.new_zeros(())
        with torch.enable_grad():
            fused = past.fused_base
            if past.gui_live is not None and adapter is not None:
                fused = fused + adapter.gui_scale * past.gui_live
            if planner is not None:
                plan = planner(past.pooled, past.state, sample=False, explore=0.0)
                fused = fused + plan["bias"]
            if iterator is not None and past.iter_grid is not None and past.task_id is not None and past.task_aux is not None:
                it = iterator(
                    past.iter_grid,
                    past.pooled,
                    past.state,
                    past.task_id,
                    past.task_aux,
                    h=past.iter_h,
                    prev_cells=past.iter_prev_cells,
                    gui_mask=past.state.new_tensor([1.0 if (past.gui or past.paused) else 0.0]),
                )
                fused = fused + it["bias"]
                moved = (
                    math.hypot(now.x - past.x, now.z - past.z) > 0.25
                    or abs(yaw_wrap_delta(past.yaw, now.yaw)) > 12.0
                    or abs(now.adv_frac - past.adv_frac) > 1e-4
                )
                tgt = it["change_logit"].new_tensor([1.0 if moved else 0.0]).expand_as(it["change_logit"].reshape(-1))
                aux_w = float(self.cfg.iterator.change_aux)
                if aux_w > 0 and not past.gui and not past.paused:
                    loss = loss + aux_w * F.binary_cross_entropy_with_logits(
                        it["change_logit"].reshape(-1), tgt
                    )
            if scene is not None and past.curl_grid is not None:
                sc = scene(
                    past.curl_grid,
                    mem=past.curl_mem,
                    gui_mask=past.state.new_tensor([1.0 if (past.gui or past.paused) else 0.0]),
                )
                fused = fused + sc["bias"]
            if inventory is not None and past.inv_ids is not None:
                inv = inventory(
                    past.inv_ids,
                    past.hand_id,
                    past.hover_id,
                    past.carry_id,
                    past.state,
                    mem=past.inv_mem,
                )
                fused = fused + inv["bias"]
                if past.gui and "pointer_logits" in inv and "pointer_idx" in inv:
                    idx = inv["pointer_idx"]
                    pl = inv["pointer_logits"]
                    valid = (idx >= 0) & (idx < pl.size(-1))
                    if bool(valid.any().item()):
                        loss = loss + 0.25 * F.cross_entropy(
                            pl[valid], idx[valid].clamp(0, pl.size(-1) - 1)
                        )
            group_bias = fused.new_zeros(fused.shape)
            sea_bias = fused.new_zeros(fused.shape)
            detect_bias = fused.new_zeros(fused.shape)
            event_bias = fused.new_zeros(fused.shape)
            if groups is not None:
                go = groups.observe(
                    fused,
                    mem=past.group_mem,
                    mem_q=past.group_mem_q,
                    mem_k=past.group_mem_k,
                    track=False,
                )
                group_bias = go["bias"]
            if sea is not None and past.sea_map is not None:
                so = sea.observe(
                    past.sea_map,
                    fused,
                    hw=None,
                    mem=past.sea_mem,
                    mem_q=past.sea_mem_q,
                    mem_k=past.sea_mem_k,
                    prev_spike=past.sea_prev_spike,
                    gui_mask=None,
                    block_pulse=past.block_pulse,
                )
                sea_bias = so["bias"]
            if detect is not None and past.sea_map is not None:
                do = detect.observe(
                    past.sea_map,
                    hw=None,
                    mem=past.detect_mem,
                    gui_mask=past.state.new_tensor([1.0 if (past.gui or past.paused) else 0.0]),
                )
                detect_bias = do["bias"]
            if event_seq is not None:
                eo = event_seq.observe(
                    fused + sea_bias + detect_bias,
                    event_tokens=past.event_tokens,
                    slots=past.event_slots,
                    ages=past.event_ages,
                    mem_q=past.event_mem_q,
                    mem_k=past.event_mem_k,
                    pointer=self.agent._pointer_feat(past.state),
                )
                event_bias = eo["bias"]
            thought = getattr(host, "thought", None)
            thought_bias = fused.new_zeros(fused.shape)
            if thought is not None:
                thought_bias = thought.observe(fused + sea_bias + detect_bias + event_bias)["bias"]
            fused = fused + group_bias + sea_bias + detect_bias + event_bias + thought_bias
            fused = grow(fused) if grow is not None else fused
            if groups is not None:
                fused = groups.apply_specialists(fused)
            if past.policy_input is not None and past.policy_input.shape == fused.shape:
                # Planner sampling, live smoothing and cached temporal state
                # cannot be bit-for-bit replayed from the compact snapshot.
                # Anchor the forward value to the context that really produced
                # the executed action, while keeping the rebuilt graph as a
                # straight-through gradient estimator for plastic modules.
                self.last_replay_error = float(
                    (fused.detach().float() - past.policy_input.float())
                    .square()
                    .mean()
                    .sqrt()
                    .item()
                )
                fused = past.policy_input + (fused - fused.detach())
            else:
                self.last_replay_error = 0.0
            pol = getattr(host, "policy", None)
            if isinstance(pol, SpikePolicy):
                pol_out = pol(fused, target, persist=False)
            else:
                pol_out = pol(fused, target)
            logits = pol_out["logits"]
            if logits is not None:
                if self.inject:
                    reward_target = max(
                        -float(self.learn.reward_clip),
                        min(float(self.learn.reward_clip), float(reward)),
                    )
                    # The behavior policy samples logits / temperature.  Use
                    # that same distribution for the actor objective; using
                    # raw logits would optimize a different policy whenever a
                    # non-unit runtime temperature is configured.
                    behavior_temp = max(float(self.cfg.policy.temperature), 1e-5)
                    value = pol_out["value"].float().reshape(-1)
                    bootstrap = value.new_zeros(value.shape)
                    if not now.dead and not now.paused and now.policy_input is not None:
                        with torch.no_grad():
                            next_tokens = now.act if now.act is not None else target
                            if isinstance(pol, SpikePolicy):
                                next_out = pol(
                                    now.policy_input,
                                    next_tokens,
                                    persist=False,
                                )
                            else:
                                next_out = pol(now.policy_input, next_tokens)
                            bootstrap = next_out["value"].float().reshape_as(value)
                    clip = float(self.learn.reward_clip)
                    ret = (
                        value.new_full(value.shape, reward_target)
                        + float(self.learn.credit_gamma) * bootstrap.clamp(-clip, clip)
                    ).clamp(-clip, clip)
                    advantage = ret - value.detach()
                    chosen_logp = []
                    entropies = []
                    for slot, student_logit in enumerate(logits):
                        logp = F.log_softmax(student_logit.float() / behavior_temp, dim=-1)
                        prob = logp.exp()
                        action = target[:, slot].long().clamp(0, student_logit.size(-1) - 1)
                        chosen_logp.append(logp.gather(-1, action.unsqueeze(-1)).squeeze(-1))
                        entropies.append(-(prob * logp).sum(dim=-1))
                    mean_logp = torch.stack(chosen_logp, dim=0).mean(dim=0)
                    entropy = torch.stack(entropies, dim=0).mean()
                    actor_loss = -(advantage * mean_logp).mean()
                    critic_loss = F.mse_loss(value, ret)
                    loss = (
                        loss
                        + actor_loss
                        + float(self.learn.value_coef) * critic_loss
                        - float(self.learn.entropy_coef) * entropy
                    )
                    surprise = float(advantage.detach().abs().mean().item())
                else:
                    slot_w = (
                        default_slot_weights(self.agent.spec).to(device=fused.device, dtype=fused.dtype)
                        if past.gui
                        else None
                    )
                    # Human actions remain ordinary positive CE targets.  The
                    # reward is diagnostic in this mode and can never reverse
                    # the demonstration gradient.
                    loss = loss + action_ce_loss(logits, target, slot_weights=slot_w)
                    surprise = abs(float(reward))
                teacher_policy = cpu.teacher_policy if cpu is not None else self._teacher_policy
                with torch.no_grad():
                    teacher_input = past.policy_input
                    if teacher_input is None or teacher_input.shape != fused.shape:
                        teacher_input = fused.detach()
                    if isinstance(teacher_policy, SpikePolicy):
                        teacher_logits = teacher_policy(teacher_input, target, persist=False)["logits"]
                    else:
                        teacher_logits = teacher_policy(teacher_input, target)["logits"]
                kl_terms = []
                kl_temp = max(float(self.cfg.policy.temperature), 1e-5) if self.inject else 1.0
                for teacher_logit, student_logit in zip(teacher_logits, logits):
                    log_teacher = F.log_softmax(teacher_logit.float() / kl_temp, dim=-1)
                    teacher_prob = log_teacher.exp()
                    log_student = F.log_softmax(student_logit.float() / kl_temp, dim=-1)
                    kl_terms.append((teacher_prob * (log_teacher - log_student)).sum(dim=-1).mean())
                if kl_terms:
                    loss = loss + float(getattr(self.learn, "kl_coef", 0.1)) * torch.stack(kl_terms).mean()
            if float(loss.detach().item()) == 0.0:
                self.last_loss = 0.0
                self.last_surprise = surprise
                return {"surprise": self.last_surprise, "born": None, "reward": reward}
            assert opt is not None
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 1.0)
            opt.step()
        if cpu is not None:
            self._cpu_dirty = True
        self.last_loss = float(loss.detach().item())
        self.last_surprise = surprise
        trigger = bool(hurt or surprise >= float(self.learn.birth_surprise))
        born = self._request_birth(trigger=trigger)
        return {"surprise": self.last_surprise, "born": born, "reward": reward}

    def _request_birth(self, *, trigger: bool) -> str | None:
        """Module-structure changes stay on the predict thread; the worker flags them."""
        if not self.background:
            return self._maybe_birth(trigger=trigger)
        self._birth_pending = bool(self._birth_pending) or bool(trigger)
        return None

    def _maybe_birth(self, *, trigger: bool) -> str | None:
        now = time.perf_counter()
        if not trigger or self.n_steps < 5:
            return None
        if (now - self._last_birth) < float(self.learn.birth_cooldown_s):
            return None
        planner = self.agent.planner
        grow = self.agent.grow
        groups = getattr(self.agent, "groups", None)
        born = None
        if planner is not None and self.n_event_births < int(self.learn.max_new_events):
            idx = planner.birth_event()
            if idx is not None:
                self.n_event_births += 1
                born = f"event:{idx}"
        if born is None and groups is not None:
            energy = getattr(self.agent, "_last_group_energy", None)
            thresh = float(getattr(self.cfg.groups, "birth_energy", 0.25) or 0.25)
            hot = False
            if energy is not None:
                e = energy.detach().float().reshape(-1)
                if e.numel() == groups.n_groups:
                    dead = groups.alive < 0.5
                    if bool(dead.any().item()) and float(e[dead].max().item()) >= thresh:
                        hot = True
            if hot:
                idx = groups.birth_specialist(energy=energy)
                if idx is not None:
                    self.n_spec_births += 1
                    born = f"group:{idx}"
        if born is None and grow is not None and self.n_layer_births < int(self.learn.max_new_layers):
            idx = grow.birth()
            if idx is not None:
                self.n_layer_births += 1
                born = f"layer:{idx}"
        if born is not None:
            self._last_birth = now
            self._cpu_stale = True
            print(f"[learn] born {born} after high surprise/damage", flush=True)
        return born

    def close(self) -> None:
        worker = self._worker
        if worker is not None:
            # Finish queued CPU work before sending the sentinel.  A short
            # daemon-thread timeout could otherwise leave a backward running
            # while the final checkpoint reads the live model.
            self.wait_idle()
            self._worker_stop.set()
            self._work_q.put(None)
            worker.join()
            self._worker = None
        self._publish_cpu(force=True)
        if self.save_path is not None and self.n_steps > 0:
            self.save_plastic()

    def status(self) -> dict:
        n_layers = 0
        n_groups = 0
        if self.agent.grow is not None:
            n_layers = int(self.agent.grow.n_alive())
        groups = getattr(self.agent, "groups", None)
        if groups is not None:
            n_groups = int(groups.n_alive())
        return {
            "learn_mode": self.mode,
            "learn_steps": self.n_steps,
            "learn_loss": self.last_loss,
            "surprise": self.last_surprise,
            "reward": self.last_reward,
            "born_events": int(self.n_event_births),
            "born_layers": n_layers,
            "born_groups": n_groups,
            "learn_ms": self.last_ms,
            "learn_dropped": int(self.n_dropped),
            "replay_rms": float(self.last_replay_error),
        }


def _signal_progress(signal: dict[str, Any] | None) -> tuple[float, float, float]:
    if not signal:
        return 0.0, 0.0, 0.0
    hp = float(signal.get("hp", 0.0) or 0.0)
    adv = float(signal.get("adv_done", 0.0) or 0.0)
    frac = float(signal.get("adv_next_frac", 0.0) or 0.0)
    return hp, adv, frac


def _signal_next(signal: dict[str, Any] | None) -> str:
    if not signal:
        return "-"
    nxt = signal.get("adv_next") or ""
    return str(nxt) if nxt else "-"
