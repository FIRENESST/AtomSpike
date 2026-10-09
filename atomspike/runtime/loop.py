"""Dual-rate runtime driven by DualRateClock, not tick%N."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from atomspike.capture.mc_state import MC_STATE_FIELDS
from atomspike.config import AtomSpikeConfig
from atomspike.data.sessions import clock_from_runtime
from atomspike.models.actions import cursor_from_logits, look_deltas_from_logits, tokens_to_controls, wheel_from_logits
from atomspike.models.agent import AtomSpikeAgent
from atomspike.models.inventory import inventory_tensors_from_signal
from atomspike.models.iterator import task_tensors_from_signal
from atomspike.models.sea import BlockPulseTracker
from atomspike.runtime.scheduler import ClockMode, DualRateClock
from atomspike.train.common import autocast_ctx


@dataclass
class TickStats:
    latencies_ms: list[float] = field(default_factory=list)
    slept_ms: list[float] = field(default_factory=list)
    n_reason: int = 0
    n_policy: int = 0
    n_skipped: int = 0

    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        arr = np.asarray(self.latencies_ms)
        return float(np.percentile(arr, 95))


def _overlay_open(game_state: NDArray, signal: dict | None) -> bool:
    """Inventory / pause / title overlay — freeze world frame-diff."""
    if signal:
        gui = str(signal.get("gui") or "").lower()
        if signal.get("gui_open") or signal.get("paused") or signal.get("inv_open"):
            return True
        if "pause" in gui or "inventory" in gui or "title" in gui or "death" in gui:
            return True
    try:
        if "gui_open" in MC_STATE_FIELDS:
            idx = MC_STATE_FIELDS.index("gui_open")
            if idx < int(np.asarray(game_state).reshape(-1).size):
                return float(np.asarray(game_state).reshape(-1)[idx]) > 0.5
    except Exception:
        return False
    return False


class DualRateLoop:
    def __init__(
        self,
        agent: AtomSpikeAgent,
        cfg: AtomSpikeConfig,
        device: torch.device,
        mode: ClockMode | None = None,
        pace: bool | None = None,
    ):
        self.agent = agent
        self.cfg = cfg
        self.device = device
        self.clock = DualRateClock.from_cfg(cfg.dual_rate, mode=mode, pace=pace)
        self.stats = TickStats()
        self._prev: torch.Tensor | None = None
        self._last_tokens: torch.Tensor | None = None
        self._last_look: tuple[float, float] | None = None
        self._last_cursor: tuple[float, float] | None = None
        self._last_wheel: float | None = None
        self._session_t = 0.0
        self._last_tick_t: float | None = None
        self._block_pulse = BlockPulseTracker()
        self.learner = None
        self._runtime_warnings: set[str] = set()
        self._infer_stream = None
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # Non-default stream: the legacy default stream waits for every other
            # stream, so a learn backward would stall the next predict tick.
            try:
                self._infer_stream = torch.cuda.Stream(priority=-1)
            except Exception:
                self._infer_stream = torch.cuda.Stream()

    def reset(self) -> None:
        self.agent.reset_runtime()
        self.clock.reset()
        self._prev = None
        self._last_tokens = None
        self._last_look = None
        self._last_cursor = None
        self._last_wheel = None
        self._session_t = 0.0
        self._last_tick_t = None
        self._block_pulse.reset()

    def _warn_once(self, area: str, exc: Exception) -> None:
        if area in self._runtime_warnings:
            return
        self._runtime_warnings.add(area)
        print(f"[runtime] {area} unavailable: {type(exc).__name__}: {exc}", flush=True)

    @torch.no_grad()
    def tick(
        self,
        frame: NDArray[np.uint8],
        game_state: NDArray[np.float32],
        deterministic: bool = True,
        human_tokens: NDArray[np.int64] | None = None,
        injected: bool = False,
        signal: dict | None = None,
    ) -> tuple[NDArray[np.int64], dict]:
        if self._infer_stream is not None:
            torch.cuda.set_stream(self._infer_stream)
        decision = self.clock.start_tick()
        t0 = time.perf_counter()
        if self.clock.mode == "sim":
            dt_s = float(self.clock.policy_dt)
            self._session_t += dt_s
        else:
            now = time.perf_counter()
            dt_s = 0.0 if self._last_tick_t is None else max(0.0, now - self._last_tick_t)
            self._last_tick_t = now
            self._session_t += dt_s
        clock_np = clock_from_runtime(self._session_t, dt_s if dt_s > 0 else float(self.clock.policy_dt))
        clock_t = torch.from_numpy(clock_np).unsqueeze(0).to(self.device)
        overlay = _overlay_open(game_state, signal)
        if signal:
            self._block_pulse.hold(signal)
        chw = np.transpose(frame, (2, 0, 1))
        frames = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(self.device)
        state = torch.from_numpy(np.asarray(game_state, dtype=np.float32)).unsqueeze(0).to(self.device)
        skipped = False
        if not decision.run_policy and self._last_tokens is not None:
            tokens = self._last_tokens
            skipped = True
            self.stats.n_skipped += 1
        else:
            sample_plan = None if deterministic else True
            explore = None if deterministic else self.cfg.planner.explore
            plan_sample = bool(self.cfg.planner.creative) if sample_plan is None else bool(sample_plan)
            exp = 0.0 if not plan_sample else (self.cfg.planner.explore if explore is None else float(explore))
            task_id = task_aux = None
            inv_kw = {}
            pulse_kw = {}
            gui_now = bool(overlay)
            if signal:
                gui_now = gui_now or bool(signal.get("gui_open") or signal.get("inv_open"))
            if decision.run_perception or self.agent._context is None:
                try:
                    n_buckets = int(getattr(self.cfg.iterator, "n_task_buckets", 16) or 16)
                    task_id, task_aux = task_tensors_from_signal(
                        signal, device=self.device, batch=1, n_buckets=n_buckets
                    )
                except Exception as exc:
                    self._warn_once("task signal", exc)
                    task_id = task_aux = None
            if decision.run_perception or gui_now or self.agent._context is None:
                try:
                    n_item_buckets = int(getattr(self.cfg.inventory, "n_buckets", 48) or 48)
                    inv_kw = inventory_tensors_from_signal(
                        signal, device=self.device, batch=1, n_buckets=n_item_buckets
                    )
                except Exception as exc:
                    self._warn_once("inventory signal", exc)
                    inv_kw = {}
            if decision.run_perception or self.agent._context is None:
                pulse = self._block_pulse.take()
                if pulse.any():
                    pulse_kw = {
                        "block_pulse": torch.as_tensor(pulse, device=self.device, dtype=torch.float32).view(1, 2)
                    }
            pol = getattr(self.agent, "policy", None)
            if pol is not None and hasattr(pol, "trace_enabled"):
                pol.trace_enabled = bool(getattr(self.agent, "_activity_trace", False) and decision.run_perception)
            prec = str(self.cfg.train.precision)
            if self.device.type != "cuda":
                prec = "fp32"
            with autocast_ctx(self.device, prec):
                out = self.agent(
                    frames,
                    state,
                    tokens=None,
                    prev_frames=self._prev,
                    reason_tick=decision.run_perception,
                    plan_sample=plan_sample,
                    explore=exp,
                    clock=clock_t,
                    task_id=task_id,
                    task_aux=task_aux,
                    overlay=overlay,
                    gui_open=gui_now,
                    **inv_kw,
                    **pulse_kw,
                    sample_actions=True,
                    action_temperature=float(self.cfg.policy.temperature),
                    deterministic_action=deterministic,
                )
            tokens = out.get("sampled_tokens")
            if not isinstance(tokens, torch.Tensor):
                raise RuntimeError("policy did not return sampled_tokens")
            packed = []
            # Deterministic evaluation keeps the smooth expectation used by
            # the legacy controller.  Self-play executes the sampled bins
            # themselves, so the action credited by REINFORCE is the action
            # that actually reached Minecraft.
            if deterministic:
                if gui_now:
                    nx_t, ny_t = cursor_from_logits(out["logits"], self.agent.spec)
                    packed.extend([nx_t.reshape(-1)[0], ny_t.reshape(-1)[0]])
                else:
                    dx_t, dy_t = look_deltas_from_logits(out["logits"], self.agent.spec)
                    packed.extend([dx_t.reshape(-1)[0], dy_t.reshape(-1)[0]])
                if self.agent.spec.has_wheel:
                    packed.append(wheel_from_logits(out["logits"], self.agent.spec).reshape(-1)[0])
                nums = torch.stack([p.float() for p in packed]).detach().cpu().tolist()
                if gui_now:
                    self._last_cursor = (float(nums[0]), float(nums[1]))
                    self._last_look = None
                else:
                    self._last_look = (float(nums[0]), float(nums[1]))
                    self._last_cursor = None
                self._last_wheel = float(nums[2]) if len(nums) > 2 else None
            else:
                self._last_look = None
                self._last_cursor = None
                self._last_wheel = None
            self._last_tokens = tokens
            if not overlay:
                self._prev = frames
            self.stats.n_policy += 1
            if decision.run_perception:
                self.stats.n_reason += 1
            self.clock.finish_tick(decision)
        tok_cpu = tokens.squeeze(0).detach().cpu()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.latencies_ms.append(dt_ms)
        self.stats.slept_ms.append(decision.slept_s * 1000.0)
        tok_np = tok_cpu.numpy().astype(np.int64)
        controls = tokens_to_controls(tok_cpu, self.agent.spec)
        if self._last_look is not None:
            controls["mouse"]["dx"] = self._last_look[0]
            controls["mouse"]["dy"] = self._last_look[1]
        if self._last_cursor is not None:
            controls["mouse"]["nx"] = self._last_cursor[0]
            controls["mouse"]["ny"] = self._last_cursor[1]
        if self._last_wheel is not None:
            controls["mouse"]["wheel"] = self._last_wheel
        hz = self.clock.scheduled_hz()
        info = {
            "controls": controls,
            "tokens": tok_np,
            "latency_ms": dt_ms,
            "slept_ms": decision.slept_s * 1000.0,
            "run_perception": decision.run_perception and not skipped,
            "run_policy": (not skipped),
            "sim_t": self.clock.sim_t,
            "policy_hz": hz["policy_hz"],
            "perception_hz": hz["perception_hz"],
            "clock_mode": self.clock.mode,
        }
        if signal:
            info["adv_next"] = signal.get("adv_next")
            try:
                info["adv_next_frac"] = float(signal.get("adv_next_frac") or 0.0)
            except (TypeError, ValueError):
                info["adv_next_frac"] = 0.0
        if self.learner is not None and info["run_perception"]:
            ht = None
            if human_tokens is not None:
                ht = torch.from_numpy(np.asarray(human_tokens, dtype=np.int64)).unsqueeze(0).to(self.device)
            act = tokens.detach() if isinstance(tokens, torch.Tensor) else None
            info.update(
                self.learner.observe(
                    pooled=getattr(self.agent, "_last_pooled", None),
                    fused_base=getattr(self.agent, "_last_fused_base", None),
                    policy_input=getattr(self.agent, "_last_policy_input", None),
                    state=state,
                    human_tokens=ht,
                    injected=bool(injected),
                    act=act,
                    signal=signal,
                )
            )
        return tok_np, info
