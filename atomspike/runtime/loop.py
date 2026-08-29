"""Dual-rate runtime driven by DualRateClock, not tick%N."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from atomspike.config import AtomSpikeConfig
from atomspike.models.actions import tokens_to_controls
from atomspike.models.agent import AtomSpikeAgent
from atomspike.runtime.scheduler import ClockMode, DualRateClock


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

    def reset(self) -> None:
        self.agent.reset_runtime()
        self.clock.reset()
        self._prev = None
        self._last_tokens = None

    @torch.no_grad()
    def tick(
        self,
        frame: NDArray[np.uint8],
        game_state: NDArray[np.float32],
        deterministic: bool = True,
    ) -> tuple[NDArray[np.int64], dict]:
        decision = self.clock.start_tick()
        t0 = time.perf_counter()
        chw = np.transpose(frame, (2, 0, 1))
        frames = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(self.device)
        state = torch.from_numpy(np.asarray(game_state, dtype=np.float32)).unsqueeze(0).to(self.device)
        skipped = False
        if not decision.run_policy and self._last_tokens is not None:
            tokens = self._last_tokens
            skipped = True
            self.stats.n_skipped += 1
        else:
            tokens = self.agent.act(
                frames,
                state,
                prev_frames=self._prev,
                deterministic=deterministic,
                reason_tick=decision.run_perception,
            )
            self._last_tokens = tokens
            self._prev = frames
            self.stats.n_policy += 1
            if decision.run_perception:
                self.stats.n_reason += 1
            self.clock.finish_tick(decision)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.latencies_ms.append(dt_ms)
        self.stats.slept_ms.append(decision.slept_s * 1000.0)
        tok_np = tokens.squeeze(0).cpu().numpy().astype(np.int64)
        controls = tokens_to_controls(tokens.squeeze(0).cpu(), self.agent.spec)
        hz = self.clock.scheduled_hz()
        return tok_np, {
            "controls": controls,
            "latency_ms": dt_ms,
            "slept_ms": decision.slept_s * 1000.0,
            "run_perception": decision.run_perception and not skipped,
            "run_policy": (not skipped),
            "sim_t": self.clock.sim_t,
            "policy_hz": hz["policy_hz"],
            "perception_hz": hz["perception_hz"],
            "clock_mode": self.clock.mode,
        }
