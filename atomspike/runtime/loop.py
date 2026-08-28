"""Dual-rate runtime: perception 5Hz, policy 30Hz, optional OS inject."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from atomspike.config import AtomSpikeConfig
from atomspike.models.actions import tokens_to_controls
from atomspike.models.agent import AtomSpikeAgent


@dataclass
class TickStats:
    latencies_ms: list[float] = field(default_factory=list)
    n_reason: int = 0
    n_policy: int = 0

    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        arr = np.asarray(self.latencies_ms)
        return float(np.percentile(arr, 95))


class DualRateLoop:
    def __init__(self, agent: AtomSpikeAgent, cfg: AtomSpikeConfig, device: torch.device):
        self.agent = agent
        self.cfg = cfg
        self.device = device
        self.stats = TickStats()
        self._prev: torch.Tensor | None = None

    def reset(self) -> None:
        self.agent.reset_runtime()
        self._prev = None

    @torch.no_grad()
    def tick(
        self,
        frame: NDArray[np.uint8],
        game_state: NDArray[np.float32],
        deterministic: bool = True,
    ) -> tuple[NDArray[np.int64], dict]:
        t0 = time.perf_counter()
        chw = np.transpose(frame, (2, 0, 1))
        frames = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        state = torch.from_numpy(np.asarray(game_state, dtype=np.float32)).unsqueeze(0).to(self.device)
        tokens = self.agent.act(
            frames,
            state,
            prev_frames=self._prev,
            deterministic=deterministic,
            dual_rate=True,
        )
        self._prev = frames
        self.stats.n_policy += 1
        every = self.cfg.dual_rate.perception_every_n
        if (self.agent._tick - 1) % every == 0:
            self.stats.n_reason += 1
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.latencies_ms.append(dt_ms)
        tok_np = tokens.squeeze(0).cpu().numpy().astype(np.int64)
        controls = tokens_to_controls(tokens.squeeze(0).cpu(), self.agent.spec)
        return tok_np, {"controls": controls, "latency_ms": dt_ms}
