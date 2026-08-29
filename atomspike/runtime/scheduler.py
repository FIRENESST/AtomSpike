"""Time-based dual-rate clock: perception 5Hz, policy 30Hz.

Call counting (tick % 6) is not a scheduler: eval tight-loops at thousands of
Hz and play would not sleep. This clock has:

- sim: each policy tick advances 1/policy_hz seconds; perception fires on
  elapsed sim time (so 30 ticks => 5 perception updates at the default rates)
- realtime: uses perf_counter; optional pace sleeps to cap policy at 30Hz
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from atomspike.config import DualRateConfig

ClockMode = Literal["sim", "realtime"]


@dataclass
class DualRateDecision:
    now: float
    run_policy: bool
    run_perception: bool
    slept_s: float = 0.0


class DualRateClock:
    def __init__(
        self,
        perception_hz: float = 5.0,
        policy_hz: float = 30.0,
        mode: ClockMode = "sim",
        pace: bool = False,
    ):
        if perception_hz <= 0 or policy_hz <= 0:
            raise ValueError("Hz must be positive")
        self.perception_hz = float(perception_hz)
        self.policy_hz = float(policy_hz)
        self.perception_dt = 1.0 / self.perception_hz
        self.policy_dt = 1.0 / self.policy_hz
        self.mode: ClockMode = mode
        self.pace = bool(pace) and mode == "realtime"
        self.sim_t = 0.0
        self.last_perception_t = -1e9
        self.last_policy_t = -1e9
        self.n_policy = 0
        self.n_perception = 0
        self._wall0 = time.perf_counter()
        self._slot = 0
        self._slept_s = 0.0

    @classmethod
    def from_cfg(cls, cfg: DualRateConfig, mode: ClockMode | None = None, pace: bool | None = None) -> "DualRateClock":
        return cls(
            perception_hz=cfg.perception_hz,
            policy_hz=cfg.policy_hz,
            mode=mode or cfg.mode,
            pace=cfg.pace if pace is None else pace,
        )

    def reset(self) -> None:
        self.sim_t = 0.0
        self.last_perception_t = -1e9
        self.last_policy_t = -1e9
        self.n_policy = 0
        self.n_perception = 0
        self._wall0 = time.perf_counter()
        self._slot = 0
        self._slept_s = 0.0

    def _now(self) -> float:
        if self.mode == "realtime":
            return time.perf_counter() - self._wall0
        return self.sim_t

    def start_tick(self) -> DualRateDecision:
        slept = 0.0
        if self.pace:
            slept = self._sleep_until_slot()
        now = self._now()
        run_perception = (now - self.last_perception_t) >= self.perception_dt - 1e-9
        if self.mode == "sim":
            run_policy = True
        elif self.pace:
            run_policy = True
        else:
            run_policy = (now - self.last_policy_t) >= self.policy_dt - 1e-9
        return DualRateDecision(now=now, run_policy=run_policy, run_perception=run_perception, slept_s=slept)

    def finish_tick(self, decision: DualRateDecision) -> None:
        if decision.run_perception:
            self.last_perception_t = decision.now
            self.n_perception += 1
        if decision.run_policy:
            self.last_policy_t = decision.now
            self.n_policy += 1
            if self.mode == "sim":
                self.sim_t += self.policy_dt
            if self.pace:
                self._slot += 1

    def _sleep_until_slot(self) -> float:
        deadline = self._wall0 + self._slot * self.policy_dt
        now = time.perf_counter()
        remaining = deadline - now
        if remaining > 0:
            time.sleep(remaining)
            self._slept_s += remaining
            return remaining
        return 0.0

    def scheduled_hz(self) -> dict[str, float]:
        if self.mode == "sim":
            dur = max(self.sim_t, self.n_policy * self.policy_dt, 1e-9)
        else:
            dur = max(time.perf_counter() - self._wall0, 1e-9)
        return {
            "clock_mode": 1.0 if self.mode == "realtime" else 0.0,
            "elapsed_s": dur,
            "policy_hz": self.n_policy / dur,
            "perception_hz": self.n_perception / dur,
            "n_policy": float(self.n_policy),
            "n_perception": float(self.n_perception),
            "slept_s": self._slept_s,
            "target_policy_hz": self.policy_hz,
            "target_perception_hz": self.perception_hz,
        }
