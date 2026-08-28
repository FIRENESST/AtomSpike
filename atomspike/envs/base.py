"""Minimal game-env protocol used by capture, train, and eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class StepResult:
    frame: NDArray[np.uint8]  # HWC RGB
    game_state: NDArray[np.float32]
    reward: float
    done: bool
    info: dict[str, Any]


class GameEnv(Protocol):
    spec_state_dim: int

    def reset(self, seed: int | None = None) -> StepResult: ...

    def step(self, tokens: NDArray[np.int64]) -> StepResult: ...

    def close(self) -> None: ...
