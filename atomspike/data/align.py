"""Frame-action timestamp alignment for raw capture logs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CaptureEvent:
    t_ns: int
    kind: str
    payload: dict


def align_frames_actions(
    frame_times: NDArray[np.int64],
    events: list[CaptureEvent],
    n_slots: int = 8,
    idle: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """Nearest-previous event for each frame (causal)."""
    idle = np.zeros(n_slots, dtype=np.int64) if idle is None else idle
    actions = np.repeat(idle.reshape(1, -1), len(frame_times), axis=0)
    event_times = np.array([e.t_ns for e in events], dtype=np.int64)
    if event_times.size == 0:
        return actions
    for i, t in enumerate(frame_times):
        j = int(np.searchsorted(event_times, t, side="right") - 1)
        if j >= 0 and "tokens" in events[j].payload:
            actions[i] = np.asarray(events[j].payload["tokens"], dtype=np.int64)
    return actions
