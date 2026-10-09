"""Observed short-horizon outcomes, not counterfactual causal effects.

An example is valid only when its future stays inside one episode, both
endpoints have fresh crosshair telemetry, and the primary action persists
through most of the window. This is deliberately conservative: ordinary
demonstrations cannot establish what would have happened under another action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from atomspike.capture.mc_state import MC_STATE_FIELDS

if TYPE_CHECKING:
    from atomspike.models.actions import ActionSpec

EFFECT_ACTIONS: tuple[str, ...] = ("attack", "use", "move", "jump", "look", "idle")
EFFECT_EVENTS: tuple[str, ...] = ("goal_progress", "selected_gain", "damage", "block_change")


def action_macros(actions: NDArray[np.integer], spec: ActionSpec) -> NDArray[np.int64]:
    """Exclusive diagnostic macro with a fixed priority for mixed controls."""
    from atomspike.models.actions import KEY_HOLD, KEY_PRESS

    a = np.asarray(actions, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != spec.n_slots:
        raise ValueError("actions must be [N,n_slots] for the configured ActionSpec")
    active = (a == KEY_PRESS) | (a == KEY_HOLD)
    result = np.full(a.shape[0], EFFECT_ACTIONS.index("idle"), dtype=np.int64)
    look = (a[:, spec.n_keys] != spec.mouse_center) | (a[:, spec.n_keys + 1] != spec.mouse_center)
    result[look] = EFFECT_ACTIONS.index("look")
    jump = [i for i, name in enumerate(spec.key_names) if name == "SPACE"]
    if jump:
        result[active[:, jump[0]]] = EFFECT_ACTIONS.index("jump")
    move = [i for i, name in enumerate(spec.key_names) if name in {"W", "A", "S", "D"}]
    if move:
        result[active[:, move].any(axis=1)] = EFFECT_ACTIONS.index("move")
    use = [i for i, name in enumerate(spec.key_names) if name == "E"]
    if spec.n_mouse_buttons > 1:
        use.append(spec.n_keys + 3)
    if use:
        result[active[:, use].any(axis=1)] = EFFECT_ACTIONS.index("use")
    if spec.n_mouse_buttons > 0:
        result[active[:, spec.n_keys + 2]] = EFFECT_ACTIONS.index("attack")
    return result


def observed_effect_targets(
    states: NDArray[np.floating],
    actions: NDArray[np.integer],
    task_aux: NDArray[np.floating],
    task_id: NDArray[np.integer],
    block_pulse: NDArray[np.floating],
    look_concept: NDArray[np.integer],
    session_id: NDArray[np.integer],
    spec: ActionSpec | None,
    *,
    horizon: int,
    recording_id: NDArray[np.integer] | None = None,
    min_action_fraction: float = 0.8,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Return [N,4] binary outcomes and [N] validity for observed windows."""
    n = len(states)
    target = np.zeros((n, len(EFFECT_EVENTS)), dtype=np.float32)
    valid = np.zeros(n, dtype=np.uint8)
    if n == 0 or spec is None or horizon < 1:
        return target, valid
    if any(len(v) != n for v in (actions, task_aux, task_id, block_pulse, look_concept, session_id)):
        raise ValueError("effect source arrays must be frame-aligned")
    if recording_id is not None and len(recording_id) != n:
        raise ValueError("recording_id must be frame-aligned")
    if states.ndim != 2 or states.shape[1] <= max(MC_STATE_FIELDS.index("hp"), MC_STATE_FIELDS.index("selected_count")):
        return target, valid
    if task_aux.ndim != 2 or task_aux.shape[1] < 8 or block_pulse.ndim != 2 or block_pulse.shape[1] < 2:
        return target, valid
    macro = action_macros(actions, spec)
    hp = states[:, MC_STATE_FIELDS.index("hp")]
    selected = states[:, MC_STATE_FIELDS.index("selected_count")]
    aux = np.asarray(task_aux, dtype=np.float32)
    task = np.asarray(task_id).reshape(-1)
    pulse = np.asarray(block_pulse, dtype=np.float32)
    seen = np.asarray(look_concept).reshape(-1) >= 0
    sid = np.asarray(session_id).reshape(-1)
    rid = None if recording_id is None else np.asarray(recording_id).reshape(-1)
    h = int(horizon)
    required = max(1, int(np.ceil(h * float(min_action_fraction))))
    for t in range(max(0, n - h)):
        end = t + h
        if sid[t] != sid[end] or (rid is not None and rid[t] != rid[end]) or not (seen[t] and seen[end]):
            continue
        if int(np.count_nonzero(macro[t:end] == macro[t])) < required:
            continue
        done_gained = aux[end, 7] > aux[t, 7] + 1e-5
        frac_gained = task[t] == task[end] and aux[end, 0] > aux[t, 0] + 1e-3
        target[t, 0] = float(done_gained or frac_gained)
        target[t, 1] = float(selected[end] > selected[t] + 0.5 / 64.0)
        target[t, 2] = float(hp[end] < hp[t] - 0.5 / 20.0)
        target[t, 3] = float(np.any(pulse[t + 1 : end + 1] > 0.5))
        valid[t] = 1
    return target, valid
