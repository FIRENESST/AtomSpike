"""8–12 slot atomic action space (keyboard + quantized mouse).

Slots (n_keys=8 Minecraft default after aligned())
-------------------------------------------------
0-7   WASD SPACE SHIFT CTRL E  (idle / press / hold / release)
8-9   look: mouse dx/dy bins; inventory/GUI: window-normalized cursor x/y
10-11 LMB RMB (idle / press / hold / release). Shift-click = SHIFT key + LMB.

Smoke / synthetic keeps n_keys=4 → 8 slots total.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from atomspike.config import ActionConfig

KEY_IDLE, KEY_PRESS, KEY_HOLD, KEY_RELEASE = 0, 1, 2, 3
KEY_STATE_NAMES = ("idle", "press", "hold", "release")


@dataclass(frozen=True)
class ActionSpec:
    n_keys: int
    n_key_states: int
    mouse_bins: int
    n_mouse_buttons: int
    key_names: tuple[str, ...]
    n_slots: int = 8
    mouse_delta_max: float = 10.0
    mouse_delta_scale: float = 4.0

    @classmethod
    def from_config(cls, cfg: ActionConfig) -> "ActionSpec":
        names = tuple(cfg.key_names)
        if len(names) < cfg.n_keys:
            names = names + tuple(f"K{i}" for i in range(len(names), cfg.n_keys))
        elif len(names) > cfg.n_keys:
            names = names[: cfg.n_keys]
        return cls(
            n_keys=cfg.n_keys,
            n_key_states=cfg.n_key_states,
            mouse_bins=cfg.mouse_bins,
            n_mouse_buttons=cfg.n_mouse_buttons,
            key_names=names,
            n_slots=cfg.n_slots,
            mouse_delta_max=float(cfg.mouse_delta_max),
            mouse_delta_scale=float(cfg.mouse_delta_scale),
        )

    @property
    def mouse_center(self) -> int:
        return self.mouse_bins // 2

    @property
    def mouse_radius(self) -> int:
        return self.mouse_bins // 2

    @property
    def slot_vocabs(self) -> tuple[int, ...]:
        keys = (self.n_key_states,) * self.n_keys
        mouse_xy = (self.mouse_bins, self.mouse_bins)
        btns = (self.n_key_states,) * self.n_mouse_buttons
        vocabs = keys + mouse_xy + btns
        if len(vocabs) != self.n_slots:
            raise ValueError(f"expected {self.n_slots} slots, got {len(vocabs)}")
        return vocabs

    @property
    def slot_kinds(self) -> tuple[str, ...]:
        kinds = ("key",) * self.n_keys + ("mouse_axis", "mouse_axis") + ("mouse_btn",) * self.n_mouse_buttons
        return kinds

    def idle_tokens(self) -> tuple[int, ...]:
        keys = (KEY_IDLE,) * self.n_keys
        mouse = (self.mouse_center, self.mouse_center)
        btns = (KEY_IDLE,) * self.n_mouse_buttons
        return keys + mouse + btns

    def mouse_delta_to_bin(self, delta: float, max_abs: float | None = None) -> int:
        """Compress look deltas with asinh so flicks do not all saturate the last bin."""
        arr = self.mouse_deltas_to_bins(np.asarray([delta], dtype=np.float64), max_abs=max_abs)
        return int(arr[0])

    def bin_to_mouse_delta(self, bin_id: int, max_abs: float | None = None) -> float:
        max_abs = self.mouse_delta_max if max_abs is None else float(max_abs)
        scale = max(1e-6, self.mouse_delta_scale)
        y = (int(bin_id) - self.mouse_center) / max(1, self.mouse_radius)
        y = max(-1.0, min(1.0, float(y)))
        return float(scale * math.sinh(y * math.asinh(max_abs / scale)))

    def mouse_deltas_to_bins(self, delta: NDArray[np.floating], max_abs: float | None = None) -> NDArray[np.int64]:
        max_abs = self.mouse_delta_max if max_abs is None else float(max_abs)
        scale = max(1e-6, self.mouse_delta_scale)
        denom = max(1e-6, math.asinh(max_abs / scale))
        y = np.arcsinh(np.asarray(delta, dtype=np.float64) / scale) / denom
        y = np.clip(y, -1.0, 1.0)
        bins = np.rint(y * self.mouse_radius + self.mouse_center)
        return np.clip(bins, 0, self.mouse_bins - 1).astype(np.int64)

    def mouse_units_to_bins(self, unit: NDArray[np.floating]) -> NDArray[np.int64]:
        last = max(0, self.mouse_bins - 1)
        u = np.clip(np.asarray(unit, dtype=np.float64), 0.0, 1.0)
        return np.clip(np.rint(u * last), 0, last).astype(np.int64)

    def mouse_unit_to_bin(self, unit: float) -> int:
        """Map window-normalized [0, 1] cursor to a mouse-axis bin (GUI / inventory)."""
        u = max(0.0, min(1.0, float(unit)))
        last = max(0, self.mouse_bins - 1)
        return int(round(u * last))

    def bin_to_mouse_unit(self, bin_id: int) -> float:
        last = max(1, self.mouse_bins - 1)
        return float(max(0, min(last, int(bin_id)))) / float(last)


class ActionStateMachine:
    """Tracks press/hold/release legality and builds per-slot masks."""

    def __init__(self, spec: ActionSpec):
        self.spec = spec
        self.key_down = [False] * spec.n_keys
        self.btn_down = [False] * spec.n_mouse_buttons

    def reset(self) -> None:
        self.key_down = [False] * self.spec.n_keys
        self.btn_down = [False] * self.spec.n_mouse_buttons

    def mask_logits(self, logits: list[Tensor]) -> list[Tensor]:
        masked = list(logits)
        for i, down in enumerate(self.key_down):
            masked[i] = _mask_button_logits(masked[i], down)
        btn_base = self.spec.n_keys + 2
        for j, down in enumerate(self.btn_down):
            masked[btn_base + j] = _mask_button_logits(masked[btn_base + j], down)
        return masked

    def apply(self, tokens: Tensor) -> Tensor:
        """tokens: [8] or [B, 8] int. Updates held state; returns tokens."""
        squeeze = tokens.ndim == 1
        if squeeze:
            tokens = tokens.unsqueeze(0)
        b = tokens.shape[0]
        if b != 1:
            # batch apply is used in envs one-at-a-time; keep last row
            row = tokens[-1]
        else:
            row = tokens[0]
        for i in range(self.spec.n_keys):
            self.key_down[i] = _next_held(self.key_down[i], int(row[i].item()))
        base = self.spec.n_keys + 2
        for j in range(self.spec.n_mouse_buttons):
            self.btn_down[j] = _next_held(self.btn_down[j], int(row[base + j].item()))
        return tokens.squeeze(0) if squeeze else tokens

    def legal_idle_fallback(self, tokens: Tensor) -> Tensor:
        """Rewrite illegal button tokens to idle/hold rather than crash."""
        out = tokens.clone()
        if out.ndim == 1:
            out = out.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        for b in range(out.shape[0]):
            for i, down in enumerate(self.key_down):
                out[b, i] = _sanitize_button(int(out[b, i]), down)
            base = self.spec.n_keys + 2
            for j, down in enumerate(self.btn_down):
                out[b, base + j] = _sanitize_button(int(out[b, base + j]), down)
        return out.squeeze(0) if squeeze else out


def _next_held(was_down: bool, state: int) -> bool:
    if state in (KEY_PRESS, KEY_HOLD):
        return True
    if state == KEY_RELEASE:
        return False
    return False


def _sanitize_button(state: int, was_down: bool) -> int:
    if was_down:
        if state == KEY_PRESS:
            return KEY_HOLD
        return state
    if state in (KEY_HOLD, KEY_RELEASE):
        return KEY_IDLE if state == KEY_HOLD else KEY_IDLE
    return state


def _mask_button_logits(logits: Tensor, was_down: bool) -> Tensor:
    """Invalid transitions get -inf. logits: [..., 4]."""
    masked = logits.clone()
    neg = torch.finfo(masked.dtype).min
    if was_down:
        masked[..., KEY_PRESS] = neg
    else:
        masked[..., KEY_HOLD] = neg
        masked[..., KEY_RELEASE] = neg
    return masked


def tokens_to_controls(tokens: Tensor, spec: ActionSpec) -> dict[str, object]:
    """Decode a single [8] token vector into OS-level controls."""
    if tokens.ndim != 1:
        tokens = tokens.reshape(-1)[: spec.n_slots]
    keys = {}
    for i, name in enumerate(spec.key_names):
        keys[name] = KEY_STATE_NAMES[int(tokens[i].item())]
    mx = int(tokens[spec.n_keys].item())
    my = int(tokens[spec.n_keys + 1].item())
    dx = spec.bin_to_mouse_delta(mx)
    dy = spec.bin_to_mouse_delta(my)
    buttons = {
        "LMB": KEY_STATE_NAMES[int(tokens[spec.n_keys + 2].item())],
        "RMB": KEY_STATE_NAMES[int(tokens[spec.n_keys + 3].item())],
    }
    return {
        "keys": keys,
        "mouse": {
            "dx": dx,
            "dy": dy,
            "nx": spec.bin_to_mouse_unit(mx),
            "ny": spec.bin_to_mouse_unit(my),
            "buttons": buttons,
        },
    }


def look_deltas_from_logits(logits: list[Tensor], spec: ActionSpec) -> tuple[Tensor, Tensor]:
    """Soft look dx/dy. Argmax on the idle bin would otherwise never turn the camera."""
    table = torch.tensor(
        [spec.bin_to_mouse_delta(i) for i in range(spec.mouse_bins)],
        device=logits[spec.n_keys].device,
        dtype=torch.float32,
    )
    mx = torch.softmax(logits[spec.n_keys].float(), dim=-1) @ table
    my = torch.softmax(logits[spec.n_keys + 1].float(), dim=-1) @ table
    return mx, my


def retokenize_mouse_slots(
    actions: NDArray[np.int64],
    spec: ActionSpec,
    *,
    mouse_xy: NDArray[np.floating] | None = None,
    cursor_xy: NDArray[np.floating] | None = None,
    gui_open: NDArray[np.floating] | None = None,
) -> NDArray[np.int64]:
    """Rebuild mouse-axis tokens from stored high-res extras (no re-collect)."""
    out = np.array(actions, dtype=np.int64, copy=True)
    n = int(out.shape[0])
    mx_i, my_i = spec.n_keys, spec.n_keys + 1
    gui = np.zeros(n, dtype=bool)
    if gui_open is not None:
        gui = np.asarray(gui_open, dtype=np.float64).reshape(-1)[:n] > 0.5
    look = ~gui
    if mouse_xy is not None:
        delta = np.asarray(mouse_xy, dtype=np.float64).reshape(-1, 2)[:n]
        mag = np.hypot(delta[:, 0], delta[:, 1])
        # HUD often flags caves/sky as inventory; raw look deltas still mean look.
        look = look | (mag > 3.0)
        out[look, mx_i] = spec.mouse_deltas_to_bins(delta[look, 0])
        out[look, my_i] = spec.mouse_deltas_to_bins(delta[look, 1])
    gui_cursor = gui & ~look
    if cursor_xy is not None and gui_cursor.any():
        cur = np.asarray(cursor_xy, dtype=np.float64).reshape(-1, 2)[:n]
        out[gui_cursor, mx_i] = spec.mouse_units_to_bins(cur[gui_cursor, 0])
        out[gui_cursor, my_i] = spec.mouse_units_to_bins(cur[gui_cursor, 1])
    return out
