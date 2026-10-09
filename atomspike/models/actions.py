"""8–13 slot atomic action space (keyboard + quantized mouse + optional wheel).

Slots (n_keys=8 Minecraft default after aligned())
-------------------------------------------------
0-7   WASD SPACE SHIFT CTRL E  (idle / press / hold / release)
8-9   look: mouse dx/dy bins; inventory/GUI: window-normalized cursor x/y
10-11 LMB RMB (idle / press / hold / release). Shift-click = SHIFT key + LMB.
12    wheel notches this tick (Minecraft hotbar / inventory scroll), if wheel_bins>=3.

Smoke / synthetic keeps n_keys=4 → 8 slots total (no wheel).
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
    wheel_bins: int = 1
    wheel_max: float = 4.0

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
            wheel_bins=int(getattr(cfg, "wheel_bins", 1) or 1),
            wheel_max=float(getattr(cfg, "wheel_max", 4.0) or 4.0),
        )

    @property
    def mouse_center(self) -> int:
        return self.mouse_bins // 2

    @property
    def mouse_radius(self) -> int:
        return self.mouse_bins // 2

    @property
    def has_wheel(self) -> bool:
        return int(self.wheel_bins) >= 3

    @property
    def wheel_center(self) -> int:
        return max(0, int(self.wheel_bins) // 2)

    @property
    def wheel_index(self) -> int:
        return self.n_keys + 2 + self.n_mouse_buttons

    @property
    def slot_vocabs(self) -> tuple[int, ...]:
        keys = (self.n_key_states,) * self.n_keys
        mouse_xy = (self.mouse_bins, self.mouse_bins)
        btns = (self.n_key_states,) * self.n_mouse_buttons
        vocabs = keys + mouse_xy + btns
        if self.has_wheel:
            vocabs = vocabs + (int(self.wheel_bins),)
        if len(vocabs) != self.n_slots:
            raise ValueError(f"expected {self.n_slots} slots, got {len(vocabs)}")
        return vocabs

    @property
    def slot_kinds(self) -> tuple[str, ...]:
        kinds = ("key",) * self.n_keys + ("mouse_axis", "mouse_axis") + ("mouse_btn",) * self.n_mouse_buttons
        if self.has_wheel:
            kinds = kinds + ("wheel",)
        return kinds

    @property
    def slot_names(self) -> tuple[str, ...]:
        names = self.key_names + ("MX", "MY") + ("LMB", "RMB")[: self.n_mouse_buttons]
        if self.has_wheel:
            names = names + ("WHL",)
        return names

    def idle_tokens(self) -> tuple[int, ...]:
        keys = (KEY_IDLE,) * self.n_keys
        mouse = (self.mouse_center, self.mouse_center)
        btns = (KEY_IDLE,) * self.n_mouse_buttons
        idle = keys + mouse + btns
        if self.has_wheel:
            idle = idle + (self.wheel_center,)
        return idle

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

    def wheel_to_bin(self, notches: float) -> int:
        arr = self.wheels_to_bins(np.asarray([notches], dtype=np.float64))
        return int(arr[0])

    def bin_to_wheel(self, bin_id: int) -> float:
        radius = max(1, self.wheel_center)
        y = (int(bin_id) - self.wheel_center) / float(radius)
        y = max(-1.0, min(1.0, float(y)))
        return float(y * self.wheel_max)

    def wheels_to_bins(self, notches: NDArray[np.floating]) -> NDArray[np.int64]:
        radius = max(1, self.wheel_center)
        y = np.clip(np.asarray(notches, dtype=np.float64) / max(1e-6, self.wheel_max), -1.0, 1.0)
        bins = np.rint(y * radius + self.wheel_center)
        return np.clip(bins, 0, max(0, self.wheel_bins - 1)).astype(np.int64)


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
    wheel = 0.0
    if spec.has_wheel and int(tokens.numel()) > spec.wheel_index:
        wheel = spec.bin_to_wheel(int(tokens[spec.wheel_index].item()))
    return {
        "keys": keys,
        "mouse": {
            "dx": dx,
            "dy": dy,
            "nx": spec.bin_to_mouse_unit(mx),
            "ny": spec.bin_to_mouse_unit(my),
            "buttons": buttons,
            "wheel": wheel,
        },
    }


_SOFT_TABLES: dict[tuple, Tensor] = {}


def _soft_table(kind: str, spec: ActionSpec, device, dtype) -> Tensor:
    key = (
        kind,
        int(spec.mouse_bins),
        float(spec.mouse_delta_max),
        float(spec.mouse_delta_scale),
        int(spec.wheel_bins),
        float(spec.wheel_max),
        str(device),
        str(dtype),
    )
    tbl = _SOFT_TABLES.get(key)
    if tbl is None or tbl.device != device:
        if kind == "look":
            vals = [spec.bin_to_mouse_delta(i) for i in range(spec.mouse_bins)]
        elif kind == "cursor":
            last = max(1, spec.mouse_bins - 1)
            vals = [i / last for i in range(spec.mouse_bins)]
        else:
            vals = [spec.bin_to_wheel(i) for i in range(spec.wheel_bins)]
        tbl = torch.tensor(vals, device=device, dtype=torch.float32)
        _SOFT_TABLES[key] = tbl
    return tbl.to(device=device, dtype=dtype)


def look_deltas_from_logits(logits: list[Tensor], spec: ActionSpec) -> tuple[Tensor, Tensor]:
    """Soft look dx/dy. Argmax on the idle bin would otherwise never turn the camera."""
    table = _soft_table("look", spec, logits[spec.n_keys].device, torch.float32)
    mx = torch.softmax(logits[spec.n_keys].float(), dim=-1) @ table
    my = torch.softmax(logits[spec.n_keys + 1].float(), dim=-1) @ table
    return mx, my


def cursor_from_logits(logits: list[Tensor], spec: ActionSpec) -> tuple[Tensor, Tensor]:
    """Soft window-normalized GUI cursor. Argmax of 41 bins is a 20px inventory snap."""
    table = _soft_table("cursor", spec, logits[spec.n_keys].device, torch.float32)
    nx = torch.softmax(logits[spec.n_keys].float(), dim=-1) @ table
    ny = torch.softmax(logits[spec.n_keys + 1].float(), dim=-1) @ table
    return nx, ny


def wheel_from_logits(logits: list[Tensor], spec: ActionSpec) -> Tensor:
    """Soft wheel notches. Center bin is rest; argmax would never scroll."""
    if not spec.has_wheel:
        return torch.zeros((), device=logits[0].device, dtype=torch.float32)
    table = _soft_table("wheel", spec, logits[spec.wheel_index].device, torch.float32)
    return torch.softmax(logits[spec.wheel_index].float(), dim=-1) @ table


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
        # HUD probe sometimes flags caves as inventory. Grabbed look keeps the
        # OS cursor near the window center; a real GUI cursor leaves it.
        if cursor_xy is not None:
            cur = np.asarray(cursor_xy, dtype=np.float64).reshape(-1, 2)[:n]
            parked = (np.abs(cur[:, 0] - 0.5) < 0.08) & (np.abs(cur[:, 1] - 0.5) < 0.08)
            look = look | ((mag > 6.0) & parked)
        else:
            look = look | (mag > 8.0)
        out[look, mx_i] = spec.mouse_deltas_to_bins(delta[look, 0])
        out[look, my_i] = spec.mouse_deltas_to_bins(delta[look, 1])
    gui_cursor = gui & ~look
    if cursor_xy is not None and gui_cursor.any():
        cur = np.asarray(cursor_xy, dtype=np.float64).reshape(-1, 2)[:n]
        out[gui_cursor, mx_i] = spec.mouse_units_to_bins(cur[gui_cursor, 0])
        out[gui_cursor, my_i] = spec.mouse_units_to_bins(cur[gui_cursor, 1])
    return out


def attach_wheel_slot(
    actions: NDArray[np.int64],
    spec: ActionSpec,
    wheel: NDArray[np.floating] | None = None,
) -> NDArray[np.int64]:
    """Pad old 12-slot demos and (re)tokenize WHL from stored notches."""
    out = np.array(actions, dtype=np.int64, copy=True)
    if out.ndim != 2:
        raise ValueError(f"actions must be [N, slots], got {out.shape}")
    n = int(out.shape[0])
    notches = np.zeros(n, dtype=np.float64)
    if wheel is not None:
        notches = np.asarray(wheel, dtype=np.float64).reshape(-1)[:n]
    if not spec.has_wheel:
        return out[:, : spec.n_slots] if out.shape[1] > spec.n_slots else out
    bins = spec.wheels_to_bins(notches).reshape(n, 1)
    if out.shape[1] == spec.n_slots:
        out[:, spec.wheel_index] = bins[:, 0]
        return out
    if out.shape[1] == spec.wheel_index:
        return np.concatenate([out, bins], axis=1)
    if out.shape[1] > spec.n_slots:
        out = out[:, : spec.n_slots]
        out[:, spec.wheel_index] = bins[:, 0]
        return out
    pad = np.zeros((n, spec.n_slots - out.shape[1]), dtype=np.int64)
    out = np.concatenate([out, pad], axis=1)
    out[:, spec.wheel_index] = bins[:, 0]
    return out
