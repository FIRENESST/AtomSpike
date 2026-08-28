"""8-token atomic action space (keyboard + quantized mouse).

Slots
-----
0-3  keyboard keys (idle / press / hold / release)
4-5  mouse dx, dy quantized bins
6-7  mouse buttons (idle / press / hold / release)

This is closer to Game-TARS device-level actions than a one-hot combo
classifier, and keeps multi-key concurrency without an exponential vocab.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
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

    @classmethod
    def from_config(cls, cfg: ActionConfig) -> "ActionSpec":
        return cls(
            n_keys=cfg.n_keys,
            n_key_states=cfg.n_key_states,
            mouse_bins=cfg.mouse_bins,
            n_mouse_buttons=cfg.n_mouse_buttons,
            key_names=tuple(cfg.key_names),
            n_slots=cfg.n_slots,
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

    def mouse_delta_to_bin(self, delta: float, max_abs: float = 10.0) -> int:
        clamped = max(-max_abs, min(max_abs, float(delta)))
        scaled = (clamped / max_abs) * self.mouse_radius
        return int(round(scaled + self.mouse_center))

    def bin_to_mouse_delta(self, bin_id: int, max_abs: float = 10.0) -> float:
        centered = int(bin_id) - self.mouse_center
        return (centered / max(1, self.mouse_radius)) * max_abs


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
    dx = spec.bin_to_mouse_delta(int(tokens[spec.n_keys].item()))
    dy = spec.bin_to_mouse_delta(int(tokens[spec.n_keys + 1].item()))
    buttons = {
        "LMB": KEY_STATE_NAMES[int(tokens[spec.n_keys + 2].item())],
        "RMB": KEY_STATE_NAMES[int(tokens[spec.n_keys + 3].item())],
    }
    return {"keys": keys, "mouse": {"dx": dx, "dy": dy, "buttons": buttons}}
