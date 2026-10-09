"""Screen-Event Coincidence Attention (SECA).

Screen-semantic spike self-attention for the 10Hz path.

A Minecraft frame is mostly repeated sky / dirt. Dense softmax over every
visual token therefore spends most of its mass on silence. SECA instead:

1. LIF-eventizes a compact spatial map of CNN tokens.
2. Ranks cells by tonic spikes plus ON-events (new spikes this tick).
3. Keeps only the top-K event cells and the decision vector.
4. Mixes that short sequence with *coincidence* weights: the number of
   co-firing Q/K channels (Hamming overlap). No softmax, so a silent
   screen cannot leak uniform attention onto the policy.

The residual starts at zero. 30Hz ticks only add the cached bias.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import ScreenEventConfig
from atomspike.models.iterator import as_hold_mask
from atomspike.models.lif import LIFCell


def coincidence_weights(q_s: Tensor, k_s: Tensor) -> Tensor:
    """[B, H, Tq, Tk] weights from spike overlap. Silent keys stay at 0."""
    overlap = torch.matmul(q_s, k_s.transpose(-2, -1))
    denom = overlap.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return overlap / denom


def compact_screen_tokens(
    tokens: Tensor,
    hw: tuple[int, int] | None,
    max_spatial: int,
) -> tuple[Tensor, tuple[int, int]]:
    """Pool visual tokens down to at most `max_spatial` cells, keep a 2D layout."""
    b, n, d = tokens.shape
    cap = max(4, int(max_spatial))
    h, w = (1, n)
    if hw is not None and int(hw[0]) * int(hw[1]) == n and int(hw[0]) > 1:
        h, w = int(hw[0]), int(hw[1])
    if n <= cap:
        return tokens, (h, w)
    if h * w == n and h > 1:
        side = max(2, min(h, w, int(cap ** 0.5)))
        grid = tokens.transpose(1, 2).reshape(b, d, h, w)
        grid = F.adaptive_avg_pool2d(grid, (side, side))
        return grid.flatten(2).transpose(1, 2).contiguous(), (side, side)
    take = min(n, cap)
    flat = tokens.transpose(1, 2).unsqueeze(-1)
    flat = F.adaptive_avg_pool2d(flat, (take, 1))
    return flat.squeeze(-1).transpose(1, 2).contiguous(), (take, 1)


def look_center_index(n: int, hw: tuple[int, int] | None) -> int:
    """Crosshair cell. World look is the middle of the token grid, not the GUI cursor."""
    n = max(1, int(n))
    if hw is not None:
        h, w = int(hw[0]), int(hw[1])
        if h > 0 and w > 0 and h * w == n:
            return (h // 2) * w + (w // 2)
    return n // 2


def innate_block_patterns(d: int, device, dtype) -> tuple[Tensor, Tensor]:
    """Unit-RMS orthogonal tags: break = even channels, place = odd."""
    d = max(2, int(d))
    brk = torch.zeros(d, device=device, dtype=dtype)
    plc = torch.zeros(d, device=device, dtype=dtype)
    brk[0::2] = 1.0
    plc[1::2] = 1.0
    brk = brk / brk.pow(2).mean().sqrt().clamp_min(1e-6)
    plc = plc / plc.pow(2).mean().sqrt().clamp_min(1e-6)
    return brk, plc


def _pulse_pair(pulse: Tensor, batch: int) -> Tensor:
    p = pulse.reshape(pulse.size(0), -1)
    if p.size(-1) < 2:
        p = F.pad(p, (0, 2 - p.size(-1)))
    p = p[:, :2].clamp(0.0, 1.0)
    if p.size(0) == batch:
        return p
    if p.size(0) == 1 and batch > 1:
        return p.expand(batch, 2)
    return p[:batch] if p.size(0) > batch else p


def apply_innate_block_pulse(
    tokens: Tensor,
    pulse: Tensor | None,
    hw: tuple[int, int] | None = None,
    amp: float = 0.08,
    hold: Tensor | None = None,
) -> Tensor:
    """Add a weak look-center tag. Block appearance stays in the same cell."""
    if pulse is None or float(amp) <= 0.0:
        return tokens
    if pulse.device.type == "cpu" and not bool(pulse.detach().abs().sum()):
        return tokens
    p = _pulse_pair(pulse.to(device=tokens.device, dtype=tokens.dtype), int(tokens.size(0)))
    held = as_hold_mask(hold, int(tokens.size(0)), tokens)
    if held is not None:
        p = p * (~held).to(dtype=p.dtype).unsqueeze(-1)
    b, n, d = tokens.shape
    idx = look_center_index(n, hw)
    brk, plc = innate_block_patterns(d, tokens.device, tokens.dtype)
    cell = tokens[:, idx]
    rms = cell.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-3)
    delta = (p[:, 0:1] * brk + p[:, 1:2] * plc) * (float(amp) * rms)
    out = tokens.clone()
    out[:, idx] = cell + delta
    return out


def block_counts_from_signal(signal: dict[str, Any] | None) -> tuple[int, int]:
    raw = signal or {}

    def _n(key: str) -> int:
        try:
            v = raw.get(key, 0)
            if v is None:
                return 0
            return max(0, int(float(v)))
        except (TypeError, ValueError):
            return 0

    return _n("broke"), _n("placed")


class BlockPulseTracker:
    """Sidecar counters -> one-shot {break, place} pulses. Counter reset is not a pulse."""

    def __init__(self) -> None:
        self._prev: tuple[int, int] | None = None
        self._held = np.zeros(2, dtype=np.float32)

    def observe(self, signal: dict[str, Any] | None) -> np.ndarray:
        broke, placed = block_counts_from_signal(signal)
        pulse = np.zeros(2, dtype=np.float32)
        if self._prev is not None:
            pb, pp = self._prev
            if broke > pb:
                pulse[0] = 1.0
            if placed > pp:
                pulse[1] = 1.0
        self._prev = (broke, placed)
        return pulse

    def hold(self, signal: dict[str, Any] | None) -> None:
        self._held = np.maximum(self._held, self.observe(signal))

    def take(self) -> np.ndarray:
        out = self._held.copy()
        self._held[:] = 0.0
        return out

    def reset(self) -> None:
        self._prev = None
        self._held[:] = 0.0


def block_pulse_from_signal(
    signal: dict[str, Any] | None,
    *,
    device: torch.device,
    batch: int = 1,
    tracker: BlockPulseTracker | None = None,
) -> Tensor:
    pulse = tracker.take() if tracker is not None else np.zeros(2, dtype=np.float32)
    t = torch.as_tensor(pulse, device=device, dtype=torch.float32).reshape(1, 2)
    b = max(1, int(batch))
    return t.expand(b, 2).contiguous()


def block_pulse_forward_kwargs(batch: dict, device: torch.device) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    if "block_pulse" not in batch:
        return out
    t = batch["block_pulse"].to(device, non_blocking=True).float()
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.size(-1) < 2:
        t = F.pad(t, (0, 2 - t.size(-1)))
    out["block_pulse"] = t.reshape(t.size(0), -1)[:, :2].contiguous()
    return out


class ScreenEventCoincidenceAttention(nn.Module):
    """Top-K screen spike events + coincidence self-attention on {decision ∪ events}."""

    def __init__(self, cfg: ScreenEventConfig, d_model: int):
        super().__init__()
        d = int(d_model)
        self.k_events = max(1, int(getattr(cfg, "k_events", 12) or 12))
        self.max_spatial = max(4, int(getattr(cfg, "max_spatial", 64) or 64))
        self.onset_w = max(0.0, float(getattr(cfg, "onset", 1.0) or 0.0))
        self.block_onset = max(0.0, float(getattr(cfg, "block_onset", 0.2) or 0.0))
        tau = float(getattr(cfg, "lif_tau", 0.5) or 0.5)
        v_th = float(getattr(cfg, "lif_v_th", 1.0) or 1.0)
        heads = max(1, int(getattr(cfg, "n_heads", 2) or 2))
        while heads > 1 and d % heads != 0:
            heads -= 1
        self.n_heads = heads
        self.head_dim = d // heads
        self.event_lif = LIFCell(tau=tau, v_th=v_th, thresh_shape=(d,))
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.lif_q = LIFCell(tau=tau, v_th=v_th, thresh_shape=(heads, 1, self.head_dim))
        self.lif_k = LIFCell(tau=tau, v_th=v_th, thresh_shape=(heads, 1, self.head_dim))
        self.resid_scale = nn.Parameter(torch.zeros(d))

    def observe(
        self,
        tokens: Tensor,
        decision: Tensor,
        hw: tuple[int, int] | None = None,
        mem: Tensor | None = None,
        mem_q: Tensor | None = None,
        mem_k: Tensor | None = None,
        prev_spike: Tensor | None = None,
        gui_mask: Tensor | None = None,
        cursor_xy: Tensor | None = None,
        cursor_w: Tensor | None = None,
        block_pulse: Tensor | None = None,
    ) -> dict[str, Tensor]:
        compact, layout = compact_screen_tokens(tokens, hw, self.max_spatial)
        b, n, d = compact.shape
        hold = as_hold_mask(gui_mask, b, compact)
        mem_in = mem if mem is not None and mem.shape == compact.shape else None
        spike, mem_new = self.event_lif(compact, mem_in)
        if hold is not None:
            vis = (~hold).to(dtype=compact.dtype).view(b, 1, 1)
            if mem_in is not None:
                mem_new = mem_new * vis + mem_in * (1.0 - vis)
            else:
                # The first overlay tick used to integrate GUI pixels into a
                # previously empty membrane even though spikes were masked.
                mem_new = mem_new * vis
            spike = spike * vis
        eventness = spike.mean(dim=-1)
        if (
            self.onset_w > 0
            and prev_spike is not None
            and prev_spike.shape == spike.shape
        ):
            onset = (spike - prev_spike).clamp_min(0.0)
            if hold is not None:
                onset = onset * (~hold).to(dtype=onset.dtype).view(b, 1, 1)
            eventness = eventness + self.onset_w * onset.mean(dim=-1)
        if cursor_xy is not None:
            gh, gw = int(layout[0]), int(layout[1])
            if gh * gw == n and gh > 1 and gw > 1:
                gx = (cursor_xy[:, 0].clamp(0, 1) * float(gw - 1)).round().long().clamp(0, gw - 1)
                gy = (cursor_xy[:, 1].clamp(0, 1) * float(gh - 1)).round().long().clamp(0, gh - 1)
                cell = (gy * gw + gx).clamp(0, n - 1).view(b, 1)
                boost = eventness.new_zeros(b, n)
                boost.scatter_(1, cell, 1.5)
                if cursor_w is not None:
                    boost = boost * cursor_w.to(dtype=boost.dtype).reshape(b, 1)
                eventness = eventness + boost
        if block_pulse is not None and self.block_onset > 0:
            p = _pulse_pair(block_pulse.to(device=eventness.device, dtype=eventness.dtype), b)
            if hold is not None:
                p = p * (~hold).to(dtype=p.dtype).unsqueeze(-1)
            center = look_center_index(n, layout)
            eventness = eventness.clone()
            eventness[:, center] = eventness[:, center] + self.block_onset * (p[:, 0] + p[:, 1])
        k = min(self.k_events, n)
        idx = eventness.detach().topk(k, dim=1).indices
        gather = idx.unsqueeze(-1).expand(-1, -1, d)
        events = compact.gather(1, gather)
        seq = torch.cat([decision.unsqueeze(1), events], dim=1)
        t = seq.size(1)
        qkv = self.qkv(seq).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, key, v = qkv[0], qkv[1], qkv[2]
        mem_q_in = mem_q if mem_q is not None and mem_q.shape == q.shape else None
        mem_k_in = mem_k if mem_k is not None and mem_k.shape == key.shape else None
        q_s, mem_q = self.lif_q(q, mem_q_in)
        k_s, mem_k = self.lif_k(key, mem_k_in)
        if hold is not None:
            vis_q = (~hold).to(dtype=q_s.dtype).view(b, 1, 1, 1)
            q_s = q_s * vis_q
            k_s = k_s * vis_q
            mem_q = mem_q * vis_q + (torch.zeros_like(mem_q) if mem_q_in is None else mem_q_in) * (1.0 - vis_q)
            mem_k = mem_k * vis_q + (torch.zeros_like(mem_k) if mem_k_in is None else mem_k_in) * (1.0 - vis_q)
        attn = coincidence_weights(q_s, k_s)
        mixed = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, d)
        delta = self.out(mixed)[:, 0]
        attn_w = attn[:, :, 0, 1:].mean(dim=1) if t > 1 else mixed.new_zeros(b, k)
        mask = compact.new_zeros(b, n)
        mask.scatter_(1, idx, 1.0)
        return {
            "bias": self.resid_scale * delta,
            "map": compact,
            "mem": mem_new,
            "mem_q": mem_q,
            "mem_k": mem_k,
            "spike": spike,
            "eventness": eventness,
            "attn_w": attn_w,
            "idx": idx,
            "mask": mask,
            "k": mixed.new_tensor(float(k)),
            "hw": layout,
            "hw_h": mixed.new_tensor(float(layout[0])),
            "hw_w": mixed.new_tensor(float(layout[1])),
        }


def overlay_events(out: dict) -> dict:
    """Top-K SEA cells and the attention mass they send to the decision token."""
    idx = out["idx"][0].detach()
    w = out["attn_w"][0].detach()
    ev = out["eventness"][0].gather(0, idx).detach()
    pack = torch.stack((idx.float(), w.float(), ev.float()), dim=0)
    hw = out.get("hw") or (0, 0)
    return {
        "idx": pack[0],
        "w": pack[1],
        "ev": pack[2],
        "hw": (int(hw[0]), int(hw[1])),
        "n": int(out["mask"].size(1)),
        "k": int(idx.numel()),
    }
