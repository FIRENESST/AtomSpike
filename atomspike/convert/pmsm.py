"""Explicitly lossy, calibrated ANN → binary-activation conversion.

One-bit activations cannot be mathematically equivalent to analog ReLU/GELU
at one time step. Conversion therefore requires real calibration data and
reports observed logit drift. ``firing_rate`` is measured on that same data;
it is not an energy measurement.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from pathlib import Path

import torch
from torch import Tensor, nn

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.models.activations import Act, iter_acts
from atomspike.models.agent import AtomSpikeAgent
from atomspike.data.sessions import stride_split_segments
from atomspike.train.common import load_agent, resolve_device, save_checkpoint


CalibrationBatch = tuple[Tensor, Tensor] | dict[str, Tensor]


def _forward_calibration(agent: AtomSpikeAgent, batch: CalibrationBatch) -> dict:
    """Run the same observable inputs used by BC/eval, not frame+state only."""
    if isinstance(batch, tuple):
        frames, state = batch
        return agent(frames, state, reason_tick=True)
    frames = batch["frame"]
    state = batch["game_state"]

    def scalar_id(key: str) -> Tensor | None:
        value = batch.get(key)
        if value is None:
            return None
        return value.reshape(value.size(0), -1)[:, 0]

    return agent(
        frames,
        state,
        prev_frames=batch.get("prev_frame"),
        reason_tick=True,
        clock=batch.get("clock"),
        task_id=batch.get("task_id"),
        task_aux=batch.get("task_aux"),
        inv_ids=batch.get("inv_ids"),
        hand_id=scalar_id("hand_id"),
        hover_id=scalar_id("hover_id"),
        carry_id=scalar_id("carry_id"),
        block_pulse=batch.get("block_pulse"),
    )


def _named_acts(module: nn.Module) -> list[tuple[str, Act]]:
    return [(name, item) for name, item in module.named_modules() if isinstance(item, Act)]


def enable_pmsm(
    agent: nn.Module,
    *,
    calibration: dict[str, dict[str, float]],
) -> int:
    """Enable a complete, finite per-layer PMSM calibration.

    There is intentionally no scalar threshold fallback.  A global threshold
    makes layers with different activation ranges degenerate, which was the
    original source of the false "T=1" conversion claim.
    """
    if not calibration:
        raise ValueError("PMSM requires a non-empty per-layer calibration; run calibrate_pmsm on real demos")
    n = 0
    for name, act in _named_acts(agent):
        item = calibration.get(name)
        if item is None:
            raise ValueError(f"PMSM calibration is missing layer {name}")
        try:
            threshold = float(item["threshold"])
            scale = float(item["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"PMSM calibration for {name} needs numeric threshold and scale") from exc
        if not isfinite(threshold) or threshold < 0.0 or not isfinite(scale) or scale <= 0.0:
            raise ValueError(f"PMSM calibration for {name} is invalid (threshold >= 0 and scale > 0 required)")
        act.set_spike(True, threshold=threshold, scale=scale)
        n += 1
    if n <= 0:
        raise RuntimeError("PMSM found 0 Act modules; encoder/reasoner must use Act")
    return n


def disable_pmsm(agent: nn.Module) -> int:
    n = 0
    for act in iter_acts(agent):
        act.set_spike(False)
        n += 1
    return n


def pmsm_status(agent: nn.Module) -> dict:
    acts = _named_acts(agent)
    if not acts:
        return {"enabled": False, "n_acts": 0, "threshold": 0.0, "layers": {}}
    enabled = all(act.is_spike for _, act in acts)
    layers = {
        name: {"threshold": float(act.threshold.item()), "scale": float(act.scale.item()), "spike": act.is_spike}
        for name, act in acts
    }
    return {
        "enabled": enabled,
        "n_acts": len(acts),
        "n_spike": sum(1 for _, act in acts if act.is_spike),
        # Compatibility field; use layers for the actual calibration.
        "threshold": float(acts[0][1].threshold.item()),
        "layers": layers,
    }


def _sample_values(x: Tensor, limit: int) -> Tensor:
    """Bound calibration memory without favoring one feature-map region."""
    flat = x.detach().float().reshape(-1)
    if flat.numel() <= limit:
        return flat.cpu()
    step = max(1, flat.numel() // limit)
    return flat[::step][:limit].cpu()


@torch.no_grad()
def calibrate_pmsm(
    agent: AtomSpikeAgent,
    batches: Iterable[CalibrationBatch],
    *,
    percentile: float,
    max_values_per_layer: int = 131_072,
) -> dict[str, dict[str, float]]:
    """Fit one threshold and scale per ``Act`` layer from ANN activations."""
    acts = _named_acts(agent)
    if not acts:
        raise RuntimeError("PMSM found 0 Act modules")
    samples: dict[str, list[Tensor]] = {name: [] for name, _ in acts}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, Tensor):
                remaining = max_values_per_layer - sum(int(x.numel()) for x in samples[name])
                if remaining > 0:
                    samples[name].append(_sample_values(output.abs(), remaining))

        return hook

    disable_pmsm(agent)
    for name, act in acts:
        handles.append(act.register_forward_hook(make_hook(name)))
    agent.eval()
    seen = 0
    try:
        for batch in batches:
            agent.reset_runtime()
            _forward_calibration(agent, batch)
            frames = batch[0] if isinstance(batch, tuple) else batch["frame"]
            seen += int(frames.size(0))
    finally:
        for handle in handles:
            handle.remove()
    if seen <= 0:
        raise ValueError("PMSM calibration needs at least one real demonstration frame")

    result: dict[str, dict[str, float]] = {}
    q = min(1.0, max(0.0, float(percentile) / 100.0))
    for name, _act in acts:
        values = torch.cat(samples[name]) if samples[name] else torch.zeros(1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            raise ValueError(f"PMSM calibration collected no finite activations for {name}")
        threshold = float(torch.quantile(values, q).item())
        fired = values > threshold
        # Least-squares constant for y ~= scale * 1[y > threshold].
        scale = float(values[fired].mean().item()) if bool(fired.any()) else 0.0
        result[name] = {
            "threshold": threshold,
            "scale": max(scale, 1e-8),
            "sample_count": float(values.numel()),
            "firing_rate": float(fired.float().mean().item()),
        }
    return result


def _flat_logits(out: dict) -> Tensor:
    return torch.cat([logit.detach().float().reshape(logit.size(0), -1) for logit in out["logits"]], dim=1)


@torch.no_grad()
def measure_conversion_fidelity(
    agent: AtomSpikeAgent,
    batches: Iterable[CalibrationBatch],
) -> dict[str, float]:
    """Compare ANN and calibrated-binary logits on real calibration examples."""
    ann_logits: list[Tensor] = []
    snn_logits: list[Tensor] = []
    agreements: list[Tensor] = []
    calibration = pmsm_status(agent)["layers"]
    agent.eval()
    for batch in batches:
        agent.reset_runtime()
        disable_pmsm(agent)
        ann = _forward_calibration(agent, batch)
        ann_logits.append(_flat_logits(ann).cpu())
        agent.reset_runtime()
        enable_pmsm(agent, calibration=calibration)
        snn = _forward_calibration(agent, batch)
        snn_logits.append(_flat_logits(snn).cpu())
        agreements.append(
            torch.stack(
                [a.argmax(dim=-1) == b.argmax(dim=-1) for a, b in zip(ann["logits"], snn["logits"])], dim=1
            )
            .float()
            .cpu()
        )
    if not ann_logits:
        raise ValueError("PMSM fidelity measurement needs at least one frame")
    ann_all = torch.cat(ann_logits)
    snn_all = torch.cat(snn_logits)
    mse = torch.mean((ann_all - snn_all).square())
    denom = ann_all.square().mean().clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(ann_all, snn_all, dim=1).mean()
    return {
        "logit_mse": float(mse.item()),
        "relative_logit_mse": float((mse / denom).item()),
        "logit_cosine": float(cosine.item()),
        "token_agreement": float(torch.cat(agreements).mean().item()),
        "n_examples": float(ann_all.size(0)),
    }


@torch.no_grad()
def measure_firing_rate(agent: AtomSpikeAgent, batches: Iterable[CalibrationBatch]) -> float:
    """Mean firing rate on calibration data, never random noise."""
    rates: list[Tensor] = []

    def hook(_module, _inputs, output):
        if isinstance(output, Tensor):
            rates.append((output > 0).float().mean().cpu())

    handles = [act.register_forward_hook(hook) for act in iter_acts(agent) if act.is_spike]
    try:
        for batch in batches:
            agent.reset_runtime()
            _forward_calibration(agent, batch)
    finally:
        for handle in handles:
            handle.remove()
    return float(torch.stack(rates).mean().item()) if rates else 0.0


def _calibration_batches(
    ds: H5DemoDataset,
    device: torch.device,
    *,
    n_samples: int,
    batch_size: int,
    indices: Iterable[int] | None = None,
) -> list[dict[str, Tensor]]:
    pool = list(range(len(ds))) if indices is None else [int(i) for i in indices]
    n = min(len(pool), max(1, int(n_samples)))
    if n <= 0:
        raise ValueError("PMSM calibration dataset is empty")
    positions = torch.linspace(0, len(pool) - 1, steps=n).round().long().tolist()
    chosen = [pool[pos] for pos in positions]
    keys = (
        "frame",
        "prev_frame",
        "game_state",
        "clock",
        "task_id",
        "task_aux",
        "inv_ids",
        "hand_id",
        "hover_id",
        "carry_id",
        "block_pulse",
    )
    out: list[dict[str, Tensor]] = []
    for start in range(0, len(chosen), max(1, batch_size)):
        selected = chosen[start : start + max(1, batch_size)]
        examples = [ds[i] for i in selected]
        packed = {
            key: torch.stack([torch.as_tensor(item[key]) for item in examples]).to(device)
            for key in keys
            if all(key in item for item in examples)
        }
        out.append(packed)
    return out


def convert_checkpoint(
    cfg: AtomSpikeConfig,
    ckpt_path: str | Path,
    out_path: str | Path,
    data_path: str | Path | list[str | Path] | None = None,
    *,
    allow_lossy: bool = False,
) -> dict:
    """Save a calibrated binary-activation experiment with measured drift."""
    if not allow_lossy:
        raise ValueError(
            "PMSM conversion is a lossy experimental binary approximation. "
            "Pass --allow-lossy-snn only after reviewing calibration data and fidelity."
        )
    if data_path is None:
        raise ValueError("PMSM conversion requires --data calibration demos; random calibration is disabled")
    device = resolve_device(cfg.train.device)
    agent = load_agent(ckpt_path, cfg, device)
    ds = H5DemoDataset.from_config(data_path, cfg)
    try:
        ds.assert_matches(cfg)
        train_idx, heldout_idx = stride_split_segments(
            len(ds),
            float(cfg.train.val_frac),
            getattr(ds, "recording_id", None),
        )
        batch_size = min(32, max(1, int(cfg.train.batch_size)))
        batches = _calibration_batches(
            ds,
            device,
            n_samples=int(cfg.convert.calibration_samples),
            batch_size=batch_size,
            indices=train_idx,
        )
        if len(heldout_idx):
            fidelity_batches = _calibration_batches(
                ds,
                device,
                n_samples=int(cfg.convert.calibration_samples),
                batch_size=batch_size,
                indices=heldout_idx,
            )
            fidelity_scope = "heldout_recordings"
        else:
            # A single recording cannot produce a temporally honest validation
            # set.  Still report drift, but label the reuse explicitly.
            fidelity_batches = batches
            fidelity_scope = "calibration_reuse_no_heldout_recording"
        calibration = calibrate_pmsm(agent, batches, percentile=cfg.convert.threshold_percentile)
        n_replaced = enable_pmsm(agent, calibration=calibration)
        fidelity = measure_conversion_fidelity(agent, fidelity_batches)
        firing_rate = measure_firing_rate(agent, fidelity_batches)
    finally:
        ds.close()
    fidelity["scope"] = fidelity_scope
    extra = {
        "pmsm": {
            "enabled": True,
            "n_acts": n_replaced,
            "method": "calibrated_binary_activation",
            "layers": calibration,
        },
        "firing_rate": firing_rate,
        "fidelity": fidelity,
        "method": "calibrated_binary_activation",
        "time_steps": 1,
        "calibration_examples": int(sum(int(batch["frame"].size(0)) for batch in batches)),
    }
    path = save_checkpoint(agent, out_path, extra=extra)
    return {
        "checkpoint": str(path),
        "replaced": n_replaced,
        "firing_rate": firing_rate,
        "fidelity": fidelity,
        "method": "calibrated_binary_activation",
        "time_steps": 1,
        "fidelity_scope": fidelity_scope,
        "status": pmsm_status(agent),
        "warning": "firing_rate is not an energy measurement; inspect fidelity before use",
    }
