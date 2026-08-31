"""Shared training utilities."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from atomspike.config import AtomSpikeConfig, dump_config
from atomspike.models.agent import AtomSpikeAgent


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_demos(batch: list[dict]) -> dict[str, Tensor]:
    def stack(key: str, dtype=None) -> Tensor:
        arr = np.stack([b[key] for b in batch])
        t = torch.from_numpy(arr)
        return t if dtype is None else t.to(dtype)

    return {
        "frame": stack("frame", torch.uint8),
        "prev_frame": stack("prev_frame", torch.uint8),
        "game_state": stack("game_state", torch.float32),
        "action": stack("action", torch.long),
        "reward": stack("reward", torch.float32).squeeze(-1),
        "weight": stack("weight", torch.float32).squeeze(-1),
        "done": stack("done").squeeze(-1),
    }


def make_loader(ds: Dataset, cfg: AtomSpikeConfig, sampler=None) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_demos,
        drop_last=False,
    )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _atomic_save(payload: dict, path: Path) -> None:
    """Write to .tmp, fsync, os.replace, fsync dir. Never half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    # Reopen in r+b so fsync works on Windows (rb-only fds can't fsync there).
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        # fsync'ing a directory is not portable; on Windows it can fail.
        # Data integrity is already guaranteed by the file-level fsync.
        pass


def _rotate_keep_k(path: Path, keep_k: int) -> None:
    """After saving path, drop older sibling checkpoints past keep_k.

    Matches every file in path.parent sharing the stem prefix and suffix.
    The just-written `path` is always kept; remaining slots go to the
    newest siblings by mtime. SHA-256 sidecars are removed alongside
    their checkpoint.
    """
    if keep_k < 1:
        return
    stem, suffix = path.stem, path.suffix
    siblings = sorted(
        [p for p in path.parent.glob(f"{stem}*{suffix}") if p != path],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # keep_k slots total; path takes one, so siblings get keep_k - 1.
    for old in siblings[max(0, keep_k - 1):]:
        try:
            old.unlink()
        except OSError:
            pass
        side = old.with_suffix(old.suffix + ".sha256")
        try:
            side.unlink()
        except OSError:
            pass


def save_checkpoint(
    agent: AtomSpikeAgent,
    path: str | Path,
    extra: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    loader_state: dict | None = None,
    keep_k: int | None = None,
    verify: bool = True,
) -> Path:
    """Save 5-state checkpoint atomically.

    5-state per README-5070Ti §5.1:
      model / optimizer / scheduler / rng / dataloader
    RNG captures python/np/torch(/cuda); optimizer & scheduler are optional
    so inference-only checkpoints (T6 convert) don't have to fake one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(extra or {})
    lora_meta = getattr(agent, "_lora_meta", None)
    if lora_meta and "lora" not in merged:
        merged["lora"] = lora_meta
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    payload = {
        "model": agent.state_dict(),
        "cfg": dump_config(agent.cfg),
        "extra": merged,
        "rng": rng,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if loader_state is not None:
        payload["loader"] = loader_state

    _atomic_save(payload, path)

    if verify:
        digest = _sha256(path)
        side = path.with_suffix(path.suffix + ".sha256")
        side.write_text(f"{digest}  {path.name}\n", encoding="utf-8")

    if keep_k:
        _rotate_keep_k(path, keep_k)
    return path


def load_checkpoint(
    path: str | Path,
    cfg: AtomSpikeConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    restore_rng: bool = True,
) -> tuple[AtomSpikeAgent, dict]:
    """Load a 5-state checkpoint. Returns (agent, leftover payload).

    Restores model + LoRA wrappers; restores optimizer / scheduler / RNG if
    handles are given. Returns the raw payload so callers can grab `loader`
    state or `extra` themselves.
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    extra = ckpt.get("extra") or {}
    agent = AtomSpikeAgent(cfg).to(device)
    lora = extra.get("lora")
    if lora:
        from atomspike.models.lora import apply_lora

        apply_lora(agent, int(lora["r"]), int(lora["alpha"]))
    agent.load_state_dict(ckpt["model"], strict=False)
    agent.to(device)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and "rng" in ckpt:
        rng = ckpt["rng"]
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return agent, ckpt


def load_agent(path: str | Path, cfg: AtomSpikeConfig, device: torch.device) -> AtomSpikeAgent:
    """Inference-only loader: model + LoRA only, no RNG/optimizer side effects."""
    agent, _ = load_checkpoint(path, cfg, device, restore_rng=False)
    return agent


# ---------------------------------------------------------------------------
# bf16 autocast + torch.compile helpers
# ---------------------------------------------------------------------------


def autocast_ctx(device: torch.device, precision: str):
    """Return a context manager for forward autocast, or nullcontext on fp32/cpu."""
    if precision == "fp32":
        from contextlib import nullcontext

        return nullcontext()
    if device.type != "cuda":
        # CPU autocast supports bf16 only; fp16 on CPU is not worth it.
        if precision == "bf16":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        from contextlib import nullcontext

        return nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


def maybe_compile(agent: AtomSpikeAgent, cfg: AtomSpikeConfig) -> AtomSpikeAgent:
    """torch.compile the agent when enabled and supported."""
    if not cfg.train.compile:
        return agent
    if not hasattr(torch, "compile"):
        return agent
    try:
        return torch.compile(agent, mode=cfg.train.compile_mode)
    except Exception:
        # compile is best-effort: never break training because of it
        return agent


def build_scheduler_if_needed(optimizer, cfg: AtomSpikeConfig, total_steps: int):
    """Create WSD scheduler if enabled; otherwise None (constant LR)."""
    if cfg.train.lr_schedule != "wsd":
        return None
    from atomspike.train.sched import build_wsd

    return build_wsd(
        optimizer,
        total_steps=total_steps,
        warmup_frac=cfg.train.wsd_warmup_frac,
        decay_frac=cfg.train.wsd_decay_frac,
        decay=cfg.train.wsd_decay,
    )


def optimizer_step(opt, scheduler, clip_norm: float, params) -> None:
    """Clip grads, step optimizer, step scheduler."""
    if clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, clip_norm)
    opt.step()
    if scheduler is not None:
        scheduler.step()
