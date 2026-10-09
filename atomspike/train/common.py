"""Shared training utilities."""

from __future__ import annotations

import hashlib
import hmac
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from atomspike.config import AtomSpikeConfig, dump_config
from atomspike.models.agent import AtomSpikeAgent


class CheckpointIntegrityError(ValueError):
    """A checkpoint failed its checksum or cannot be safely deserialized."""


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

    out = {
        "frame": stack("frame", torch.uint8),
        "prev_frame": stack("prev_frame", torch.uint8),
        "game_state": stack("game_state", torch.float32),
        "action": stack("action", torch.long),
        "reward": stack("reward", torch.float32).squeeze(-1),
        "return": stack("return", torch.float32).squeeze(-1),
        "weight": stack("weight", torch.float32).squeeze(-1),
        "done": stack("done").squeeze(-1),
        "time_frac": stack("time_frac", torch.float32).squeeze(-1),
        "phase": stack("phase", torch.long).squeeze(-1),
        "next_phase": stack("next_phase", torch.long).squeeze(-1),
        "future_state": stack("future_state", torch.float32),
        "future_state_near": stack("future_state_near", torch.float32),
        "future_state_far": stack("future_state_far", torch.float32),
        "clock": stack("clock", torch.float32),
        "session_id": stack("session_id", torch.long).squeeze(-1),
    }
    if "task_id" in batch[0]:
        out["task_id"] = stack("task_id", torch.long).reshape(len(batch), -1).squeeze(-1)
    if "task_aux" in batch[0]:
        out["task_aux"] = stack("task_aux", torch.float32)
    if "inv_ids" in batch[0]:
        out["inv_ids"] = stack("inv_ids", torch.long)
    if "block_pulse" in batch[0]:
        out["block_pulse"] = stack("block_pulse", torch.float32)
    for key in ("hand_id", "hover_id", "carry_id"):
        if key in batch[0]:
            out[key] = stack(key, torch.long).reshape(len(batch), -1).squeeze(-1)
    return out


def make_loader(ds: Dataset, cfg: AtomSpikeConfig, sampler=None, shuffle: bool | None = None) -> DataLoader:
    pin = torch.cuda.is_available() and str(cfg.train.device) != "cpu"
    if shuffle is None:
        shuffle = sampler is None
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_demos,
        drop_last=False,
        pin_memory=pin,
    )


def grad_accum_divisor(step: int, n_batches: int, accumulation: int) -> int:
    """Return this micro-batch group's true size, including the final tail."""
    accum = max(1, int(accumulation))
    total = max(0, int(n_batches))
    tail = total % accum
    if tail and int(step) >= total - tail:
        return tail
    return accum


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


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically publish a tiny sidecar file as well as the checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _safe_rng_state() -> dict:
    """Serialize RNG state using tensors/primitives accepted by weights_only."""
    py_ver, py_words, py_gauss = random.getstate()
    np_name, np_words, np_pos, np_has_gauss, np_cached = np.random.get_state()
    rng: dict = {
        "format": 2,
        "python_version": int(py_ver),
        "python_words": [int(v) for v in py_words],
        "python_gauss": None if py_gauss is None else float(py_gauss),
        "numpy_name": str(np_name),
        "numpy_words": torch.from_numpy(np.asarray(np_words, dtype=np.uint32).copy()),
        "numpy_pos": int(np_pos),
        "numpy_has_gauss": int(np_has_gauss),
        "numpy_cached": float(np_cached),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def _cpu_byte_state(state: torch.Tensor) -> torch.Tensor:
    """RNG blobs are byte tensors. map_location='cuda' would move them off CPU."""
    if not isinstance(state, torch.Tensor):
        raise TypeError("RNG state is not a tensor")
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng_state(rng: dict) -> None:
    """Restore the safe v2 format; accept primitive legacy tuples when present."""
    if int(rng.get("format", 1)) >= 2:
        try:
            random.setstate(
                (
                    int(rng["python_version"]),
                    tuple(int(v) for v in rng["python_words"]),
                    rng.get("python_gauss"),
                )
            )
            words = rng["numpy_words"]
            if isinstance(words, torch.Tensor):
                words = words.detach().cpu().numpy()
            np.random.set_state(
                (
                    str(rng["numpy_name"]),
                    np.asarray(words, dtype=np.uint32),
                    int(rng["numpy_pos"]),
                    int(rng["numpy_has_gauss"]),
                    float(rng["numpy_cached"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            # A partial RNG restore is worse than leaving a known seed alone.
            pass
    else:
        # Legacy checkpoints are only reachable through an explicit unsafe
        # load opt-in below; retain compatibility for that migration path.
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
    if "torch" in rng:
        try:
            torch.set_rng_state(_cpu_byte_state(rng["torch"]))
        except (TypeError, RuntimeError, ValueError):
            pass
    if "cuda" in rng and torch.cuda.is_available():
        try:
            states = rng["cuda"]
            if isinstance(states, (list, tuple)):
                torch.cuda.set_rng_state_all([_cpu_byte_state(s) for s in states])
        except (TypeError, RuntimeError, ValueError):
            pass


def _verify_checksum(path: Path) -> None:
    side = path.with_suffix(path.suffix + ".sha256")
    if not side.is_file():
        return
    try:
        expected = side.read_text(encoding="utf-8").strip().split()[0].lower()
    except (OSError, IndexError):
        raise CheckpointIntegrityError(f"invalid checksum sidecar: {side}") from None
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise CheckpointIntegrityError(f"invalid SHA-256 checksum in {side}")
    actual = _sha256(path)
    if not hmac.compare_digest(expected, actual):
        raise CheckpointIntegrityError(
            f"checkpoint checksum mismatch for {path}; refuse to load a corrupted or substituted file"
        )


def _rotate_keep_k(path: Path, keep_k: int) -> None:
    """After saving path, drop older sibling checkpoints past keep_k.

    Only explicit ``name.step-N.pt`` siblings participate.  A normal fixed
    output path (``bc.pt``) has no generations to rotate, and files such as
    ``bc.plastic.pt`` must never be mistaken for one.
    """
    if keep_k < 1:
        return
    stem, suffix = path.stem, path.suffix
    siblings = sorted(
        [p for p in path.parent.glob(f"{stem}.step-*{suffix}") if p != path],
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


def _archive_current(path: Path) -> Path | None:
    """Copy the currently published checkpoint into the next numbered generation."""
    if not path.is_file():
        return None
    pattern = re.compile(
        rf"^{re.escape(path.stem)}\.step-(\d+){re.escape(path.suffix)}$"
    )
    latest = 0
    for candidate in path.parent.glob(f"{path.stem}.step-*{path.suffix}"):
        match = pattern.match(candidate.name)
        if match is not None:
            latest = max(latest, int(match.group(1)))
    archived = path.with_name(f"{path.stem}.step-{latest + 1:06d}{path.suffix}")
    shutil.copy2(path, archived)
    side = path.with_suffix(path.suffix + ".sha256")
    if side.is_file():
        _atomic_write_text(
            archived.with_suffix(archived.suffix + ".sha256"),
            f"{_sha256(archived)}  {archived.name}\n",
        )
    return archived


def save_checkpoint(
    agent: AtomSpikeAgent,
    path: str | Path,
    extra: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    loader_state: dict | None = None,
    keep_k: int | None = None,
    verify: bool = True,
) -> Path:
    """Save an integrity-checked checkpoint atomically.

    Model and safe RNG state are always included. Optimizer, scheduler, and
    DataLoader state are included only when the caller actually supplies
    them; the normal trainers currently checkpoint at epoch boundaries and
    do not claim mid-epoch DataLoader-cursor recovery.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(extra or {})
    lora_meta = getattr(agent, "_lora_meta", None)
    if lora_meta and "lora" not in merged:
        merged["lora"] = lora_meta
    rng = _safe_rng_state()
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
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    if loader_state is not None:
        payload["loader"] = loader_state

    # Preserve the last published generation before atomically replacing the
    # stable path.  A fixed ``bc.pt`` can therefore have a real keep-k history
    # without ever matching or touching ``bc.plastic.pt``.
    if keep_k is not None and int(keep_k) > 1:
        _archive_current(path)
    _atomic_save(payload, path)

    if verify:
        digest = _sha256(path)
        side = path.with_suffix(path.suffix + ".sha256")
        _atomic_write_text(side, f"{digest}  {path.name}\n")

    if keep_k:
        _rotate_keep_k(path, keep_k)
    return path


def expand_state_dict(module: torch.nn.Module, incoming: dict) -> dict:
    """Copy overlapping rows/cols when a Linear/Embedding grew (12→13 slots, extra events)."""
    own = module.state_dict()
    out: dict = {}
    for key, value in incoming.items():
        if key not in own:
            continue
        target = own[key]
        if target.shape == value.shape:
            out[key] = value
            continue
        if target.ndim == value.ndim and all(int(target.shape[i]) >= int(value.shape[i]) for i in range(target.ndim)):
            filled = target.clone()
            sl = tuple(slice(0, int(s)) for s in value.shape)
            filled[sl] = value.to(device=filled.device, dtype=filled.dtype)
            out[key] = filled
    return out


def load_checkpoint(
    path: str | Path,
    cfg: AtomSpikeConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    restore_rng: bool = True,
    allow_unsafe_legacy: bool = False,
) -> tuple[AtomSpikeAgent, dict]:
    """Load an AtomSpike checkpoint. Returns (agent, raw payload).

    Restores model + LoRA wrappers; restores optimizer / scheduler / RNG if
    handles are given. Returns the raw payload so callers can grab `loader`
    state or `extra` themselves.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. "
            "Record Minecraft demos then train-bc before predict-mc."
        )
    _verify_checksum(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise CheckpointIntegrityError(
            "this PyTorch version lacks weights_only checkpoint loading; upgrade PyTorch rather than unsafely loading it"
        ) from exc
    except Exception as exc:
        if not allow_unsafe_legacy:
            raise CheckpointIntegrityError(
                f"refusing unsafe or legacy checkpoint {path}: {type(exc).__name__}. "
                "Re-export it with this AtomSpike version, or use the explicit migration API only after verifying its source."
            ) from exc
        # This opt-in exists solely for an offline, user-audited migration.
        # Normal CLI and inference paths never enable it.
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise CheckpointIntegrityError(f"invalid AtomSpike checkpoint structure: {path}")
    extra = ckpt.get("extra") or {}
    agent = AtomSpikeAgent(cfg).to(device)
    lora = extra.get("lora")
    if lora:
        from atomspike.models.lora import apply_lora

        apply_lora(agent, int(lora["r"]), int(lora["alpha"]))
    agent.load_state_dict(expand_state_dict(agent, ckpt["model"]), strict=False)
    agent.to(device)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and "rng" in ckpt:
        rng = ckpt["rng"]
        if isinstance(rng, dict):
            _restore_rng_state(rng)
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
    except Exception as exc:
        # Compilation is optional, but hiding a graph error makes a requested
        # performance mode indistinguishable from actually being enabled.
        print(f"[compile] disabled: {type(exc).__name__}: {exc}", flush=True)
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


def build_grad_scaler(device: torch.device, precision: str):
    """Enable dynamic loss scaling only for CUDA fp16 training."""
    enabled = device.type == "cuda" and str(precision) == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def backward_loss(loss: torch.Tensor, scaler) -> None:
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()


def optimizer_step(opt, scheduler, clip_norm: float, params, *, scaler=None) -> None:
    """Unscale, clip, update, and advance LR only when an update occurred."""
    scaled = scaler is not None and scaler.is_enabled()
    if scaled:
        scaler.unscale_(opt)
    if clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, clip_norm)
    stepped = True
    if scaled:
        old_scale = float(scaler.get_scale())
        scaler.step(opt)
        scaler.update()
        # GradScaler lowers the scale when it skipped an inf/nan update.
        stepped = float(scaler.get_scale()) >= old_scale
    else:
        opt.step()
    if scheduler is not None and stepped:
        scheduler.step()
