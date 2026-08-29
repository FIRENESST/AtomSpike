"""Shared training utilities."""

from __future__ import annotations

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


def save_checkpoint(agent: AtomSpikeAgent, path: str | Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(extra or {})
    lora_meta = getattr(agent, "_lora_meta", None)
    if lora_meta and "lora" not in merged:
        merged["lora"] = lora_meta
    payload = {
        "model": agent.state_dict(),
        "cfg": dump_config(agent.cfg),
        "extra": merged,
    }
    torch.save(payload, path)
    return path


def load_agent(path: str | Path, cfg: AtomSpikeConfig, device: torch.device) -> AtomSpikeAgent:
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
    return agent
