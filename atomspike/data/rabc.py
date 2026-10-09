"""RA-BC: score trajectories with a tiny reward model, then reweight.

High-quality sparse demos beat large noisy dumps (Chen et al. 2025).
Also implements a PostBc-style coverage upsample: rare action n-grams
get higher sampling weight so later RL sees the tails.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import WeightedRandomSampler

from atomspike.data.io import H5DemoDataset


def reward_signal_summary(ds: H5DemoDataset) -> dict[str, float | int]:
    """Describe whether a dataset can support return-aware training."""
    rewards = np.asarray(ds.rewards, dtype=np.float32).reshape(-1)
    finite = np.isfinite(rewards)
    nonzero = np.abs(rewards[finite]) > 1e-7
    returns = episode_returns(np.where(finite, rewards, 0.0), ds.dones)
    return {
        "n": int(rewards.size),
        "finite": int(finite.sum()),
        "nonzero": int(nonzero.sum()),
        "reward_std": float(rewards[finite].std()) if finite.any() else 0.0,
        "return_std": float(returns.std()) if returns.size else 0.0,
        "episodes": int(np.asarray(ds.dones, dtype=np.uint8).sum()),
    }


def require_informative_rewards(ds: H5DemoDataset, *, purpose: str) -> dict[str, float | int]:
    """Fail closed rather than silently degrading RA-BC/AWAC to plain BC."""
    info = reward_signal_summary(ds)
    if (
        info["finite"] != info["n"]
        or info["nonzero"] < 2
        or float(info["return_std"]) <= 1e-7
        or info["episodes"] < 1
    ):
        raise ValueError(
            f"{purpose} requires finite, non-constant episodic rewards; got "
            f"nonzero={info['nonzero']}/{info['n']}, return_std={info['return_std']:.3g}, "
            f"episodes={info['episodes']}. Recollect with the current collector or run plain train-bc."
        )
    return info


def episode_returns(rewards: NDArray[np.float32], dones: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Broadcast episode return onto every step of that episode."""
    out = np.zeros_like(rewards, dtype=np.float32)
    acc = 0.0
    start = 0
    for i, (r, d) in enumerate(zip(rewards, dones)):
        acc += float(r)
        if d:
            out[start : i + 1] = acc
            acc = 0.0
            start = i + 1
    if start < len(rewards):
        out[start:] = acc
    return out


def discounted_returns(rewards: NDArray[np.float32], dones: NDArray[np.uint8], gamma: float = 0.99) -> NDArray[np.float32]:
    """Reward-to-go that never crosses explicit or inferred episode cuts."""
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.uint8).reshape(-1)
    out = np.zeros_like(rewards, dtype=np.float32)
    carry = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if i == len(rewards) - 1 or dones[i]:
            carry = 0.0
        carry = float(rewards[i]) + float(gamma) * carry
        out[i] = carry
    return out


def coverage_weights(actions: NDArray[np.int64], power: float = 0.5) -> NDArray[np.float32]:
    """Inverse-frequency weights over action tuples (PostBc-lite)."""
    keys, inv, counts = np.unique(actions, axis=0, return_inverse=True, return_counts=True)
    freq = counts[inv].astype(np.float32)
    w = (freq.mean() / np.clip(freq, 1.0, None)) ** power
    return w / w.mean()


def rabc_weights(ds: H5DemoDataset, return_temp: float = 1.0, coverage_power: float = 0.5) -> NDArray[np.float32]:
    require_informative_rewards(ds, purpose="RA-BC")
    ret = episode_returns(ds.rewards, ds.dones)
    # softmax-like positive reweight on episode return
    z = (ret - ret.mean()) / max(1e-6, float(ret.std()))
    w_ret = np.exp(np.clip(z / max(return_temp, 1e-3), -8, 8))
    w_cov = coverage_weights(ds.actions, power=coverage_power)
    w = (w_ret * w_cov).astype(np.float32)
    return w / w.mean()


def make_weighted_sampler(weights: NDArray[np.float32]) -> WeightedRandomSampler:
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)
