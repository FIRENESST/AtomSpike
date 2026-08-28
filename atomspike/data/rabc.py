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


def coverage_weights(actions: NDArray[np.int64], power: float = 0.5) -> NDArray[np.float32]:
    """Inverse-frequency weights over action tuples (PostBc-lite)."""
    keys, inv, counts = np.unique(actions, axis=0, return_inverse=True, return_counts=True)
    freq = counts[inv].astype(np.float32)
    w = (freq.mean() / np.clip(freq, 1.0, None)) ** power
    return w / w.mean()


def rabc_weights(ds: H5DemoDataset, return_temp: float = 1.0, coverage_power: float = 0.5) -> NDArray[np.float32]:
    ret = episode_returns(ds.rewards, ds.dones)
    # softmax-like positive reweight on episode return
    z = (ret - ret.mean()) / max(1e-6, float(ret.std()))
    w_ret = np.exp(np.clip(z / max(return_temp, 1e-3), -8, 8))
    w_cov = coverage_weights(ds.actions, power=coverage_power)
    w = (w_ret * w_cov).astype(np.float32)
    return w / w.mean()


def make_weighted_sampler(weights: NDArray[np.float32]) -> WeightedRandomSampler:
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)
