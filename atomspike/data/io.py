"""HDF5 episode store: (frame, game_state, action, reward, done) tuples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset


class EpisodeWriter:
    def __init__(self, path: str | Path, frame_size: int, state_dim: int, n_slots: int = 8):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames: list[NDArray[np.uint8]] = []
        self.states: list[NDArray[np.float32]] = []
        self.actions: list[NDArray[np.int64]] = []
        self.rewards: list[float] = []
        self.dones: list[int] = []
        self.frame_size = frame_size
        self.state_dim = state_dim
        self.n_slots = n_slots

    def add(
        self,
        frame: NDArray[np.uint8],
        state: NDArray[np.float32],
        action: NDArray[np.int64],
        reward: float,
        done: bool,
    ) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8))
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.int64))
        self.rewards.append(float(reward))
        self.dones.append(int(done))

    def close(self) -> Path:
        n = len(self.frames)
        if n == 0:
            raise ValueError("no samples to write")
        with h5py.File(self.path, "w") as f:
            f.create_dataset("frames", data=np.stack(self.frames), compression="gzip", compression_opts=3)
            f.create_dataset("game_state", data=np.stack(self.states))
            f.create_dataset("actions", data=np.stack(self.actions))
            f.create_dataset("rewards", data=np.asarray(self.rewards, dtype=np.float32))
            f.create_dataset("dones", data=np.asarray(self.dones, dtype=np.uint8))
            f.attrs["n"] = n
            f.attrs["frame_size"] = self.frame_size
            f.attrs["state_dim"] = self.state_dim
        return self.path


class H5DemoDataset(Dataset):
    def __init__(self, path: str | Path, weights: NDArray[np.float32] | None = None):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.frames = np.array(f["frames"])
            self.states = np.array(f["game_state"])
            self.actions = np.array(f["actions"])
            self.rewards = np.array(f["rewards"])
            self.dones = np.array(f["dones"])
        self.weights = (
            np.ones(len(self.frames), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        )
        if len(self.weights) != len(self.frames):
            raise ValueError("weight length mismatch")

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        frame = np.transpose(self.frames[idx], (2, 0, 1))  # CHW
        prev_idx = idx
        if idx > 0 and int(self.dones[idx - 1]) == 0:
            prev_idx = idx - 1
        prev_frame = np.transpose(self.frames[prev_idx], (2, 0, 1))
        return {
            "frame": frame,
            "prev_frame": prev_frame,
            "game_state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx : idx + 1],
            "weight": self.weights[idx : idx + 1],
            "done": self.dones[idx : idx + 1],
        }


def collect_expert_demos(
    env,
    path: str | Path,
    episodes: int,
    seed: int = 0,
) -> Path:
    writer = EpisodeWriter(path, env.frame_size, env.spec_state_dim)
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        while True:
            act = env.expert_tokens()
            nxt = env.step(act)
            writer.add(obs.frame, obs.game_state, act, nxt.reward, nxt.done)
            obs = nxt
            if nxt.done:
                break
    return writer.close()
