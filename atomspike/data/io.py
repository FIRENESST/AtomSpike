"""HDF5 episode store: (frame, game_state, action, reward, done) tuples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from numpy.typing import NDArray
from atomspike.config import AtomSpikeConfig
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


def resolve_data_paths(data: str | Path | list) -> list[Path]:
    """Accept a file, a directory of *.h5, or a list of either; return sorted files."""
    items = [data] if isinstance(data, (str, Path)) else list(data)
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.h5")))
        else:
            paths.append(p)
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"no .h5 demo files found for {data}")
    return paths


class H5DemoDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        weights: NDArray[np.float32] | None = None,
        *,
        spec=None,
        indices: NDArray[np.integer] | None = None,
        retokenize_mouse: bool = True,
    ):
        self.path = Path(path)
        self.spec = spec
        with h5py.File(self.path, "r") as f:
            self.frames = np.array(f["frames"])
            self.states = np.array(f["game_state"])
            self.actions = np.array(f["actions"])
            self.rewards = np.array(f["rewards"])
            self.dones = np.array(f["dones"])
            self._raw_dones = np.array(self.dones, copy=True)
            self.timestamps = np.array(f["timestamps_ns"], dtype=np.int64) if "timestamps_ns" in f else None
            state_fields = f.attrs.get("state_fields", "")
            if isinstance(state_fields, bytes):
                state_fields = state_fields.decode("utf-8")
            self.state_fields = [s for s in str(state_fields).split(",") if s]
            mouse_xy = np.array(f["mouse_xy"]) if "mouse_xy" in f else None
            cursor_xy = np.array(f["cursor_xy"]) if "cursor_xy" in f else None
        self.gui_index = self.state_fields.index("gui_open") if "gui_open" in self.state_fields else None
        n = int(self.frames.shape[0])
        if spec is not None and retokenize_mouse and (mouse_xy is not None or cursor_xy is not None):
            from atomspike.models.actions import retokenize_mouse_slots

            gui = None if self.gui_index is None else self.states[:, self.gui_index]
            self.actions = retokenize_mouse_slots(
                self.actions, spec, mouse_xy=mouse_xy, cursor_xy=cursor_xy, gui_open=gui
            )
        self._index = np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        base_w = np.ones(n, dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        if base_w.shape[0] != n:
            raise ValueError("weight length mismatch")
        self.weights = base_w
        self.attach_timeline(16, 15)

    def attach_timeline(
        self,
        n_phases: int,
        future_steps: int,
        *,
        gap_s: float = 2.0,
        future_near: int = 5,
        future_far: int = 45,
        fps: float = 30.0,
    ) -> None:
        from atomspike.data.sessions import build_episode_timeline

        tl = build_episode_timeline(
            self.states,
            getattr(self, "_raw_dones", self.dones),
            n_phases=n_phases,
            future_steps=future_steps,
            timestamps_ns=getattr(self, "timestamps", None),
            gap_s=gap_s,
            future_near=future_near,
            future_far=future_far,
            fps=fps,
        )
        self.dones = np.asarray(tl["dones"], dtype=np.uint8)
        self.time_frac = tl["time_frac"]
        self.session_t = tl["session_t"]
        self.dt_s = tl["dt_s"]
        self.phase = tl["phase"]
        self.next_phase = tl["next_phase"]
        self.future_state = tl["future_state"]
        self.future_state_near = tl["future_state_near"]
        self.future_state_far = tl["future_state_far"]
        self.clock = tl["clock"]
        self.session_id = tl["session_id"]
        self.n_sessions = int(tl["n_sessions"])
        self.session_lengths = list(tl["session_lengths"])
        self._n_phases = int(n_phases)
        self._future_steps = int(future_steps)
        self._gap_s = float(gap_s)
        self._future_near = int(future_near)
        self._future_far = int(future_far)
        self._fps = float(fps)

    def describe_sessions(self) -> str:
        lengths = getattr(self, "session_lengths", None) or []
        n_sess = int(getattr(self, "n_sessions", 1) or 1)
        bits = "+".join(str(x) for x in lengths[:12])
        extra = "" if len(lengths) <= 12 else f"+{len(lengths)-12}more"
        return f"{int(self.frames.shape[0])} frames  {n_sess} recordings  ({bits}{extra})"

    @classmethod
    def from_config(cls, path, cfg: AtomSpikeConfig, weights=None, **kwargs) -> "H5DemoDataset":
        from atomspike.models.actions import ActionSpec

        spec = ActionSpec.from_config(cfg.action)
        paths = resolve_data_paths(path)
        if len(paths) == 1:
            ds = cls(paths[0], weights=weights, spec=spec, **kwargs)
        else:
            if weights is not None:
                raise ValueError("per-frame weights are only supported for a single data file")
            parts = [cls(p, spec=spec, **kwargs) for p in paths]
            ds = cls.concat(parts)
        ds.attach_timeline(
            cfg.planner.n_phases,
            cfg.planner.future_steps,
            gap_s=float(getattr(cfg.planner, "session_gap_s", 2.0)),
            future_near=int(getattr(cfg.planner, "future_near", 5)),
            future_far=int(getattr(cfg.planner, "future_far", 45)),
            fps=float(cfg.capture.fps or 30),
        )
        print(f"[data] {ds.describe_sessions()}", flush=True)
        return ds

    @classmethod
    def concat(cls, parts: list["H5DemoDataset"]) -> "H5DemoDataset":
        """Merge several loaded demo files into one in-memory dataset."""
        if not parts:
            raise ValueError("concat needs at least one dataset")
        ref = parts[0]
        for p in parts:
            if p.frames.shape[1:] != ref.frames.shape[1:]:
                raise ValueError(f"frame size mismatch: {p.path} {p.frames.shape[1:]} vs {ref.frames.shape[1:]}")
            if p.states.shape[1] != ref.states.shape[1] or p.actions.shape[1] != ref.actions.shape[1]:
                raise ValueError(f"state/action dim mismatch in {p.path}")
            if p.state_fields != ref.state_fields:
                raise ValueError(f"state_fields mismatch in {p.path}")
        out = cls.__new__(cls)
        out.path = ref.path if len(parts) == 1 else Path(f"<{len(parts)} files>")
        out.spec = ref.spec
        out.frames = np.concatenate([p.frames for p in parts])
        out.states = np.concatenate([p.states for p in parts])
        out.actions = np.concatenate([p.actions for p in parts])
        out.rewards = np.concatenate([p.rewards for p in parts])
        raw_parts = [np.asarray(getattr(p, "_raw_dones", p.dones), dtype=np.uint8) for p in parts]
        out._raw_dones = np.concatenate(raw_parts)
        out.dones = out._raw_dones.copy()
        if all(getattr(p, "timestamps", None) is not None for p in parts):
            out.timestamps = np.concatenate([p.timestamps for p in parts])
        else:
            out.timestamps = None
        # force episode boundaries so prev_frame never crosses files
        end = 0
        for p in parts[:-1]:
            end += int(p.frames.shape[0])
            out._raw_dones[end - 1] = 1
            out.dones[end - 1] = 1
        out.weights = np.concatenate([p.weights for p in parts]).astype(np.float32)
        out.state_fields = ref.state_fields
        out.gui_index = ref.gui_index
        out._index = np.arange(int(out.frames.shape[0]), dtype=np.int64)
        n_ph = getattr(ref, "_n_phases", 16)
        fut = getattr(ref, "_future_steps", 15)
        out.attach_timeline(
            n_ph,
            fut,
            gap_s=getattr(ref, "_gap_s", 2.0),
            future_near=getattr(ref, "_future_near", 5),
            future_far=getattr(ref, "_future_far", 45),
            fps=getattr(ref, "_fps", 30.0),
        )
        return out

    def subset(self, indices: NDArray[np.integer] | list[int]) -> "H5DemoDataset":
        other = H5DemoDataset.__new__(H5DemoDataset)
        other.path = self.path
        other.spec = self.spec
        other.frames = self.frames
        other.states = self.states
        other.actions = self.actions
        other.rewards = self.rewards
        other.dones = self.dones
        other._raw_dones = getattr(self, "_raw_dones", self.dones)
        other.timestamps = getattr(self, "timestamps", None)
        other.state_fields = self.state_fields
        other.gui_index = self.gui_index
        other.weights = self.weights
        other.time_frac = self.time_frac
        other.session_t = getattr(self, "session_t", None)
        other.dt_s = getattr(self, "dt_s", None)
        other.phase = self.phase
        other.next_phase = self.next_phase
        other.future_state = self.future_state
        other.future_state_near = getattr(self, "future_state_near", self.future_state)
        other.future_state_far = getattr(self, "future_state_far", self.future_state)
        other.clock = getattr(self, "clock", None)
        other.session_id = getattr(self, "session_id", None)
        other.n_sessions = getattr(self, "n_sessions", 1)
        other.session_lengths = list(getattr(self, "session_lengths", []) or [])
        other._n_phases = getattr(self, "_n_phases", 16)
        other._future_steps = getattr(self, "_future_steps", 15)
        other._gap_s = getattr(self, "_gap_s", 2.0)
        other._future_near = getattr(self, "_future_near", 5)
        other._future_far = getattr(self, "_future_far", 45)
        other._fps = getattr(self, "_fps", 30.0)
        other._index = np.asarray(indices, dtype=np.int64)
        return other

    def assert_matches(self, cfg: AtomSpikeConfig) -> None:
        n_slots = int(self.actions.shape[1])
        state_dim = int(self.states.shape[1])
        if n_slots != cfg.action.n_slots:
            raise ValueError(
                f"HDF5 actions have {n_slots} slots but config n_slots={cfg.action.n_slots}. "
                "Train with the same YAML used for collect-mc (default.yaml / minecraft.yaml)."
            )
        if state_dim != cfg.reasoner.game_state_dim:
            raise ValueError(
                f"HDF5 game_state dim is {state_dim} but config game_state_dim={cfg.reasoner.game_state_dim}."
            )
        if cfg.env.kind.lower() in {"minecraft", "mc", "minerl"} and self.frames.ndim == 4:
            h, w = int(self.frames.shape[1]), int(self.frames.shape[2])
            want_h, want_w = int(cfg.capture.store_height), int(cfg.capture.store_width)
            if h != want_h or w != want_w:
                raise ValueError(
                    f"HDF5 frames are {w}x{h} but config store is {want_w}x{want_h}. "
                    "Recapture at 480p (854x480) or pass matching YAML; do not mix resolutions."
                )

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        idx = int(self._index[i])
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
            "time_frac": self.time_frac[idx : idx + 1],
            "phase": self.phase[idx : idx + 1],
            "next_phase": self.next_phase[idx : idx + 1],
            "future_state": self.future_state[idx],
            "future_state_near": self.future_state_near[idx],
            "future_state_far": self.future_state_far[idx],
            "clock": self.clock[idx],
            "session_id": self.session_id[idx : idx + 1],
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
