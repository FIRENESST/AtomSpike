"""HDF5 episode store: (frame, game_state, action, reward, done) tuples."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


class _LazyH5Frames:
    """Fork-safe, random-access HDF5 frame source.

    Labels remain inexpensive NumPy arrays, while full-resolution video is
    opened lazily in the worker that asks for a frame.  This prevents a long
    Minecraft recording from allocating tens of GiB before the first batch.
    """

    def __init__(self, path: str | Path, shape: tuple[int, ...], dtype: np.dtype, *, dataset: str = "frames"):
        self.path = Path(path)
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.dataset = str(dataset)
        self._file: h5py.File | None = None
        self._pid: int | None = None

    def _open(self):
        pid = os.getpid()
        if self._file is None or self._pid != pid or not self._file.id.valid:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = pid
        return self._file

    def __getitem__(self, index: int) -> NDArray[np.uint8]:
        return np.asarray(self._open()[self.dataset][int(index)], dtype=self.dtype)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
        self._file = None
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_file"] = None
        state["_pid"] = None
        return state


class _ConcatFrames:
    """A logical concatenation that keeps each source file lazy."""

    def __init__(self, parts: list[Any]):
        if not parts:
            raise ValueError("empty frame source list")
        tail = tuple(int(v) for v in parts[0].shape[1:])
        if any(tuple(int(v) for v in p.shape[1:]) != tail for p in parts):
            raise ValueError("frame source shapes differ")
        self.parts = parts
        self.offsets = np.cumsum([0] + [int(p.shape[0]) for p in parts], dtype=np.int64)
        self.shape = (int(self.offsets[-1]),) + tail
        self.ndim = len(self.shape)
        self.dtype = np.dtype(parts[0].dtype)

    def __getitem__(self, index: int) -> NDArray[np.uint8]:
        i = int(index)
        if i < 0:
            i += int(self.shape[0])
        if i < 0 or i >= int(self.shape[0]):
            raise IndexError(i)
        part = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.parts[part][i - int(self.offsets[part])]

    def close(self) -> None:
        for part in self.parts:
            close = getattr(part, "close", None)
            if callable(close):
                close()


def _committed_row_count(f: h5py.File, n_total: int) -> int:
    """Return the contiguous committed prefix of a streaming HDF5 file."""
    if "committed" not in f:
        return n_total
    marks = np.asarray(f["committed"], dtype=np.uint8).reshape(-1)
    if marks.size != n_total:
        raise ValueError(f"{f.filename}: committed length does not match frames")
    bad = np.flatnonzero(marks != 1)
    if bad.size == 0:
        return n_total
    first = int(bad[0])
    if np.any(marks[first:] == 1):
        raise ValueError(f"{f.filename}: committed rows are non-contiguous")
    return first


class H5DemoDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        weights: NDArray[np.float32] | None = None,
        *,
        spec=None,
        indices: NDArray[np.integer] | None = None,
        retokenize_mouse: bool = True,
        max_resident_frames_mib: float | int | None = None,
    ):
        self.path = Path(path)
        self.spec = spec
        with h5py.File(self.path, "r") as f:
            frame_ds = f["frames"]
            n_total = int(frame_ds.shape[0])
            n_rows = _committed_row_count(f, n_total)
            if frame_ds.ndim != 4 or int(frame_ds.shape[-1]) != 3:
                raise ValueError(f"{self.path}: frames must be [N,H,W,3], got {tuple(frame_ds.shape)}")
            for name in ("game_state", "actions", "rewards", "dones"):
                if name not in f or int(f[name].shape[0]) != n_total:
                    raise ValueError(f"{self.path}: {name} is not frame-aligned")
            for name in (
                "timestamps_ns", "mouse_xy", "cursor_xy", "wheel", "task_id", "task_aux",
                "adv_next", "inv_ids", "hand_id", "hover_id", "carry_id", "block_pulse",
                "look_concept", "recording_id",
            ):
                if name in f and int(f[name].shape[0]) != n_total:
                    raise ValueError(f"{self.path}: {name} is not frame-aligned")
            frame_bytes = int(np.prod(frame_ds.shape, dtype=np.int64)) * int(frame_ds.dtype.itemsize)
            if n_rows != n_total:
                frame_bytes = int(np.prod((n_rows,) + tuple(frame_ds.shape[1:]), dtype=np.int64)) * int(frame_ds.dtype.itemsize)
            limit_bytes = None
            if max_resident_frames_mib is not None:
                limit_bytes = int(float(max_resident_frames_mib) * 1024 * 1024)
            self.frame_bytes = frame_bytes
            self.frame_storage = "lazy" if limit_bytes is not None and frame_bytes > limit_bytes else "resident"
            if self.frame_storage == "lazy":
                self.frames = _LazyH5Frames(self.path, (n_rows,) + tuple(frame_ds.shape[1:]), frame_ds.dtype)
            else:
                self.frames = np.asarray(frame_ds[:n_rows])
            self.states = np.asarray(f["game_state"][:n_rows])
            self.actions = np.asarray(f["actions"][:n_rows])
            self.rewards = np.asarray(f["rewards"][:n_rows])
            self.dones = np.asarray(f["dones"][:n_rows])
            for name, arr in (
                ("game_state", self.states),
                ("actions", self.actions),
                ("rewards", self.rewards),
                ("dones", self.dones),
            ):
                if arr.ndim < 1 or int(arr.shape[0]) != n_rows:
                    raise ValueError(f"{self.path}: {name} has {arr.shape[0] if arr.ndim else 0} rows; frames has {n_rows}")
            if self.states.ndim != 2 or self.actions.ndim != 2:
                raise ValueError(f"{self.path}: game_state and actions must be rank-2 arrays")
            if not np.isfinite(self.states).all():
                raise ValueError(f"{self.path}: game_state contains NaN or infinity")
            if not np.isfinite(self.rewards).all():
                raise ValueError(f"{self.path}: rewards contains NaN or infinity")
            if not np.isin(self.dones, (0, 1)).all():
                raise ValueError(f"{self.path}: dones must contain only 0 or 1")
            self._raw_dones = np.array(self.dones, copy=True)
            self.timestamps = np.array(f["timestamps_ns"][:n_rows], dtype=np.int64) if "timestamps_ns" in f else None
            if "timestamps_ns" in f and int(f["timestamps_ns"].shape[0]) != n_total:
                raise ValueError(f"{self.path}: timestamps_ns length does not match frames")
            self._recording_id_raw = (
                np.asarray(f["recording_id"][:n_rows], dtype=np.int64).reshape(-1)
                if "recording_id" in f
                else None
            )
            if self._recording_id_raw is not None:
                if self._recording_id_raw.shape[0] != n_rows:
                    raise ValueError(f"{self.path}: recording_id length does not match frames")
                if np.any(self._recording_id_raw < 0):
                    raise ValueError(f"{self.path}: recording_id must be non-negative")
            self._recording_starts = np.zeros(n_rows, dtype=bool)
            state_fields = f.attrs.get("state_fields", "")
            if isinstance(state_fields, bytes):
                state_fields = state_fields.decode("utf-8")
            self.state_fields = [s for s in str(state_fields).split(",") if s]
            key_names = f.attrs.get("key_names", "")
            if isinstance(key_names, bytes):
                key_names = key_names.decode("utf-8")
            self.key_names = tuple(s.strip().upper() for s in str(key_names).split(",") if s.strip())
            mouse_xy = np.array(f["mouse_xy"][:n_rows]) if "mouse_xy" in f else None
            cursor_xy = np.array(f["cursor_xy"][:n_rows]) if "cursor_xy" in f else None
            wheel = np.array(f["wheel"][:n_rows]) if "wheel" in f else None
            for name, arr in (("mouse_xy", mouse_xy), ("cursor_xy", cursor_xy), ("wheel", wheel)):
                if name in f and int(f[name].shape[0]) != n_total:
                    raise ValueError(f"{self.path}: {name} length does not match frames")
            self.task_id, self.task_aux = _load_task_arrays(f, n=n_rows, states=self.states, state_fields=self.state_fields)
            self.inv_ids, self.hand_id, self.hover_id, self.carry_id = _load_inv_arrays(f, n=n_rows)
            self.block_pulse = _load_block_pulse(f, n=n_rows)
            if "look_concept" in f:
                self.look_concept = np.asarray(f["look_concept"][:n_rows], dtype=np.int64).reshape(-1)
                from atomspike.data.visual_concepts import CONCEPT_KINDS

                if np.any((self.look_concept < -1) | (self.look_concept >= len(CONCEPT_KINDS))):
                    raise ValueError(f"{self.path}: look_concept contains an out-of-ontology ID")
            else:
                self.look_concept = np.full(n_rows, -1, dtype=np.int64)
        self.gui_index = self.state_fields.index("gui_open") if "gui_open" in self.state_fields else None
        n = int(self.frames.shape[0])
        if spec is not None and retokenize_mouse and (mouse_xy is not None or cursor_xy is not None):
            from atomspike.models.actions import retokenize_mouse_slots

            gui = None if self.gui_index is None else self.states[:, self.gui_index]
            self.actions = retokenize_mouse_slots(
                self.actions, spec, mouse_xy=mouse_xy, cursor_xy=cursor_xy, gui_open=gui
            )
        if spec is not None:
            from atomspike.models.actions import attach_wheel_slot

            self.actions = attach_wheel_slot(self.actions, spec, wheel)
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
        from atomspike.data.sessions import build_episode_timeline, recording_ids, session_lengths

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
        self.recording_id = recording_ids(
            len(self.states),
            getattr(self, "timestamps", None),
            dones=getattr(self, "_raw_dones", self.dones),
            explicit_ids=getattr(self, "_recording_id_raw", None),
            forced_starts=getattr(self, "_recording_starts", None),
            gap_s=gap_s,
        )
        from atomspike.data.effects import observed_effect_targets

        self.effect_target, self.effect_valid = observed_effect_targets(
            self.states,
            self.actions,
            self.task_aux,
            self.task_id,
            self.block_pulse,
            self.look_concept,
            self.session_id,
            self.spec,
            horizon=max(1, int(future_far)),
            recording_id=self.recording_id,
        )
        self.n_recordings = int(self.recording_id.max()) + 1 if self.recording_id.size else 0
        self.recording_lengths = session_lengths(self.recording_id)
        self._n_phases = int(n_phases)
        self._future_steps = int(future_steps)
        self._gap_s = float(gap_s)
        self._future_near = int(future_near)
        self._future_far = int(future_far)
        self._fps = float(fps)

    def describe_sessions(self) -> str:
        lengths = getattr(self, "recording_lengths", None) or []
        n_rec = int(getattr(self, "n_recordings", 1) or 1)
        n_ep = int(getattr(self, "n_sessions", 1) or 1)
        bits = "+".join(str(x) for x in lengths[:12])
        extra = "" if len(lengths) <= 12 else f"+{len(lengths)-12}more"
        return f"{int(self.frames.shape[0])} frames  {n_rec} recordings / {n_ep} episodes  ({bits}{extra})"

    @classmethod
    def from_config(cls, path, cfg: AtomSpikeConfig, weights=None, **kwargs) -> "H5DemoDataset":
        from atomspike.models.actions import ActionSpec

        spec = ActionSpec.from_config(cfg.action)
        kwargs = dict(kwargs)
        kwargs.setdefault("max_resident_frames_mib", float(cfg.train.max_resident_frames_mib))
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
        from atomspike.data.rabc import discounted_returns

        ds.returns = discounted_returns(ds.rewards, ds.dones, gamma=float(cfg.train.rl_gamma))
        storage = getattr(ds, "frame_storage", "resident")
        gib = float(getattr(ds, "frame_bytes", 0)) / float(1024**3)
        print(f"[data] {ds.describe_sessions()}  frames={storage} ({gib:.2f} GiB raw)", flush=True)
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
            if p.key_names != ref.key_names:
                raise ValueError(f"key_names mismatch in {p.path}")
        out = cls.__new__(cls)
        out.path = ref.path if len(parts) == 1 else Path(f"<{len(parts)} files>")
        out.spec = ref.spec
        if all(isinstance(p.frames, np.ndarray) for p in parts):
            out.frames = np.concatenate([p.frames for p in parts])
            out.frame_storage = "resident"
        else:
            out.frames = _ConcatFrames([p.frames for p in parts])
            out.frame_storage = "lazy"
        out.frame_bytes = sum(int(getattr(p, "frame_bytes", 0)) for p in parts)
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
        out._recording_id_raw = np.concatenate(
            [np.asarray(p.recording_id, dtype=np.int64) for p in parts]
        )
        out._recording_starts = np.zeros(int(out.frames.shape[0]), dtype=bool)
        start = 0
        for p in parts:
            part_n = int(p.frames.shape[0])
            if part_n:
                out._recording_starts[start] = True
            start += part_n
        # force episode boundaries so prev_frame never crosses files
        end = 0
        for p in parts[:-1]:
            end += int(p.frames.shape[0])
            out._raw_dones[end - 1] = 1
            out.dones[end - 1] = 1
        out.weights = np.concatenate([p.weights for p in parts]).astype(np.float32)
        out.returns = np.concatenate(
            [getattr(p, "returns", np.asarray(p.rewards, dtype=np.float32)) for p in parts]
        ).astype(np.float32)
        out.task_id = np.concatenate([p.task_id for p in parts])
        out.task_aux = np.concatenate([p.task_aux for p in parts])
        out.inv_ids = np.concatenate([p.inv_ids for p in parts])
        out.hand_id = np.concatenate([p.hand_id for p in parts])
        out.hover_id = np.concatenate([p.hover_id for p in parts])
        out.carry_id = np.concatenate([p.carry_id for p in parts])
        out.block_pulse = np.concatenate([p.block_pulse for p in parts])
        out.look_concept = np.concatenate([p.look_concept for p in parts])
        out.state_fields = ref.state_fields
        out.key_names = ref.key_names
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
        other.frame_storage = getattr(self, "frame_storage", "resident")
        other.frame_bytes = int(getattr(self, "frame_bytes", 0))
        other.states = self.states
        other.actions = self.actions
        other.rewards = self.rewards
        other.dones = self.dones
        other._raw_dones = getattr(self, "_raw_dones", self.dones)
        other.timestamps = getattr(self, "timestamps", None)
        other._recording_id_raw = getattr(self, "_recording_id_raw", None)
        other._recording_starts = getattr(self, "_recording_starts", None)
        other.state_fields = self.state_fields
        other.key_names = self.key_names
        other.gui_index = self.gui_index
        other.task_id = self.task_id
        other.task_aux = self.task_aux
        other.inv_ids = self.inv_ids
        other.hand_id = self.hand_id
        other.hover_id = self.hover_id
        other.carry_id = self.carry_id
        other.block_pulse = self.block_pulse
        other.look_concept = self.look_concept
        other.effect_target = self.effect_target
        other.effect_valid = self.effect_valid
        other.weights = self.weights
        other.returns = getattr(self, "returns", self.rewards)
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
        other.recording_id = getattr(self, "recording_id", None)
        other.n_recordings = getattr(self, "n_recordings", 1)
        other.recording_lengths = list(getattr(self, "recording_lengths", []) or [])
        other._n_phases = getattr(self, "_n_phases", 16)
        other._future_steps = getattr(self, "_future_steps", 15)
        other._gap_s = getattr(self, "_gap_s", 2.0)
        other._future_near = getattr(self, "_future_near", 5)
        other._future_far = getattr(self, "_future_far", 45)
        other._fps = getattr(self, "_fps", 30.0)
        other._index = np.asarray(indices, dtype=np.int64)
        return other

    def close(self) -> None:
        close = getattr(self.frames, "close", None)
        if callable(close):
            close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def assert_matches(self, cfg: AtomSpikeConfig) -> None:
        n_slots = int(self.actions.shape[1])
        state_dim = int(self.states.shape[1])
        if n_slots != cfg.action.n_slots:
            raise ValueError(
                f"HDF5 actions have {n_slots} slots but config n_slots={cfg.action.n_slots}. "
                "Old 12-slot files should grow a WHL column from extras on load; "
                "if this still fires, collect with the current YAML or pass matching --config."
            )
        if state_dim != cfg.reasoner.game_state_dim:
            raise ValueError(
                f"HDF5 game_state dim is {state_dim} but config game_state_dim={cfg.reasoner.game_state_dim}."
            )
        spec = self.spec
        if spec is not None:
            for i, vocab in enumerate(spec.slot_vocabs):
                col = self.actions[:, i]
                if col.size and (int(col.min()) < 0 or int(col.max()) >= int(vocab)):
                    raise ValueError(f"HDF5 action slot {i} has values outside [0, {int(vocab) - 1}]")
        if cfg.env.kind.lower() in {"minecraft", "mc", "minerl"} and self.frames.ndim == 4:
            h, w = int(self.frames.shape[1]), int(self.frames.shape[2])
            want_h, want_w = int(cfg.capture.store_height), int(cfg.capture.store_width)
            if h != want_h or w != want_w:
                raise ValueError(
                    f"HDF5 frames are {w}x{h} but config store is {want_w}x{want_h}. "
                    "Recapture at 480p (854x480) or pass matching YAML; do not mix resolutions."
                )
            from atomspike.capture.mc_state import MC_STATE_FIELDS

            expected_keys = tuple(str(k).upper() for k in cfg.action.key_names)
            if not self.key_names:
                raise ValueError(
                    "Minecraft HDF5 has no key_names metadata, so action-slot semantics cannot be verified. "
                    "Do not train on it; recollect with the current collector."
                )
            if self.key_names != expected_keys:
                raise ValueError(
                    f"HDF5 key_names={self.key_names} do not match config={expected_keys}; "
                    "do not relabel slots by position—recollect or use the matching config."
                )
            if tuple(self.state_fields) != tuple(MC_STATE_FIELDS):
                raise ValueError(
                    "Minecraft HDF5 state_fields do not exactly match the current 19-field schema; "
                    "recollect rather than silently padding/reinterpreting state labels."
                )

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        idx = int(self._index[i])
        frame = np.transpose(self.frames[idx], (2, 0, 1))  # CHW
        prev_idx = idx
        same_recording = True
        recording_id = getattr(self, "recording_id", None)
        if recording_id is not None and idx > 0:
            same_recording = int(recording_id[idx]) == int(recording_id[idx - 1])
        if idx > 0 and int(self.dones[idx - 1]) == 0 and same_recording:
            prev_idx = idx - 1
        prev_frame = np.transpose(self.frames[prev_idx], (2, 0, 1))
        return {
            "frame": frame,
            "prev_frame": prev_frame,
            "game_state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx : idx + 1],
            "return": self.returns[idx : idx + 1] if hasattr(self, "returns") else self.rewards[idx : idx + 1],
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
            "recording_id": self.recording_id[idx : idx + 1],
            "task_id": self.task_id[idx : idx + 1],
            "task_aux": self.task_aux[idx],
            "inv_ids": self.inv_ids[idx],
            "hand_id": self.hand_id[idx : idx + 1],
            "hover_id": self.hover_id[idx : idx + 1],
            "carry_id": self.carry_id[idx : idx + 1],
            "block_pulse": self.block_pulse[idx],
            "look_concept": self.look_concept[idx : idx + 1],
            "effect_target": self.effect_target[idx],
            "effect_valid": self.effect_valid[idx : idx + 1],
        }


def _load_task_arrays(f, n: int, states, state_fields) -> tuple[np.ndarray, np.ndarray]:
    from atomspike.models.iterator import TASK_AUX_DIM, task_id_from_name

    n = int(n)
    if "task_id" in f:
        task_id = np.asarray(f["task_id"][:n], dtype=np.int64).reshape(-1)
        if task_id.shape[0] != n:
            raise ValueError("task_id length does not match frames")
    else:
        task_id = np.zeros(n, dtype=np.int64)
    if "task_aux" in f:
        task_aux = np.asarray(f["task_aux"][:n], dtype=np.float32)
        if task_aux.ndim == 1:
            task_aux = task_aux.reshape(-1, 1)
        if task_aux.shape[0] != n:
            raise ValueError("task_aux length does not match frames")
        if task_aux.shape[1] != TASK_AUX_DIM:
            raise ValueError(f"task_aux width {task_aux.shape[1]} != expected {TASK_AUX_DIM}")
        if not np.isfinite(task_aux).all():
            raise ValueError("task_aux contains NaN or infinity")
        task_aux = np.ascontiguousarray(task_aux)
    else:
        task_aux = np.zeros((n, TASK_AUX_DIM), dtype=np.float32)
    if (not np.any(task_id)) and "adv_next" in f:
        raw = f["adv_next"]
        if int(raw.shape[0]) < n:
            raise ValueError("adv_next length does not match frames")
        for i in range(n):
            v = raw[i]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            task_id[i] = task_id_from_name(str(v) if v is not None else "")
    fields = list(state_fields or [])
    if states is not None and "gui_open" in fields and states.shape[1] > fields.index("gui_open"):
        empty = np.abs(task_aux).sum(axis=1) < 1e-8
        if empty.any():
            gi = fields.index("gui_open")
            g = np.asarray(states[:n, gi], dtype=np.float32)
            task_aux[empty, 1] = g[empty]
            task_aux[empty, 3] = g[empty]
    return task_id, task_aux


def _load_inv_arrays(f, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from atomspike.models.inventory import INV_SLOTS

    n = int(n)
    n_slots = INV_SLOTS
    if "inv_ids" in f:
        inv = np.asarray(f["inv_ids"][:n], dtype=np.int64)
        if inv.ndim == 1:
            inv = inv.reshape(-1, 1)
        if inv.shape[0] != n:
            raise ValueError("inv_ids length does not match frames")
        if inv.shape[1] != n_slots:
            raise ValueError(f"inv_ids width {inv.shape[1]} != expected {n_slots}")
    else:
        inv = np.zeros((n, n_slots), dtype=np.int64)

    def _ids(key: str) -> np.ndarray:
        if key in f:
            arr = np.asarray(f[key][:n], dtype=np.int64).reshape(-1)
            if arr.shape[0] != n:
                raise ValueError(f"{key} length does not match frames")
            return arr
        return np.zeros(n, dtype=np.int64)

    return inv, _ids("hand_id"), _ids("hover_id"), _ids("carry_id")


def _load_block_pulse(f, n: int) -> np.ndarray:
    n = int(n)
    if "block_pulse" in f:
        arr = np.asarray(f["block_pulse"][:n], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != n:
            raise ValueError("block_pulse length does not match frames")
        if arr.shape[1] != 2:
            raise ValueError(f"block_pulse width {arr.shape[1]} != expected 2")
        if not np.isfinite(arr).all():
            raise ValueError("block_pulse contains NaN or infinity")
        return np.clip(arr, 0.0, 1.0)
    pulse = np.zeros((n, 2), dtype=np.float32)
    if "broke" in f and "placed" in f:
        broke = np.asarray(f["broke"], dtype=np.int64).reshape(-1)
        placed = np.asarray(f["placed"], dtype=np.int64).reshape(-1)
        m = min(n, int(broke.shape[0]), int(placed.shape[0]))
        if m > 1:
            pulse[1:m, 0] = (broke[1:m] > broke[: m - 1]).astype(np.float32)
            pulse[1:m, 1] = (placed[1:m] > placed[: m - 1]).astype(np.float32)
    return pulse


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
