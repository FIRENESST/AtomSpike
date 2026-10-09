"""Append-only HDF5 writer so a 30fps Minecraft session is not held in RAM."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray


_FRAME_ALIGNED = (
    "frames", "game_state", "actions", "rewards", "dones", "timestamps_ns",
    "mouse_xy", "keys_held", "facts", "cursor_xy", "wheel", "task_id",
    "task_aux", "inv_ids", "hand_id", "hover_id", "carry_id", "block_pulse",
    "look_concept", "recording_id", "committed",
)


def _truncate_to_committed(f: h5py.File) -> int:
    """Recover an interrupted append without retaining forged zero rows.

    New files keep a ``committed`` bit for every row.  Rows are marked only
    after all aligned datasets have been written.  Older files have no such
    journal, so they are treated as legacy-complete rather than guessed at.
    """
    n = int(f["frames"].shape[0])
    for name in _FRAME_ALIGNED:
        if name in f and int(f[name].shape[0]) != n:
            raise ValueError(f"{f.filename}: {name} length differs from frames; refuse to append")
    if "committed" not in f:
        f.create_dataset("committed", data=np.ones(n, dtype=np.uint8), maxshape=(None,), chunks=(1024,), dtype="uint8")
        f.attrs["committed_schema"] = 1
        return n
    marks = np.asarray(f["committed"], dtype=np.uint8).reshape(-1)
    bad = np.flatnonzero(marks != 1)
    if bad.size == 0:
        return n
    first = int(bad[0])
    if np.any(marks[first:] == 1):
        raise ValueError(f"{f.filename}: non-contiguous committed rows; refuse to append")
    for name in _FRAME_ALIGNED:
        if name in f:
            f[name].resize(first, axis=0)
    f.flush()
    return first


def _canon_key_names(names) -> str:
    """Normalize representation only; never rewrite a key's meaning.

    In particular, treating an old ``F`` slot as the current ``E`` slot
    silently relabels every historical action in an appended recording.
    """
    if not names:
        return ""
    if isinstance(names, (bytes, bytearray)):
        names = names.decode("utf-8")
    if isinstance(names, str):
        parts = [p.strip() for p in names.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in names]
    return ",".join(parts)


class StreamingDemoWriter:
    def __init__(
        self,
        path: str | Path,
        frame_hw: tuple[int, int],
        state_dim: int,
        n_slots: int,
        *,
        key_names: tuple[str, ...] = (),
        state_fields: tuple[str, ...] = (),
        extra_attrs: dict | None = None,
        chunk: int = 32,
        fact_dim: int = 8,
        append: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h, self.w = int(frame_hw[0]), int(frame_hw[1])
        self.state_dim = int(state_dim)
        self.n_slots = int(n_slots)
        self._fact_dim = fact_dim
        self._key_names = key_names
        self._last_flush = 0
        self._recording_id = 0
        self.slow_write = False
        if append and self.path.exists():
            self._open_append(key_names, state_fields)
        else:
            self._create(key_names, state_fields, extra_attrs, chunk)

    def _create(self, key_names, state_fields, extra_attrs, chunk) -> None:
        self.n = 0
        self._f = h5py.File(self.path, "w")
        frame_kw = {"chunks": (1, self.h, self.w, 3), "dtype": "uint8"}
        try:
            self.frames = self._f.create_dataset(
                "frames",
                shape=(0, self.h, self.w, 3),
                maxshape=(None, self.h, self.w, 3),
                compression="lzf",
                **frame_kw,
            )
        except (ValueError, RuntimeError, OSError):
            self.frames = self._f.create_dataset(
                "frames",
                shape=(0, self.h, self.w, 3),
                maxshape=(None, self.h, self.w, 3),
                **frame_kw,
            )
        self.slow_write = False
        self.states = self._f.create_dataset(
            "game_state", shape=(0, self.state_dim), maxshape=(None, self.state_dim), chunks=(256, self.state_dim), dtype="float32"
        )
        self.actions = self._f.create_dataset(
            "actions", shape=(0, self.n_slots), maxshape=(None, self.n_slots), chunks=(256, self.n_slots), dtype="int64"
        )
        self.rewards = self._f.create_dataset("rewards", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="float32")
        self.dones = self._f.create_dataset("dones", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="uint8")
        self.recording_id = self._f.create_dataset(
            "recording_id", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int64"
        )
        # A row becomes 1 only after every frame-aligned dataset is written.
        # The reader/append path drops an all-zero uncommitted tail after a
        # process crash instead of training on fabricated labels.
        self.committed = self._f.create_dataset("committed", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="uint8")
        self.times = self._f.create_dataset("timestamps_ns", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int64")
        self.mouse = self._f.create_dataset(
            "mouse_xy", shape=(0, 2), maxshape=(None, 2), chunks=(256, 2), dtype="float32"
        )
        n_keys = max(1, len(self._key_names))
        self.keys_held = self._f.create_dataset(
            "keys_held", shape=(0, n_keys), maxshape=(None, n_keys), chunks=(256, n_keys), dtype="uint8"
        )
        self.facts = self._f.create_dataset(
            "facts", shape=(0, self._fact_dim), maxshape=(None, self._fact_dim), chunks=(256, self._fact_dim), dtype="float32"
        )
        self.cursor = self._f.create_dataset(
            "cursor_xy", shape=(0, 2), maxshape=(None, 2), chunks=(256, 2), dtype="float32"
        )
        self.wheel = self._f.create_dataset("wheel", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="float32")
        from atomspike.models.iterator import TASK_AUX_DIM
        from atomspike.models.inventory import INV_SLOTS

        self.task_id = self._f.create_dataset("task_id", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int64")
        self.task_aux = self._f.create_dataset(
            "task_aux",
            shape=(0, TASK_AUX_DIM),
            maxshape=(None, TASK_AUX_DIM),
            chunks=(256, TASK_AUX_DIM),
            dtype="float32",
        )
        self.inv_ids = self._f.create_dataset(
            "inv_ids",
            shape=(0, INV_SLOTS),
            maxshape=(None, INV_SLOTS),
            chunks=(256, INV_SLOTS),
            dtype="int64",
        )
        self.hand_id = self._f.create_dataset("hand_id", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int64")
        self.hover_id = self._f.create_dataset("hover_id", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int64")
        self.carry_id = self._f.create_dataset("carry_id", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int64")
        self.block_pulse = self._f.create_dataset(
            "block_pulse", shape=(0, 2), maxshape=(None, 2), chunks=(256, 2), dtype="float32"
        )
        self.look_concept = self._f.create_dataset(
            "look_concept", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="int16"
        )
        self._f.attrs["frame_size_h"] = self.h
        self._f.attrs["frame_size_w"] = self.w
        self._f.attrs["state_dim"] = self.state_dim
        self._f.attrs["n_slots"] = self.n_slots
        self._f.attrs["committed_schema"] = 1
        self._f.attrs["recording_schema"] = 1
        self._f.attrs["look_concept_schema"] = "coarse_crosshair_v1_-1_unobserved"
        self._f.attrs["source"] = "minecraft"
        self._f.attrs["fact_dim"] = self._fact_dim
        if key_names:
            self._f.attrs["key_names"] = ",".join(key_names)
        if state_fields:
            self._f.attrs["state_fields"] = ",".join(state_fields)
        self._f.attrs["fact_fields"] = "x,y,z,yaw,pitch,hp,hunger,hotbar_slot"
        for k, v in (extra_attrs or {}).items():
            self._f.attrs[k] = v

    def _open_append(self, key_names, state_fields) -> None:
        f = h5py.File(self.path, "r+")
        try:
            recovered_n = _truncate_to_committed(f)
            checks = {
                "frame_size_h": self.h,
                "frame_size_w": self.w,
                "state_dim": self.state_dim,
            }
            for k, v in checks.items():
                if int(f.attrs.get(k, -1)) != int(v):
                    raise ValueError(
                        f"{self.path} was recorded with {k}={f.attrs.get(k)}, current config wants {v}. "
                        "Use a new --out file or --overwrite."
                    )
            stored_slots = int(f["actions"].shape[1])
            if stored_slots != self.n_slots:
                if self.n_slots == stored_slots + 1 and "wheel" in f:
                    _upgrade_actions_add_wheel(f, self.n_slots)
                else:
                    raise ValueError(
                        f"{self.path} was recorded with n_slots={stored_slots}, current config wants {self.n_slots}. "
                        "Use a new --out file or --overwrite."
                    )
            f.attrs["n_slots"] = self.n_slots
            if key_names and _canon_key_names(f.attrs.get("key_names", "")) != _canon_key_names(key_names):
                raise ValueError(f"{self.path} key_names differ; use a new --out file or --overwrite")
            if key_names:
                f.attrs["key_names"] = ",".join(key_names)
            if state_fields and f.attrs.get("state_fields", "") != ",".join(state_fields):
                raise ValueError(f"{self.path} state_fields differ; use a new --out file or --overwrite")
            _ensure_recording_dataset(f, int(f["frames"].shape[0]))
            self.frames = f["frames"]
            self.states = f["game_state"]
            self.actions = f["actions"]
            self.rewards = f["rewards"]
            self.dones = f["dones"]
            self.committed = f["committed"]
            self.times = f["timestamps_ns"]
            self.mouse = f["mouse_xy"]
            self.keys_held = f["keys_held"]
            self.facts = f["facts"]
            self.cursor = f["cursor_xy"]
            if "wheel" not in f:
                n0 = int(f["frames"].shape[0])
                f.create_dataset("wheel", data=np.zeros(n0, dtype=np.float32), maxshape=(None,), chunks=(1024,), dtype="float32")
            self.wheel = f["wheel"]
            _ensure_task_datasets(f, int(f["frames"].shape[0]))
            _ensure_inv_datasets(f, int(f["frames"].shape[0]))
            _ensure_block_pulse_dataset(f, int(f["frames"].shape[0]))
            _ensure_look_concept_dataset(f, int(f["frames"].shape[0]))
            self.task_id = f["task_id"]
            self.task_aux = f["task_aux"]
            self.inv_ids = f["inv_ids"]
            self.hand_id = f["hand_id"]
            self.hover_id = f["hover_id"]
            self.carry_id = f["carry_id"]
            self.block_pulse = f["block_pulse"]
            self.look_concept = f["look_concept"]
            self.recording_id = f["recording_id"]
        except Exception:
            f.close()
            raise
        self._f = f
        self.n = int(recovered_n)
        self._last_flush = self.n
        self._recording_id = (
            int(np.max(np.asarray(self.recording_id[: self.n], dtype=np.int64))) + 1
            if self.n
            else 0
        )
        ch = self.frames.chunks
        comp = str(self.frames.compression or "")
        self.slow_write = bool(ch and int(ch[0]) > 1) or comp in {"gzip", "szip"}

    def add(
        self,
        frame: NDArray[np.uint8],
        state: NDArray[np.float32],
        action: NDArray[np.int64],
        reward: float,
        done: bool,
        t_ns: int,
        mouse_xy: tuple[float, float] = (0.0, 0.0),
        keys_held: NDArray[np.uint8] | None = None,
        facts: NDArray[np.float32] | None = None,
        cursor_xy: tuple[float, float] = (0.5, 0.5),
        wheel: float = 0.0,
        task_id: int = 0,
        task_aux: NDArray[np.floating] | None = None,
        inv_ids: NDArray[np.integer] | None = None,
        hand_id: int = 0,
        hover_id: int = 0,
        carry_id: int = 0,
        block_pulse: tuple[float, float] | NDArray[np.floating] | None = None,
        look_concept: int = -1,
    ) -> None:
        # Validate and materialize the complete row before resizing anything.
        # A bad sidecar field must leave the HDF5 file untouched.
        fr = np.asarray(frame, dtype=np.uint8)
        if fr.shape != (self.h, self.w, 3):
            raise ValueError(f"frame shape {fr.shape} != {(self.h, self.w, 3)}")
        st = np.asarray(state, dtype=np.float32).reshape(-1)
        act = np.asarray(action, dtype=np.int64).reshape(-1)
        if st.size != self.state_dim:
            raise ValueError(f"game_state has {st.size} values; expected {self.state_dim}")
        if act.size != self.n_slots:
            raise ValueError(f"action has {act.size} slots; expected {self.n_slots}")
        if not np.isfinite(st).all():
            raise ValueError("game_state contains NaN or infinity")
        scalar_reward = float(reward)
        scalar_wheel = float(wheel)
        if not np.isfinite(scalar_reward) or not np.isfinite(scalar_wheel):
            raise ValueError("reward and wheel must be finite")
        mouse = np.asarray(mouse_xy, dtype=np.float32).reshape(-1)
        cursor = np.asarray(cursor_xy, dtype=np.float32).reshape(-1)
        if mouse.size != 2 or cursor.size != 2 or not np.isfinite(mouse).all() or not np.isfinite(cursor).all():
            raise ValueError("mouse_xy and cursor_xy must each contain two finite values")
        n_keys = max(1, len(self._key_names))
        kh = np.zeros((n_keys,), dtype=np.uint8)
        if keys_held is not None:
            arr = np.asarray(keys_held, dtype=np.uint8).reshape(-1)
            if arr.size != n_keys:
                raise ValueError(f"keys_held has {arr.size} values; expected {n_keys}")
            kh[:] = arr
        ft = np.zeros((self._fact_dim,), dtype=np.float32)
        if facts is not None:
            arr = np.asarray(facts, dtype=np.float32).reshape(-1)
            if arr.size != self._fact_dim or not np.isfinite(arr).all():
                raise ValueError(f"facts must contain {self._fact_dim} finite values")
            ft[:] = arr
        from atomspike.models.iterator import TASK_AUX_DIM
        from atomspike.models.inventory import INV_SLOTS

        aux = np.zeros((TASK_AUX_DIM,), dtype=np.float32)
        if task_aux is not None:
            arr = np.asarray(task_aux, dtype=np.float32).reshape(-1)
            if arr.size != TASK_AUX_DIM or not np.isfinite(arr).all():
                raise ValueError(f"task_aux must contain {TASK_AUX_DIM} finite values")
            aux[:] = arr
        row = np.zeros((INV_SLOTS,), dtype=np.int64)
        if inv_ids is not None:
            arr = np.asarray(inv_ids, dtype=np.int64).reshape(-1)
            if arr.size != INV_SLOTS:
                raise ValueError(f"inv_ids has {arr.size} values; expected {INV_SLOTS}")
            row[:] = arr
        bp = np.zeros((2,), dtype=np.float32)
        if block_pulse is not None:
            arr = np.asarray(block_pulse, dtype=np.float32).reshape(-1)
            if arr.size != 2 or not np.isfinite(arr).all():
                raise ValueError("block_pulse must contain two finite values")
            bp[:] = np.clip(arr, 0.0, 1.0)
        from atomspike.data.visual_concepts import CONCEPT_KINDS, UNOBSERVED

        if isinstance(look_concept, bool) or int(look_concept) != look_concept or not UNOBSERVED <= int(look_concept) < len(CONCEPT_KINDS):
            raise ValueError("look_concept is outside the declared ontology")

        i = self.n
        for ds in (
            self.frames,
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.times,
            self.mouse,
            self.keys_held,
            self.facts,
            self.cursor,
            self.wheel,
            self.task_id,
            self.task_aux,
            self.inv_ids,
            self.hand_id,
            self.hover_id,
            self.carry_id,
            self.block_pulse,
            self.look_concept,
            self.recording_id,
            self.committed,
        ):
            ds.resize(i + 1, axis=0)
        self.committed[i] = np.uint8(0)
        try:
            self.frames[i] = fr
            self.states[i] = st
            self.actions[i] = act
            self.rewards[i] = scalar_reward
            self.dones[i] = np.uint8(1 if done else 0)
            self.times[i] = int(t_ns)
            self.mouse[i] = mouse
            self.keys_held[i] = kh
            self.facts[i] = ft
            self.cursor[i] = cursor
            self.wheel[i] = scalar_wheel
            self.task_id[i] = int(task_id)
            self.task_aux[i] = aux
            self.inv_ids[i] = row
            self.hand_id[i] = int(hand_id)
            self.hover_id[i] = int(hover_id)
            self.carry_id[i] = int(carry_id)
            self.block_pulse[i] = bp
            self.look_concept[i] = np.int16(look_concept)
            self.recording_id[i] = self._recording_id
        except Exception:
            # The marker remains zero, so the next append/reader will trim
            # this tail even if an HDF5 backend cannot shrink immediately.
            raise
        self.n += 1
        if self.n - self._last_flush >= 64:
            self._commit_pending()

    def _commit_pending(self) -> None:
        if self.n <= self._last_flush:
            return
        # Persist all payload datasets before publishing their valid markers.
        self._f.flush()
        self.committed[self._last_flush : self.n] = np.uint8(1)
        self._f.flush()
        self._last_flush = self.n

    def close(self) -> Path:
        if self.n > 0:
            self.dones[self.n - 1] = np.uint8(1)
        self._commit_pending()
        self._f.attrs["n"] = self._last_flush
        self._f.flush()
        self._f.close()
        return self.path


def _upgrade_actions_add_wheel(f: h5py.File, n_slots: int) -> None:
    """Rewrite actions from 12 to 13 slots using stored wheel extras."""
    old = np.asarray(f["actions"], dtype=np.int64)
    n = int(old.shape[0])
    wheel = np.asarray(f["wheel"], dtype=np.float64).reshape(-1)[:n] if "wheel" in f else np.zeros(n)
    n_bins = int(f.attrs.get("wheel_bins", 11) or 11)
    n_bins = n_bins if n_bins >= 3 else 11
    wheel_max = float(f.attrs.get("wheel_max", 4.0) or 4.0)
    center = n_bins // 2
    y = np.clip(wheel / max(1e-6, wheel_max), -1.0, 1.0)
    bins = np.clip(np.rint(y * center + center), 0, n_bins - 1).astype(np.int64).reshape(n, 1)
    new = np.concatenate([old, bins], axis=1)
    if "actions" in f:
        del f["actions"]
    f.create_dataset(
        "actions",
        data=new,
        maxshape=(None, n_slots),
        chunks=(256, n_slots),
        dtype="int64",
    )
    f.attrs["n_slots"] = int(n_slots)
    f.attrs["wheel_bins"] = int(n_bins)
    f.attrs["wheel_max"] = float(wheel_max)


def _ensure_recording_dataset(f: h5py.File, n: int) -> None:
    """Upgrade a legacy file and validate explicit collector-invocation IDs."""
    n = max(0, int(n))
    if "recording_id" not in f:
        f.create_dataset(
            "recording_id",
            data=np.zeros(n, dtype=np.int64),
            maxshape=(None,),
            chunks=(1024,),
            dtype="int64",
        )
    ids = f["recording_id"]
    if ids.ndim != 1 or int(ids.shape[0]) != n:
        raise ValueError(f"{f.filename}: recording_id must be a frame-aligned vector")
    if n and np.any(np.asarray(ids, dtype=np.int64) < 0):
        raise ValueError(f"{f.filename}: recording_id must be non-negative")
    f.attrs["recording_schema"] = 1


def _ensure_task_datasets(f: h5py.File, n: int) -> None:
    from atomspike.models.iterator import TASK_AUX_DIM

    n = max(0, int(n))
    if "task_id" not in f:
        f.create_dataset(
            "task_id",
            data=np.zeros(n, dtype=np.int64),
            maxshape=(None,),
            chunks=(1024,),
            dtype="int64",
        )
    if "task_aux" not in f:
        f.create_dataset(
            "task_aux",
            data=np.zeros((n, TASK_AUX_DIM), dtype=np.float32),
            maxshape=(None, TASK_AUX_DIM),
            chunks=(256, TASK_AUX_DIM),
            dtype="float32",
        )


def _ensure_inv_datasets(f: h5py.File, n: int) -> None:
    from atomspike.models.inventory import INV_SLOTS

    n = max(0, int(n))
    if "inv_ids" not in f:
        f.create_dataset(
            "inv_ids",
            data=np.zeros((n, INV_SLOTS), dtype=np.int64),
            maxshape=(None, INV_SLOTS),
            chunks=(256, INV_SLOTS),
            dtype="int64",
        )
    for name in ("hand_id", "hover_id", "carry_id"):
        if name not in f:
            f.create_dataset(
                name,
                data=np.zeros(n, dtype=np.int64),
                maxshape=(None,),
                chunks=(1024,),
                dtype="int64",
            )


def _ensure_block_pulse_dataset(f: h5py.File, n: int) -> None:
    n = max(0, int(n))
    if "block_pulse" not in f:
        f.create_dataset(
            "block_pulse",
            data=np.zeros((n, 2), dtype=np.float32),
            maxshape=(None, 2),
            chunks=(256, 2),
            dtype="float32",
        )


def _ensure_look_concept_dataset(f: h5py.File, n: int) -> None:
    """Legacy rows have no semantic teacher; never turn them into air."""
    if "look_concept" not in f:
        f.create_dataset(
            "look_concept",
            data=np.full(max(0, int(n)), -1, dtype=np.int16),
            maxshape=(None,),
            chunks=(1024,),
            dtype="int16",
        )
    f.attrs["look_concept_schema"] = "coarse_crosshair_v1_-1_unobserved"
