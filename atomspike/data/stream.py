"""Append-only HDF5 writer so a 30fps Minecraft session is not held in RAM."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray


def _canon_key_names(names) -> str:
    if not names:
        return ""
    if isinstance(names, (bytes, bytearray)):
        names = names.decode("utf-8")
    if isinstance(names, str):
        parts = [p.strip() for p in names.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in names]
    return ",".join("E" if p.upper() == "F" else p for p in parts)


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
        self._f.attrs["frame_size_h"] = self.h
        self._f.attrs["frame_size_w"] = self.w
        self._f.attrs["state_dim"] = self.state_dim
        self._f.attrs["n_slots"] = self.n_slots
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
            checks = {
                "frame_size_h": self.h,
                "frame_size_w": self.w,
                "state_dim": self.state_dim,
                "n_slots": self.n_slots,
            }
            for k, v in checks.items():
                if int(f.attrs.get(k, -1)) != int(v):
                    raise ValueError(
                        f"{self.path} was recorded with {k}={f.attrs.get(k)}, current config wants {v}. "
                        "Use a new --out file or --overwrite."
                    )
            if key_names and _canon_key_names(f.attrs.get("key_names", "")) != _canon_key_names(key_names):
                raise ValueError(f"{self.path} key_names differ; use a new --out file or --overwrite")
            if key_names:
                f.attrs["key_names"] = ",".join(key_names)
            if state_fields and f.attrs.get("state_fields", "") != ",".join(state_fields):
                raise ValueError(f"{self.path} state_fields differ; use a new --out file or --overwrite")
            self.frames = f["frames"]
            self.states = f["game_state"]
            self.actions = f["actions"]
            self.rewards = f["rewards"]
            self.dones = f["dones"]
            self.times = f["timestamps_ns"]
            self.mouse = f["mouse_xy"]
            self.keys_held = f["keys_held"]
            self.facts = f["facts"]
            self.cursor = f["cursor_xy"]
            self.wheel = f["wheel"]
        except Exception:
            f.close()
            raise
        self._f = f
        self.n = int(self.frames.shape[0])
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
    ) -> None:
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
        ):
            ds.resize(i + 1, axis=0)
        fr = np.asarray(frame, dtype=np.uint8)
        if fr.shape[0] != self.h or fr.shape[1] != self.w:
            raise ValueError(f"frame shape {fr.shape} != {(self.h, self.w, 3)}")
        self.frames[i] = fr
        self.states[i] = np.asarray(state, dtype=np.float32).reshape(-1)[: self.state_dim]
        self.actions[i] = np.asarray(action, dtype=np.int64).reshape(-1)[: self.n_slots]
        self.rewards[i] = float(reward)
        self.dones[i] = np.uint8(1 if done else 0)
        self.times[i] = int(t_ns)
        self.mouse[i] = np.asarray(mouse_xy, dtype=np.float32)
        kh = np.zeros((max(1, len(self._key_names)),), dtype=np.uint8)
        if keys_held is not None:
            kh[: min(kh.size, keys_held.size)] = np.asarray(keys_held, dtype=np.uint8).reshape(-1)[: kh.size]
        self.keys_held[i] = kh
        ft = np.zeros((self._fact_dim,), dtype=np.float32)
        if facts is not None:
            arr = np.asarray(facts, dtype=np.float32).reshape(-1)
            ft[: min(ft.size, arr.size)] = arr[: ft.size]
        self.facts[i] = ft
        self.cursor[i] = np.asarray(cursor_xy, dtype=np.float32)
        self.wheel[i] = float(wheel)
        self.n += 1
        if self.n - self._last_flush >= 64:
            self._f.flush()
            self._last_flush = self.n

    def close(self) -> Path:
        if self.n > 0:
            self.dones[self.n - 1] = np.uint8(1)
        self._f.attrs["n"] = self.n
        self._f.flush()
        self._f.close()
        return self.path
