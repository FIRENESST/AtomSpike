"""Episode cuts, independent recording IDs, and per-segment clocks.

An F8 mark closes an episode so prev-frame, future-state, and time_frac never
cross it. Separately, validation uses collector-invocation IDs (with timestamp
gaps as a legacy fallback), so an F8 press cannot leak adjacent video frames
across train and validation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

CLOCK_PERIODS_S = (30.0, 120.0, 480.0)
CLOCK_DIM = 2 + 2 * len(CLOCK_PERIODS_S)
DEFAULT_GAP_S = 2.0


def apply_recording_boundaries(
    dones: NDArray | list,
    timestamps_ns: NDArray | None = None,
    *,
    gap_s: float = DEFAULT_GAP_S,
) -> NDArray[np.uint8]:
    """Keep F8 marks; also cut on timestamp gaps / clock resets; close the last frame."""
    out = np.asarray(dones, dtype=np.uint8).reshape(-1).copy()
    n = int(out.shape[0])
    if n == 0:
        return out
    if timestamps_ns is not None:
        t = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
        if t.shape[0] != n:
            raise ValueError(f"timestamps length {t.shape[0]} != dones length {n}")
        dt = np.diff(t.astype(np.float64))
        gap_ns = max(0.05, float(gap_s)) * 1e9
        out[np.where(dt > gap_ns)[0]] = 1
        out[np.where(dt <= 0)[0]] = 1
    out[-1] = 1
    return out


def segment_ids(dones: NDArray) -> NDArray[np.int32]:
    """0-based id; increments after a boundary frame."""
    ends = np.asarray(dones, dtype=np.uint8).reshape(-1) > 0
    n = int(ends.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    carry = np.concatenate([[0], ends[:-1].astype(np.int32)])
    return np.cumsum(carry, dtype=np.int32)


def session_lengths(ids: NDArray[np.int32]) -> list[int]:
    if ids.size == 0:
        return []
    counts = np.bincount(ids.astype(np.int64))
    return [int(c) for c in counts if c > 0]


def clock_vector(time_frac: float, session_t: float, dt_s: float) -> NDArray[np.float32]:
    """Session-local clock: progress, frame dt, and Fourier time-of-session."""
    frac = float(np.clip(time_frac, 0.0, 1.0))
    dt_norm = float(np.clip(float(dt_s) * 30.0, 0.0, 4.0) / 4.0)
    t = max(0.0, float(session_t))
    feats = [frac, dt_norm]
    for period in CLOCK_PERIODS_S:
        ang = 2.0 * math.pi * t / period
        feats.append(math.sin(ang))
        feats.append(math.cos(ang))
    return np.asarray(feats, dtype=np.float32)


def infer_time_frac(session_t: float, tau_s: float = 180.0) -> float:
    """Play-time stand-in for demo time_frac (saturates instead of needing episode length)."""
    tau = max(1.0, float(tau_s))
    return float(1.0 - math.exp(-max(0.0, float(session_t)) / tau))


def clock_from_runtime(session_t: float, dt_s: float, *, tau_s: float = 180.0) -> NDArray[np.float32]:
    return clock_vector(infer_time_frac(session_t, tau_s=tau_s), session_t, dt_s)


def recording_ids(
    n: int,
    timestamps_ns: NDArray | None = None,
    *,
    dones: NDArray | None = None,
    explicit_ids: NDArray | None = None,
    forced_starts: NDArray | None = None,
    gap_s: float = DEFAULT_GAP_S,
) -> NDArray[np.int32]:
    """Identify independent recordings without treating in-run F8 marks as recordings.

    Current Minecraft files carry timestamps and an explicit ID written once
    per collector invocation. Timestamp gaps remain a recovery fallback for
    older files. Generic synthetic datasets have no timestamps, so only there
    may terminal ``dones`` stand in for independent trajectories.
    """
    n = max(0, int(n))
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    starts = np.zeros(n, dtype=bool)
    starts[0] = True
    if timestamps_ns is not None:
        times = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
        if times.shape[0] != n:
            raise ValueError("timestamps length mismatch")
        delta = np.diff(times.astype(np.float64))
        starts[1:] |= (delta > max(0.05, float(gap_s)) * 1e9) | (delta <= 0.0)
    elif dones is not None:
        ends = np.asarray(dones, dtype=np.uint8).reshape(-1)
        if ends.shape[0] != n:
            raise ValueError("dones length mismatch")
        starts[1:] |= ends[:-1] > 0
    if explicit_ids is not None:
        raw = np.asarray(explicit_ids, dtype=np.int64).reshape(-1)
        if raw.shape[0] != n:
            raise ValueError("recording_id length mismatch")
        if np.any(raw < 0):
            raise ValueError("recording_id must be non-negative")
        starts[1:] |= raw[1:] != raw[:-1]
    if forced_starts is not None:
        forced = np.asarray(forced_starts, dtype=bool).reshape(-1)
        if forced.shape[0] != n:
            raise ValueError("forced recording-start length mismatch")
        starts |= forced
    return (np.cumsum(starts, dtype=np.int32) - 1).astype(np.int32, copy=False)


def stride_split_segments(
    n: int,
    val_frac: float,
    segment_id: NDArray | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split by complete recordings so validation cannot see adjacent train frames.

    A video frame is far too correlated with its neighbours for a frame-stride
    split to be a meaningful validation set.  ``segment_id`` therefore denotes
    a recording/episode, and whole trailing segments are reserved for
    validation.  A single recording has no honest held-out split; callers get
    an empty validation array and should report that fact instead of quietly
    evaluating on training examples.
    """
    if segment_id is None:
        # There is no provenance that lets us prevent temporal leakage.  Keep
        # every example for training rather than manufacture a misleading val
        # score.
        return np.arange(n, dtype=np.int64), np.zeros((0,), dtype=np.int64)
    ids = np.asarray(segment_id).reshape(-1)
    if ids.shape[0] != n:
        raise ValueError("segment_id length mismatch")
    ordered = np.unique(ids)
    if val_frac <= 0.0 or ordered.size < 2:
        return np.arange(n, dtype=np.int64), np.zeros((0,), dtype=np.int64)
    n_val_segments = min(
        int(ordered.size) - 1,
        max(1, int(math.ceil(float(val_frac) * int(ordered.size)))),
    )
    val_ids = ordered[-n_val_segments:]
    val_mask = np.isin(ids, val_ids)
    val = np.flatnonzero(val_mask).astype(np.int64)
    train = np.flatnonzero(~val_mask).astype(np.int64)
    if train.size == 0 or val.size == 0:
        # Defensive fallback for malformed ids; never return an all-val split.
        idx = np.arange(n, dtype=np.int64)
        return idx, np.zeros((0,), dtype=np.int64)
    return train, val


def build_episode_timeline(
    states: object,
    dones: object,
    *,
    n_phases: int,
    future_steps: int,
    timestamps_ns: object | None = None,
    gap_s: float = DEFAULT_GAP_S,
    future_near: int = 5,
    future_far: int = 45,
    fps: float = 30.0,
) -> dict[str, object]:
    """Per-episode clocks and near-future game_state (never across an episode cut)."""
    states = np.asarray(states)
    raw_dones = np.asarray(dones).reshape(-1)
    n = int(states.shape[0])
    times = None if timestamps_ns is None else np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    cuts = apply_recording_boundaries(raw_dones, times, gap_s=gap_s)
    n_phases = max(1, int(n_phases))
    future_steps = max(1, int(future_steps))
    near = max(1, int(future_near))
    far = max(near, int(future_far))
    fps = max(1.0, float(fps))
    period = 1.0 / fps

    time_frac = np.zeros(n, dtype=np.float32)
    session_t = np.zeros(n, dtype=np.float32)
    dt_s = np.full(n, period, dtype=np.float32)
    phase = np.zeros(n, dtype=np.int64)
    next_phase = np.zeros(n, dtype=np.int64)
    future_state = np.array(states, dtype=np.float32, copy=True)
    future_near_s = np.array(states, dtype=np.float32, copy=True)
    future_far_s = np.array(states, dtype=np.float32, copy=True)
    clock = np.zeros((n, CLOCK_DIM), dtype=np.float32)
    ids = segment_ids(cuts)

    start = 0
    for i in range(n):
        if not bool(cuts[i]):
            continue
        end = i + 1
        length = end - start
        if length <= 0:
            start = end
            continue
        if length == 1:
            clock[start] = clock_vector(0.0, 0.0, period)
            start = end
            continue
        frac = np.linspace(0.0, 1.0, length, dtype=np.float32)
        time_frac[start:end] = frac
        ph = np.minimum((frac * n_phases).astype(np.int64), n_phases - 1)
        phase[start:end] = ph
        nxt = np.minimum(ph + 1, n_phases - 1)
        nxt[-1] = ph[-1]
        next_phase[start:end] = nxt
        if times is not None:
            tseg = times[start:end].astype(np.float64)
            session_t[start:end] = ((tseg - tseg[0]) / 1e9).astype(np.float32)
            dlt = np.diff(tseg) / 1e9
            dt_s[start] = period
            dt_s[start + 1 : end] = np.clip(dlt, 1e-4, 4.0).astype(np.float32)
        else:
            session_t[start:end] = (np.arange(length, dtype=np.float32) * period)
            dt_s[start:end] = period
        for off, dest in ((future_steps, future_state), (near, future_near_s), (far, future_far_s)):
            fut = np.minimum(np.arange(length) + off, length - 1)
            dest[start:end] = states[start + fut]
        for j in range(length):
            clock[start + j] = clock_vector(float(frac[j]), float(session_t[start + j]), float(dt_s[start + j]))
        start = end

    return {
        "dones": cuts,
        "session_id": ids,
        "time_frac": time_frac,
        "session_t": session_t,
        "dt_s": dt_s,
        "phase": phase,
        "next_phase": next_phase,
        "future_state": future_state,
        "future_state_near": future_near_s,
        "future_state_far": future_far_s,
        "clock": clock,
        "n_sessions": int(ids.max()) + 1 if n else 0,
        "session_lengths": session_lengths(ids),
    }
