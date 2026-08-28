"""Research-only synthetic FPS-like task: move, aim, click a target.

No commercial game, no OS injection. Used to close the full T0-T6 loop
on a laptop CPU. Frame is numpy-rendered RGB; game_state holds coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atomspike.config import ActionConfig
from atomspike.envs.base import StepResult
from atomspike.models.actions import (
    KEY_HOLD,
    KEY_IDLE,
    KEY_PRESS,
    KEY_RELEASE,
    ActionSpec,
    ActionStateMachine,
)


@dataclass
class _Sim:
    body: NDArray[np.float32]
    aim: NDArray[np.float32]
    target: NDArray[np.float32]
    hp: float
    ammo: float


class SyntheticAimEnv:
    spec_state_dim = 8

    def __init__(
        self,
        frame_size: int = 84,
        max_steps: int = 128,
        seed: int = 0,
        action_cfg: ActionConfig | None = None,
        move_speed: float = 0.045,
        aim_speed: float = 0.05,
        hit_radius: float = 0.08,
        body_radius: float = 0.18,
    ):
        self.frame_size = int(frame_size)
        self.max_steps = int(max_steps)
        self.move_speed = move_speed
        self.aim_speed = aim_speed
        self.hit_radius = hit_radius
        self.body_radius = body_radius
        self.spec = ActionSpec.from_config(action_cfg or ActionConfig())
        self.machine = ActionStateMachine(self.spec)
        self.rng = np.random.default_rng(seed)
        self._step_i = 0
        self._sim = self._spawn()
        self._prev_aim_dist = 1.0
        self._success = False

    def close(self) -> None:
        return None

    def reset(self, seed: int | None = None) -> StepResult:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.machine.reset()
        self._step_i = 0
        self._success = False
        self._sim = self._spawn()
        self._prev_aim_dist = self._aim_dist()
        return self._observe(0.0, False, {"success": False})

    def step(self, tokens: NDArray[np.int64]) -> StepResult:
        tok = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if tok.size < self.spec.n_slots:
            raise ValueError("token vector too short")
        self._apply(tok)
        self._step_i += 1
        reward, done, info = self._score()
        return self._observe(reward, done, info)

    def expert_tokens(self) -> NDArray[np.int64]:
        """Scripted expert used to generate T0 demonstrations."""
        sim = self._sim
        delta_body = sim.target - sim.body
        delta_aim = sim.target - sim.aim
        keys = [KEY_IDLE] * self.spec.n_keys
        # W/A/S/D as up/left/down/right in normalized [0,1] coords (y down).
        mapping = [
            (0, delta_body[1] < -0.02),  # W up
            (1, delta_body[0] < -0.02),  # A left
            (2, delta_body[1] > 0.02),  # S down
            (3, delta_body[0] > 0.02),  # D right
        ]
        for idx, need in mapping:
            if idx >= self.spec.n_keys:
                continue
            if need:
                keys[idx] = KEY_HOLD if self.machine.key_down[idx] else KEY_PRESS
            elif self.machine.key_down[idx]:
                keys[idx] = KEY_RELEASE
        dx = float(np.clip(delta_aim[0] / self.aim_speed, -10, 10))
        dy = float(np.clip(delta_aim[1] / self.aim_speed, -10, 10))
        # add small noise so BC is not a delta function
        dx += float(self.rng.normal(0, 0.4))
        dy += float(self.rng.normal(0, 0.4))
        mx = self.spec.mouse_delta_to_bin(dx)
        my = self.spec.mouse_delta_to_bin(dy)
        lmb = KEY_IDLE
        if self._aim_dist() < self.hit_radius and self._body_dist() < self.body_radius:
            lmb = KEY_HOLD if self.machine.btn_down[0] else KEY_PRESS
        elif self.machine.btn_down[0]:
            lmb = KEY_RELEASE
        rmb = KEY_RELEASE if self.machine.btn_down[1] else KEY_IDLE
        out = np.array(keys + [mx, my, lmb, rmb], dtype=np.int64)
        return out

    def _spawn(self) -> _Sim:
        body = self.rng.uniform(0.15, 0.85, size=2).astype(np.float32)
        aim = body + self.rng.uniform(-0.1, 0.1, size=2).astype(np.float32)
        target = self.rng.uniform(0.15, 0.85, size=2).astype(np.float32)
        while float(np.linalg.norm(target - body)) < 0.25:
            target = self.rng.uniform(0.15, 0.85, size=2).astype(np.float32)
        return _Sim(body=body, aim=np.clip(aim, 0, 1), target=target, hp=1.0, ammo=1.0)

    def _apply(self, tok: NDArray[np.int64]) -> None:
        import torch

        self.machine.apply(torch.from_numpy(tok.copy()))
        sim = self._sim
        # movement
        if self.machine.key_down[0]:
            sim.body[1] -= self.move_speed
        if self.machine.key_down[2] and self.spec.n_keys > 2:
            sim.body[1] += self.move_speed
        if self.spec.n_keys > 1 and self.machine.key_down[1]:
            sim.body[0] -= self.move_speed
        if self.spec.n_keys > 3 and self.machine.key_down[3]:
            sim.body[0] += self.move_speed
        sim.body = np.clip(sim.body, 0.02, 0.98)
        dx = self.spec.bin_to_mouse_delta(int(tok[self.spec.n_keys]))
        dy = self.spec.bin_to_mouse_delta(int(tok[self.spec.n_keys + 1]))
        sim.aim[0] += dx * self.aim_speed * 0.1
        sim.aim[1] += dy * self.aim_speed * 0.1
        sim.aim = np.clip(sim.aim, 0.0, 1.0)

    def _aim_dist(self) -> float:
        return float(np.linalg.norm(self._sim.target - self._sim.aim))

    def _body_dist(self) -> float:
        return float(np.linalg.norm(self._sim.target - self._sim.body))

    def _score(self) -> tuple[float, bool, dict[str, Any]]:
        aim_d = self._aim_dist()
        body_d = self._body_dist()
        reward = (self._prev_aim_dist - aim_d) * 2.0 + (0.5 - body_d) * 0.05
        self._prev_aim_dist = aim_d
        clicked = self.machine.btn_down[0]
        success = clicked and aim_d < self.hit_radius and body_d < self.body_radius
        if success:
            reward += 1.0
            self._success = True
        done = success or self._step_i >= self.max_steps
        return float(reward), bool(done), {"success": bool(self._success), "aim_dist": aim_d}

    def _observe(self, reward: float, done: bool, info: dict[str, Any]) -> StepResult:
        return StepResult(
            frame=self._render(),
            game_state=self._state_vec(),
            reward=reward,
            done=done,
            info=info,
        )

    def _state_vec(self) -> NDArray[np.float32]:
        sim = self._sim
        t = self._step_i / max(1, self.max_steps)
        return np.array(
            [
                sim.body[0],
                sim.body[1],
                sim.aim[0],
                sim.aim[1],
                sim.target[0],
                sim.target[1],
                sim.hp,
                t,
            ],
            dtype=np.float32,
        )

    def _render(self) -> NDArray[np.uint8]:
        n = self.frame_size
        img = np.zeros((n, n, 3), dtype=np.uint8)
        img[:, :] = (18, 20, 28)
        self._dot(img, self._sim.target, (40, 220, 90), r=max(2, n // 18))
        self._dot(img, self._sim.body, (70, 140, 255), r=max(2, n // 22))
        self._cross(img, self._sim.aim, (240, 240, 240))
        return img

    def _dot(self, img: NDArray[np.uint8], xy: NDArray[np.float32], color: tuple[int, int, int], r: int) -> None:
        n = img.shape[0]
        cx, cy = int(xy[0] * (n - 1)), int(xy[1] * (n - 1))
        y0, y1 = max(0, cy - r), min(n, cy + r + 1)
        x0, x1 = max(0, cx - r), min(n, cx + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[y0:y1, x0:x1][mask] = color

    def _cross(self, img: NDArray[np.uint8], xy: NDArray[np.float32], color: tuple[int, int, int]) -> None:
        n = img.shape[0]
        cx, cy = int(xy[0] * (n - 1)), int(xy[1] * (n - 1))
        arm = max(2, n // 16)
        img[max(0, cy - 1) : min(n, cy + 2), max(0, cx - arm) : min(n, cx + arm + 1)] = color
        img[max(0, cy - arm) : min(n, cy + arm + 1), max(0, cx - 1) : min(n, cx + 2)] = color
