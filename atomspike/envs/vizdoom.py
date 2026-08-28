"""Optional ViZDoom wrapper (research env). Import is deferred so CPU smoke still runs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from atomspike.config import ActionConfig
from atomspike.envs.base import StepResult
from atomspike.models.actions import KEY_HOLD, KEY_PRESS, ActionSpec, ActionStateMachine


class ViZDoomEnv:
    spec_state_dim = 8

    def __init__(
        self,
        scenario: str = "basic",
        frame_size: int = 84,
        max_steps: int = 300,
        action_cfg: ActionConfig | None = None,
        **_: Any,
    ):
        try:
            import vizdoom as vzd
        except ImportError as exc:
            raise ImportError("ViZDoom is optional. pip install vizdoom") from exc
        self._vzd = vzd
        self.game = vzd.DoomGame()
        self.game.set_doom_scenario_path(scenario)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_window_visible(False)
        self.game.init()
        self.frame_size = frame_size
        self.max_steps = max_steps
        self.spec = ActionSpec.from_config(action_cfg or ActionConfig())
        self.machine = ActionStateMachine(self.spec)
        self._step_i = 0

    def close(self) -> None:
        self.game.close()

    def reset(self, seed: int | None = None) -> StepResult:
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        self.machine.reset()
        self._step_i = 0
        return self._observe(0.0, False, {})

    def step(self, tokens: NDArray[np.int64]) -> StepResult:
        import torch

        tok = np.asarray(tokens, dtype=np.int64).reshape(-1)
        self.machine.apply(torch.from_numpy(tok.copy()))
        buttons = self._buttons()
        reward = float(self.game.make_action(buttons, 4))
        self._step_i += 1
        done = bool(self.game.is_episode_finished() or self._step_i >= self.max_steps)
        return self._observe(reward, done, {"success": reward > 0})

    def _buttons(self) -> list[int]:
        # Map W/A/S/D + LMB onto a small ViZDoom button set when available.
        mapping = [
            self._vzd.Button.MOVE_FORWARD,
            self._vzd.Button.TURN_LEFT,
            self._vzd.Button.MOVE_BACKWARD,
            self._vzd.Button.TURN_RIGHT,
        ]
        avail = list(self.game.get_available_buttons())
        action = [0] * len(avail)
        for i, btn in enumerate(mapping):
            if i < self.spec.n_keys and self.machine.key_down[i] and btn in avail:
                action[avail.index(btn)] = 1
        attack = getattr(self._vzd.Button, "ATTACK", None)
        if attack in avail and self.machine.btn_down[0]:
            action[avail.index(attack)] = 1
        return action

    def _observe(self, reward: float, done: bool, info: dict[str, Any]) -> StepResult:
        if done:
            frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            state = np.zeros(8, dtype=np.float32)
            return StepResult(frame, state, reward, True, info)
        st = self.game.get_state()
        raw = np.transpose(st.screen_buffer, (1, 2, 0)) if st.screen_buffer.ndim == 3 else st.screen_buffer
        frame = _resize_rgb(raw, self.frame_size)
        game_state = np.zeros(8, dtype=np.float32)
        game_vars = list(st.game_variables) if st.game_variables is not None else []
        for i, v in enumerate(game_vars[:8]):
            game_state[i] = float(v)
        return StepResult(frame, game_state, reward, done, info)


def _resize_rgb(frame: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    from PIL import Image

    img = Image.fromarray(frame)
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)
