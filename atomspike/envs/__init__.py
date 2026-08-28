"""Environment protocol + factory."""

from atomspike.envs.base import GameEnv
from atomspike.envs.synthetic import SyntheticAimEnv

__all__ = ["GameEnv", "SyntheticAimEnv", "make_env"]


def make_env(kind: str, **kwargs) -> GameEnv:
    kind = kind.lower()
    if kind in {"synthetic", "aim", "synthetic_aim"}:
        return SyntheticAimEnv(**kwargs)
    if kind in {"vizdoom", "doom"}:
        from atomspike.envs.vizdoom import ViZDoomEnv

        return ViZDoomEnv(**kwargs)
    raise ValueError(f"unknown env kind: {kind}")
