"""T0-T6 pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import collect_expert_demos
from atomspike.data.rabc import rabc_weights
from atomspike.data.io import H5DemoDataset
from atomspike.envs import make_env
from atomspike.train.bc import train_bc
from atomspike.train.offline_rl import train_offline_rl
from atomspike.train.peft_train import train_peft
from atomspike.convert.pmsm import convert_checkpoint


def run_pipeline(
    cfg: AtomSpikeConfig,
    workdir: str | Path,
    episodes: int = 40,
    skip_rl: bool = False,
    skip_peft: bool = False,
    skip_convert: bool = False,
) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        cfg.env.kind,
        frame_size=cfg.env.frame_size,
        max_steps=cfg.env.max_steps,
        seed=cfg.env.seed,
        action_cfg=cfg.action,
    )
    demo = collect_expert_demos(env, workdir / "demos.h5", episodes=episodes, seed=cfg.train.seed)
    ds = H5DemoDataset(demo)
    weights = rabc_weights(ds)
    bc = train_bc(cfg, demo, workdir / "bc.pt", weights=weights)
    result: dict = {"demo": str(demo), "bc": bc}
    teacher = bc["checkpoint"]
    if not skip_rl:
        result["rl"] = train_offline_rl(cfg, demo, teacher, workdir / "rl.pt", epochs=max(1, cfg.train.epochs // 2))
        teacher = result["rl"]["checkpoint"]
    if not skip_peft:
        result["peft"] = train_peft(cfg, demo, teacher, workdir / "peft.pt", epochs=max(1, cfg.train.epochs // 2))
        teacher = result["peft"]["checkpoint"]
    if not skip_convert:
        result["snn"] = convert_checkpoint(cfg, teacher, workdir / "snn.pt", data_path=demo)
    env.close()
    return result
