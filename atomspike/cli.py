"""AtomSpike CLI: collect, train, convert, eval, play, smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _cfg(args) -> "AtomSpikeConfig":
    from atomspike.config import load_config

    path = getattr(args, "config", None)
    if path:
        return load_config(path)
    default = ROOT / "configs" / "default.yaml"
    if default.exists():
        return load_config(default)
    return load_config(None)


def cmd_collect(args) -> int:
    from atomspike.data.io import collect_expert_demos
    from atomspike.envs import make_env

    cfg = _cfg(args)
    env = make_env(
        cfg.env.kind,
        frame_size=cfg.env.frame_size,
        max_steps=cfg.env.max_steps,
        seed=cfg.env.seed,
        action_cfg=cfg.action,
    )
    path = collect_expert_demos(env, args.out, episodes=args.episodes, seed=cfg.train.seed)
    env.close()
    print(path)
    return 0


def cmd_train_bc(args) -> int:
    from atomspike.data.io import H5DemoDataset
    from atomspike.data.rabc import rabc_weights
    from atomspike.train.bc import train_bc

    cfg = _cfg(args)
    weights = None
    if args.rabc:
        weights = rabc_weights(H5DemoDataset(args.data))
    result = train_bc(cfg, args.data, args.out, weights=weights)
    print(json.dumps(result, indent=2))
    return 0


def cmd_train_rl(args) -> int:
    from atomspike.train.offline_rl import train_offline_rl

    cfg = _cfg(args)
    result = train_offline_rl(cfg, args.data, args.teacher, args.out)
    print(json.dumps(result, indent=2))
    return 0


def cmd_train_peft(args) -> int:
    from atomspike.train.peft_train import train_peft

    cfg = _cfg(args)
    result = train_peft(cfg, args.data, args.backbone, args.out)
    print(json.dumps(result, indent=2))
    return 0


def cmd_convert(args) -> int:
    from atomspike.convert.pmsm import convert_checkpoint
    from atomspike.convert.spiked_attention import convert_spiked_attention

    cfg = _cfg(args)
    if args.method == "spiked_attention":
        result = convert_spiked_attention(cfg, args.ckpt, args.out)
    else:
        result = convert_checkpoint(cfg, args.ckpt, args.out, data_path=args.data)
    print(json.dumps(result, indent=2))
    return 0


def cmd_eval(args) -> int:
    from atomspike.eval.metrics import evaluate_policy

    cfg = _cfg(args)
    result = evaluate_policy(cfg, args.ckpt, episodes=args.episodes)
    print(json.dumps(result, indent=2))
    return 0


def cmd_play(args) -> int:
    from atomspike.eval.metrics import evaluate_policy

    if args.inject:
        print(
            "Refusing OS inject in play. This project only runs synthetic / "
            "ViZDoom research envs. Do not attach to online games.",
            file=sys.stderr,
        )
        return 2
    cfg = _cfg(args)
    result = evaluate_policy(cfg, args.ckpt, episodes=args.episodes, deterministic=not args.sample)
    print(json.dumps(result, indent=2))
    return 0


def cmd_pipeline(args) -> int:
    from atomspike.train.pipeline import run_pipeline

    cfg = _cfg(args)
    result = run_pipeline(
        cfg,
        args.workdir,
        episodes=args.episodes,
        skip_rl=args.skip_rl,
        skip_peft=not args.peft,
        skip_convert=args.skip_convert,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_smoke(args) -> int:
    from atomspike.config import load_config
    from atomspike.eval.metrics import evaluate_policy
    from atomspike.train.pipeline import run_pipeline

    cfg_path = ROOT / "configs" / "smoke.yaml"
    cfg = load_config(cfg_path if cfg_path.exists() else None)
    workdir = Path(args.workdir)
    result = run_pipeline(
        cfg,
        workdir,
        episodes=args.episodes,
        skip_rl=False,
        skip_peft=True,
        skip_convert=False,
    )
    metrics = evaluate_policy(cfg, result["bc"]["checkpoint"], episodes=8)
    metrics_rl = None
    if "rl" in result:
        metrics_rl = evaluate_policy(cfg, result["rl"]["checkpoint"], episodes=8)
    metrics_snn = None
    if "snn" in result:
        metrics_snn = evaluate_policy(cfg, result["snn"]["checkpoint"], episodes=8)
    report = {
        "pipeline": result,
        "eval_bc": metrics,
        "eval_rl": metrics_rl,
        "eval_snn": metrics_snn,
        "ok": metrics["latency_p95_ms"] >= 0 and result["bc"]["params"] > 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_info(args) -> int:
    from atomspike.config import dump_config
    from atomspike.models.agent import AtomSpikeAgent

    cfg = _cfg(args)
    agent = AtomSpikeAgent(cfg)
    info = {
        "params": agent.parameter_count(),
        "d_model": cfg.encoder.d_model,
        "action_slots": cfg.action.n_slots,
        "slot_vocabs": list(agent.spec.slot_vocabs),
        "perception_every_n": cfg.dual_rate.perception_every_n,
        "policy_kind": cfg.policy.kind,
        "action_decode": cfg.policy.action_decode,
        "config": dump_config(cfg),
    }
    print(json.dumps(info, indent=2, default=_json_default))
    return 0


def _add_config_arg(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    default: object = argparse.SUPPRESS if suppress else None
    parser.add_argument("--config", type=str, default=default, help="YAML config path")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atomspike", description="AtomSpike atomic-level game agent")
    _add_config_arg(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="print model size and config")
    _add_config_arg(s, suppress=True)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("collect", help="T0: record scripted/human demos to HDF5")
    _add_config_arg(s, suppress=True)
    s.add_argument("--out", required=True)
    s.add_argument("--episodes", type=int, default=40)
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("train-bc", help="T1: behavior cloning warm-start")
    _add_config_arg(s, suppress=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rabc", action="store_true", help="T2 RA-BC + coverage reweight")
    s.set_defaults(func=cmd_train_bc)

    s = sub.add_parser("train-rl", help="T3: advantage-weighted offline RL")
    _add_config_arg(s, suppress=True)
    s.add_argument("--data", required=True)
    s.add_argument("--teacher", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_train_rl)

    s = sub.add_parser("train-peft", help="T4: freeze backbone + LoRA")
    _add_config_arg(s, suppress=True)
    s.add_argument("--data", required=True)
    s.add_argument("--backbone", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_train_peft)

    s = sub.add_parser("convert", help="T6: training-free ANN to SNN")
    _add_config_arg(s, suppress=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--data", default=None)
    s.add_argument("--method", choices=["pmsm", "spiked_attention"], default="pmsm")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("eval", help="rollout success / latency / token acc")
    _add_config_arg(s, suppress=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--episodes", type=int, default=12)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("play", help="closed-loop rollout in the research env")
    _add_config_arg(s, suppress=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--episodes", type=int, default=4)
    s.add_argument("--sample", action="store_true")
    s.add_argument("--inject", action="store_true", help="OS inject (refused)")
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("pipeline", help="run T0-T6 on the synthetic env")
    _add_config_arg(s, suppress=True)
    s.add_argument("--workdir", default="runs/default")
    s.add_argument("--episodes", type=int, default=40)
    s.add_argument("--skip-rl", action="store_true")
    s.add_argument("--peft", action="store_true")
    s.add_argument("--skip-convert", action="store_true")
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("smoke", help="tiny CPU end-to-end sanity check")
    _add_config_arg(s, suppress=True)
    s.add_argument("--workdir", default="runs/smoke")
    s.add_argument("--episodes", type=int, default=16)
    s.set_defaults(func=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
