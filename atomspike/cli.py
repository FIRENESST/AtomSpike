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


def _require_ckpt(path: str | Path) -> Path:
    ckpt = Path(path)
    if ckpt.is_file():
        return ckpt
    print(
        f"找不到权重：{ckpt}\n"
        "清过数据之后还没有 Minecraft 模型。先采集再训练：\n"
        "  python -m atomspike collect-mc --config configs/minecraft.yaml --out runs/mc.h5\n"
        "  python -m atomspike train-bc --data runs/mc.h5 --out runs/bc.pt --rabc\n"
        "然后再：\n"
        "  python -m atomspike predict-mc --config configs/minecraft.yaml --ckpt runs/bc.pt --seconds 30",
        file=sys.stderr,
    )
    raise SystemExit(2)


def cmd_collect(args) -> int:
    from atomspike.data.io import collect_expert_demos
    from atomspike.envs import make_env

    cfg = _cfg(args)
    if cfg.env.kind.lower() in {"minecraft", "mc", "minerl"}:
        print(
            "Minecraft T0 is the 30fps human collector. Example:\n"
            "  python -m atomspike collect-mc --out runs/mc.h5 --seconds 120",
            file=sys.stderr,
        )
        return 2
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


def cmd_collect_mc(args) -> int:
    from atomspike.capture.mc_collector import run_minecraft_collector
    from atomspike.capture.window import list_windows

    cfg = _cfg(args)
    if getattr(args, "list_windows", False):
        hits = list_windows(args.title or "")
        for w in hits:
            print(f"{w.width}x{w.height}\t{w.left},{w.top}\t{w.title}")
        if not hits:
            print("(no matching windows)")
        return 0
    region = None
    if args.region:
        parts = [int(x) for x in args.region.split(",")]
        if len(parts) != 4:
            print("--region must be left,top,width,height", file=sys.stderr)
            return 2
        region = (parts[0], parts[1], parts[2], parts[3])
    if args.title:
        cfg.minecraft.window_title = args.title
        cfg.capture.window_title = args.title
    if getattr(args, "saves_dir", None):
        cfg.minecraft.saves_dir = args.saves_dir
    seconds = args.seconds
    if args.dry_run and seconds is None:
        seconds = 2.0
    print(
        "Recording local Minecraft at "
        f"{cfg.capture.fps} fps. Stop={cfg.minecraft.stop_key}  mark-episode={cfg.minecraft.mark_key}. "
        f"UDP sidecar {cfg.minecraft.sidecar_host}:{cfg.minecraft.sidecar_udp_port} (127.0.0.1 JSON only). "
        "A capture preview window shows the stored frames. Singleplayer / LAN research only.",
        file=sys.stderr,
    )
    try:
        result = run_minecraft_collector(
            cfg,
            args.out,
            seconds=seconds,
            dry_run=args.dry_run,
            desktop=args.desktop,
            region=region,
            log_every_s=0.0 if args.quiet else args.log_every,
            append=not getattr(args, "overwrite", False),
            preview=(
                True
                if getattr(args, "preview", False)
                else (False if getattr(args, "no_preview", False) or args.dry_run else None)
            ),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if result.get("n_frames", 0) > 0 else 1


def cmd_sidecar_send(args) -> int:
    """Push sample JSON to the collector UDP port (localhost only)."""
    import time

    from atomspike.capture.sidecar import SIDECAR_HOST, SIDECAR_PORT, send_sidecar

    if args.json:
        payload = json.loads(args.json)
        if not isinstance(payload, dict):
            print("--json must be a JSON object", file=sys.stderr)
            return 2
    else:
        payload = {
            "hp": 20.0,
            "max_hp": 20.0,
            "hunger": 18,
            "air": 300,
            "xp_progress": 0.4,
            "x": 12.5,
            "y": 64.0,
            "z": -8.1,
            "yaw": 90.0,
            "pitch": 12.0,
            "vx": 0.1,
            "vy": 0.0,
            "vz": 0.0,
            "hotbar_slot": 0,
            "selected_count": 1,
            "on_ground": True,
            "sneaking": False,
            "sprinting": True,
            "in_water": False,
            "gui_open": False,
            "mouse_nx": 0.5,
            "mouse_ny": 0.5,
            "hovered_slot": -1,
            "carried_count": 0,
        }
    host = args.host or SIDECAR_HOST
    port = int(args.port)
    n = max(1, int(args.count))
    hz = max(0.5, float(args.hz))
    for i in range(n):
        send_sidecar(payload, host=host, port=port)
        if i + 1 < n:
            time.sleep(1.0 / hz)
    print(f"sent {n} packets to {host}:{port}")
    return 0


def cmd_predict_mc(args) -> int:
    from atomspike.capture.mc_predict import run_minecraft_predict
    from atomspike.capture.window import list_windows

    inject = bool(getattr(args, "inject", False))
    cfg = _cfg(args)
    if getattr(args, "list_windows", False):
        hits = list_windows(args.title or "")
        for w in hits:
            print(f"{w.width}x{w.height}\t{w.left},{w.top}\t{w.title}")
        if not hits:
            print("(no matching windows)")
        return 0
    region = None
    if args.region:
        parts = [int(x) for x in args.region.split(",")]
        if len(parts) != 4:
            print("--region must be left,top,width,height", file=sys.stderr)
            return 2
        region = (parts[0], parts[1], parts[2], parts[3])
    if args.title:
        cfg.minecraft.window_title = args.title
        cfg.capture.window_title = args.title
    _require_ckpt(args.ckpt)
    seconds = args.seconds
    if args.dry_run and seconds is None:
        seconds = 2.0
    if inject and args.dry_run:
        print("--inject has no effect in --dry-run", file=sys.stderr)
        inject = False
    if inject and seconds is None:
        seconds = 30.0
    if inject:
        print(
            f"CLOSED LOOP: the model will drive your keyboard/mouse for up to {seconds:.0f}s. "
            f"Singleplayer only. Emergency stop={cfg.minecraft.stop_key}. "
            "After the model loads, click into Minecraft.",
            file=sys.stderr,
        )
    elif not args.dry_run:
        print(
            "Predicting local Minecraft at "
            f"{cfg.capture.fps} fps (no OS inject). Stop={cfg.minecraft.stop_key}. "
            "After the model loads, click into Minecraft. Singleplayer / LAN research only.",
            file=sys.stderr,
        )
    else:
        print(
            "Predicting local Minecraft at "
            f"{cfg.capture.fps} fps (no OS inject). Stop={cfg.minecraft.stop_key}. "
            "A small SNN window opens at 120Hz. Singleplayer / LAN research only.",
            file=sys.stderr,
        )
    try:
        result = run_minecraft_predict(
            cfg,
            args.ckpt,
            seconds=seconds,
            dry_run=args.dry_run,
            desktop=args.desktop,
            region=region,
            dump_path=args.dump,
            inject=inject,
            sample=bool(getattr(args, "sample", False)),
            greedy=bool(getattr(args, "greedy", False)),
            explore=getattr(args, "explore", None),
            viz=(
                True
                if getattr(args, "viz", False)
                else (False if getattr(args, "no_viz", False) or args.dry_run else None)
            ),
            viz_hz=float(getattr(args, "viz_hz", 120.0) or 120.0),
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if result.get("n_ticks", 0) > 0 else 1


def cmd_train_bc(args) -> int:
    from atomspike.train.bc import train_bc

    cfg = _cfg(args)
    result = train_bc(cfg, args.data, args.out, rabc=bool(args.rabc), reset=bool(getattr(args, "reset", False)))
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
    from atomspike.eval.metrics import evaluate_offline, evaluate_policy

    cfg = _cfg(args)
    if getattr(args, "data", None):
        result = evaluate_offline(cfg, args.ckpt, args.data)
        print(json.dumps(result, indent=2, default=_json_default))
        return 0
    if cfg.env.kind.lower() in {"minecraft", "mc", "minerl"}:
        print(
            "Minecraft has no scripted env. Offline eval on demos:\n"
            "  python -m atomspike eval --ckpt runs/bc.pt --data runs/mc.h5\n"
            "Live 30fps prediction (no inject):\n"
            "  python -m atomspike predict-mc --ckpt runs/bc.pt --seconds 30",
            file=sys.stderr,
        )
        return 2
    result = evaluate_policy(cfg, args.ckpt, episodes=args.episodes)
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


def cmd_play(args) -> int:
    from atomspike.envs import make_env
    from atomspike.eval.metrics import evaluate_policy
    from atomspike.runtime.loop import DualRateLoop
    from atomspike.train.common import load_agent, resolve_device

    if args.inject:
        print(
            "For Minecraft use predict-mc --inject (singleplayer only):\n"
            "  python -m atomspike predict-mc --ckpt runs/bc.pt --inject --seconds 30",
            file=sys.stderr,
        )
        return 2
    cfg = _cfg(args)
    if cfg.env.kind.lower() in {"minecraft", "mc", "minerl"}:
        print(
            "Minecraft closed-loop play is predict-mc (capture + model; --inject to drive OS).\n"
            "  python -m atomspike predict-mc --ckpt runs/bc.pt --inject --seconds 30",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "eval_only", False):
        result = evaluate_policy(cfg, args.ckpt, episodes=args.episodes, deterministic=not args.sample)
        print(json.dumps(result, indent=2))
        return 0
    device = resolve_device(cfg.train.device)
    agent = load_agent(args.ckpt, cfg, device)
    agent.eval()
    env = make_env(
        cfg.env.kind,
        frame_size=cfg.env.frame_size,
        max_steps=cfg.env.max_steps,
        seed=cfg.env.seed,
        action_cfg=cfg.action,
    )
    loop = DualRateLoop(agent, cfg, device, mode="realtime", pace=not args.unpaced)
    successes = 0
    for ep in range(args.episodes):
        loop.reset()
        obs = env.reset(seed=cfg.env.seed + ep)
        while True:
            tokens, info = loop.tick(obs.frame, obs.game_state, deterministic=not args.sample)
            nxt = env.step(tokens)
            obs = nxt
            if nxt.done:
                successes += int(bool(nxt.info.get("success", False)))
                break
    env.close()
    hz = loop.clock.scheduled_hz()
    print(
        json.dumps(
            {
                "success_rate": successes / max(1, args.episodes),
                "clock_mode": loop.clock.mode,
                "paced": loop.clock.pace,
                "policy_hz": hz["policy_hz"],
                "perception_hz": hz["perception_hz"],
                "n_policy": loop.stats.n_policy,
                "n_reason": loop.stats.n_reason,
                "latency_p95_ms": loop.stats.p95_ms(),
            },
            indent=2,
        )
    )
    return 0


def cmd_pipeline(args) -> int:
    from atomspike.train.pipeline import run_pipeline

    cfg = _cfg(args)
    result = run_pipeline(
        cfg,
        args.workdir,
        episodes=args.episodes,
        skip_rl=args.skip_rl,
        skip_peft=args.skip_peft,
        skip_convert=args.skip_convert,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_smoke(args) -> int:
    from atomspike.config import load_config
    from atomspike.convert.pmsm import pmsm_status
    from atomspike.eval.metrics import evaluate_policy
    from atomspike.train.common import load_agent, resolve_device
    from atomspike.train.pipeline import run_pipeline
    from atomspike.verify import run_verify

    cfg_path = ROOT / "configs" / "smoke.yaml"
    cfg = load_config(cfg_path if cfg_path.exists() else None)
    verify = run_verify(cfg_path if cfg_path.exists() else Path("configs/smoke.yaml"))
    workdir = Path(args.workdir)
    result = run_pipeline(
        cfg,
        workdir,
        episodes=args.episodes,
        skip_rl=False,
        skip_peft=False,
        skip_convert=False,
    )
    metrics = evaluate_policy(cfg, result["bc"]["checkpoint"], episodes=6)
    metrics_rl = evaluate_policy(cfg, result["rl"]["checkpoint"], episodes=6) if "rl" in result else None
    metrics_peft = evaluate_policy(cfg, result["peft"]["checkpoint"], episodes=6) if "peft" in result else None
    metrics_snn = evaluate_policy(cfg, result["snn"]["checkpoint"], episodes=6) if "snn" in result else None
    snn_status = None
    if "snn" in result:
        device = resolve_device(cfg.train.device)
        snn_agent = load_agent(result["snn"]["checkpoint"], cfg, device)
        snn_status = pmsm_status(snn_agent)
    lora_ok = bool(result.get("peft", {}).get("lora_wrapped", 0) > 0)
    convert_ok = bool(snn_status and snn_status.get("enabled"))
    clock_ok = bool(metrics.get("clock_mode") == "sim" and metrics.get("scheduled_policy_hz", 0) > 0)
    mc_ok = bool(verify.get("mc_collector", {}).get("ok") and verify.get("mc_train_predict", {}).get("ok"))
    report = {
        "verify": verify,
        "pipeline": result,
        "eval_bc": metrics,
        "eval_rl": metrics_rl,
        "eval_peft": metrics_peft,
        "eval_snn": metrics_snn,
        "snn_status": snn_status,
        "lora_ok": lora_ok,
        "convert_persisted": convert_ok,
        "clock_ok": clock_ok,
        "mc_collector_ok": mc_ok,
        "ok": bool(verify and lora_ok and convert_ok and clock_ok and mc_ok and result["bc"]["params"] > 0),
    }
    print(json.dumps(report, indent=2, default=_json_default))
    return 0 if report["ok"] else 1


def cmd_verify(args) -> int:
    from atomspike.config import load_config
    from atomspike.verify import run_verify

    cfg_path = Path(args.config) if getattr(args, "config", None) else ROOT / "configs" / "smoke.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "configs" / "default.yaml"
    load_config(cfg_path)
    report = run_verify(cfg_path)
    print(json.dumps(report, indent=2, default=_json_default))
    return 0


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
        "policy_hz": cfg.dual_rate.policy_hz,
        "perception_hz": cfg.dual_rate.perception_hz,
        "clock_mode": cfg.dual_rate.mode,
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

    s = sub.add_parser("collect-mc", help="T0 Minecraft: 30fps window + HUD/NBT/sidecar + key/mouse")
    _add_config_arg(s, suppress=True)
    s.add_argument("--out", default="runs/mc.h5")
    s.add_argument("--overwrite", action="store_true", help="discard existing --out file instead of appending")
    s.add_argument("--seconds", type=float, default=None, help="stop after N seconds (default: until stop key)")
    s.add_argument("--title", default=None, help="window title substring (default: Minecraft)")
    s.add_argument("--desktop", action="store_true", help="capture the whole monitor instead of the game window")
    s.add_argument("--region", default=None, help="left,top,width,height")
    s.add_argument("--dry-run", action="store_true", help="no game window; write dummy frames for a smoke test")
    s.add_argument("--list-windows", action="store_true", help="print visible windows and exit")
    s.add_argument("--saves-dir", default=None, help="Minecraft saves folder (default: APPDATA/.minecraft/saves)")
    s.add_argument("--log-every", type=float, default=1.0, help="seconds between live status lines (0=off)")
    s.add_argument("--quiet", action="store_true", help="no live status; only start/end summary")
    s.add_argument("--preview", action="store_true", help="force the capture preview window (including --dry-run)")
    s.add_argument("--no-preview", action="store_true", help="do not open the capture preview window")
    s.set_defaults(func=cmd_collect_mc)

    s = sub.add_parser(
        "sidecar-send",
        help="send localhost UDP JSON so collect-mc live log shows src=sidecar",
    )
    s.add_argument("--json", default=None, help="JSON object string (default: demo player stats)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=21337)
    s.add_argument("--count", type=int, default=30, help="how many packets")
    s.add_argument("--hz", type=float, default=20.0, help="send rate")
    s.set_defaults(func=cmd_sidecar_send)

    s = sub.add_parser("predict-mc", help="Minecraft 30fps live prediction (tokens/controls, no OS inject)")
    _add_config_arg(s, suppress=True)
    s.add_argument("--ckpt", required=True, help="trained checkpoint")
    s.add_argument("--seconds", type=float, default=None)
    s.add_argument("--title", default=None)
    s.add_argument("--desktop", action="store_true")
    s.add_argument("--region", default=None, help="left,top,width,height")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--dump", default=None, help="JSONL path for tokens/controls")
    s.add_argument("--inject", action="store_true", help="drive local keyboard/mouse (singleplayer only, F9 stop)")
    s.add_argument("--sample", action="store_true", help="sample actions instead of argmax (more motor variation)")
    s.add_argument("--greedy", action="store_true", help="disable latent/task-line sampling")
    s.add_argument("--explore", type=float, default=None, help="chance of jumping to a random task-line phase (0-1)")
    s.add_argument("--viz", action="store_true", help="force the SNN activity window (including --dry-run)")
    s.add_argument("--no-viz", action="store_true", help="do not open the SNN activity window")
    s.add_argument("--viz-hz", type=float, default=120.0, help="SNN window refresh rate (default 120)")
    s.set_defaults(func=cmd_predict_mc)

    s = sub.add_parser("train-bc", help="T1: behavior cloning warm-start")
    _add_config_arg(s, suppress=True)
    s.add_argument("--data", required=True, nargs="+", help="h5 file(s) or directory of h5 files")
    s.add_argument("--out", required=True)
    s.add_argument("--rabc", action="store_true", help="T2 RA-BC + coverage reweight")
    s.add_argument(
        "--reset",
        action="store_true",
        help="wipe existing --out weights and train from scratch (default: continue fine-tuning)",
    )
    s.set_defaults(func=cmd_train_bc)

    s = sub.add_parser("train-rl", help="T3: advantage-weighted offline RL")
    _add_config_arg(s, suppress=True)
    s.add_argument("--data", required=True, nargs="+", help="h5 file(s) or directory of h5 files")
    s.add_argument("--teacher", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_train_rl)

    s = sub.add_parser("train-peft", help="T4: freeze backbone + LoRA")
    _add_config_arg(s, suppress=True)
    s.add_argument("--data", required=True, nargs="+", help="h5 file(s) or directory of h5 files")
    s.add_argument("--backbone", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_train_peft)

    s = sub.add_parser("convert", help="T6: training-free ANN to SNN")
    _add_config_arg(s, suppress=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--data", default=None, nargs="+", help="h5 file(s) or directory for PMSM calibration")
    s.add_argument("--method", choices=["pmsm", "spiked_attention"], default="pmsm")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("eval", help="rollout success / latency / token acc")
    _add_config_arg(s, suppress=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--episodes", type=int, default=12)
    s.add_argument("--data", default=None, nargs="+", help="h5 file(s) or directory of h5 files (Minecraft offline eval)")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("play", help="closed-loop rollout paced at 30Hz")
    _add_config_arg(s, suppress=True)
    s.add_argument("--ckpt", required=True)
    s.add_argument("--episodes", type=int, default=4)
    s.add_argument("--sample", action="store_true")
    s.add_argument("--unpaced", action="store_true", help="realtime clock without sleep")
    s.add_argument("--eval-only", action="store_true", help="sim-time eval instead of paced play")
    s.add_argument("--inject", action="store_true", help="drive local keyboard/mouse (singleplayer only, F9 stop)")
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("verify", help="regression tests: LoRA reload, PMSM persist, 30Hz clock")
    _add_config_arg(s, suppress=True)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("pipeline", help="run T0-T6 on the synthetic env")
    _add_config_arg(s, suppress=True)
    s.add_argument("--workdir", default="runs/default")
    s.add_argument("--episodes", type=int, default=40)
    s.add_argument("--skip-rl", action="store_true")
    s.add_argument("--skip-peft", action="store_true")
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
