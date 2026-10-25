"""Run the whole experiment unattended, one subprocess per step.

  python -m lrsft.pipeline --config configs/default.yaml --stages A,B,extras
  python -m lrsft.pipeline --config configs/default.yaml --dry-run      # show plan + status

Each step declares its output files; a step whose outputs exist is skipped, so re-running the same
command resumes (SFT and RL also resume from their own Trainer checkpoints). A failing step is
retried once, then the pipeline moves on; later steps whose inputs are missing are skipped.
Each step runs in its own process so vLLM and HF never compete for GPU memory.
Status: results/pipeline_status.json, logs: <logs>/<step>.log.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .common import read_json, write_json
from .config import Paths, add_config_args, config_cli_args, config_from_args

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Step:
    name: str
    stage: str
    cmd: Callable[[], list[str]]          # built lazily: some args (R) only exist after earlier steps
    outputs: list[Path] = field(default_factory=list)
    inputs: list[Path] = field(default_factory=list)
    gpu: bool = True
    skip_if: Callable[[], str | None] | None = None  # checked at run time; returns a reason to skip

    def done(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs)

    def ready(self) -> bool:
        return all(p.exists() for p in self.inputs)


def build_steps(cfg, paths: Paths, cfg_args: list[str]) -> list[Step]:
    py = [sys.executable, "-m"]
    R = lambda: ",".join(map(str, read_json(paths.layers)["R"]))  # noqa: E731
    steps: list[Step] = []
    data_done = paths.data / "DONE"
    model_cfg = lambda name: paths.ckpt(name) / "config.json"  # noqa: E731
    epochs = list(cfg.sft.save_epochs)

    def add(name, stage, module, args, outputs, inputs, gpu=True) -> Step:
        steps.append(Step(name, stage, lambda: py + [module] + cfg_args + (args() if callable(args) else args),
                          outputs, inputs, gpu))
        return steps[-1]

    def probe(ck, stage):
        add(f"rollout_probe__{ck}", stage, "lrsft.rollout", ["--ckpt", ck, "--task", "probe"],
            [paths.result("probe_rollouts", ck), paths.rollouts(ck, "probe")], [data_done, model_cfg(ck)])
        add(f"probe__{ck}", stage, "lrsft.probe", ["--ckpt", ck],
            [paths.result("probe", ck)], [paths.result("probe_rollouts", ck)])

    def evaluate(ck, stage, diag=True):
        add(f"eval__{ck}", stage, "lrsft.rollout", ["--ckpt", ck, "--task", "eval"],
            [paths.result("eval", ck), paths.rollouts(ck, "eval")], [data_done, model_cfg(ck)])
        if diag:
            add(f"diag__{ck}", stage, "lrsft.diagnostics", ["--ckpt", ck],
                [paths.result("diag", ck)], [paths.result("eval", ck)])

    def report(stage):
        steps.append(Step(f"report__{stage}", stage, lambda: [sys.executable, str(REPO / "plot.py")] + cfg_args,
                          [], [], gpu=False))

    def rl(init, layers_arg, name, stage) -> Step:
        step = add(f"rl__{name}", stage, "lrsft.rl",
                   lambda: ["--init", init, "--layers", layers_arg(), "--name", name],
                   [paths.result("rl", name), model_cfg(f"rl__{name}")], [model_cfg(init), paths.layers])
        evaluate(f"rl__{name}", stage, diag=False)
        return step

    # ---- A: layer ranking at theta_0, SFT, re-ranking, pre-RL diagnostics
    add("prep", "A", "lrsft.data", [], [data_done], [], gpu=False)
    probe("base", "A")
    add("select_R", "A", "lrsft.select_layers", ["--from", "base"], [paths.layers], [paths.result("probe", "base")],
        gpu=False)
    evaluate("base", "A")
    add("sft_full", "A", "lrsft.sft", ["--mode", "full"], [paths.result("sft", "sft_full")], [data_done])
    for e in epochs:
        probe(f"full_ep{e}", "A")  # A3: does the theta_0 ranking survive SFT?
    report("A3")
    add("sft_lr", "A", "lrsft.sft", lambda: ["--mode", "lr", "--layers", R()], [paths.result("sft", "sft_lr")],
        [data_done, paths.layers])
    add("verify_lr", "A", "lrsft.verify", lambda: ["--layers", R()], [paths.results / "verify_lr.json"],
        [paths.result("sft", "sft_lr")], gpu=False)
    for e in epochs:
        probe(f"lr_ep{e}", "A")
    for kind in ("full", "lr"):
        for e in epochs:
            evaluate(f"{kind}_ep{e}", "A")
    report("A")

    # ---- B: GRPO on R only, same R and config for both rows, deepest SFT first
    for d in cfg.rl.depths:
        for kind in ("full", "lr"):
            rl(f"{kind}_ep{d}", R, f"{kind}_ep{d}__R", "B")
    report("B")

    # ---- extras: controls at the deepest full-SFT checkpoint, and a full-parameter baseline
    deep = f"full_ep{max(epochs)}"
    ex = cfg.extras
    if ex.random_R:
        rl(deep, lambda: ",".join(map(str, read_json(paths.layers)["R_random"])), f"{deep}__Rrand", "extras")
    if ex.reselected_R:
        resel = paths.results / "layers_reselected.json"
        add("select_Rre", "extras", "lrsft.select_layers", ["--from", deep, "--reselect"], [resel],
            [paths.result("probe", deep)], gpu=False)
        step = rl(deep, lambda: ",".join(map(str, read_json(resel)["R_reselected"])), f"{deep}__Rre", "extras")
        step.inputs.append(resel)  # waits for R'
        step.skip_if = lambda: ("R' == R, same run as R" if read_json(resel)["R_reselected"] == read_json(
            paths.layers)["R"] else None)
    if ex.full_param:
        rl(deep, lambda: "all", f"{deep}__all", "extras")
    if ex.rl_from_base:
        rl("base", R, "base__R", "extras")
    report("extras")
    return steps


def run_step(step: Step, log_dir: Path, retries: int) -> tuple[str, float]:
    log = log_dir / f"{step.name}.log"
    t0 = time.time()
    for attempt in range(retries + 1):
        cmd = step.cmd()
        with open(log, "a") as f:
            f.write(f"\n===== {time.strftime('%F %T')} attempt {attempt + 1}: {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
        if rc == 0:
            return "ok", time.time() - t0
        print(f"[pipeline] {step.name} failed (rc={rc}), see {log}", flush=True)
    return "failed", time.time() - t0


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--stages", default="A,B,extras", help="comma list of A, A3, B, extras (report steps follow)")
    parser.add_argument("--only", default="", help="comma list of step names to run (ignores --stages)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    steps = build_steps(cfg, paths, config_cli_args(args))
    stages = set(args.stages.split(","))
    if "A" in stages:
        stages.add("A3")
    only = set(filter(None, args.only.split(",")))
    selected = [s for s in steps if (s.name in only if only else s.stage in stages)]

    if args.dry_run:
        for s in selected:
            state = "done" if s.done() else ("ready" if s.ready() else "waiting")
            print(f"{s.stage:7} {s.name:40} {state}")
        return

    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)
    # One pipeline per results dir: a second run_all.sh would fight the first for the GPU.
    lock = open(paths.results / ".pipeline.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another pipeline is already running on {paths.results} (lock held); not starting")
    status_path = paths.results / "pipeline_status.json"
    status = read_json(status_path) if status_path.exists() else {}
    t_all = time.time()
    for s in selected:
        if s.done():
            print(f"[pipeline] {s.name}: done, skip", flush=True)
            continue
        if not s.ready():
            missing = [str(p) for p in s.inputs if not p.exists()]
            print(f"[pipeline] {s.name}: inputs missing {missing}, skip", flush=True)
            status[s.name] = {"state": "blocked", "missing": missing}
            write_json(status_path, status)
            continue
        reason = s.skip_if() if s.skip_if else None
        if reason:
            print(f"[pipeline] {s.name}: skipped ({reason})", flush=True)
            status[s.name] = {"state": "skipped", "reason": reason}
            write_json(status_path, status)
            continue
        print(f"[pipeline] {time.strftime('%F %T')} start {s.name}", flush=True)
        state, secs = run_step(s, paths.logs, args.retries)
        prev = status.get(s.name, {}).get("seconds", 0.0)
        status[s.name] = {"state": state, "seconds": prev + secs, "gpu": s.gpu, "finished": time.strftime("%F %T")}
        write_json(status_path, status)
        print(f"[pipeline] {s.name}: {state} in {secs / 60:.1f} min", flush=True)
    gpu_h = sum(v.get("seconds", 0) for v in status.values() if v.get("gpu")) / 3600
    print(f"[pipeline] finished in {(time.time() - t_all) / 3600:.2f} h; GPU-hours logged so far: {gpu_h:.2f}")
    print(json.dumps({k: v["state"] for k, v in status.items()}, indent=1))


if __name__ == "__main__":
    main()
