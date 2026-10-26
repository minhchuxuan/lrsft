"""Second pass of part B: a short lr sweep for GRPO on R, pick the lr, then run B in priority order.

  MODULE=lrsft.rl_rerun CONFIGS="configs/default.yaml configs/h100.yaml" ./run_all.sh

Why: with rl.lr_subset=5e-6 the first B run (full_ep8, 60 rounds) did not learn at all (reward flat,
clip ratio ~0). Steps:
  1. sweep: GRPO on R from rerun.sweep_init for rerun.sweep_rounds rounds at each lr in rerun.sweep_lrs.
     Results go to results/lr_sweep/ (outside results/rl, so plots and the pipeline ignore them);
     sweep snapshots are deleted afterwards.
  2. choose the lr (rule below) -> results/lr_sweep.json and configs/rl_rerun.yaml
     (rl.lr_subset = chosen, rl.rounds = rerun.rounds).
  3. run each B run in rerun.order through `lrsft.pipeline --only rl__X,eval__rl__X`, then the plots.
Re-running the same command resumes: finished sweep runs, the choice and finished B runs are kept.

Rule (approved by the user): take the larger lr if its reward rises and it is stable; if the larger lr
is unstable take the smaller; if the larger is stable but flat while the smaller rises, take the
smaller; if both are flat take the larger and say so.
  stable = run finished, no NaN/inf, mean reward of the last 3 rounds not more than 0.05 below the
           first 2 rounds, token entropy (last 4 logged steps vs first 4) within [0.5x, 2x]
  rises  = last-3 mean reward exceeds first-2 mean by more than rerun.rise_threshold
"""

import argparse
import math
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from .common import read_json, read_jsonl, write_json
from .config import Paths, add_config_args, config_cli_args, config_from_args
from .pipeline import REPO, build_steps


def log(msg):
    print(f"[rl_rerun] {time.strftime('%F %T')} {msg}", flush=True)


def run(cmd, log_path) -> int:
    with open(log_path, "a") as f:
        f.write(f"\n===== {time.strftime('%F %T')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode


def sweep_metrics(log_rows, finished: bool) -> dict:
    gen = [r for r in log_rows if "reward" in r]
    rew = np.array([float(r["reward"]) for r in gen])
    ent = np.array([float(r["entropy"]) for r in log_rows if "entropy" in r])
    watched = [float(r[k]) for r in log_rows for k in ("reward", "entropy", "loss", "grad_norm") if k in r]
    clip = [float(r["clip_ratio/region_mean"]) for r in log_rows if "clip_ratio/region_mean" in r]
    m = {"finished": finished, "rounds_logged": len(rew), "rewards": rew.round(4).tolist(),
         "nan": not all(math.isfinite(x) for x in watched)}
    if len(rew) >= 3 and len(ent) >= 2:
        k = min(4, len(ent) // 2)
        m.update(
            reward_first=float(rew[:2].mean()), reward_last=float(rew[-3:].mean()),
            reward_slope_per_round=float(np.polyfit(np.arange(len(rew)), rew, 1)[0]),
            entropy_first=float(ent[:k].mean()), entropy_last=float(ent[-k:].mean()),
            clip_ratio_mean=float(np.mean(clip)) if clip else None,
            dead_last=float(np.mean([float(r.get("frac_reward_zero_std", np.nan)) for r in gen[-3:]])),
            length_last=float(np.mean([float(r.get("completions/mean_length", np.nan)) for r in gen[-3:]])),
        )
    return m


def judge(m: dict, rise_threshold: float) -> tuple[bool, bool]:
    if not m["finished"] or m["nan"] or "reward_first" not in m:
        return False, False
    ratio = m["entropy_last"] / max(m["entropy_first"], 1e-8)
    stable = m["reward_last"] >= m["reward_first"] - 0.05 and 0.5 <= ratio <= 2.0
    rises = m["reward_last"] - m["reward_first"] > rise_threshold
    return stable, rises


def choose(results: dict, rise_threshold: float) -> tuple[float, str]:
    lo, hi = sorted(results, key=float)[0], sorted(results, key=float)[-1]
    s_hi, r_hi = judge(results[hi], rise_threshold)
    s_lo, r_lo = judge(results[lo], rise_threshold)
    for lr, s, r in ((hi, s_hi, r_hi), (lo, s_lo, r_lo)):
        results[lr]["stable"], results[lr]["rises"] = s, r
    if s_hi and r_hi:
        return float(hi), f"{hi}: reward rises and is stable"
    if not s_hi:
        return float(lo), f"{hi} is not stable; took {lo}" + ("" if s_lo else f" ({lo} is not stable either)")
    if s_lo and r_lo:
        return float(lo), f"{hi} is stable but flat while {lo} rises; took {lo}"
    return float(hi), f"both flat on reward; took {hi} (larger steps, see clip ratio)"


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    rr = cfg.rerun
    base_args = config_cli_args(args)
    py = [sys.executable, "-m"]
    paths.logs.mkdir(parents=True, exist_ok=True)
    R = ",".join(map(str, read_json(paths.layers)["R"]))
    decision_path = paths.results / "lr_sweep.json"
    sweep_dir = paths.results / "lr_sweep"

    # ---- 1. sweep
    if decision_path.exists():
        decision = read_json(decision_path)
        log(f"lr already chosen: {decision['chosen_lr']} ({decision['reason']})")
    else:
        results = {}
        for lr in rr.sweep_lrs:
            name = f"sweep_{rr.sweep_init}__R_lr{lr:g}"
            out = sweep_dir / "rl" / f"{name}.json"
            if not out.exists():
                log(f"sweep: lr={lr:g} from {rr.sweep_init}, {rr.sweep_rounds} rounds")
                t0 = time.time()
                rc = run(py + ["lrsft.rl"] + base_args + [
                    "--set", f"paths.results={sweep_dir}", f"rl.lr_subset={lr}", f"rl.rounds={rr.sweep_rounds}",
                    "--init", rr.sweep_init, "--layers", R, "--name", name], paths.logs / f"{name}.log")
                log(f"sweep: lr={lr:g} rc={rc} in {(time.time() - t0) / 60:.1f} min")
            snap = paths.ckpt(f"rl__{name}")
            if snap.exists():
                shutil.rmtree(snap)  # not evaluated; only the training curves are used
            log_path = sweep_dir / "rl" / f"{name}_log.jsonl"
            rows = read_jsonl(log_path) if log_path.exists() else []
            results[f"{lr:g}"] = {"name": name, **sweep_metrics(rows, out.exists())}
        chosen, reason = choose(results, rr.rise_threshold)
        decision = {"chosen_lr": chosen, "reason": reason, "init": rr.sweep_init, "layers": R,
                    "rounds": rr.sweep_rounds, "rise_threshold": rr.rise_threshold, "runs": results,
                    "rule": "larger lr if rising and stable; smaller if larger unstable, or if larger flat and "
                            "smaller rising; larger if both flat. stable: finished, no NaN, last-3 reward >= "
                            "first-2 - 0.05, entropy last/first in [0.5, 2]."}
        if not any(r["finished"] for r in results.values()):
            write_json(paths.results / "lr_sweep_failed.json", decision)
            raise SystemExit("both sweep runs failed; not starting B (see logs/sweep_*.log)")
        write_json(decision_path, decision)
        log(f"chose lr={chosen:g}: {reason}")

    rerun_cfg = REPO / "configs" / "rl_rerun.yaml"
    rerun_cfg.write_text(
        "# Written by lrsft.rl_rerun from results/lr_sweep.json; used for the second pass of part B.\n"
        + yaml.safe_dump({"rl": {"lr_subset": float(decision["chosen_lr"]), "rounds": int(rr.rounds)}}))
    cfg_args = base_args + ["--config", str(rerun_cfg)]

    # ---- 2. part B in priority order
    known = {s.name for s in build_steps(config_from_args(argparse.Namespace(
        config=args.config + [str(rerun_cfg)], overrides=args.overrides)), paths, [])}
    for name in rr.order:
        only = (["select_Rre"] if name.endswith("__Rre") else []) + [f"rl__{name}", f"eval__rl__{name}"]
        missing = [s for s in only if s not in known]
        if missing:
            log(f"skip {name}: no such pipeline steps {missing} (check rl.depths / extras)")
            continue
        log(f"B: {name}")
        rc = subprocess.run(py + ["lrsft.pipeline"] + cfg_args + ["--only", ",".join(only)], cwd=REPO).returncode
        if rc == 3:
            raise SystemExit("pipeline stopped on low disk; fix and re-run the same command")
        subprocess.run([sys.executable, str(REPO / "plot.py")] + cfg_args, cwd=REPO,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("done")


if __name__ == "__main__":
    main()
