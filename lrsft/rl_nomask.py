"""Third pass of part B: GRPO without masking truncated completions.

  MODULE=lrsft.rl_nomask CONFIGS="configs/default.yaml configs/h100.yaml" ./run_all.sh

Why: at lr 5e-5 with mask_truncated=true, 40 rounds of GRPO on R lowered MATH-500 pass@1 at ep8
(full 33.7 -> 30.9, LR 33.6 -> 30.4) while the share of answers cut at 4096 tokens doubled: masked
truncations carry no penalty, so lengths drift up. Without the mask a truncated answer scores 0 and
stays in the loss (plain GRPO).

Steps (re-running the same command resumes):
  1. wait until no other pipeline holds results/.pipeline.lock (the previous driver's last run)
  2. move the masked runs (nomask.archive_runs) to results/<archive_dir>/ and <work>/<archive_dir>/,
     so they are kept as an ablation and the pipeline does not skip the new runs
  3. write configs/rl_nomask.yaml (mask off, lr, rounds) and run nomask.order through
     `lrsft.pipeline --only rl__X,eval__rl__X`, then the plots
  - time guard: a run is not started if its estimated end (measured durations of this pass, else
    nomask.est_hours_R; full-parameter x nomask.est_factor_all until measured) is after
    nomask.deadline_utc; the remaining runs are then skipped too
  - early check: after nomask.early_check_rounds rounds of each run, log completion length and
    truncation rate against the first round (log only, the run continues)
Status: results/rl_nomask_status.json, driver log: the run_all log.
"""

import argparse
import calendar
import fcntl
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
    print(f"[rl_nomask] {time.strftime('%F %T')} {msg}", flush=True)


def wait_for_lock(paths: Paths):
    lock_path = paths.results / ".pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another pipeline holds the lock (previous run still going); waiting")
            fcntl.flock(f, fcntl.LOCK_EX)
        fcntl.flock(f, fcntl.LOCK_UN)


def _move(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def archive_masked(cfg, paths: Paths):
    nm = cfg.nomask
    dst_r, dst_o = paths.results / nm.archive_dir, paths.work / nm.archive_dir
    marker = dst_r / "ARCHIVED"
    if marker.exists():
        return
    status_path = paths.results / "pipeline_status.json"
    status = read_json(status_path) if status_path.exists() else {}
    moved_status, notes = {}, []
    for name in nm.archive_runs:
        done = paths.result("rl", name).exists() and paths.result("eval", f"rl__{name}").exists()
        sub = "" if done else "partial/"
        notes.append(f"{name}: {'complete' if done else 'partial'}")
        _move(paths.result("rl", name), dst_r / sub / "rl" / f"{name}.json")
        _move(paths.results / "rl" / f"{name}_log.jsonl", dst_r / sub / "rl" / f"{name}_log.jsonl")
        _move(paths.result("eval", f"rl__{name}"), dst_r / sub / "eval" / f"rl__{name}.json")
        _move(paths.ckpt(f"rl__{name}"), dst_o / sub / "models" / f"rl__{name}")
        _move(paths.rollouts(f"rl__{name}", "eval"), dst_o / sub / "rollouts" / f"rl__{name}__eval.jsonl")
        _move(paths.trainer_dir(f"rl_{name}"), dst_o / sub / "runs" / f"rl_{name}")
        for step in (f"rl__{name}", f"eval__rl__{name}"):
            _move(paths.logs / f"{step}.log", dst_o / sub / "logs" / f"{step}.log")
            if step in status:
                moved_status[step] = status.pop(step)
    # anything else the previous driver may have started (partial run dirs / logs)
    for run_dir in (paths.work / "runs").glob("rl_*") if (paths.work / "runs").exists() else []:
        notes.append(f"{run_dir.name}: partial run dir moved")
        _move(run_dir, dst_o / "partial" / "runs" / run_dir.name)
    for f in ("lr_sweep.json",):
        if (paths.results / f).exists():
            shutil.copy2(paths.results / f, dst_r / f)
    if (REPO / "configs" / "rl_rerun.yaml").exists():
        shutil.copy2(REPO / "configs" / "rl_rerun.yaml", dst_r / "rl_rerun.yaml")
    write_json(dst_r / "pipeline_status_masked.json", moved_status)
    write_json(status_path, status)
    (dst_r / "README.txt").write_text(
        "GRPO on R (and controls) with lr_subset=5e-5, 40 rounds, mask_truncated=true (configs in "
        "rl_rerun.yaml). Kept as an ablation: truncated answers carry no penalty and lengths drift up. "
        "Moved here by lrsft.rl_nomask before the mask_truncated=false pass.\n" + "\n".join(notes) + "\n")
    marker.write_text(time.strftime("%F %T\n"))
    log("archived masked runs: " + "; ".join(notes))


def early_check(log_path, k):
    rows = [r for r in read_jsonl(log_path) if "reward" in r] if log_path.exists() else []
    if len(rows) < k:
        return None
    first, last = rows[0], rows[max(1, k - 3):k]
    f = lambda rs, key: float(np.mean([float(r[key]) for r in rs]))  # noqa: E731
    out = {"rounds": k, "len_first": f([first], "completions/mean_length"), "len_now": f(last, "completions/mean_length"),
           "trunc_first": f([first], "completions/clipped_ratio"), "trunc_now": f(last, "completions/clipped_ratio"),
           "reward_first": f([first], "reward"), "reward_now": f(last, "reward")}
    out["rising_strongly"] = bool(out["len_now"] > 1.10 * out["len_first"] or out["trunc_now"] > out["trunc_first"] + 0.05)
    return out


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    nm = cfg.nomask
    paths.logs.mkdir(parents=True, exist_ok=True)
    deadline = calendar.timegm(time.strptime(nm.deadline_utc, "%Y-%m-%d %H:%M"))

    wait_for_lock(paths)
    archive_masked(cfg, paths)

    nomask_cfg = REPO / "configs" / "rl_nomask.yaml"
    nomask_cfg.write_text("# Written by lrsft.rl_nomask: part B without masking truncated completions.\n"
                          + yaml.safe_dump({"rl": {"mask_truncated": False, "lr_subset": float(nm.lr_subset),
                                                   "rounds": int(nm.rounds)}}))
    cfg_args = config_cli_args(args) + ["--config", str(nomask_cfg)]
    full_cfg = config_from_args(argparse.Namespace(config=args.config + [str(nomask_cfg)], overrides=args.overrides))
    assert full_cfg.rl.mask_truncated is False
    known = {s.name for s in build_steps(full_cfg, paths, [])}

    status_path = paths.results / "rl_nomask_status.json"
    st = read_json(status_path) if status_path.exists() else {"runs": {}}
    st.update(config={"mask_truncated": False, "lr_subset": float(nm.lr_subset), "lr_full": float(cfg.rl.lr_full),
                      "rounds": int(nm.rounds)}, order=list(nm.order), deadline_utc=nm.deadline_utc)
    write_json(status_path, st)
    py = [sys.executable, "-m"]
    stop_all = None
    for name in nm.order:
        rec = st["runs"].setdefault(name, {})
        steps = (["select_Rre"] if name.endswith("__Rre") else []) + [f"rl__{name}", f"eval__rl__{name}"]
        if paths.result("eval", f"rl__{name}").exists():
            rec.setdefault("state", "done")
            continue
        if [s for s in steps if s not in known]:
            rec.update(state="skipped", reason="no such pipeline steps")
            continue
        if stop_all:
            rec.update(state="skipped", reason=stop_all)
            write_json(status_path, st)
            continue
        done_R = [r["hours"] for n, r in st["runs"].items() if r.get("hours") and not n.endswith("__all")]
        done_all = [r["hours"] for n, r in st["runs"].items() if r.get("hours") and n.endswith("__all")]
        est_R = float(np.mean(done_R)) if done_R else float(nm.est_hours_R)
        est = (done_all[0] if done_all else est_R * float(nm.est_factor_all)) if name.endswith("__all") else est_R
        end = time.time() + est * 3600
        if end > deadline:
            stop_all = (f"time guard: estimated end {time.strftime('%F %H:%M UTC', time.gmtime(end))} "
                        f"is after {nm.deadline_utc} UTC")
            rec.update(state="skipped", reason=stop_all)
            log(f"skip {name} and later runs: {stop_all}")
            write_json(status_path, st)
            continue
        log(f"B (no mask): {name}, estimated {est:.1f} h, ends ~{time.strftime('%F %H:%M UTC', time.gmtime(end))}")
        rec.update(state="running", started=time.strftime("%F %T UTC", time.gmtime()), est_hours=round(est, 2))
        write_json(status_path, st)
        t0 = time.time()
        proc = subprocess.Popen(py + ["lrsft.pipeline"] + cfg_args + ["--only", ",".join(steps)], cwd=REPO)
        checked = False
        while proc.poll() is None:
            time.sleep(60)
            if not checked:
                chk = early_check(paths.results / "rl" / f"{name}_log.jsonl", int(nm.early_check_rounds))
                if chk:
                    checked = True
                    rec["early_check"] = chk
                    write_json(status_path, st)
                    log(f"early check {name} after {chk['rounds']} rounds: mean length {chk['len_first']:.0f} -> "
                        f"{chk['len_now']:.0f}, truncated {chk['trunc_first']:.2f} -> {chk['trunc_now']:.2f}, "
                        f"reward {chk['reward_first']:.3f} -> {chk['reward_now']:.3f}"
                        + ("  << STILL RISING STRONGLY (run continues)" if chk["rising_strongly"] else ""))
        hours = (time.time() - t0) / 3600
        ok = paths.result("eval", f"rl__{name}").exists()
        pstat = read_json(paths.results / "pipeline_status.json").get(f"rl__{name}", {})
        state = "done" if ok else ("skipped" if pstat.get("state") == "skipped" else "failed")
        rec.update(state=state, hours=round(hours, 2) if ok else None, reason=pstat.get("reason"),
                   finished=time.strftime("%F %T UTC", time.gmtime()))
        write_json(status_path, st)
        log(f"{name}: {state} in {hours:.2f} h")
        if proc.returncode == 3:
            raise SystemExit("pipeline stopped on low disk; free space and re-run the same command")
        subprocess.run([sys.executable, str(REPO / "plot.py")] + cfg_args, cwd=REPO,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("done")


if __name__ == "__main__":
    main()
