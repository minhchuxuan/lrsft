"""Fixed rerun of part B for the ep8 pair, after the audit (lrsft_audit.md):

  MODULE=lrsft.rl_fixed CONFIGS="configs/default.yaml configs/h100.yaml configs/rl_fix.yaml" ./run_all.sh

Why: the 40-round no-mask runs were flat. (F1) ~55% of groups were dead and 4 updates/round of 128
rollouts made each Adam step mostly noise: the drift of R matched a random walk. (F2) lr_full=1e-6 sits
below the bf16 rounding floor, so the full-parameter policy hardly changed. Changes (configs/rl_fix.yaml):
  - pre-filter: only prompts with 1..7 of 8 correct under fix.prefilter_init (RL sampling); one ordered
    list (fixed seed, several shuffled epochs if short) used by every arm
  - rl.grad_accum=64: one on-policy optimizer step per round on all 512 rollouts, no warmup
  - lr_full 5e-6; drift diagnostics (bf16-changed share, coherence) in rl.py
Steps (re-running the same command resumes):
  1. pre-filter -> <fix.results_dir>/prefilter.json, rl_train_fix.jsonl
  2. fix.order: lrsft.rl + MATH-500 eval per run, named <run><fix.suffix>, results in fix.results_dir
  - time guard: a run is not started if its estimated end (measured hours of this pass, else
    fix.est_hours) is after fix.deadline_utc; the remaining runs are then skipped too
  - early check after fix.early_check_rounds rounds: reward, drift (log only)
Status: <fix.results_dir>/status.json, driver log: the run_all log.
"""

import argparse
import calendar
import fcntl
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import encode_prompt, load_tokenizer, read_json, read_jsonl, stop_token_ids, write_json, write_jsonl
from .config import Paths, add_config_args, config_cli_args, config_from_args
from .pipeline import REPO, free_gb


def log(msg):
    print(f"[rl_fixed] {time.strftime('%F %T')} {msg}", flush=True)


def ordered_prompts(keep: list[int], total: int, per_round: int, seed: int) -> list[tuple[int, int]]:
    """`total` (index, epoch) pairs: shuffled epochs of `keep`, no prompt twice within one round."""
    if len(keep) < per_round:
        raise SystemExit(f"pre-filter kept {len(keep)} prompts, fewer than one round ({per_round})")
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int]] = []
    epoch = 0
    while len(out) < total:
        perm = [int(i) for i in rng.permutation(sorted(keep))]
        open_round = {i for i, _ in out[len(out) - len(out) % per_round:]} if len(out) % per_round else set()
        perm = [i for i in perm if i not in open_round] + [i for i in perm if i in open_round]
        out += [(i, epoch) for i in perm]
        epoch += 1
    return out[:total]


def prefilter(cfg, paths: Paths, fdir: Path) -> dict:
    fx, r = cfg.fix, cfg.rl
    summary_path, train_path = fdir / "prefilter.json", fdir / "rl_train_fix.jsonl"
    if summary_path.exists() and train_path.exists():
        return read_json(summary_path)
    n = int(r.num_generations)
    rows = read_jsonl(paths.data / "rl_train.jsonl")
    by_prompt = {x["prompt"]: i for i, x in enumerate(rows)}
    rewards: dict[int, list[float]] = {}

    probe_path = paths.rollouts(fx.prefilter_init, "probe")
    reused = 0
    if fx.reuse_probe and probe_path.exists() and int(cfg.probe.num_samples) == n:
        # probe rollouts use the RL sampling (rl.temperature, top_p 1, rl.max_completion_tokens)
        pool = read_jsonl(paths.data / "probe_pool.jsonl")
        for g in read_jsonl(probe_path):
            i = by_prompt.get(pool[g["idx"]]["prompt"])
            if i is not None:
                rewards[i] = [float(x) for x in g["rewards"]]
        reused = len(rewards)
        log(f"pre-filter: reused {reused} probe groups of {fx.prefilter_init} (prompts in the RL set)")

    rng = np.random.default_rng(cfg.seed)
    fresh = [i for i in range(len(rows)) if i not in rewards]
    new = sorted(int(i) for i in rng.choice(fresh, size=min(int(fx.prefilter_new_prompts), len(fresh)),
                                            replace=False))
    gen_seconds = 0.0
    if new:
        from .reward import score_parallel
        from .rollout import generate_hf, generate_vllm

        model_dir = paths.ckpt(fx.prefilter_init)
        tok = load_tokenizer(model_dir)
        ids = [encode_prompt(tok, rows[i]["prompt"]) for i in new]
        log(f"pre-filter: sampling {len(new)} prompts x {n} on {fx.prefilter_init} "
            f"(T={r.temperature}, max {r.max_completion_tokens} tokens)")
        gen = generate_vllm if cfg.gen.backend == "vllm" else generate_hf
        t0 = time.time()
        groups = gen(model_dir, ids, n, r.temperature, 1.0, r.max_completion_tokens, cfg.seed, cfg,
                     stop_token_ids(tok))
        gen_seconds = time.time() - t0
        texts = [tok.decode(c, skip_special_tokens=True) for g in groups for c, _ in g]
        golds = [rows[i]["answer"] for i, g in zip(new, groups) for _ in g]
        if cfg.debug.get("fake_reward"):
            import zlib

            scores = [float(zlib.crc32(t.encode()) % 2) for t in texts]
        else:
            scores = score_parallel(texts, golds, workers=cfg.gen.get("score_workers", 8))
        k = 0
        for i, g in zip(new, groups):
            rewards[i] = scores[k : k + len(g)]
            k += len(g)

    correct = {i: int(round(sum(v))) for i, v in rewards.items()}
    keep = sorted(i for i, c in correct.items() if 1 <= c <= n - 1)
    total = int(r.rounds) * int(r.prompts_per_round)
    order = ordered_prompts(keep, total, int(r.prompts_per_round), int(cfg.seed))
    write_jsonl(train_path, [{"prompt": rows[i]["prompt"], "answer": rows[i]["answer"],
                              "level": rows[i].get("level"), "rl_idx": i, "epoch": e} for i, e in order])
    scored = len(correct)
    summary = {
        "init": fx.prefilter_init, "n_samples": n, "temperature": r.temperature,
        "max_tokens": r.max_completion_tokens, "rl_set": len(rows), "scored": scored,
        "reused_probe": reused, "new": len(new), "gen_seconds": gen_seconds,
        "kept": len(keep), "informative_rate": len(keep) / scored,
        "all_wrong_rate": sum(c == 0 for c in correct.values()) / scored,
        "all_correct_rate": sum(c == n for c in correct.values()) / scored,
        "kept_by_level": {str(lv): sum(rows[i].get("level") == lv for i in keep)
                          for lv in sorted({rows[i].get("level") for i in keep})},
        "slots": total, "epochs": total / len(keep), "seed": cfg.seed,
    }
    write_json(summary_path, summary)
    return summary


def paired_delta(before: Path, after: Path) -> dict | None:
    """Pass@1 change on MATH-500 with its paired standard error (same problems, same seed)."""
    if not (before.exists() and after.exists()):
        return None
    a = {x["idx"]: np.mean(x["rewards"]) for x in read_jsonl(before)}
    b = {x["idx"]: np.mean(x["rewards"]) for x in read_jsonl(after)}
    both = [i for i in a if i in b]
    d = np.array([b[i] - a[i] for i in both])
    return {"pass1_before": float(np.mean([a[i] for i in both])), "pass1_after": float(np.mean([b[i] for i in both])),
            "delta": float(d.mean()), "paired_se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": int(len(d))}


def early_check(fdir: Path, name: str, k: int) -> dict | None:
    log_path, drift_path = fdir / "rl" / f"{name}_log.jsonl", fdir / "rl" / f"{name}_drift.jsonl"
    rows = [x for x in read_jsonl(log_path) if "reward" in x] if log_path.exists() else []
    drift = [x for x in read_jsonl(drift_path) if x["round"] >= k] if drift_path.exists() else []
    if len(rows) < k or not drift:
        return None
    f = lambda rs, key: float(np.mean([float(x[key]) for x in rs if key in x]))  # noqa: E731
    head, now = rows[:1], rows[max(1, k - 3):k]
    t = drift[0]["trainable"]
    return {"rounds": k, "reward_first": f(head, "reward"), "reward_now": f(now, "reward"),
            "dead_first": f(head, "frac_reward_zero_std"), "dead_now": f(now, "frac_reward_zero_std"),
            "len_first": f(head, "completions/mean_length"), "len_now": f(now, "completions/mean_length"),
            "trunc_first": f(head, "completions/clipped_ratio"), "trunc_now": f(now, "completions/clipped_ratio"),
            "frac_bf16_changed": t["frac_bf16_changed"], "coherence": t["coherence"],
            "random_walk_coherence": drift[0]["random_walk_coherence"], "drift_round": drift[0]["round"],
            "layers": drift[0].get("layers")}


def run(cmd, log_path: Path) -> int:
    with open(log_path, "a") as f:
        f.write(f"\n===== {time.strftime('%F %T')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode


def disk_ok(cfg, paths: Paths) -> bool:
    low = [(str(p), round(free_gb(p), 1), lim) for p, lim in ((paths.work, cfg.disk.get("min_free_gb", 0)),
                                                              (Path.home(), cfg.disk.get("min_free_home_gb", 0)))
           if lim and free_gb(p) < lim]
    if low:
        log(f"low disk, stopping: {low}")
    return not low


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    fx, r = cfg.fix, cfg.rl
    assert r.mask_truncated is False and not r.shuffle, "run with configs/rl_fix.yaml on top"
    fdir = Path(fx.results_dir)
    (fdir / "rl").mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    deadline = calendar.timegm(time.strptime(fx.deadline_utc, "%Y-%m-%d %H:%M"))

    # One GPU job at a time: hold the pipeline lock for the whole driver.
    lock = open(paths.results / ".pipeline.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another pipeline holds the lock; waiting")
        fcntl.flock(lock, fcntl.LOCK_EX)

    status_path = fdir / "status.json"
    st = read_json(status_path) if status_path.exists() else {"runs": {}}
    st.update(config={k: r[k] for k in ("mask_truncated", "lr_subset", "lr_full", "rounds", "grad_accum",
                                        "warmup_rounds", "prompts_per_round", "num_generations")},
              order=list(fx.order), deadline_utc=fx.deadline_utc)
    write_json(status_path, st)

    if not disk_ok(cfg, paths):
        raise SystemExit(3)
    pf = prefilter(cfg, paths, fdir)
    st["prefilter"] = pf
    write_json(status_path, st)
    log(f"pre-filter: kept {pf['kept']} of {pf['scored']} (informative {pf['informative_rate']:.2f}), "
        f"{pf['slots']} slots = {pf['epochs']:.2f} epochs")

    cfg_args = config_cli_args(args) + ["--set", f"paths.results={fdir}",
                                        f"rl.train_file={fdir / 'rl_train_fix.jsonl'}"]
    py = [sys.executable, "-m"]
    R = ",".join(map(str, read_json(paths.layers)["R"]))
    stop_all = None
    for base in fx.order:
        name = base + fx.suffix
        init, _, which = base.partition("__")
        layers = {"R": R, "all": "all"}.get(which)
        rec = st["runs"].setdefault(name, {})
        rl_out, ev_out = fdir / "rl" / f"{name}.json", fdir / "eval" / f"rl__{name}.json"
        if ev_out.exists():
            rec["state"] = "done"
            continue
        if layers is None:
            rec.update(state="skipped", reason=f"unknown layer set {which!r}")
            continue
        if stop_all:
            rec.update(state="skipped", reason=stop_all)
            write_json(status_path, st)
            continue
        done = [x["hours"] for x in st["runs"].values() if x.get("hours")]
        est = float(np.mean(done)) if done else float(fx.est_hours)
        end = time.time() + est * 3600
        if end > deadline:
            stop_all = (f"time guard: estimated end {time.strftime('%F %H:%M UTC', time.gmtime(end))} "
                        f"is after {fx.deadline_utc} UTC")
            rec.update(state="skipped", reason=stop_all)
            log(f"skip {name} and later runs: {stop_all}")
            write_json(status_path, st)
            continue
        if not disk_ok(cfg, paths):
            raise SystemExit(3)
        log(f"{name}: init={init} layers={layers}, estimated {est:.1f} h, "
            f"ends ~{time.strftime('%F %H:%M UTC', time.gmtime(end))}")
        rec.update(state="running", started=time.strftime("%F %T UTC", time.gmtime()), est_hours=round(est, 2))
        write_json(status_path, st)
        t0 = time.time()
        rl_cmd = py + ["lrsft.rl"] + cfg_args + ["--init", init, "--layers", layers, "--name", name]
        for attempt in range(1 + int(fx.retries)):
            if rl_out.exists():
                break
            if attempt:
                log(f"{name}: rl failed, retry {attempt} (resumes from the last checkpoint)")
            with open(paths.logs / f"rl__{name}.log", "a") as f:
                f.write(f"\n===== {time.strftime('%F %T')} attempt {attempt + 1}: {' '.join(rl_cmd)}\n")
                f.flush()
                proc = subprocess.Popen(rl_cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
                while proc.poll() is None:
                    time.sleep(60)
                    if "early_check" not in rec:
                        chk = early_check(fdir, name, int(fx.early_check_rounds))
                        if chk:
                            rec["early_check"] = chk
                            write_json(status_path, st)
                            log(f"early check {name} after {chk['rounds']} rounds: reward {chk['reward_first']:.3f}"
                                f" -> {chk['reward_now']:.3f}, dead {chk['dead_first']:.2f} -> {chk['dead_now']:.2f}"
                                f", bf16 changed {chk['frac_bf16_changed']:.3f}, coherence {chk['coherence']:.3f}"
                                f" (random walk ~{chk['random_walk_coherence']:.3f}), length {chk['len_first']:.0f}"
                                f" -> {chk['len_now']:.0f}, truncated {chk['trunc_first']:.2f} -> "
                                f"{chk['trunc_now']:.2f}")
        if rl_out.exists():
            ev_cmd = py + ["lrsft.rollout"] + cfg_args + ["--ckpt", f"rl__{name}", "--task", "eval"]
            for attempt in range(1 + int(fx.retries)):
                if ev_out.exists() or run(ev_cmd, paths.logs / f"eval__rl__{name}.log") == 0:
                    break
        hours = (time.time() - t0) / 3600
        ok = ev_out.exists()
        rec.update(state="done" if ok else "failed", hours=round(hours, 2) if ok else None,
                   finished=time.strftime("%F %T UTC", time.gmtime()))
        if ok:
            res = read_json(rl_out)
            rec.update(drift_final=res.get("drift_final"), reward_first=res.get("reward_first"),
                       reward_last20pct=res.get("reward_last20pct"),
                       dead_first=res.get("dead_first"), dead_last20pct=res.get("dead_last20pct"),
                       math500=paired_delta(paths.rollouts(init, "eval"), paths.rollouts(f"rl__{name}", "eval")))
        write_json(status_path, st)
        m = rec.get("math500") or {}
        log(f"{name}: {rec['state']} in {hours:.2f} h" + (
            f", MATH-500 pass@1 {m['pass1_before']:.3f} -> {m['pass1_after']:.3f} "
            f"(delta {m['delta']:+.3f}, paired SE {m['paired_se']:.3f})" if m else ""))
    log("done")


if __name__ == "__main__":
    main()
