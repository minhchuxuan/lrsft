"""Tables and slide figures from results/. Safe to run at any time: missing pieces are skipped.

  python plot.py --config configs/default.yaml

Writes results/summary_*.csv and results/figures/*.png:
  fig_a3_layer_ranking.png   layer scores at theta_0 vs after full SFT, R marked; rank agreement vs depth
  fig_a4_diagnostics.png     pre-RL pass@1, pass@k, entropy, dead groups vs SFT depth (full vs LR)
  fig_a4_reserved_layers.png CKA of R's input vs base, weight drift of R
  fig_b_rl_vs_depth.png      MATH-500 after GRPO-on-R vs SFT depth (full vs LR), RL gain
  fig_b_controls_cost.png    deepest checkpoint: R vs R' vs random vs all-params (accuracy, memory, time)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lrsft.config import Paths, add_config_args, config_from_args  # noqa: E402
from lrsft.metrics import spearman, top_k, top_k_overlap  # noqa: E402

# Light-surface tokens of the reference palette (dataviz skill): slides are light.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FULL, LR = "#2a78d6", "#eb6834"                       # categorical slots 1, 2
DEPTH_RAMP = ["#86b6ef", "#2a78d6", "#104281", "#0d366b"]  # ordinal blue ramp for SFT epochs
BAND = "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "font.family": "sans-serif",
    "axes.formatter.useoffset": False, "axes.axisbelow": True,
})
MARK = dict(marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def depth_of(ck):
    return 0 if ck == "base" else int(ck.split("_ep")[1])


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)


def save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------- tables

def checkpoint_table(paths, cfg, R):
    rows = []
    cks = ["base"] + [f"{k}_ep{e}" for k in ("full", "lr") for e in cfg.sft.save_epochs]
    base_probe = load(paths.result("probe", "base"))
    metric = cfg.select.metric
    for ck in cks:
        ev, dg, pr = load(paths.result("eval", ck)), load(paths.result("diag", ck)), load(paths.result("probe", ck))
        pro = load(paths.result("probe_rollouts", ck))
        if not any([ev, dg, pr, pro]):
            continue
        row = {"ckpt": ck, "kind": ck.split("_")[0], "epoch": depth_of(ck)}
        if ev:
            row.update({f"pass@{k}": v for k, v in ev["pass_at_k"].items()})
            row.update(eval_len=ev["mean_length"], eval_trunc=ev["trunc_rate"])
        if dg:
            row.update(entropy=dg["entropy"])
            if R:
                row.update(cka_R=float(np.mean([dg["cka_input_vs_base"][i] for i in R])),
                           drift_R=float(np.mean([dg["rel_drift"][i] for i in R])),
                           drift_other=float(np.mean([d for i, d in enumerate(dg["rel_drift"]) if i not in R])))
        if pro:
            row.update(dead_rate=pro["dead_rate"], dead_all_correct=pro["dead_all_correct"],
                       dead_all_wrong=pro["dead_all_wrong"], probe_reward=pro["mean_reward"],
                       probe_len=pro["mean_length"], probe_trunc=pro["trunc_rate"])
        if pr:
            s = pr["mean"][metric]
            row.update(rollout_entropy=pr["rollout_entropy"], top2=str(top_k(s, 2)),
                       rank_stability=pr["rank_stability"][metric])
            if R:
                order = list(np.argsort(-np.asarray(s)))
                row["R_ranks"] = str([order.index(i) + 1 for i in R])
                row["R_grad_share"] = float(sum(s[i] ** 2 for i in R) / sum(x ** 2 for x in s))
                row["snr_R"] = float(np.mean([pr["mean"]["snr"][i] for i in R]))
            if base_probe:
                b = base_probe["mean"][metric]
                row.update(spearman_vs_base=spearman(b, s), top2_overlap=top_k_overlap(b, s, 2),
                           top4_overlap=top_k_overlap(b, s, 4))
        rows.append(row)
    return pd.DataFrame(rows)


def rl_table(paths, cfg):
    rows = []
    rdir = paths.results / "rl"
    for f in sorted(rdir.glob("*.json")) if rdir.exists() else []:
        r = json.loads(f.read_text())
        pre, post = load(paths.result("eval", r["init"])), load(paths.result("eval", f"rl__{r['name']}"))
        row = {"name": r["name"], "init": r["init"], "kind": r["init"].split("_")[0], "epoch": depth_of(r["init"]),
               "layers": r["layers"], "variant": r["name"].split("__")[-1], "trainable_params": r["trainable_params"],
               "hours": r["seconds"] / 3600, "rollout_tokens_M": r["rollout_tokens"] / 1e6,
               "mem_train_peak_gb": r.get("mem_train_peak_gb"), "mem_device_peak_gb": r.get("mem_device_peak_gb"),
               "reward_first": r["reward_first"], "reward_last": r["reward_last20pct"],
               "dead_first": r["dead_first"], "dead_last": r["dead_last20pct"]}
        for tag, ev in (("pre", pre), ("post", post)):
            if ev:
                for k, v in ev["pass_at_k"].items():
                    row[f"{tag}_pass@{k}"] = v
        if pre and post:
            row["gain_pass@1"] = post["pass_at_k"]["1"] - pre["pass_at_k"]["1"]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def fig_a3(paths, cfg, R, out):
    base = load(paths.result("probe", "base"))
    if not base:
        return
    metric = cfg.select.metric
    eps = [e for e in cfg.sft.save_epochs if paths.result("probe", f"full_ep{e}").exists()]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    L = base["n_layers"]
    x = np.arange(L)
    for i in R or []:
        ax.axvspan(i - 0.45, i + 0.45, color=BAND, zorder=0)
        ax.text(i, 0.02, "R", ha="center", va="bottom", color=INK2, fontsize=9, transform=ax.get_xaxis_transform())
    series = [("θ₀ (base)", base, INK2)] + [
        (f"full SFT ep{e}", load(paths.result("probe", f"full_ep{e}")), DEPTH_RAMP[min(j, 3)])
        for j, e in enumerate(eps)]
    for label, pr, color in series:
        m, s = np.asarray(pr["mean"][metric]), np.asarray(pr["std"][metric])
        ax.fill_between(x, (m - s) / m.max(), (m + s) / m.max(), color=color, alpha=0.12, linewidth=0)
        ax.plot(x, m / m.max(), color=color, label=label, **{**MARK, "markersize": 5})
    ax.set_xlabel("decoder layer")
    ax.set_ylabel(f"{metric} (÷ max per checkpoint)")
    ax.set_title(f"GRPO gradient by layer: θ₀ vs full SFT (mean ± sd, {base['num_batches']} batches)")
    ax.set_xticks(x[:: max(1, L // 14)])
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    # rank agreement with theta_0, by depth
    for kind, color in (("full", FULL), ("lr", LR)):
        pts = [(e, load(paths.result("probe", f"{kind}_ep{e}"))) for e in cfg.sft.save_epochs]
        pts = [(e, p) for e, p in pts if p]
        if not pts:
            continue
        xs = [0] + [e for e, _ in pts]
        ys = [1.0] + [spearman(base["mean"][metric], p["mean"][metric]) for _, p in pts]
        ax2.plot(xs, ys, color=color, label="full SFT" if kind == "full" else "LR-SFT", **MARK)
        end_label(ax2, xs[-1], ys[-1], f"{ys[-1]:.2f}")
    ax2.set_xticks([0] + list(cfg.sft.save_epochs))
    ax2.set_xlabel("SFT epochs")
    ax2.set_ylabel("Spearman ρ vs θ₀ ranking")
    ax2.set_title("Does the θ₀ ranking survive SFT?")
    ax2.set_ylim(min(-0.1, ax2.get_ylim()[0]), 1.05)
    ax2.legend(loc="lower left", fontsize=8)
    save(fig, out)


def lines_vs_depth(ax, df, col, title, ylabel, pct=False):
    if col not in df:
        ax.set_visible(False)
        return
    base = df[df.ckpt == "base"]
    for kind, color, label in (("full", FULL, "full SFT"), ("lr", LR, "LR-SFT")):
        d = pd.concat([base, df[df.kind == kind]]).dropna(subset=[col]).sort_values("epoch")
        if len(d) < 1:
            continue
        ys = d[col] * (100 if pct else 1)
        ax.plot(d.epoch, ys, color=color, label=label, **MARK)
        end_label(ax, d.epoch.iloc[-1], ys.iloc[-1], f"{ys.iloc[-1]:.1f}" if pct else f"{ys.iloc[-1]:.2f}")
    ax.set_title(title)
    ax.set_xticks(sorted(df.epoch.unique()))
    ax.set_xlabel("SFT epochs (0 = base)")
    ax.set_ylabel(ylabel)


def fig_a4(ck, cfg, out):
    if ck.empty or "pass@1" not in ck:
        return
    kmax = max(k for k in cfg.eval.ks if f"pass@{k}" in ck)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    lines_vs_depth(axes[0], ck, "pass@1", "MATH-500 pass@1 (pre-RL)", "%", pct=True)
    lines_vs_depth(axes[1], ck, f"pass@{kmax}", f"MATH-500 pass@{kmax} (pre-RL)", "%", pct=True)
    lines_vs_depth(axes[2], ck, "entropy", "Token entropy on own samples", "nats / token")
    lines_vs_depth(axes[3], ck, "dead_rate", "Dead GRPO groups (RL prompts)", "% of prompts", pct=True)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save(fig, out)


def fig_a4_reserved(ck, out):
    if ck.empty or "cka_R" not in ck:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    lines_vs_depth(axes[0], ck, "cka_R", "CKA of R's input vs θ₀", "linear CKA")
    lines_vs_depth(axes[1], ck, "drift_R", "Weight drift of R", "‖W − W₀‖ / ‖W₀‖")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save(fig, out)


def fig_b(rl, cfg, out):
    if rl.empty:
        return
    main = rl[rl.variant == "R"]
    if main.empty or "post_pass@1" not in main:
        return
    kmax = max(k for k in cfg.eval.ks if f"post_pass@{k}" in main)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for kind, color, label in (("full", FULL, "full SFT → RL on R"), ("lr", LR, "LR-SFT → RL on R")):
        d = main[main.kind == kind].sort_values("epoch")
        if d.empty:
            continue
        for ax, col in ((axes[0], "post_pass@1"), (axes[1], "gain_pass@1"), (axes[2], f"post_pass@{kmax}")):
            if col in d and d[col].notna().any():
                ys = d[col] * 100
                ax.plot(d.epoch, ys, color=color, label=label, **MARK)
                end_label(ax, d.epoch.iloc[-1], ys.iloc[-1], f"{ys.iloc[-1]:.1f}")
    axes[0].set_title("MATH-500 pass@1 after RL")
    axes[1].set_title("RL gain in pass@1")
    axes[2].set_title(f"MATH-500 pass@{kmax} after RL")
    axes[1].axhline(0, color=AXIS, linewidth=1)
    for ax in axes:
        ax.set_xticks(sorted(main.epoch.unique()))
        ax.set_xlabel("SFT epochs before RL")
        ax.set_ylabel("percentage points" if ax is axes[1] else "%")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save(fig, out)


def fig_controls(rl, cfg, out):
    deep = f"full_ep{max(cfg.sft.save_epochs)}"
    if rl.empty:
        return
    d = rl[rl.init == deep]
    if len(d) < 2 or "post_pass@1" not in d:
        return
    names = {"R": "R (θ₀)", "Rre": "R′ (after SFT)", "Rrand": "random 2", "all": "all params"}
    d = d.assign(order=d.variant.map({k: i for i, k in enumerate(names)})).sort_values("order")
    labels = [names.get(v, v) for v in d.variant]
    panels = [("post_pass@1", "MATH-500 pass@1 after RL (%)", 100), ("mem_train_peak_gb", "Training-side peak memory (GB)", 1),
              ("hours", "Wall-clock (h)", 1), ("rollout_tokens_M", "Rollout tokens (M)", 1)]
    panels = [p for p in panels if p[0] in d and d[p[0]].notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.2))
    for ax, (col, title, scale) in zip(np.atleast_1d(axes), panels):
        vals = d[col].fillna(0) * scale
        ax.bar(range(len(vals)), vals, width=0.55, color=FULL)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.3g}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", color=INK2,
                        fontsize=9)
        ax.set_xticks(range(len(vals)), labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"RL from {deep}: layer choice and cost", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out)


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    layers = load(paths.layers)
    R = layers["R"] if layers else None
    figs = paths.results / "figures"

    ck = checkpoint_table(paths, cfg, R)
    rl = rl_table(paths, cfg)
    if not ck.empty:
        ck.to_csv(paths.results / "summary_checkpoints.csv", index=False)
        print(ck.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not rl.empty:
        rl.to_csv(paths.results / "summary_rl.csv", index=False)
        print(rl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig_a3(paths, cfg, R, figs / "fig_a3_layer_ranking.png")
    fig_a4(ck, cfg, figs / "fig_a4_diagnostics.png")
    fig_a4_reserved(ck, figs / "fig_a4_reserved_layers.png")
    fig_b(rl, cfg, figs / "fig_b_rl_vs_depth.png")
    fig_controls(rl, cfg, figs / "fig_b_controls_cost.png")


if __name__ == "__main__":
    main()
