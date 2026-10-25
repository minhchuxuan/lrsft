"""Stage A1/A3: rank decoder layers by the GRPO gradient at a checkpoint.

Reads the probe rollouts of this checkpoint (rollout.py --task probe), keeps the informative
prompt groups (not all rollouts with the same reward), splits them into `probe.num_batches`
batches of `probe.batch_prompts` groups, and computes per-layer gradient statistics on each batch
(one backward per group, no optimizer step: nothing is updated or saved).

-> results/probe/<ckpt>.json : per-batch and mean +- std of grad_norm / fisher / snr / rel_grad,
   between-batch Spearman (rank stability), top-k layers per metric, dead-group stats.
"""

import argparse
import itertools
import random
import time

import numpy as np
import torch

from .common import device, load_model, load_tokenizer, read_json, read_jsonl, set_seed, write_json
from .config import Paths, add_config_args, config_from_args
from .metrics import spearman, top_k
from .scoring import batch_layer_stats

METRICS = ("grad_norm", "fisher", "snr", "rel_grad")


def summarize(per_batch: list[dict], k: int) -> dict:
    out = {"mean": {}, "std": {}, "topk": {}, "rank_stability": {}}
    for m in METRICS:
        arr = np.array([b[m] for b in per_batch])  # [batches, layers]
        out["mean"][m] = arr.mean(0).tolist()
        out["std"][m] = (arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])).tolist()
        out["topk"][m] = top_k(arr.mean(0), k)
        pairs = [spearman(a, b) for a, b in itertools.combinations(arr, 2)]
        out["rank_stability"][m] = float(np.mean(pairs)) if pairs else None
        # how often each layer lands in the per-batch top-k
        freq = np.zeros(arr.shape[1])
        for row in arr:
            freq[top_k(row, k)] += 1
        out.setdefault("topk_freq", {})[m] = (freq / len(arr)).tolist()
    return out


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    out = paths.result("probe", args.ckpt)
    if out.exists():
        print(f"probe {args.ckpt}: already done")
        return
    set_seed(cfg.seed)
    rows = read_jsonl(paths.rollouts(args.ckpt, "probe"))
    roll = read_json(paths.result("probe_rollouts", args.ckpt))

    # Informative = non-zero GRPO advantages and at least one completion that the loss keeps.
    informative = [r for r in rows if len(set(r["rewards"])) > 1
                   and not (cfg.rl.mask_truncated and all(r["truncated"]))]
    random.Random(cfg.seed).shuffle(informative)
    bp, nb = cfg.probe.batch_prompts, cfg.probe.num_batches
    n_batches = min(nb, len(informative) // bp)
    note = None
    if n_batches == 0:
        n_batches, bp = 1, len(informative)
        note = f"only {len(informative)} informative prompts; one batch of all of them"
    batches = [informative[i * bp : (i + 1) * bp] for i in range(n_batches)]

    dev = device()
    model = load_model(paths.ckpt(args.ckpt), dtype=torch.float32, attn_implementation=cfg.model.attn_implementation,
                       device_=dev)
    for name, p in model.named_parameters():
        p.requires_grad_(".layers." in name)  # all decoder layers; embeddings / lm_head (tied) frozen
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # HF only checkpoints in train mode; Qwen3 has no dropout, so this changes nothing else
    tok = load_tokenizer(paths.ckpt(args.ckpt))
    bf16 = dev.type == "cuda" and cfg.train.bf16

    per_batch = []
    t0 = time.time()
    for b, groups in enumerate(batches):
        stats = batch_layer_stats(model, groups, tok.pad_token_id, dev, cfg.probe.micro_batch_size, bf16,
                                  scale=cfg.rl.scale_rewards, mask_truncated=cfg.rl.mask_truncated)
        per_batch.append(stats)
        print(f"probe {args.ckpt} batch {b + 1}/{len(batches)}: top-{cfg.select.k} grad_norm "
              f"{top_k(stats['grad_norm'], cfg.select.k)} ({time.time() - t0:.0f}s)")

    result = {
        "ckpt": args.ckpt,
        "n_layers": len(per_batch[0]["layers"]),
        "num_batches": len(batches),
        "batch_prompts": bp,
        "informative_prompts": len(informative),
        "note": note,
        "seconds": time.time() - t0,
        "dead": {k: roll[k] for k in ("dead_rate", "dead_all_wrong", "dead_all_correct", "mean_reward")},
        "rollout_entropy": float(np.mean([b["rollout_entropy"] for b in per_batch])),
        "weight_norm": per_batch[0]["weight_norm"],
        "per_batch": [{m: b[m] for m in (*METRICS, "groups_used", "tokens")} for b in per_batch],
        **summarize(per_batch, cfg.select.k),
    }
    write_json(out, result)
    print(f"probe {args.ckpt}: topk={result['topk']} stability={result['rank_stability']}")


if __name__ == "__main__":
    main()
