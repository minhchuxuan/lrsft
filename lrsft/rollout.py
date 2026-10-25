"""Sampling with vLLM (GPU) or HF generate (CPU smoke tests), then math-verify scoring.

  --task probe : RL prompts of the probe pool, n=probe.num_samples at the RL temperature.
                 Rollouts feed the layer probe; the summary gives the dead-group rate.
                 -> <work>/rollouts/<ckpt>__probe.jsonl, results/probe_rollouts/<ckpt>.json
  --task eval  : MATH-500, n=eval.n -> pass@k (unbiased), lengths, truncation.
                 -> <work>/rollouts/<ckpt>__eval.jsonl, results/eval/<ckpt>.json
"""

import argparse
import gc
import os
import time
import zlib

import numpy as np

from .common import encode_prompt, load_tokenizer, read_jsonl, set_seed, stop_token_ids, write_json, write_jsonl
from .config import Paths, add_config_args, config_from_args
from .metrics import group_stats, mean_pass_at_k
from .reward import score_parallel


def generate_vllm(model_dir, prompt_ids, n, temperature, top_p, max_tokens, seed, cfg, stops):
    # The engine core runs in a child process; with "fork" it dies if CUDA was touched in this process
    # first (seen on a Turing GPU). "spawn" is safe everywhere.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    max_prompt = max(len(p) for p in prompt_ids)
    llm = LLM(
        model=str(model_dir),
        dtype="auto",  # = the snapshot's config.dtype (bf16; fp16 on pre-Ampere debug GPUs)
        seed=seed,
        gpu_memory_utilization=cfg.gen.gpu_memory_utilization,
        max_model_len=max_prompt + max_tokens,
        enable_prefix_caching=True,
        enforce_eager=bool(cfg.gen.get("enforce_eager", False)),
    )
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stops)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sampling_params=sp, use_tqdm=True)
    groups = [[(list(o.token_ids), o.finish_reason == "length") for o in out.outputs] for out in outs]
    del llm  # release the engine before scoring spawns workers
    gc.collect()
    return groups


def generate_hf(model_dir, prompt_ids, n, temperature, top_p, max_tokens, seed, cfg, stops):
    import torch

    from .common import device, load_model

    dev = device()
    model = load_model(model_dir, dtype=torch.float32, attn_implementation=cfg.model.attn_implementation, device_=dev)
    model.eval()
    results = []
    for p in prompt_ids:
        torch.manual_seed(seed + len(results))
        x = torch.tensor([p], device=dev)
        with torch.no_grad():
            out = model.generate(
                x, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens,
                num_return_sequences=n, eos_token_id=stops, pad_token_id=stops[-1], use_cache=True,
            )
        group = []
        for seq in out[:, len(p):].tolist():
            ids = []
            for t in seq:
                ids.append(t)
                if t in stops:
                    break
            group.append((ids, ids[-1] not in stops if ids else True))
        results.append(group)
    return results


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--ckpt", required=True, help="model name under <work>/models")
    parser.add_argument("--task", choices=["probe", "eval"], required=True)
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    kind = "probe_rollouts" if args.task == "probe" else "eval"
    out_json = paths.result(kind, args.ckpt)
    out_rows = paths.rollouts(args.ckpt, args.task)
    if out_json.exists() and out_rows.exists():
        print(f"rollout {args.ckpt} {args.task}: already done")
        return
    set_seed(cfg.seed)
    model_dir = paths.ckpt(args.ckpt)
    tok = load_tokenizer(model_dir)
    stops = stop_token_ids(tok)

    if args.task == "probe":
        data = read_jsonl(paths.data / "probe_pool.jsonl")
        n, temp, top_p = cfg.probe.num_samples, cfg.rl.temperature, 1.0
        max_tokens = cfg.rl.max_completion_tokens
    else:
        data = read_jsonl(paths.data / "math500.jsonl")
        if cfg.eval.get("limit"):
            data = data[: cfg.eval.limit]
        n, temp, top_p = cfg.eval.n, cfg.eval.temperature, cfg.eval.top_p
        max_tokens = cfg.eval.max_tokens
    prompt_ids = [encode_prompt(tok, r["prompt"]) for r in data]

    gen = generate_vllm if cfg.gen.backend == "vllm" else generate_hf
    t0 = time.time()
    groups = gen(model_dir, prompt_ids, n, temp, top_p, max_tokens, cfg.seed, cfg, stops)
    gen_time = time.time() - t0

    texts = [tok.decode(ids, skip_special_tokens=True) for g in groups for ids, _ in g]
    golds = [r["answer"] for r, g in zip(data, groups) for _ in g]
    if cfg.debug.get("fake_reward"):
        # Smoke tests only: a random-but-deterministic reward so that "informative" groups exist.
        rewards = [float(zlib.crc32(t.encode()) % 2) for t in texts]
    else:
        rewards = score_parallel(texts, golds, workers=cfg.gen.get("score_workers", 8))

    rows, k = [], 0
    for i, (r, g) in enumerate(zip(data, groups)):
        rr = rewards[k : k + len(g)]
        k += len(g)
        row = {"idx": i, "answer": r["answer"], "level": r.get("level"), "rewards": rr,
               "lengths": [len(ids) for ids, _ in g], "truncated": [bool(t) for _, t in g],
               "prompt_ids": prompt_ids[i]}
        # Probe needs every completion; eval keeps two samples per problem for the entropy diagnostic.
        keep = g if args.task == "probe" else g[: cfg.eval.keep_completions]
        row["completion_ids"] = [ids for ids, _ in keep]
        rows.append(row)

    lengths = np.array([x for r in rows for x in r["lengths"]])
    summary = {
        "ckpt": args.ckpt, "task": args.task, "n_prompts": len(rows), "n_samples": n,
        "temperature": temp, "top_p": top_p, "max_tokens": max_tokens,
        "mean_length": float(lengths.mean()), "trunc_rate": float(np.mean([t for r in rows for t in r["truncated"]])),
        "gen_seconds": gen_time, "generated_tokens": int(lengths.sum()),
        **group_stats([r["rewards"] for r in rows]),
    }
    if args.task == "eval":
        cs = [int(sum(r["rewards"])) for r in rows]
        ns = [len(r["rewards"]) for r in rows]
        summary["pass_at_k"] = {str(kk): mean_pass_at_k(ns, cs, kk) for kk in cfg.eval.ks if kk <= n}
        summary["pass1_by_level"] = {
            str(lv): mean_pass_at_k([x for x, r in zip(ns, rows) if r["level"] == lv],
                                    [x for x, r in zip(cs, rows) if r["level"] == lv], 1)
            for lv in sorted({r["level"] for r in rows})
        }
    write_jsonl(out_rows, rows)
    write_json(out_json, summary)
    print(f"rollout {args.ckpt} {args.task}:", {k: v for k, v in summary.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
