"""Stage B: GRPO with DAPO tricks on the reserved layers only (or on all parameters, as a baseline).

  python -m lrsft.rl --config ... --init full_ep8 --layers 12,14 --name full_ep8__R
  python -m lrsft.rl --config ... --init full_ep8 --layers all   --name full_ep8__all

TRL GRPOTrainer, vLLM colocated on the same GPU. DAPO pieces used: token-level loss
(loss_type="dapo"), clip-higher (epsilon_high), masking of truncated completions, no KL.
Dynamic sampling is not available in TRL; dead groups are logged instead (frac_reward_zero_std).
Only parameters with requires_grad reach the optimizer; TRL pushes every parameter to vLLM after
each update (cheap at 0.6B), so frozen layers are synced too and stay consistent.

-> <work>/models/rl__<name> (bf16), results/rl/<name>.json, results/rl/<name>_log.jsonl
"""

import argparse
import os
import shutil
import time

import torch
from transformers import TrainerCallback

from .common import (
    count_params,
    encode_prompt,
    load_model,
    load_tokenizer,
    parse_layers,
    patch_generation_config,
    read_jsonl,
    save_snapshot,
    set_seed,
    snapshot_dtype,
    stop_token_ids,
    train_only_layers,
    write_json,
)
from .config import Paths, add_config_args, config_from_args
from .reward import score
from .sft import JsonlLog


class MemoryCallback(TrainerCallback):
    """Training-side peak memory. vLLM's pool is allocated before training starts, so the peak above
    that baseline is what the optimizer, gradients and activations cost."""

    def __init__(self):
        self.baseline = self.peak = self.device_peak = 0

    def on_train_begin(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kw):
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            self.device_peak = max(self.device_peak, total - free)

    def report(self):
        if not torch.cuda.is_available():
            return {}
        peak = torch.cuda.max_memory_allocated()
        gb = 2**30
        return {"mem_baseline_gb": self.baseline / gb, "mem_peak_alloc_gb": peak / gb,
                "mem_train_peak_gb": (peak - self.baseline) / gb, "mem_device_peak_gb": self.device_peak / gb}


def make_reward(counter: dict, fake: bool):
    def math_reward(completions, answer, completion_ids=None, **kwargs):
        counter["rollouts"] += len(completions)
        counter["rollout_tokens"] += sum(len(c) for c in completion_ids) if completion_ids is not None else 0
        if fake:  # CPU smoke tests only
            import zlib

            return [float(zlib.crc32(c.encode()) % 2) for c in completions]
        return score(completions, answer)

    return math_reward


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--init", required=True, help="start checkpoint under <work>/models")
    parser.add_argument("--layers", required=True, help="comma list of decoder layers, or 'all'")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    out = paths.result("rl", args.name)
    final = paths.ckpt(f"rl__{args.name}")
    if out.exists():  # the snapshot may already be deleted after its eval (disk budget)
        print(f"rl {args.name}: already done")
        return
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # needs nvcc to JIT; see rollout.py
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    set_seed(cfg.seed)
    r = cfg.rl
    tok = load_tokenizer(paths.ckpt(args.init))
    tok.padding_side = "left"
    model = load_model(paths.ckpt(args.init), dtype=torch.float32, attn_implementation=cfg.model.attn_implementation)
    patch_generation_config(model, tok)  # TRL reads eos ids (incl. <|im_end|>) from here for masking
    full = args.layers == "all"
    layers = None if full else parse_layers(args.layers)
    if not full:
        train_only_layers(model, layers)
    trainable = count_params(model, trainable_only=True)

    rows = read_jsonl(paths.data / "rl_train.jsonl")
    ds = Dataset.from_list([{"prompt": x["prompt"], "answer": x["answer"]} for x in rows])
    max_prompt = max(len(encode_prompt(tok, x["prompt"])) for x in rows)

    gen_batch = r.prompts_per_round * r.num_generations
    per_update = r.per_device_batch_size * r.grad_accum
    if gen_batch % per_update:
        raise ValueError(f"prompts_per_round*num_generations ({gen_batch}) must be a multiple of "
                         f"per_device_batch_size*grad_accum ({per_update})")
    updates_per_round = gen_batch // per_update
    run_dir = paths.trainer_dir(f"rl_{args.name}")
    gargs = GRPOConfig(
        output_dir=str(run_dir),
        learning_rate=r.lr_full if full else r.lr_subset,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=r.warmup_rounds * updates_per_round,
        max_steps=r.rounds * updates_per_round,
        per_device_train_batch_size=r.per_device_batch_size,
        gradient_accumulation_steps=r.grad_accum,
        generation_batch_size=gen_batch,
        num_generations=r.num_generations,
        max_completion_length=r.max_completion_tokens,
        temperature=r.temperature,
        top_p=1.0,
        loss_type="dapo",
        epsilon=r.epsilon,
        epsilon_high=r.epsilon_high,
        mask_truncated_completions=r.mask_truncated,
        beta=0.0,
        scale_rewards=r.scale_rewards,
        max_grad_norm=r.max_grad_norm,
        weight_decay=0.0,
        use_vllm=r.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=r.vllm_gpu_memory_utilization,
        vllm_max_model_length=max_prompt + r.max_completion_tokens,
        # Per-token truncated IS for the vLLM-vs-HF logprob mismatch. TRL's default (sequence_mask) sums the
        # log-ratio over all tokens, so a small per-token bias can mask whole 4k-token answers.
        vllm_importance_sampling_mode=r.is_mode,
        generation_kwargs={"stop_token_ids": stop_token_ids(tok)} if r.use_vllm else None,
        bf16=cfg.train.bf16,
        use_cpu=cfg.train.get("use_cpu", False),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        # Resume checkpoints only for R-only runs (~2.7 GB); full-parameter ones would be ~7 GB each.
        save_strategy="no" if full or not r.save_every_rounds else "steps",
        save_steps=(r.save_every_rounds or 1) * updates_per_round,
        save_total_limit=1,
        seed=cfg.seed,
        report_to="none",
    )
    counter = {"rollouts": 0, "rollout_tokens": 0}
    mem = MemoryCallback()
    (paths.results / "rl").mkdir(parents=True, exist_ok=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward(counter, bool(cfg.debug.get("fake_reward"))),
        args=gargs,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[mem, JsonlLog(paths.results / "rl" / f"{args.name}_log.jsonl")],
    )
    n_opt = sum(p.numel() for g in trainer.create_optimizer().param_groups for p in g["params"])
    assert n_opt == trainable, (n_opt, trainable)  # the optimizer only sees the reserved layers
    print(f"rl {args.name}: init={args.init} layers={args.layers} trainable={trainable:,} "
          f"rounds={r.rounds} updates/round={updates_per_round}")

    # Frozen weights must come out of RL untouched (guards against e.g. a stray weight decay).
    frozen_ref = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if not p.requires_grad}

    resume = any(run_dir.glob("checkpoint-*"))
    t0 = time.time()
    train_out = trainer.train(resume_from_checkpoint=True if resume else None)
    seconds = time.time() - t0
    changed = [n for n, p in trainer.model.named_parameters() if n in frozen_ref and not torch.equal(
        p.detach().cpu(), frozen_ref[n])]
    if changed:
        raise RuntimeError(f"frozen parameters changed during RL: {changed[:5]}")
    save_snapshot(trainer.model, tok, final, snapshot_dtype(cfg))
    if cfg.disk.get("delete_run_dirs"):
        shutil.rmtree(run_dir, ignore_errors=True)

    hist = [h for h in trainer.state.log_history if "reward" in h]
    tail = hist[-max(1, len(hist) // 5):]
    write_json(out, {
        "name": args.name, "init": args.init, "layers": args.layers, "trainable_params": trainable,
        "total_params": count_params(model), "rounds": r.rounds, "updates_per_round": updates_per_round,
        "lr": gargs.learning_rate, "seconds": seconds, "resumed": resume, "frozen_unchanged": True,
        "rollouts": counter["rollouts"], "rollout_tokens": counter["rollout_tokens"],
        "reward_first": hist[0]["reward"] if hist else None,
        "reward_last20pct": sum(h["reward"] for h in tail) / len(tail) if tail else None,
        "dead_first": hist[0].get("frac_reward_zero_std") if hist else None,
        "dead_last20pct": sum(h.get("frac_reward_zero_std", 0) for h in tail) / len(tail) if tail else None,
        **mem.report(), "train_metrics": train_out.metrics,
    })


if __name__ == "__main__":
    main()
