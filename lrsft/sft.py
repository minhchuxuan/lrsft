"""Stage A2: full SFT or LR-SFT (reserved layers R frozen), same data / seed / schedule.

  python -m lrsft.sft --config ... --mode full
  python -m lrsft.sft --config ... --mode lr --layers 12,14

Saves bf16 snapshots <work>/models/{full,lr}_ep{E} at every E in sft.save_epochs (constant LR after
warmup, so "depth" = number of epochs, not a point on a decay schedule). HF Trainer checkpoints in
<work>/runs/sft_<mode> make the run resumable.
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from .common import (
    count_params,
    freeze_layers,
    is_model_dir,
    load_model,
    load_tokenizer,
    parse_layers,
    save_snapshot,
    set_seed,
    snapshot_dtype,
    write_json,
)
from .config import Paths, add_config_args, config_from_args


class EpochSnapshots(TrainerCallback):
    def __init__(self, epochs, out_dir_fn, tokenizer, dtype):
        self.epochs = set(epochs)
        self.out_dir_fn = out_dir_fn
        self.tokenizer = tokenizer
        self.dtype = dtype

    def on_epoch_end(self, args, state, control, model=None, **kw):
        epoch = int(round(state.epoch))
        out = self.out_dir_fn(epoch)
        if epoch in self.epochs and not is_model_dir(out) and state.is_world_process_zero:
            save_snapshot(model, self.tokenizer, out, self.dtype)
            print(f"sft: saved {out}")


class JsonlLog(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kw):
        import json

        if logs and state.is_world_process_zero:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}) + "\n")


def build_trainer(cfg, paths: Paths, mode: str, layers, model=None, train_dataset=None):
    from datasets import load_from_disk
    from trl import SFTConfig, SFTTrainer

    tok = load_tokenizer(paths.ckpt("base"))
    if model is None:
        model = load_model(paths.ckpt("base"), dtype=torch.float32, attn_implementation=cfg.model.attn_implementation)
    if mode == "lr":
        freeze_layers(model, layers)
    if train_dataset is None:
        train_dataset = load_from_disk(str(paths.data / "sft")).select_columns(["input_ids", "completion_mask"])
    s = cfg.sft
    name = f"sft_{mode}"
    run_dir = paths.trainer_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=max(s.save_epochs),
        per_device_train_batch_size=s.per_device_batch_size,
        gradient_accumulation_steps=s.grad_accum,
        learning_rate=s.lr,
        lr_scheduler_type=s.scheduler,
        warmup_steps=s.warmup,
        weight_decay=s.weight_decay,
        max_grad_norm=s.max_grad_norm,
        bf16=cfg.train.bf16,
        use_cpu=cfg.train.get("use_cpu", False),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # reentrant can drop grads around frozen blocks
        max_length=None,            # data is pre-filtered to sft_max_tokens
        completion_only_loss=True,  # loss on the trace + <|im_end|> only
        train_sampling_strategy="group_by_length",
        logging_steps=s.logging_steps,
        save_strategy="steps",
        save_steps=s.save_steps,
        save_total_limit=1,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to="none",
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset, processing_class=tok)
    snapshots = EpochSnapshots(s.save_epochs, lambda e: paths.ckpt(f"{mode}_ep{e}"), tok, snapshot_dtype(cfg))
    trainer.add_callback(snapshots)
    trainer.add_callback(JsonlLog(paths.results / "sft" / f"{name}_log.jsonl"))
    return trainer


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--mode", choices=["full", "lr"], required=True)
    parser.add_argument("--layers", default="", help="reserved layers R (mode=lr)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    layers = parse_layers(args.layers)
    if args.mode == "lr" and not layers:
        raise SystemExit("--mode lr needs --layers")
    out = paths.result("sft", f"sft_{args.mode}")
    targets = [paths.ckpt(f"{args.mode}_ep{e}") for e in cfg.sft.save_epochs]
    if out.exists() and all(is_model_dir(t) for t in targets):
        print(f"sft {args.mode}: already done")
        return
    set_seed(cfg.seed)
    (paths.results / "sft").mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(cfg, paths, args.mode, layers)
    model = trainer.model
    trainable = count_params(model, trainable_only=True)
    print(f"sft {args.mode}: trainable {trainable:,} / {count_params(model):,} params, frozen layers {layers}")
    if args.mode == "lr":
        bad = [n for n, p in trainer.model.named_parameters() if p.requires_grad and any(
            n.startswith(f"model.layers.{i}.") for i in layers)]
        assert not bad, bad

    resume = any(paths.trainer_dir(f"sft_{args.mode}").glob("checkpoint-*"))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_out = trainer.train(resume_from_checkpoint=True if resume else None)
    # A resumed run may have skipped an epoch end that was already saved; make sure the last one exists.
    last = paths.ckpt(f"{args.mode}_ep{max(cfg.sft.save_epochs)}")
    if not is_model_dir(last):
        save_snapshot(trainer.model, trainer.processing_class, last, snapshot_dtype(cfg))
    missing = [str(t) for t in targets if not is_model_dir(t)]
    if missing:
        raise RuntimeError(f"missing epoch snapshots: {missing} (resumed past them? rerun with a fresh run dir)")
    write_json(out, {
        "mode": args.mode, "layers": layers, "trainable_params": trainable, "total_params": count_params(model),
        "seconds": time.time() - t0, "resumed": resume, "metrics": train_out.metrics,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "epochs": list(cfg.sft.save_epochs),
    })


if __name__ == "__main__":
    main()
