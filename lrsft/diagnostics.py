"""Stage A4 (HF part): pre-RL diagnostics of a checkpoint, compared with the base model.

  entropy   : mean next-token entropy on the checkpoint's own MATH-500 samples (from rollout --task eval)
  cka       : linear CKA between the inputs of every decoder layer at this checkpoint and at the base,
              on the same token sequences (MATH-500 prompt + reference solution)
  weights   : per-layer weight norm and relative drift ||W - W0|| / ||W0||

-> results/diag/<ckpt>.json
"""

import argparse
import random

import numpy as np
import torch

from .common import END_OF_TURN, device, layer_of, load_model, load_tokenizer, read_jsonl, set_seed, write_json
from .config import Paths, add_config_args, config_from_args
from .metrics import linear_cka
from .scoring import autocast_ctx, pad_batch, token_logprobs


@torch.no_grad()
def layer_inputs(model, seqs, dev, bf16, token_idx):
    """Inputs of each decoder layer at the chosen (sequence, position) pairs -> list of [N, d] tensors."""
    captured = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def hook(mod, args, kwargs, i=i):
            h = args[0] if args else kwargs["hidden_states"]
            captured[i] = h.detach().float()
        hooks.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
    feats = {i: [] for i in range(len(model.model.layers))}
    try:
        for s, ids in enumerate(seqs):
            pos = [p for (si, p) in token_idx if si == s]
            if not pos:
                continue
            x = torch.tensor([ids], device=dev)
            with autocast_ctx(dev, bf16):
                model.model(input_ids=x)
            for i, h in captured.items():
                feats[i].append(h[0, pos].cpu())
    finally:
        for h in hooks:
            h.remove()
    return [torch.cat(feats[i]) for i in sorted(feats)]


@torch.no_grad()
def mean_entropy(model, rows, tok, dev, bf16, limit):
    tot, n = 0.0, 0
    for r in rows[:limit]:
        for comp in r["completion_ids"][:1]:
            if not comp:
                continue
            ids, attn, mask = pad_batch([(r["prompt_ids"], comp)], tok.pad_token_id, dev)
            with autocast_ctx(dev, bf16):
                _, ent, _ = token_logprobs(model, ids, attn, mask)
            tot += float(ent.sum())
            n += ent.numel()
    return tot / max(n, 1), n


def weight_stats(model, base):
    base_sd = {k: v for k, v in base.state_dict().items()}
    per = {}
    for name, p in model.state_dict().items():
        layer = layer_of(name)
        if layer is None:
            continue
        w, w0 = p.float(), base_sd[name].float().to(p.device)
        d = per.setdefault(layer, [0.0, 0.0, 0.0])
        d[0] += float(w.pow(2).sum())
        d[1] += float(w0.pow(2).sum())
        d[2] += float((w - w0).pow(2).sum())
    layers = sorted(per)
    return {
        "weight_norm": [float(np.sqrt(per[i][0])) for i in layers],
        "rel_drift": [float(np.sqrt(per[i][2] / per[i][1])) for i in layers],
    }


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    out = paths.result("diag", args.ckpt)
    if out.exists():
        print(f"diag {args.ckpt}: already done")
        return
    set_seed(cfg.seed)
    dev = device()
    bf16 = dev.type == "cuda" and cfg.train.bf16
    tok = load_tokenizer(paths.ckpt(args.ckpt))
    kw = dict(dtype=torch.float32, attn_implementation=cfg.model.attn_implementation, device_=dev)
    model = load_model(paths.ckpt(args.ckpt), **kw).eval()
    base = load_model(paths.ckpt("base"), **kw).eval()

    eval_rows = read_jsonl(paths.rollouts(args.ckpt, "eval"))
    ent, n_ent = mean_entropy(model, eval_rows, tok, dev, bf16, cfg.eval.entropy_samples)

    # Fixed probe text for CKA: the same sequences and token positions for every checkpoint.
    m500 = read_jsonl(paths.data / "math500.jsonl")[: cfg.eval.cka_samples]
    seqs = [tok(text=r["prompt"] + r["solution"] + END_OF_TURN)["input_ids"] for r in m500]
    all_idx = [(s, p) for s, ids in enumerate(seqs) for p in range(len(ids))]
    token_idx = sorted(random.Random(cfg.seed).sample(all_idx, min(cfg.eval.cka_tokens, len(all_idx))))
    feats = layer_inputs(model, seqs, dev, bf16, token_idx)
    feats0 = layer_inputs(base, seqs, dev, bf16, token_idx)
    cka = [linear_cka(a, b) for a, b in zip(feats, feats0)]

    write_json(out, {"ckpt": args.ckpt, "entropy": ent, "entropy_tokens": n_ent, "cka_input_vs_base": cka,
                     "cka_tokens": len(token_idx), **weight_stats(model, base)})
    print(f"diag {args.ckpt}: entropy={ent:.3f} cka(min)={min(cka):.3f}")


if __name__ == "__main__":
    main()
