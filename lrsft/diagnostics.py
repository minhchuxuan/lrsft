"""Stage A4 (HF part): pre-RL diagnostics of a checkpoint, compared with the base model.

  entropy   : mean next-token entropy on the checkpoint's own MATH-500 samples (from rollout --task eval)
  cka       : linear CKA between the inputs of every decoder layer at this checkpoint and at the base,
              on the same token sequences (MATH-500 prompt + reference solution), three ways:
              raw residual (cka_input_vs_base; dominated by massive activations, ~1.0 everywhere),
              after the layer's input_layernorm, i.e. what its attention reads (cka_normed_vs_base),
              and raw without the base model's outlier tokens/channels (cka_robust_vs_base)
  weights   : per-layer weight norm and relative drift ||W - W0|| / ||W0||

-> results/diag/<ckpt>.json
"""

import argparse
import random

import numpy as np
import torch

from .common import END_OF_TURN, device, layer_of, load_model, load_tokenizer, read_json, read_jsonl, set_seed, write_json
from .config import Paths, add_config_args, config_from_args
from .metrics import linear_cka
from .scoring import autocast_ctx, pad_batch, token_logprobs


@torch.no_grad()
def layer_inputs(model, seqs, dev, bf16, token_idx):
    """Residual-stream input and input_layernorm output of each decoder layer at the chosen
    (sequence, position) pairs -> (raw, normed), each a list of [N, d] tensors."""
    captured, captured_n = {}, {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def hook(mod, args, kwargs, i=i):
            h = args[0] if args else kwargs["hidden_states"]
            captured[i] = h.detach().float()

        def hook_norm(mod, args, out, i=i):
            captured_n[i] = out.detach().float()
        hooks.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
        hooks.append(layer.input_layernorm.register_forward_hook(hook_norm))
    n = len(model.model.layers)
    feats, feats_n = {i: [] for i in range(n)}, {i: [] for i in range(n)}
    try:
        for s, ids in enumerate(seqs):
            pos = [p for (si, p) in token_idx if si == s]
            if not pos:
                continue
            x = torch.tensor([ids], device=dev)
            with autocast_ctx(dev, bf16):
                model.model(input_ids=x)
            for i in range(n):
                feats[i].append(captured[i][0, pos].cpu())
                feats_n[i].append(captured_n[i][0, pos].cpu())
    finally:
        for h in hooks:
            h.remove()
    return [torch.cat(feats[i]) for i in range(n)], [torch.cat(feats_n[i]) for i in range(n)]


def outlier_masks(x0, factor=10.0):
    """Tokens / channels of the base representation with massive activations (> factor x median)."""
    norms = x0.norm(dim=1)
    keep_tok = norms <= factor * norms.median()
    mag = x0[keep_tok].abs().mean(0)
    keep_dim = mag <= factor * mag.median()
    return keep_tok, keep_dim


def cka_report(feats, feats0, feats_n, feats0_n) -> dict:
    out = {"cka_input_vs_base": [], "cka_normed_vs_base": [], "cka_robust_vs_base": [],
           "base_outlier_tokens": [], "base_outlier_channels": []}
    for x, x0, xn, x0n in zip(feats, feats0, feats_n, feats0_n):
        keep_tok, keep_dim = outlier_masks(x0)
        out["cka_input_vs_base"].append(linear_cka(x, x0))
        out["cka_normed_vs_base"].append(linear_cka(xn, x0n))
        out["cka_robust_vs_base"].append(linear_cka(x[keep_tok][:, keep_dim], x0[keep_tok][:, keep_dim]))
        out["base_outlier_tokens"].append(int((~keep_tok).sum()))
        out["base_outlier_channels"].append(int((~keep_dim).sum()))
    return out


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


def weight_stats(model, base_sd):
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
    parser.add_argument("--cka-only", action="store_true",
                        help="recompute the CKA fields of an existing diag JSON (no entropy pass)")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    out = paths.result("diag", args.ckpt)
    if args.threads:
        torch.set_num_threads(args.threads)
    if out.exists() and not args.cka_only:
        print(f"diag {args.ckpt}: already done")
        return
    if args.cka_only and not out.exists():
        raise SystemExit(f"--cka-only needs an existing {out}")
    set_seed(cfg.seed)
    dev = torch.device(args.device) if args.device else device()
    bf16 = dev.type == "cuda" and cfg.train.bf16
    tok = load_tokenizer(paths.ckpt(args.ckpt))
    kw = dict(dtype=torch.float32, attn_implementation=cfg.model.attn_implementation, device_=dev)

    # Fixed probe text for CKA: the same sequences and token positions for every checkpoint.
    m500 = read_jsonl(paths.data / "math500.jsonl")[: cfg.eval.cka_samples]
    seqs = [tok(text=r["prompt"] + r["solution"] + END_OF_TURN)["input_ids"] for r in m500]
    all_idx = [(s, p) for s, ids in enumerate(seqs) for p in range(len(ids))]
    token_idx = sorted(random.Random(cfg.seed).sample(all_idx, min(cfg.eval.cka_tokens, len(all_idx))))

    # One model on the GPU at a time (fits small debug GPUs); the base weights wait on the CPU.
    base = load_model(paths.ckpt("base"), **kw).eval()
    feats0, feats0_n = layer_inputs(base, seqs, dev, bf16, token_idx)
    base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    del base
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    model = load_model(paths.ckpt(args.ckpt), **kw).eval()
    feats, feats_n = layer_inputs(model, seqs, dev, bf16, token_idx)
    cka = cka_report(feats, feats0, feats_n, feats0_n)

    if args.cka_only:
        result = read_json(out)
        result.update(cka, cka_tokens=len(token_idx))
    else:
        eval_rows = read_jsonl(paths.rollouts(args.ckpt, "eval"))
        ent, n_ent = mean_entropy(model, eval_rows, tok, dev, bf16, cfg.eval.entropy_samples)
        result = {"ckpt": args.ckpt, "entropy": ent, "entropy_tokens": n_ent, **cka,
                  "cka_tokens": len(token_idx), **weight_stats(model, base_sd)}
    write_json(out, result)
    print(f"diag {args.ckpt}: entropy={result['entropy']:.3f} cka raw/normed/robust (min) = "
          f"{min(cka['cka_input_vs_base']):.3f}/{min(cka['cka_normed_vs_base']):.3f}/"
          f"{min(cka['cka_robust_vs_base']):.3f}")


if __name__ == "__main__":
    main()
