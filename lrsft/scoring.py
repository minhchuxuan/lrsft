"""HF-side scoring: per-token log-probs / entropy without materialising full logits, GRPO advantages,
and per-layer statistics of the GRPO policy gradient at the current weights."""

from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .common import layer_of


def _chunk_logp_entropy(h, targets, weight):
    logits = (h @ weight.T).float()
    logz = torch.logsumexp(logits, dim=-1)
    logp = logits.gather(-1, targets[:, None]).squeeze(-1) - logz
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)
        entropy = logz - (p * logits).sum(-1)
    return logp, entropy


def token_logprobs(model, input_ids, attention_mask, target_mask, chunk=1024):
    """log p(token) and entropy at every position whose *token* is marked in `target_mask`.

    Returns flat tensors over the selected tokens and the row index of each one. The LM head runs
    in checkpointed chunks, so a 4k-token batch never holds [B, L, 152k] logits for backward.
    """
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    sel = target_mask[:, 1:].bool()
    h = hidden[:, :-1][sel]
    t = input_ids[:, 1:][sel]
    rows = sel.nonzero()[:, 0]
    weight = model.lm_head.weight
    lps, ents = [], []
    for s in range(0, h.shape[0], chunk):
        if torch.is_grad_enabled():
            lp, en = checkpoint(_chunk_logp_entropy, h[s : s + chunk], t[s : s + chunk], weight, use_reentrant=False)
        else:
            lp, en = _chunk_logp_entropy(h[s : s + chunk], t[s : s + chunk], weight)
        lps.append(lp)
        ents.append(en)
    if not lps:
        z = hidden.new_zeros(0, dtype=torch.float32)
        return z, z, rows
    return torch.cat(lps), torch.cat(ents), rows


def grpo_advantages(rewards, scale="group", eps=1e-4) -> np.ndarray:
    """Same formula as TRL's GRPOTrainer (unbiased std within the group)."""
    r = torch.tensor(rewards, dtype=torch.float32)
    a = r - r.mean()
    if scale == "group":
        a = a / (r.std() + eps)
    return a.numpy()


def pad_batch(seqs, pad_id, device):
    """Right-padded [B, L] ids, attention mask, and mask of completion tokens."""
    L = max(len(p) + len(c) for p, c in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    comp = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (p, c) in enumerate(seqs):
        n = len(p) + len(c)
        ids[i, :n] = torch.tensor(p + c)
        attn[i, :n] = 1
        comp[i, len(p) : n] = 1
    return ids.to(device), attn.to(device), comp.to(device)


def autocast_ctx(device, bf16: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if bf16 else nullcontext()


def group_backward(model, group, pad_id, device, micro_bs, bf16, scale="group", mask_truncated=True, chunk=1024):
    """Backward of one prompt group's GRPO loss at ratio = 1 (on-policy), token-mean over the group:

        L = - sum_i A_i sum_t log pi(o_i,t) / sum_i |o_i|

    At ratio 1 the clipped DAPO surrogate has exactly this gradient. Returns (n_tokens, entropy_sum).
    """
    adv = grpo_advantages(group["rewards"], scale)
    keep = [i for i in range(len(adv)) if not (mask_truncated and group["truncated"][i])]
    n_tok = sum(len(group["completion_ids"][i]) for i in keep)
    if n_tok == 0 or np.all(adv[keep] == 0):
        return 0, 0.0
    ent_sum = 0.0
    for s in range(0, len(keep), micro_bs):
        idx = keep[s : s + micro_bs]
        seqs = [(group["prompt_ids"], group["completion_ids"][i]) for i in idx]
        ids, attn, comp = pad_batch(seqs, pad_id, device)
        with autocast_ctx(device, bf16):
            logp, ent, rows = token_logprobs(model, ids, attn, comp, chunk=chunk)
        a = torch.tensor(adv[idx], dtype=torch.float32, device=device)[rows]
        loss = -(a * logp).sum() / n_tok
        loss.backward()
        ent_sum += float(ent.sum())
    return n_tok, ent_sum


def layer_params(model) -> dict[int, list]:
    out = defaultdict(list)
    for name, p in model.named_parameters():
        layer = layer_of(name)
        if layer is not None and p.requires_grad:
            out[layer].append(p)
    return dict(sorted(out.items()))


def batch_layer_stats(model, groups, pad_id, device, micro_bs=4, bf16=True, scale="group", mask_truncated=True,
                      acc_device=None, chunk=1024):
    """Per-layer statistics of the GRPO gradient over a batch of prompt groups.

    g_j = gradient of group j's loss; g = mean_j g_j (the batch gradient GRPO would follow).
      grad_norm[l] = ||g restricted to layer l||
      fisher[l]    = mean_j ||g_j[l]||^2      (empirical Fisher trace at group level)
      snr[l]       = ||g[l]||^2 / tr Var_j(g_j[l])
    All decoder layers of Qwen3 have identical shapes, so normalising by parameter count does not
    change the ranking between layers; rel_grad = grad_norm / ||W_l|| is reported as a scale-free view.
    """
    lp = layer_params(model)
    params = [p for ps in lp.values() for p in ps]
    # acc_device="cpu" keeps the running gradient sum off a small GPU (one D2H copy per group)
    acc = [torch.zeros_like(p, dtype=torch.float32, device=acc_device or p.device) for p in params]
    sq = []
    tokens, ent = 0, 0.0
    used = 0
    for group in groups:
        model.zero_grad(set_to_none=True)
        n_tok, e = group_backward(model, group, pad_id, device, micro_bs, bf16, scale, mask_truncated, chunk)
        if n_tok == 0:
            continue
        used += 1
        tokens += n_tok
        ent += e
        row = []
        for layer, ps in lp.items():
            row.append(sum(float(p.grad.float().pow(2).sum()) for p in ps if p.grad is not None))
        sq.append(row)
        for a, p in zip(acc, params):
            if p.grad is not None:
                a.add_(p.grad.float().to(a.device))
    model.zero_grad(set_to_none=True)
    if used == 0:
        raise RuntimeError("no group with non-zero advantage and untruncated completions")
    sq = np.array(sq)  # [groups, layers]
    mean_sq = []
    i = 0
    for layer, ps in lp.items():
        mean_sq.append(sum(float((acc[i + j] / used).pow(2).sum()) for j in range(len(ps))))
        i += len(ps)
    mean_sq = np.array(mean_sq)
    fisher = sq.mean(0)
    var = np.maximum(fisher - mean_sq, 0.0) * used / max(used - 1, 1)
    wnorm = np.array([np.sqrt(sum(float(p.detach().float().pow(2).sum()) for p in ps)) for ps in lp.values()])
    gn = np.sqrt(mean_sq)
    return {
        "layers": list(lp.keys()),
        "grad_norm": gn.tolist(),
        "fisher": fisher.tolist(),
        "snr": (mean_sq / np.maximum(var, 1e-30)).tolist(),
        "rel_grad": (gn / np.maximum(wnorm, 1e-30)).tolist(),
        "weight_norm": wnorm.tolist(),
        "groups_used": used,
        "tokens": tokens,
        "rollout_entropy": ent / max(tokens, 1),
    }
