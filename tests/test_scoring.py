import numpy as np
import pytest
import torch

from lrsft.scoring import batch_layer_stats, group_backward, grpo_advantages, layer_params, pad_batch, token_logprobs


def make_group(rng, rewards, vocab=128, prompt_len=5, truncated=None):
    prompt = rng.integers(0, vocab, prompt_len).tolist()
    comps = [rng.integers(0, vocab, rng.integers(3, 9)).tolist() for _ in rewards]
    return {"prompt_ids": prompt, "completion_ids": comps, "rewards": list(rewards),
            "truncated": truncated or [False] * len(rewards)}


def test_token_logprobs_match_full_softmax(tiny_model):
    rng = np.random.default_rng(0)
    seqs = [(rng.integers(0, 128, 4).tolist(), rng.integers(0, 128, n).tolist()) for n in (3, 7)]
    ids, attn, comp = pad_batch(seqs, 0, torch.device("cpu"))
    tiny_model.eval()
    lp, ent, rows = token_logprobs(tiny_model, ids, attn, comp, chunk=4)
    logits = tiny_model(input_ids=ids, attention_mask=attn).logits.float()
    logp_all = torch.log_softmax(logits[:, :-1], -1)
    sel = comp[:, 1:].bool()
    ref = logp_all.gather(-1, ids[:, 1:, None]).squeeze(-1)[sel]
    ref_ent = -(logp_all.exp() * logp_all).sum(-1)[sel]
    assert torch.allclose(lp, ref, atol=1e-5)
    assert torch.allclose(ent, ref_ent, atol=1e-4)
    assert rows.tolist() == [0] * 3 + [1] * 7


def test_advantages_match_trl_formula():
    r = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    t = torch.tensor(r)
    assert np.allclose(grpo_advantages(r), ((t - t.mean()) / (t.std() + 1e-4)).numpy())
    assert np.allclose(grpo_advantages(r, scale="none"), (t - t.mean()).numpy())


def test_zero_advantage_group_is_skipped(tiny_model):
    rng = np.random.default_rng(1)
    for p in tiny_model.parameters():
        p.requires_grad_(True)
    n_tok, _ = group_backward(tiny_model, make_group(rng, [1, 1, 1, 1]), 0, torch.device("cpu"), 2, False)
    assert n_tok == 0 and all(p.grad is None for p in tiny_model.parameters())


def _direct_grads(model, group):
    model.zero_grad(set_to_none=True)
    group_backward(model, group, 0, torch.device("cpu"), micro_bs=99, bf16=False)
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for ps in layer_params(model).values()
            for p in ps]


def test_micro_batching_does_not_change_the_gradient(tiny_model):
    rng = np.random.default_rng(2)
    for n, p in tiny_model.named_parameters():
        p.requires_grad_(".layers." in n)
    g = make_group(rng, [1, 0, 0, 1, 0, 1])
    full = _direct_grads(tiny_model, g)
    tiny_model.zero_grad(set_to_none=True)
    group_backward(tiny_model, g, 0, torch.device("cpu"), micro_bs=1, bf16=False)
    micro = [p.grad for ps in layer_params(tiny_model).values() for p in ps]
    for a, b in zip(full, micro):
        assert torch.allclose(a, b, atol=1e-6)


def test_truncated_completions_are_masked(tiny_model):
    rng = np.random.default_rng(3)
    for n, p in tiny_model.named_parameters():
        p.requires_grad_(".layers." in n)
    g = make_group(rng, [1, 0, 1, 0], truncated=[True, True, True, True])
    n_tok, _ = group_backward(tiny_model, g, 0, torch.device("cpu"), 2, False, mask_truncated=True)
    assert n_tok == 0


def test_batch_layer_stats_definitions(tiny_model):
    """grad_norm = ||mean_j g_j||, fisher = mean_j ||g_j||^2, per layer, recomputed by hand."""
    rng = np.random.default_rng(4)
    for n, p in tiny_model.named_parameters():
        p.requires_grad_(".layers." in n)
    tiny_model.train()
    groups = [make_group(rng, r) for r in ([1, 0, 0, 0], [0, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0])]
    stats = batch_layer_stats(tiny_model, groups, 0, torch.device("cpu"), micro_bs=2, bf16=False)
    assert stats["groups_used"] == 3  # the all-zero group carries no signal
    assert all(p.grad is None for p in tiny_model.parameters())  # nothing left behind, nothing updated

    per_group = [_direct_grads(tiny_model, g) for g in groups[:3]]
    sizes = [len(ps) for ps in layer_params(tiny_model).values()]
    bounds = np.cumsum([0] + sizes)
    for li in range(len(sizes)):
        sl = slice(bounds[li], bounds[li + 1])
        mean_g = [sum(pg[k] for pg in per_group) / 3 for k in range(bounds[li], bounds[li + 1])]
        gn = float(torch.sqrt(sum((m ** 2).sum() for m in mean_g)))
        fisher = float(np.mean([sum(float((t ** 2).sum()) for t in pg[sl]) for pg in per_group]))
        assert stats["grad_norm"][li] == pytest.approx(gn, rel=1e-4)
        assert stats["fisher"][li] == pytest.approx(fisher, rel=1e-4)
        assert stats["snr"][li] > 0
