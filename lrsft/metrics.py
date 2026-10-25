"""Small, dependency-free metrics: pass@k, dead groups, rank agreement, linear CKA."""

import numpy as np
import torch


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def mean_pass_at_k(ns, cs, k) -> float:
    return float(np.mean([pass_at_k(n, c, k) for n, c in zip(ns, cs, strict=True)]))


def group_stats(rewards_per_group) -> dict:
    """Dead groups = all rollouts of a prompt got the same reward, so GRPO advantages are all zero."""
    r = [np.asarray(g, dtype=float) for g in rewards_per_group]
    all_wrong = float(np.mean([np.all(g == 0) for g in r]))
    all_right = float(np.mean([np.all(g == 1) for g in r]))
    dead = float(np.mean([np.all(g == g[0]) for g in r]))
    return {
        "dead_rate": dead,
        "dead_all_wrong": all_wrong,
        "dead_all_correct": all_right,
        "informative_rate": 1.0 - dead,
        "mean_reward": float(np.mean(np.concatenate(r))) if r else float("nan"),
    }


def rankdata(x) -> np.ndarray:
    """Average ranks (1 = smallest), ties share the mean rank, like scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        idx = np.where(x == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(a, b) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def top_k(scores, k: int, exclude=()) -> list[int]:
    order = [i for i in np.argsort(-np.asarray(scores, dtype=float), kind="mergesort") if i not in set(exclude)]
    return sorted(int(i) for i in order[:k])


def top_k_overlap(a, b, k: int) -> float:
    return len(set(top_k(a, k)) & set(top_k(b, k))) / k


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two [n_tokens, d] representation matrices of the same tokens."""
    x = x.double()
    y = y.double()
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    hsic = (x.T @ y).pow(2).sum()
    nx = (x.T @ x).pow(2).sum().sqrt()
    ny = (y.T @ y).pow(2).sum().sqrt()
    return float(hsic / (nx * ny))
