import itertools
from math import comb

import numpy as np
import pytest
import torch

from lrsft.metrics import group_stats, linear_cka, pass_at_k, rankdata, spearman, top_k, top_k_overlap


def brute_pass_at_k(n, c, k):
    samples = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(any(samples[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("n", [1, 4, 6])
def test_pass_at_k_matches_brute_force(n):
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_at_k(n, c, k) == pytest.approx(brute_pass_at_k(n, c, k), abs=1e-12)
            assert pass_at_k(n, c, k) == pytest.approx(1 - comb(n - c, k) / comb(n, k), abs=1e-12)


def test_pass_at_1_is_mean_accuracy():
    assert pass_at_k(16, 5, 1) == pytest.approx(5 / 16)


def test_group_stats_dead_groups():
    s = group_stats([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0], [1, 1, 0, 1]])
    assert s["dead_rate"] == 0.5
    assert s["dead_all_wrong"] == 0.25 and s["dead_all_correct"] == 0.25
    assert s["informative_rate"] == 0.5
    assert s["mean_reward"] == pytest.approx(8 / 16)


def test_rank_agreement():
    a = [5.0, 1.0, 3.0, 2.0]
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, [-x for x in a]) == pytest.approx(-1.0)
    assert list(rankdata([3, 1, 3, 2])) == [3.5, 1.0, 3.5, 2.0]  # ties share the average rank
    assert top_k(a, 2) == [0, 2]
    assert top_k(a, 2, exclude=[0]) == [2, 3]
    assert top_k_overlap(a, [5, 1, 0, 9], 2) == 0.5


def test_linear_cka_invariances():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 16, generator=g)
    q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g))
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, x @ q) == pytest.approx(1.0)          # orthogonal transform
    assert linear_cka(x, 3.0 * x + 7.0) == pytest.approx(1.0)  # isotropic scale + shift
    y = torch.randn(500, 16, generator=g)
    assert linear_cka(x, y) < 0.2
    assert 0.0 <= linear_cka(x, x + 0.5 * y) <= 1.0
    assert np.isfinite(linear_cka(x, x[:, :4]))
