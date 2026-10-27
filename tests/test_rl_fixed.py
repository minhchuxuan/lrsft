import math

import torch

from lrsft.rl import DriftLog
from lrsft.rl_fixed import ordered_prompts


def test_ordered_prompts_cycles_without_repeats_in_a_round():
    keep = list(range(0, 300, 3))  # 100 prompts
    order = ordered_prompts(keep, 40 * 64, 64, seed=1)
    assert len(order) == 40 * 64
    idx = [i for i, _ in order]
    assert set(idx) == set(keep)
    for k in range(40):
        assert len(set(idx[k * 64:(k + 1) * 64])) == 64
    assert order == ordered_prompts(list(reversed(keep)), 40 * 64, 64, seed=1)  # same list for every arm
    assert [e for _, e in order][:100] == [0] * 100 and order[100][1] == 1


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.zeros(10))
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])


def test_drift_coherence_and_bf16_share(tmp_path):
    m = Tiny()
    with torch.no_grad():
        m.layers[0].weight.copy_(torch.full((4, 4), 0.5))
    d = DriftLog(m, tmp_path / "d.jsonl", every=10, rounds=[1], updates_per_round=1, lr=1e-3, warmup_steps=0)
    with torch.no_grad():  # 5 coherent steps of lr on half of the layer, a tiny sub-ulp change on the rest
        m.layers[0].weight[:2] += 5e-3
        m.layers[0].weight[2:] += 1e-5
    rec = d.measure(m, 5)
    lay = rec["layers"]
    assert lay["n"] == 16 and rec["trainable"]["n"] == 26
    assert lay["frac_bf16_changed"] == 0.5  # 1e-5 is below half an ulp of 0.5 in bf16
    assert math.isclose(lay["coherence"], (8 * 5e-3 + 8 * 1e-5) / 16 / 5e-3, rel_tol=1e-3)
    assert math.isclose(rec["random_walk_coherence"], 0.8 / math.sqrt(5))
    assert math.isclose(DriftLog(m, None, 10, [], 1, 1e-3, warmup_steps=4).lr_sum(4), 1e-3 * 1.5)
