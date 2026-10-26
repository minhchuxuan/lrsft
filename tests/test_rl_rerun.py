import pytest

from lrsft.config import load_config
from lrsft.rl_rerun import choose, judge, sweep_metrics


def rows(rewards, entropies=(0.3,) * 8, nan=False):
    out = []
    for i, r in enumerate(rewards):
        out.append({"reward": float("nan") if nan and i == 3 else r, "entropy": entropies[i % len(entropies)],
                    "loss": 0.1, "grad_norm": 0.2, "clip_ratio/region_mean": 1e-3})
    return out


def m(rewards, **kw):
    return sweep_metrics(rows(rewards, **kw), finished=True)


FLAT = [0.22] * 8
RISE = [0.20, 0.21, 0.22, 0.24, 0.25, 0.26, 0.27, 0.28]
DROP = [0.25, 0.25, 0.22, 0.20, 0.18, 0.15, 0.14, 0.13]


def test_judge():
    assert judge(m(RISE), 0.01) == (True, True)
    assert judge(m(FLAT), 0.01) == (True, False)
    assert judge(m(DROP), 0.01)[0] is False                      # reward fell by > 0.05
    assert judge(m(RISE, entropies=(0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9)), 0.01)[0] is False  # entropy x3
    assert judge(m(RISE, nan=True), 0.01)[0] is False
    assert judge(sweep_metrics(rows(RISE), finished=False), 0.01)[0] is False


@pytest.mark.parametrize("lo,hi,want", [
    (FLAT, RISE, "5e-05"),   # larger rises and is stable
    (RISE, DROP, "2e-05"),   # larger unstable
    (FLAT, DROP, "2e-05"),   # larger unstable, smaller flat: still the smaller
    (RISE, FLAT, "2e-05"),   # larger flat, smaller rises
    (FLAT, FLAT, "5e-05"),   # both flat: larger
])
def test_choose(lo, hi, want):
    lr, reason = choose({"2e-05": m(lo), "5e-05": m(hi)}, 0.01)
    assert f"{lr:g}" == want, reason


def test_override_numbers_without_dot():
    c = load_config(["configs/default.yaml"], ["rl.lr_subset=2e-05", "rl.rounds=40", "select.override=[3,4]"])
    assert c.rl.lr_subset == 2e-05 and isinstance(c.rl.lr_subset, float)
    assert c.rl.rounds == 40 and c.select.override == [3, 4]
