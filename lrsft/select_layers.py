"""Pick the reserved layers.

  --from base           -> results/layers.json            {R, R_random, metric, scores}
  --from full_ep8 --reselect
                        -> results/layers_reselected.json {R_reselected, ...}   (control: R' chosen after SFT)

`select.override: [i, j]` in the config forces R (e.g. after looking at the probe by hand).
"""

import argparse
import random

from .common import read_json, write_json
from .config import Paths, add_config_args, config_from_args
from .metrics import top_k


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--from", dest="src", default="base")
    parser.add_argument("--reselect", action="store_true")
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    out = paths.results / ("layers_reselected.json" if args.reselect else "layers.json")
    if out.exists():
        print(f"select: {out} exists")
        return
    probe = read_json(paths.result("probe", args.src))
    metric, k = cfg.select.metric, cfg.select.k
    scores = probe["mean"][metric]
    exclude = list(cfg.select.get("exclude_layers") or [])
    if args.reselect:
        R = top_k(scores, k, exclude)
        write_json(out, {"R_reselected": R, "from": args.src, "metric": metric, "scores": scores})
        print(f"select: R' = {R} (from {args.src})")
        return
    R = sorted(cfg.select.override) if cfg.select.get("override") else top_k(scores, k, exclude)
    others = [i for i in range(probe["n_layers"]) if i not in R and i not in exclude]
    R_random = sorted(random.Random(cfg.seed).sample(others, k))
    write_json(out, {"R": R, "R_random": R_random, "from": args.src, "metric": metric, "scores": scores,
                     "overridden": bool(cfg.select.get("override"))})
    print(f"select: R = {R}, random control = {R_random}")


if __name__ == "__main__":
    main()
