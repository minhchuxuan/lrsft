"""Check that the reserved layers of every LR-SFT checkpoint are bit-identical to the base model,
and that the other layers did change. -> results/verify_lr.json (raises if the check fails)."""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from .common import layer_of, parse_layers, write_json
from .config import Paths, add_config_args, config_from_args


def load_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                out[k] = sf.get_tensor(k)
    return out


def compare(base_dir, ckpt_dir, layers) -> dict:
    base, ckpt = load_tensors(base_dir), load_tensors(ckpt_dir)
    reserved_equal, others_changed, n_reserved = True, 0, 0
    mismatched = []
    for name, w0 in base.items():
        layer = layer_of(name)
        if layer is None or name not in ckpt:
            continue
        w = ckpt[name]
        if layer in layers:
            n_reserved += 1
            if w.dtype != w0.dtype or not torch.equal(w, w0):
                reserved_equal = False
                mismatched.append(name)
        elif not torch.equal(w, w0):
            others_changed += 1
    return {"reserved_bit_identical": reserved_equal and n_reserved > 0, "reserved_tensors": n_reserved,
            "mismatched": mismatched, "other_tensors_changed": others_changed}


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--layers", required=True)
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    layers = set(parse_layers(args.layers))
    report = {"layers": sorted(layers), "checkpoints": {}}
    for e in cfg.sft.save_epochs:
        name = f"lr_ep{e}"
        report["checkpoints"][name] = compare(paths.ckpt("base"), paths.ckpt(name), layers)
    ok = all(c["reserved_bit_identical"] and c["other_tensors_changed"] > 0 for c in report["checkpoints"].values())
    report["ok"] = ok
    print("verify:", report)
    if not ok:
        raise SystemExit("reserved layers changed during LR-SFT (or nothing else changed)")
    write_json(paths.results / "verify_lr.json", report)


if __name__ == "__main__":
    main()
