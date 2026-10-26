"""Config: YAML files merged left to right, then `--set a.b=value` overrides.

Every stage module takes the same `--config ... --set ...` arguments, so the pipeline can
pass them through unchanged.
"""

import argparse
import copy
import os
import re
from pathlib import Path

import yaml


_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


class Cfg(dict):
    """dict with attribute access (cfg.rl.lr). Nested dicts are wrapped on access."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    if not key or not _:
        raise ValueError(f"override must look like a.b=value, got {assignment!r}")
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    value = yaml.safe_load(raw)
    # PyYAML (YAML 1.1) reads "2e-05" as a string; numbers written without a dot must still be numbers
    if isinstance(value, str) and _NUMBER.fullmatch(value.strip()):
        value = float(value)
    node[parts[-1]] = value


def load_config(paths, overrides=()) -> Cfg:
    cfg: dict = {}
    for p in paths:
        with open(p) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    # Big files (models, checkpoints, rollouts) can live on another disk; an explicit --set still wins.
    if os.environ.get("LRSFT_WORK"):
        cfg.setdefault("paths", {})["work"] = os.environ["LRSFT_WORK"]
    for o in overrides:
        apply_override(cfg, o)
    return Cfg(cfg)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", action="append", required=True, help="YAML file; repeat to layer overrides")
    parser.add_argument("--set", dest="overrides", action="extend", nargs="+", default=[],
                        help="overrides, e.g. --set rl.rounds=30 sft.lr=2e-5")
    return parser


def config_from_args(args) -> Cfg:
    return load_config(args.config, args.overrides)


def config_cli_args(args) -> list[str]:
    """Re-emit the config arguments so a child process sees the same config."""
    out = []
    for c in args.config:
        out += ["--config", c]
    for o in args.overrides:
        out += ["--set", o]
    return out


class Paths:
    """All file locations in one place. `work` holds big artifacts, `results` small JSON/CSV/figures."""

    def __init__(self, cfg: Cfg):
        self.work = Path(cfg.paths.work)
        self.results = Path(cfg.paths.results)
        self.logs = Path(cfg.paths.logs)

    # data
    @property
    def data(self):
        return self.work / "data"

    def ckpt(self, name: str) -> Path:
        """bf16 model dirs: base, full_ep3, lr_ep8, rl__full_ep8__R, ..."""
        return self.work / "models" / name

    def trainer_dir(self, name: str) -> Path:
        """HF Trainer output (resume checkpoints, fp32 + optimizer)."""
        return self.work / "runs" / name

    def rollouts(self, ckpt: str, task: str) -> Path:
        return self.work / "rollouts" / f"{ckpt}__{task}.jsonl"

    def result(self, kind: str, name: str) -> Path:
        return self.results / kind / f"{name}.json"

    @property
    def layers(self) -> Path:
        return self.results / "layers.json"
