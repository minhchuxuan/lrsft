"""Shared pieces: the single prompt format, model loading/freezing/saving, small IO helpers."""

import json
import os
import random
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

# One prompt format for SFT, RL, probes and eval. The base model has no chat template of its
# own, but Qwen base models have seen ChatML, and <|im_end|> gives a clean stop token.
PROMPT_TEMPLATE = (
    "<|im_start|>user\n{problem}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
END_OF_TURN = "<|im_end|>"


def build_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem.strip())


def stop_token_ids(tokenizer) -> list[int]:
    """Generation stops at <|im_end|> (end of turn) or <|endoftext|> (Qwen base eos)."""
    ids = [tokenizer.convert_tokens_to_ids(END_OF_TURN), tokenizer.eos_token_id]
    return sorted({i for i in ids if i is not None})


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    # Same call TRL's GRPOTrainer makes for non-conversational prompts.
    return tokenizer(text=prompt)["input_ids"]


# ----------------------------------------------------------------------------- misc

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    # MPS is skipped on purpose: CPU is the reference for the Mac smoke tests.
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def write_json(path, obj) -> None:
    """Atomic write: a stage's output file exists only once it is complete (idempotent pipeline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
    os.replace(tmp, path)


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_layers(text) -> list[int]:
    if isinstance(text, (list, tuple)):
        return sorted(int(x) for x in text)
    return sorted(int(x) for x in str(text).split(",") if x.strip())


# ----------------------------------------------------------------------------- models

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def layer_of(param_name: str):
    """Decoder-layer index of a parameter name, or None (embeddings, final norm, lm_head)."""
    m = _LAYER_RE.match(param_name)
    return int(m.group(1)) if m else None


def num_layers(model) -> int:
    return len(model.model.layers)


def load_model(path, dtype=torch.float32, attn_implementation="sdpa", device_=None):
    """Training/probing always uses fp32 master weights (bf16 autocast for compute).

    bf16 weights cannot hold RL-sized updates (lr ~1e-6), and bf16 -> fp32 is lossless, so the
    frozen layers stay bit-identical to the bf16 checkpoint they came from.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype, attn_implementation=attn_implementation)
    model.config.use_cache = False
    if device_ is not None:
        model.to(device_)
    return model


def load_tokenizer(path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def freeze_layers(model, layers) -> None:
    """LR-SFT: everything trains except the reserved decoder layers R."""
    layers = set(layers)
    for name, p in model.named_parameters():
        p.requires_grad_(layer_of(name) not in layers)


def train_only_layers(model, layers) -> None:
    """RL on R: only the reserved decoder layers train (embeddings / lm_head stay frozen; they are tied)."""
    layers = set(layers)
    for name, p in model.named_parameters():
        p.requires_grad_(layer_of(name) in layers)


def count_params(model, trainable_only=False) -> int:
    seen, total = set(), 0
    for p in model.parameters():  # parameters() already yields tied weights once
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def patch_generation_config(model_or_cfg, tokenizer) -> None:
    gen = model_or_cfg.generation_config if hasattr(model_or_cfg, "generation_config") else model_or_cfg
    gen.eos_token_id = stop_token_ids(tokenizer)
    gen.pad_token_id = tokenizer.pad_token_id


def save_bf16(model, tokenizer, out_dir) -> None:
    """Save a bf16 copy (what vLLM and later stages load) without touching the training model.

    A fresh model is built from the config and filled with bf16 copies of the weights, so tied
    embeddings stay tied and config.dtype is bfloat16 (vLLM picks its dtype from it).
    """
    from transformers import AutoModelForCausalLM

    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.state_dict().items()}
    config = model.config.__class__.from_dict(model.config.to_dict())
    config.use_cache = True
    fresh = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]  # tied, restored by tie_weights()
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    fresh.tie_weights()
    patch_generation_config(fresh, tokenizer)
    fresh.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    replace_dir(tmp, out_dir)


def replace_dir(src: Path, dst: Path) -> None:
    import shutil

    if dst.exists():
        old = dst.with_name(dst.name + ".old")
        os.replace(dst, old)
        os.replace(src, dst)
        shutil.rmtree(old)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)


def is_model_dir(path) -> bool:
    path = Path(path)
    return (path / "config.json").exists() and any(path.glob("*.safetensors"))
