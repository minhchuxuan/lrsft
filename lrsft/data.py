"""Stage `prep`: base model + all datasets, written once under <work>/.

Outputs
  models/base/              bf16 Qwen3 base with eos = [<|endoftext|>, <|im_end|>] (or a tiny random Qwen3)
  data/sft/                 tokenized SFT set (input_ids, completion_mask), saved with datasets.save_to_disk
  data/rl_train.jsonl       RL prompts  {problem, prompt, answer, level}
  data/probe_pool.jsonl     fixed subset of the RL prompts, used by every layer probe
  data/math500.jsonl        eval set    {problem, prompt, answer, level, solution}
  results/data_stats.json
"""

import argparse
import random
import re
from pathlib import Path

from .common import (
    END_OF_TURN,
    build_prompt,
    encode_prompt,
    is_model_dir,
    load_tokenizer,
    patch_generation_config,
    write_json,
    write_jsonl,
)
from .config import Paths, add_config_args, config_from_args
from .reward import last_boxed


def norm_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# ----------------------------------------------------------------------------- base model

def prepare_base(cfg, paths: Paths) -> None:
    out = paths.ckpt("base")
    if is_model_dir(out):
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if cfg.model.get("tiny"):
        # Random tiny Qwen3 with the real tokenizer: exercises the same code paths on a laptop CPU.
        from transformers import Qwen3Config, Qwen3ForCausalLM

        t = cfg.model.tiny
        config = Qwen3Config(
            vocab_size=len(tok),
            hidden_size=t["hidden_size"],
            intermediate_size=t["intermediate_size"],
            num_hidden_layers=t["num_layers"],
            num_attention_heads=t["num_heads"],
            num_key_value_heads=t["num_kv_heads"],
            head_dim=t["hidden_size"] // t["num_heads"],
            max_position_embeddings=4096,
            tie_word_embeddings=True,
            bos_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        torch.manual_seed(cfg.seed)
        model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
    assert model.config.tie_word_embeddings, "the code assumes tied embeddings (Qwen3-0.6B)"
    patch_generation_config(model, tok)
    tmp = out.with_name("base.tmp")
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    from .common import replace_dir

    replace_dir(tmp, out)


# ----------------------------------------------------------------------------- datasets

def load_math500(cfg):
    from datasets import load_dataset

    ds = load_dataset(cfg.data.eval_source, split="test")
    return [
        {"problem": r["problem"], "answer": r["answer"], "level": int(r["level"]), "solution": r["solution"],
         "subject": r["subject"]}
        for r in ds
    ]


def load_rl_prompts(cfg, exclude: set):
    from datasets import load_dataset

    rows = []
    if cfg.data.rl_source == "math":
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
        levels = {f"Level {lv}" for lv in cfg.data.rl_levels}
        for r in ds:
            if r["level"] not in levels:
                continue
            ans = last_boxed(r["solution"])
            if ans:
                rows.append({"problem": r["problem"], "answer": ans, "level": int(r["level"].split()[-1])})
    elif cfg.data.rl_source == "dapo":
        ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
        rows = [{"problem": r["prompt"], "answer": r["solution"], "level": -1} for r in ds]
    else:
        raise ValueError(cfg.data.rl_source)
    return [r for r in rows if norm_problem(r["problem"]) not in exclude]


def load_sft_rows(cfg, tok, exclude: set, stats: dict):
    """OpenR1-Math: keep problems with a complete, math-verify-correct R1 trace whose prompt+trace fits
    in `sft_max_tokens`; use the shortest such trace per problem."""
    from datasets import load_dataset

    ds = load_dataset(cfg.data.sft_source, cfg.data.sft_config, split="train").select_columns(
        ["problem", "answer", "generations", "correctness_math_verify", "is_reasoning_complete"])
    max_tok = cfg.data.sft_max_tokens
    max_chars = int(max_tok * 5)  # cheap pre-filter before tokenizing; ~3.5 chars/token for this text
    candidates = []
    for r in ds:
        if not r["answer"] or norm_problem(r["problem"]) in exclude:
            continue
        gens = []
        complete = r["is_reasoning_complete"] or [True] * len(r["generations"])
        correct = r["correctness_math_verify"] or [False] * len(r["generations"])
        for g, ok, done in zip(r["generations"], correct, complete):
            if ok and done and "</think>" in g and last_boxed(g.split("</think>")[-1]) is not None:
                if len(g) + len(r["problem"]) < max_chars:
                    gens.append(g)
        if gens:
            candidates.append((r["problem"], r["answer"], min(gens, key=len)))
    stats["sft_candidates"] = len(candidates)
    random.Random(cfg.seed).shuffle(candidates)

    rows, n_long = [], 0
    for i in range(0, len(candidates), 2048):
        chunk = candidates[i : i + 2048]
        prompts = [build_prompt(p) for p, _, _ in chunk]
        p_ids = tok(text=prompts)["input_ids"]
        c_ids = tok(text=[g.strip() + END_OF_TURN for _, _, g in chunk])["input_ids"]
        for (prob, ans, _), pi, ci in zip(chunk, p_ids, c_ids):
            if len(pi) + len(ci) > max_tok:
                n_long += 1
                continue
            rows.append({"problem": prob, "answer": ans, "input_ids": pi + ci,
                         "completion_mask": [0] * len(pi) + [1] * len(ci)})
            if len(rows) >= cfg.data.sft_num_samples:
                break
        if len(rows) >= cfg.data.sft_num_samples:
            break
    stats["sft_dropped_too_long"] = n_long
    return rows


def synthetic(cfg, tok):
    """Arithmetic toy data with the same fields as the real sets (CPU smoke tests only)."""
    rng = random.Random(cfg.seed)

    def item():
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        return {"problem": f"Compute {a} + {b}.", "answer": str(a + b), "level": 1,
                "solution": f"{a} + {b} = \\boxed{{{a + b}}}", "subject": "toy"}

    n = cfg.data.synthetic
    rl = [item() for _ in range(n["rl"])]
    ev = [item() for _ in range(n["eval"])]
    sft = []
    for _ in range(n["sft"]):
        it = item()
        a, b = it["problem"][8:-1].split(" + ")
        trace = f"<think>\n{a} plus {b} is {it['answer']}.\n</think>\n\nThe answer is \\boxed{{{it['answer']}}}."
        pi = encode_prompt(tok, build_prompt(it["problem"]))
        ci = tok(text=trace + END_OF_TURN)["input_ids"]
        sft.append({"problem": it["problem"], "answer": it["answer"], "input_ids": pi + ci,
                    "completion_mask": [0] * len(pi) + [1] * len(ci)})
    return sft, rl, ev


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = Paths(cfg)
    done = paths.data / "DONE"
    if done.exists():
        print("prep: already done")
        return
    prepare_base(cfg, paths)
    tok = load_tokenizer(paths.ckpt("base"))
    stats = {}

    if cfg.data.get("synthetic"):
        sft_rows, rl_rows, eval_rows = synthetic(cfg, tok)
    else:
        eval_rows = load_math500(cfg)
        exclude = {norm_problem(r["problem"]) for r in eval_rows}
        rl_rows = load_rl_prompts(cfg, exclude)
        sft_rows = load_sft_rows(cfg, tok, exclude, stats)

    for r in eval_rows + rl_rows:
        r["prompt"] = build_prompt(r["problem"])
    rng = random.Random(cfg.seed)
    rng.shuffle(rl_rows)
    if cfg.data.get("rl_num_prompts"):
        rl_rows = rl_rows[: cfg.data.rl_num_prompts]
    probe_pool = rl_rows[: cfg.data.probe_pool_size]

    from datasets import Dataset

    sft_ds = Dataset.from_list(sft_rows)
    lengths = [len(x) for x in sft_ds["input_ids"]]
    stats.update(
        sft_samples=len(sft_ds),
        sft_tokens_total=int(sum(lengths)),
        sft_tokens_mean=float(sum(lengths) / max(1, len(lengths))),
        sft_tokens_max=int(max(lengths)),
        rl_prompts=len(rl_rows),
        probe_pool=len(probe_pool),
        eval_prompts=len(eval_rows),
    )
    paths.data.mkdir(parents=True, exist_ok=True)
    sft_ds.save_to_disk(str(paths.data / "sft"))
    write_jsonl(paths.data / "rl_train.jsonl", rl_rows)
    write_jsonl(paths.data / "probe_pool.jsonl", probe_pool)
    write_jsonl(paths.data / "math500.jsonl", eval_rows)
    write_json(paths.results / "data_stats.json", stats)
    Path(done).write_text("ok\n")
    print("prep:", stats)


if __name__ == "__main__":
    main()
