"""LR-SFT end to end on a tiny model: reserved layers come out bit-identical, the rest trains,
the loss mask covers only the completion, and the bf16 snapshot keeps tied embeddings."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lrsft.common import layer_of, load_model, save_bf16
from lrsft.config import Paths, load_config

REPO = Path(__file__).resolve().parent.parent
CONFIGS = [str(REPO / "configs/default.yaml"), str(REPO / "configs/tiny.yaml")]


@pytest.fixture
def tiny_setup(tmp_path, qwen_tokenizer):
    from datasets import Dataset

    from lrsft.data import prepare_base, synthetic

    cfg = load_config(CONFIGS, [f"paths.work={tmp_path / 'w'}", f"paths.results={tmp_path / 'r'}",
                                "sft.save_epochs=[1]", "data.synthetic.sft=32"])
    paths = Paths(cfg)
    prepare_base(cfg, paths)
    sft_rows, _, _ = synthetic(cfg, qwen_tokenizer)
    ds = Dataset.from_list([{"input_ids": r["input_ids"], "completion_mask": r["completion_mask"]} for r in sft_rows])
    return cfg, paths, ds


def test_lr_sft_keeps_reserved_layers_bit_identical(tiny_setup):
    from lrsft.sft import build_trainer
    from lrsft.verify import compare

    cfg, paths, ds = tiny_setup
    R = [1, 2]
    trainer = build_trainer(cfg, paths, "lr", R, train_dataset=ds)
    base = load_model(paths.ckpt("base"), dtype=torch.float32)
    before = {n: p.detach().clone() for n, p in base.named_parameters()}

    # loss mask: prompt tokens are -100, completion tokens (incl. <|im_end|>) are trained
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels, ids = batch["labels"][0], batch["input_ids"][0]
    mask = torch.tensor(ds[0]["completion_mask"]).bool()
    assert torch.equal(labels[: len(mask)][~mask], torch.full(((~mask).sum(),), -100))
    assert torch.equal(labels[: len(mask)][mask], ids[: len(mask)][mask])

    trainer.train()
    changed_other = 0
    for name, p in trainer.model.named_parameters():
        if layer_of(name) in R:
            assert torch.equal(p.detach().cpu(), before[name]), name
        elif not torch.equal(p.detach().cpu(), before[name]):
            changed_other += 1
    assert changed_other > 0

    # the epoch snapshot written by the callback (bf16 on disk) matches the base bitwise on R
    snap = paths.ckpt("lr_ep1")
    report = compare(paths.ckpt("base"), snap, set(R))
    assert report["reserved_bit_identical"] and report["other_tensors_changed"] > 0, report


def test_save_bf16_keeps_tied_embeddings(tiny_setup, tmp_path):
    cfg, paths, _ = tiny_setup
    model = load_model(paths.ckpt("base"), dtype=torch.float32)
    with torch.no_grad():
        model.model.layers[0].mlp.up_proj.weight.add_(0.123)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(paths.ckpt("base"))
    save_bf16(model, tok, tmp_path / "snap")
    back = load_model(tmp_path / "snap", dtype=torch.bfloat16)
    assert back.config.dtype == torch.bfloat16
    assert back.lm_head.weight.data_ptr() == back.model.embed_tokens.weight.data_ptr()
    for (n, a), (_, b) in zip(model.state_dict().items(), back.state_dict().items()):
        assert torch.equal(a.to(torch.bfloat16), b), n
    assert set(back.generation_config.eos_token_id) == {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}


@pytest.mark.slow
def test_tiny_pipeline_end_to_end(tmp_path, qwen_tokenizer):
    """prep -> probe -> select -> SFT (full, LR) -> verify -> eval/diag -> GRPO on R -> controls -> plots."""
    args = [sys.executable, "-m", "lrsft.pipeline"]
    for c in CONFIGS:
        args += ["--config", c]
    args += ["--set", f"paths.work={tmp_path / 'w'}", "--set", f"paths.results={tmp_path / 'r'}",
             "--set", f"paths.logs={tmp_path / 'l'}", "--retries", "0"]
    subprocess.run(args, check=True, cwd=REPO)
    import json

    status = json.loads((tmp_path / "r" / "pipeline_status.json").read_text())
    failed = {k: v for k, v in status.items() if v["state"] == "failed"}
    assert not failed, failed
    # anything not "ok" must be the R' control being skipped because R' == R (and its eval)
    other = {k for k, v in status.items() if v["state"] != "ok"}
    assert other <= {"rl__full_ep2__Rre", "eval__rl__full_ep2__Rre"}, other
    assert (tmp_path / "r" / "rl" / "full_ep2__R.json").exists() and (tmp_path / "r" / "rl" / "lr_ep2__R.json").exists()
    assert (tmp_path / "r" / "figures" / "fig_a3_layer_ranking.png").exists()
    assert json.loads((tmp_path / "r" / "verify_lr.json").read_text())["ok"]
