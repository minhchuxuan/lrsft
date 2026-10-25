import torch

from lrsft.common import count_params, freeze_layers, layer_of, train_only_layers


def test_layer_of():
    assert layer_of("model.layers.13.mlp.up_proj.weight") == 13
    assert layer_of("model.embed_tokens.weight") is None
    assert layer_of("lm_head.weight") is None
    assert layer_of("model.norm.weight") is None


def test_freeze_layers_lr_sft(tiny_model):
    freeze_layers(tiny_model, [1, 2])
    for name, p in tiny_model.named_parameters():
        assert p.requires_grad == (layer_of(name) not in (1, 2)), name
    # tied embeddings: the lm_head is the embedding matrix and stays trainable in LR-SFT
    assert tiny_model.lm_head.weight is tiny_model.model.embed_tokens.weight
    assert tiny_model.lm_head.weight.requires_grad


def test_train_only_layers_rl(tiny_model):
    train_only_layers(tiny_model, [2])
    trainable = [n for n, p in tiny_model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("model.layers.2.") for n in trainable)
    per_layer = count_params(tiny_model, trainable_only=True)
    assert per_layer == sum(p.numel() for p in tiny_model.model.layers[2].parameters())


def test_trainer_optimizer_only_sees_trainable(tiny_model, tmp_path):
    from transformers import Trainer, TrainingArguments

    train_only_layers(tiny_model, [1, 3])
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", weight_decay=0.1)
    opt = Trainer(model=tiny_model, args=args).create_optimizer()
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    for name, p in tiny_model.named_parameters():
        assert (id(p) in in_opt) == p.requires_grad, name


def test_grads_flow_with_frozen_blocks_and_checkpointing(tiny_model):
    """Only layer 2 trains, embeddings frozen, non-reentrant checkpointing: same grads as without it."""
    x = torch.randint(0, 128, (2, 12))
    train_only_layers(tiny_model, [2])
    tiny_model.train()
    tiny_model(input_ids=x, labels=x).loss.backward()
    ref = {n: p.grad.clone() for n, p in tiny_model.named_parameters() if p.grad is not None}
    tiny_model.zero_grad(set_to_none=True)

    tiny_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tiny_model(input_ids=x, labels=x).loss.backward()
    got = {n: p.grad for n, p in tiny_model.named_parameters() if p.grad is not None}
    assert set(got) == set(ref) and all(n.startswith("model.layers.2.") for n in got)
    for n in ref:
        assert torch.allclose(got[n], ref[n], atol=1e-6), n
    assert sum(float(g.norm()) for g in got.values()) > 0
