import os

import pytest
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def make_tiny(vocab_size=128, layers=4, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
                      max_position_embeddings=512, use_cache=False)
    return Qwen3ForCausalLM(cfg).float()


@pytest.fixture
def tiny_model():
    return make_tiny()


@pytest.fixture(scope="session")
def qwen_tokenizer():
    """Real Qwen3 tokenizer (a few MB, cached in HF_HOME). Tests that need it skip when offline."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {e}")
