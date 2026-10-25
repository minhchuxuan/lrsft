import pytest

from lrsft.common import PROMPT_TEMPLATE, build_prompt, stop_token_ids
from lrsft.reward import is_correct, last_boxed, score


@pytest.mark.parametrize("text,expected", [
    ("so \\boxed{42}.", "42"),
    ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
    ("first \\boxed{1}, final \\boxed{(1,2)}", "(1,2)"),
    ("nested \\boxed{\\sqrt{\\frac{a}{b}}}", "\\sqrt{\\frac{a}{b}}"),
    ("\\fbox{7}", "7"),
    ("no answer", None),
    ("truncated \\boxed{12", None),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize("completion,gold,ok", [
    ("<think>..</think> \\boxed{\\frac{1}{2}}", "0.5", True),
    ("\\boxed{x=3}", "3", True),
    ("\\boxed{3}", "4", False),
    ("\\boxed{\\dfrac{\\sqrt{2}}{2}}", "\\frac{\\sqrt2}{2}", True),
    ("\\boxed{10^\\circ}", "10", True),
    ("\\boxed{\\text{(C)}}", "C", True),
    ("\\boxed{3,2}", "2, 3", True),
    ("\\boxed{[-2,7]}", "[-2, 7]", True),
    ("\\boxed{5\\text{ cm}}", "5", True),
    ("\\boxed{\\text{Evelyn}}", "\\text{Evelyn}", True),
    ("the answer is 5", "5", False),     # no \boxed -> 0, same rule everywhere
    ("\\boxed{}", "5", False),
])
def test_is_correct(completion, gold, ok):
    assert is_correct(completion, gold) is ok


def test_score_list():
    assert score(["\\boxed{2}", "\\boxed{3}"], ["2", "2"]) == [1.0, 0.0]


def test_prompt_format_is_shared(qwen_tokenizer):
    p = build_prompt("  What is 1+1? ")
    assert p == PROMPT_TEMPLATE.format(problem="What is 1+1?")
    assert p.endswith("<|im_start|>assistant\n")
    ids = stop_token_ids(qwen_tokenizer)
    assert qwen_tokenizer.convert_tokens_to_ids("<|im_end|>") in ids and qwen_tokenizer.eos_token_id in ids
