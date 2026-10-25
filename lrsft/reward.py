"""Binary math reward: last \\boxed{...} in the completion, checked against the gold answer with math-verify.

The same function scores probe rollouts, GRPO rollouts and MATH-500 eval samples.
"""

import logging
import threading

logging.getLogger("math_verify").setLevel(logging.ERROR)


def last_boxed(text: str):
    """Content of the last \\boxed{...} (or \\fbox{...}), with nested braces; None if absent."""
    for key in ("\\boxed", "\\fbox"):
        start = text.rfind(key)
        if start >= 0:
            break
    else:
        return None
    i = start + len(key)
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return None  # unbalanced (e.g. truncated)


def _parse(expr: str, timeout):
    # Wrapping in $...$ with math-verify's default extraction normalises things like \sqrt2,
    # units, \% and \text{(C)} better than re-wrapping in \boxed (checked in tests/test_reward.py).
    from math_verify import parse

    return parse("$" + expr + "$", parsing_timeout=timeout)


def is_correct(completion: str, gold: str) -> bool:
    from math_verify import verify

    pred = last_boxed(completion)
    if pred is None or not pred.strip():
        return False
    # math-verify's timeouts use signal.alarm, which only works in the main thread.
    main = threading.current_thread() is threading.main_thread()
    try:
        gold_p = _parse(gold, 5 if main else None)
        pred_p = _parse(pred, 5 if main else None)
        if not gold_p or not pred_p:
            return pred.strip() == gold.strip()
        return bool(verify(gold_p, pred_p, timeout_seconds=5 if main else None))
    except Exception:
        return False


def score(completions: list[str], golds: list[str]) -> list[float]:
    return [float(is_correct(c, g)) for c, g in zip(completions, golds, strict=True)]


def _score_chunk(args):
    return score(*args)


def score_parallel(completions: list[str], golds: list[str], workers: int = 8, chunk: int = 256) -> list[float]:
    """Process pool for the big eval batches (each worker runs math-verify in its own main thread)."""
    if workers <= 1 or len(completions) <= chunk:
        return score(completions, golds)
    import multiprocessing as mp

    chunks = [(completions[i : i + chunk], golds[i : i + chunk]) for i in range(0, len(completions), chunk)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_score_chunk, chunks)
    return [x for part in parts for x in part]
