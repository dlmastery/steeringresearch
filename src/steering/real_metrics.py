"""real_metrics.py — REAL capability + coherence loaders/scorers.

This module replaces the SYNTHETIC surrogates with genuine community benchmarks:

  * ``eval.mcq_accuracy`` is a deterministic *tripwire* on a hand-written 20-item
    "mmlu_tiny" set — NOT a real capability number. ``mmlu_accuracy`` here does
    proper cloze MCQ logprob scoring on real MMLU items (``cais/mmlu``).
  * GSM8K (``openai/gsm8k``) adds a real *arithmetic-reasoning* capability axis via
    greedy generation + final-number extraction.
  * WikiText-103 (``Salesforce/wikitext``) supplies real held-out English prose so
    ``eval.perplexity`` measures real coherence instead of the tiny bundled slice.

Network discipline (mirrors ``axbench.py``): truststore is injected for the OS
trust store (the SSL-intercepting middlebox breaks plain urllib), downloads are
cached, and EVERY loader degrades to a tiny OFFLINE fallback (the bundled
``data/*.json`` slices / small inline sets) so the unit tests run with NO network.
When the real data is required and unavailable AND the offline fallback is
disabled, a clear ``RealMetricsUnavailable`` is raised — never a cryptic stack.

House style mirrors axbench.py / datasets.py: ``from __future__ import
annotations``, typed, ``lru_cache`` on the network calls, offline-first tests.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from .eval import perplexity
from .model import encode_to_device

try:  # OS trust store for the SSL-intercepting middlebox (same as axbench.py)
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional
    pass


class RealMetricsUnavailable(RuntimeError):
    """Raised when a real benchmark cannot be loaded and fallback is disabled.

    Carries the underlying cause so the caller can tell a gated/SSL/offline
    failure apart from a genuinely missing dataset (mirrors model.load_model).
    """


_ids = encode_to_device


# ---------------------------------------------------------------------------
# Loaders — real HF dataset, cached, with a tiny OFFLINE fallback.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_hf_split(repo: str, config: Optional[str], split: str):
    """Load one HF dataset split (cached). Raises if `datasets`/network fails."""
    from datasets import load_dataset

    if config is not None:
        return load_dataset(repo, config, split=split)
    return load_dataset(repo, split=split)


def _offline_mmlu() -> list[dict]:
    """Tiny offline MCQ fallback (the bundled mmlu_tiny tripwire, re-typed)."""
    from .datasets import load_mmlu_tiny

    out: list[dict] = []
    for q in load_mmlu_tiny():
        out.append({
            "question": str(q["question"]),
            "options": [str(o) for o in q["options"]],
            "answer_idx": int(q["answer_idx"]),
            "real": False,
        })
    return out


def load_mmlu(n: int = 500, *, allow_offline_fallback: bool = True) -> list[dict]:
    """Real MMLU questions: ``[{question, options[4], answer_idx, real}]``.

    Pulls the first ``n`` items of ``cais/mmlu`` (config ``all``, ``test`` split).
    On any failure (no ``datasets``, no network, SSL, gating) falls back to the
    bundled ``mmlu_tiny`` slice when ``allow_offline_fallback`` (so offline tests
    pass), else raises ``RealMetricsUnavailable``. ``real`` tags the provenance.
    """
    try:
        ds = _load_hf_split("cais/mmlu", "all", "test")
        out: list[dict] = []
        for row in ds:
            opts = [str(c) for c in row["choices"]]
            out.append({
                "question": str(row["question"]),
                "options": opts,
                "answer_idx": int(row["answer"]),
                "real": True,
            })
            if len(out) >= n:
                break
        if not out:
            raise RealMetricsUnavailable("cais/mmlu returned no rows")
        return out
    except Exception as err:  # network / datasets / SSL / gating / empty
        if allow_offline_fallback:
            return _offline_mmlu()[:n]
        raise RealMetricsUnavailable(f"MMLU unavailable: {err}") from err


# Tiny, clearly-synthetic GSM8K-shaped fallback (hand-written single-step word
# problems; the gold ``answer`` is the final number, matching GSM8K's #### form).
_OFFLINE_GSM8K: list[dict] = [
    {"question": "Tom has 3 apples and buys 4 more. How many apples does he have?",
     "answer": 7.0, "raw_answer": "3 + 4 = 7\n#### 7", "real": False},
    {"question": "A box holds 6 pens. How many pens are in 5 boxes?",
     "answer": 30.0, "raw_answer": "6 * 5 = 30\n#### 30", "real": False},
    {"question": "Sara had 20 sweets and ate 8. How many are left?",
     "answer": 12.0, "raw_answer": "20 - 8 = 12\n#### 12", "real": False},
    {"question": "There are 24 cookies shared equally among 4 kids. How many each?",
     "answer": 6.0, "raw_answer": "24 / 4 = 6\n#### 6", "real": False},
]


def load_gsm8k(n: int = 200, *, allow_offline_fallback: bool = True) -> list[dict]:
    """Real GSM8K problems: ``[{question, answer(float), raw_answer, real}]``.

    Pulls the first ``n`` items of ``openai/gsm8k`` (config ``main``, ``test``).
    ``answer`` is the parsed final number (after ``####``). Falls back to a tiny
    offline set when ``allow_offline_fallback``, else raises
    ``RealMetricsUnavailable``.
    """
    try:
        ds = _load_hf_split("openai/gsm8k", "main", "test")
        out: list[dict] = []
        for row in ds:
            gold = _extract_final_number(str(row["answer"]))
            if gold is None:
                continue
            out.append({
                "question": str(row["question"]),
                "answer": gold,
                "raw_answer": str(row["answer"]),
                "real": True,
            })
            if len(out) >= n:
                break
        if not out:
            raise RealMetricsUnavailable("openai/gsm8k returned no usable rows")
        return out
    except Exception as err:
        if allow_offline_fallback:
            return list(_OFFLINE_GSM8K[:n])
        raise RealMetricsUnavailable(f"GSM8K unavailable: {err}") from err


def load_wikitext103(n: int = 40, *, min_chars: int = 120,
                     allow_offline_fallback: bool = True) -> list[str]:
    """Real WikiText-103 test passages (>= ``min_chars`` chars), first ``n``.

    From ``Salesforce/wikitext`` (``wikitext-103-raw-v1``, ``test``). Falls back to
    the bundled real WikiText-2 slice when ``allow_offline_fallback``, else raises
    ``RealMetricsUnavailable``. Feed the result to ``eval.perplexity`` for real
    coherence (or use ``wikitext_perplexity``).
    """
    try:
        ds = _load_hf_split("Salesforce/wikitext", "wikitext-103-raw-v1", "test")
        out: list[str] = []
        for row in ds:
            t = str(row["text"]).strip()
            if len(t) >= min_chars and not t.startswith("="):  # skip headings
                out.append(t)
            if len(out) >= n:
                break
        if not out:
            raise RealMetricsUnavailable("WikiText-103 returned no usable passages")
        return out
    except Exception as err:
        if allow_offline_fallback:
            from .datasets import load_wikitext2_real

            passages = [str(p) for p in load_wikitext2_real() if len(str(p)) >= min_chars]
            return passages[:n]
        raise RealMetricsUnavailable(f"WikiText-103 unavailable: {err}") from err


# ---------------------------------------------------------------------------
# Scorers — real capability accuracy.
# ---------------------------------------------------------------------------
def _sequence_logprob(model: nn.Module, tokenizer, prefix: str, continuation: str) -> float:
    """Mean per-token log-prob of ``continuation`` conditioned on ``prefix``.

    Teacher-forced: encode ``prefix + continuation`` once, sum log p(token_t |
    token_<t) over ONLY the continuation positions, divide by the continuation
    length (length-normalised cloze score). Works on any model exposing
    ``.logits`` (FakeResidualLM and real HF causal LMs alike).
    """
    pre_ids = _ids(tokenizer, prefix, model)
    full_ids = _ids(tokenizer, prefix + continuation, model)
    n_pre = pre_ids.shape[1]
    n_full = full_ids.shape[1]
    n_cont = n_full - n_pre
    if n_cont <= 0:
        return float("-inf")
    with torch.no_grad():
        logits = model(full_ids).logits.float()
    logprobs = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    # token at position i is predicted by logits at position i-1.
    for i in range(n_pre, n_full):
        tok = int(full_ids[0, i])
        total += float(logprobs[i - 1, tok])
    return total / n_cont


def mmlu_accuracy(
    model: nn.Module,
    tokenizer,
    questions: Sequence[dict],
    *,
    scorer: Optional[Callable] = None,
) -> float:
    """REAL cloze-MCQ accuracy: pick the option with the highest length-normalised

    conditional log-prob given the question stem. Replaces ``eval.mcq_accuracy``'s
    deterministic token-id surrogate with genuine logprob scoring. ``scorer``, if
    given, overrides with ``scorer(model, tokenizer, questions) -> float``.
    Returns accuracy in [0, 1].
    """
    if scorer is not None:
        return float(scorer(model, tokenizer, questions))
    if not questions:
        return 0.0
    correct = 0
    for q in questions:
        stem = str(q["question"]).rstrip() + "\nAnswer: "
        options = [str(o) for o in q["options"]]
        gold = int(q["answer_idx"])
        scores = [_sequence_logprob(model, tokenizer, stem, opt) for opt in options]
        pred = int(max(range(len(options)), key=lambda i: scores[i]))
        if pred == gold:
            correct += 1
    return correct / len(questions)


def _extract_final_number(text: str) -> Optional[float]:
    """Parse the final answer number from GSM8K-style text.

    Prefers the value after ``####`` (GSM8K gold format); otherwise takes the LAST
    number anywhere in the text (the convention for grading generated solutions).
    Strips thousands separators. Returns None when no number is present.
    """
    if "####" in text:
        tail = text.split("####", 1)[1]
        m = re.findall(r"-?\d[\d,]*\.?\d*", tail)
        if m:
            return _to_float(m[0])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        return _to_float(nums[-1])
    return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:  # pragma: no cover - regex already constrains the shape
        return None


def gsm8k_accuracy(
    model: nn.Module,
    tokenizer,
    problems: Sequence[dict],
    *,
    max_new_tokens: int = 256,
    generate_fn: Optional[Callable[..., str]] = None,
) -> float:
    """REAL GSM8K accuracy: greedily generate a solution, extract the final number,

    compare to the gold answer (exact numeric match, 1e-6 tolerance). ``generate_fn
    (model, tokenizer, prompt, max_new_tokens) -> str`` overrides the default greedy
    generator (used by the offline tests; the default routes through
    ``eval._greedy_generate`` and so requires a model with ``.generate``).
    Returns accuracy in [0, 1].
    """
    if not problems:
        return 0.0
    gen = generate_fn if generate_fn is not None else _default_generate
    correct = 0
    for p in problems:
        prompt = (
            "Solve the problem. End with '#### <answer>'.\n"
            f"Question: {p['question']}\nAnswer: "
        )
        text = gen(model, tokenizer, prompt, max_new_tokens)
        pred = _extract_final_number(text)
        gold = p.get("answer")
        if pred is not None and gold is not None and math.isclose(
            pred, float(gold), rel_tol=0.0, abs_tol=1e-6
        ):
            correct += 1
    return correct / len(problems)


def _default_generate(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Default greedy generation for gsm8k_accuracy (real models only)."""
    from .eval import _greedy_generate

    return _greedy_generate(model, tokenizer, prompt, max_new_tokens)


def wikitext_perplexity(
    model: nn.Module, tokenizer, n: int = 40, *, allow_offline_fallback: bool = True
) -> float:
    """Convenience: load real WikiText passages and return ``eval.perplexity``.

    Real coherence in one call (loader + the existing teacher-forced PPL scorer).
    """
    passages = load_wikitext103(n, allow_offline_fallback=allow_offline_fallback)
    return perplexity(model, tokenizer, passages)
