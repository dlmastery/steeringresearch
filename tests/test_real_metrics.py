"""Offline unit checks for real_metrics.py — the REAL capability/coherence layer.

Everything here runs with NO network: the HF loader is monkeypatched to raise so
we exercise (a) the documented OFFLINE FALLBACK and (b) the
``RealMetricsUnavailable`` path, and the scorers run on the FakeResidualLM / a
tiny generation stub. No real dataset is downloaded.
"""

import math

import pytest
import torch

from steering import real_metrics
from steering.model import load_model
from steering.real_metrics import (
    RealMetricsUnavailable,
    _extract_final_number,
    gsm8k_accuracy,
    load_gsm8k,
    load_mmlu,
    load_wikitext103,
    mmlu_accuracy,
    wikitext_perplexity,
)


def _force_no_network(monkeypatch):
    """Make the HF split loader fail, so loaders take the offline path."""
    def _raise(*_a, **_k):
        raise RuntimeError("offline: no network in tests")

    monkeypatch.setattr(real_metrics, "_load_hf_split", _raise)


# --------------------------------------------------------------------------- #
# Loaders — offline fallback
# --------------------------------------------------------------------------- #
def test_load_mmlu_offline_fallback(monkeypatch):
    _force_no_network(monkeypatch)
    qs = load_mmlu(n=5)
    assert 0 < len(qs) <= 5
    q = qs[0]
    assert set(("question", "options", "answer_idx", "real")) <= set(q)
    assert len(q["options"]) == 4
    assert 0 <= q["answer_idx"] < 4
    assert q["real"] is False  # fallback is tagged synthetic


def test_load_gsm8k_offline_fallback(monkeypatch):
    _force_no_network(monkeypatch)
    probs = load_gsm8k(n=3)
    assert 0 < len(probs) <= 3
    p = probs[0]
    assert set(("question", "answer", "raw_answer", "real")) <= set(p)
    assert isinstance(p["answer"], float)
    assert p["real"] is False


def test_load_wikitext103_offline_fallback(monkeypatch):
    _force_no_network(monkeypatch)
    passages = load_wikitext103(n=4, min_chars=120)
    assert 0 < len(passages) <= 4
    assert all(isinstance(t, str) and len(t) >= 120 for t in passages)


# --------------------------------------------------------------------------- #
# Loaders — RealMetricsUnavailable when fallback disabled
# --------------------------------------------------------------------------- #
def test_real_metrics_unavailable_path(monkeypatch):
    _force_no_network(monkeypatch)
    with pytest.raises(RealMetricsUnavailable):
        load_mmlu(n=5, allow_offline_fallback=False)
    with pytest.raises(RealMetricsUnavailable):
        load_gsm8k(n=5, allow_offline_fallback=False)
    with pytest.raises(RealMetricsUnavailable):
        load_wikitext103(n=5, allow_offline_fallback=False)


# --------------------------------------------------------------------------- #
# mmlu_accuracy — real logprob scoring, runs on FakeLM
# --------------------------------------------------------------------------- #
def test_mmlu_accuracy_on_fakelm(monkeypatch):
    _force_no_network(monkeypatch)
    model, tok = load_model("fake")
    qs = load_mmlu(n=6)
    acc = mmlu_accuracy(model, tok, qs)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_mmlu_accuracy_empty_is_zero():
    model, tok = load_model("fake")
    assert mmlu_accuracy(model, tok, []) == 0.0


def test_mmlu_accuracy_scorer_override():
    model, tok = load_model("fake")
    qs = [{"question": "q", "options": ["a", "b"], "answer_idx": 0}]
    # A scorer override short-circuits the logprob path.
    acc = mmlu_accuracy(model, tok, qs, scorer=lambda m, t, q: 1.0)
    assert acc == 1.0


# --------------------------------------------------------------------------- #
# gsm8k_accuracy — extraction + scoring via a generation stub
# --------------------------------------------------------------------------- #
def test_extract_final_number():
    assert _extract_final_number("blah blah\n#### 42") == 42.0
    assert _extract_final_number("the answer is 1,234") == 1234.0
    assert _extract_final_number("first 3 then 7 finally 9") == 9.0
    assert _extract_final_number("#### -5") == -5.0
    assert _extract_final_number("no numbers here") is None


def test_gsm8k_accuracy_with_generate_stub():
    model, tok = load_model("fake")
    problems = [
        {"question": "q1", "answer": 7.0},
        {"question": "q2", "answer": 30.0},
        {"question": "q3", "answer": 12.0},
    ]
    # Stub generator: correct for q1 and q3, wrong for q2 -> 2/3 accuracy.
    canned = {"q1": "#### 7", "q2": "#### 99", "q3": "the result is 12"}

    def gen(_model, _tok, prompt, _max):
        for k, v in canned.items():
            if k in prompt:
                return v
        return ""

    acc = gsm8k_accuracy(model, tok, problems, generate_fn=gen)
    assert abs(acc - 2 / 3) < 1e-9


def test_gsm8k_accuracy_empty_is_zero():
    model, tok = load_model("fake")
    assert gsm8k_accuracy(model, tok, [], generate_fn=lambda *a: "") == 0.0


# --------------------------------------------------------------------------- #
# wikitext_perplexity — real-prose coherence via eval.perplexity, on FakeLM
# --------------------------------------------------------------------------- #
def test_wikitext_perplexity_on_fakelm(monkeypatch):
    _force_no_network(monkeypatch)
    model, tok = load_model("fake")
    ppl = wikitext_perplexity(model, tok, n=3)
    assert isinstance(ppl, float)
    assert math.isfinite(ppl)
    assert ppl > 0.0


def test_sequence_logprob_is_negative_meanlogprob():
    # A sanity check that the cloze scorer returns a finite mean log-prob (<=0).
    model, tok = load_model("fake")
    lp = real_metrics._sequence_logprob(model, tok, "the capital is ", "paris")
    assert isinstance(lp, float)
    assert math.isfinite(lp)
    assert lp <= 0.0 + 1e-6  # log-probs are non-positive
    # an empty continuation -> -inf sentinel
    assert real_metrics._sequence_logprob(model, tok, "abc", "") == float("-inf")


def test_no_torch_grad_needed():
    # Scorers must not require grad (eval-only). A trivial smoke that a forward
    # pass through the stub works under no_grad inside mmlu_accuracy.
    model, tok = load_model("fake")
    with torch.no_grad():
        acc = mmlu_accuracy(
            model, tok, [{"question": "x", "options": ["yes", "no"], "answer_idx": 1}]
        )
    assert 0.0 <= acc <= 1.0
