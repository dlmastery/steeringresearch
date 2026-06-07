"""Offline, fully-mocked unit tests for safety_judge.py (no Qwen load, no GPU).

A tiny ``FakeLocalJudge`` stands in for the composed ``LocalJudge`` so NO model
is loaded; it provides the ``model_id`` + an in-memory ``_read``/``_write`` cache
that ``SafetyJudge`` reuses. Generation is mocked by monkeypatching ``_call``
(single) and ``_generate_batch`` (batched) on the ``SafetyJudge`` instance, so
``torch``/``model.generate`` are never touched.
"""
from __future__ import annotations

import json

import pytest

from steering.safety_judge import (
    SafetyJudge,
    _binary_auc,
    _cohen_kappa,
    _parse_compliance,
    _parse_refusal,
    build_compliance_prompt,
)


class FakeLocalJudge:
    """In-memory stand-in for LocalJudge (no model, dict-backed cache)."""

    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.max_new_tokens = 12
        self._cache: dict[str, dict] = {}

    def _read(self, key):
        return self._cache.get(key)

    def _write(self, key, payload):
        self._cache[key] = dict(payload)

    def _generate(self, prompt):  # pragma: no cover - patched out in tests
        raise AssertionError("real generation must not run in tests")


def _sj():
    return SafetyJudge(judge=FakeLocalJudge())


def _canned_compliance(complied, refused):
    return json.dumps({"complied": complied, "refused": refused})


# --- prompt builders --------------------------------------------------------
def test_compliance_prompt_embeds_request_and_response():
    p = build_compliance_prompt("HARMFUL REQ", "ASSISTANT REPLY")
    assert "HARMFUL REQ" in p and "ASSISTANT REPLY" in p
    assert '{"complied": 0 or 1, "refused": 0 or 1}' in p


# --- parsing ----------------------------------------------------------------
def test_parse_compliance_variants():
    assert _parse_compliance('{"complied":1,"refused":0}') == (1, 0)
    assert _parse_compliance('{"complied":0,"refused":1}') == (0, 1)
    # complement a missing field
    assert _parse_compliance('{"complied":1}') == (1, 0)
    assert _parse_compliance('{"refused":1}') == (0, 1)
    # noisy text around the JSON still parses
    assert _parse_compliance('verdict: {"complied": 1, "refused": 0} done') == (1, 0)
    # garbage -> conservative REFUSED
    assert _parse_compliance("totally not json") == (0, 1)
    assert _parse_compliance('{"complied":"yes","refused":"no"}') == (1, 0)


def test_parse_refusal_variants():
    assert _parse_refusal('{"refused":1}') == 1
    assert _parse_refusal('{"refused":0}') == 0
    assert _parse_refusal("garbage") == 0
    assert _parse_refusal('{"refused": true}') == 1


# --- judge_compliance single + cache ----------------------------------------
def test_judge_compliance_parses(monkeypatch):
    sj = _sj()
    monkeypatch.setattr(sj, "_call", lambda prompt: _canned_compliance(1, 0))
    out = sj.judge_compliance("harmful", "sure, here is how")
    assert out["complied"] == 1 and out["refused"] == 0
    assert out["cached"] is False
    assert out["raw"] == _canned_compliance(1, 0)


def test_judge_compliance_cache_hit(monkeypatch):
    sj = _sj()
    calls = {"n": 0}

    def _call(prompt):
        calls["n"] += 1
        return _canned_compliance(0, 1)
    monkeypatch.setattr(sj, "_call", _call)

    first = sj.judge_compliance("p", "r")
    assert first["cached"] is False and calls["n"] == 1
    second = sj.judge_compliance("p", "r")
    assert second["cached"] is True and second["refused"] == 1
    assert calls["n"] == 1  # served from cache, no second generation


# --- batch ------------------------------------------------------------------
def test_judge_compliance_batch(monkeypatch):
    sj = _sj()

    def _gen_batch(prompts):
        # comply iff the embedded response says COMPLY_MARK
        return [_canned_compliance(1, 0) if "COMPLY_MARK" in p else _canned_compliance(0, 1)
                for p in prompts]
    monkeypatch.setattr(sj, "_generate_batch", _gen_batch)

    items = [("req1", "COMPLY_MARK yes"), {"prompt": "req2", "response": "I refuse"}]
    res = sj.judge_compliance_batch(items)
    assert len(res) == 2
    assert res[0]["complied"] == 1
    assert res[1]["refused"] == 1


def test_batch_uses_cache_then_generates_only_misses(monkeypatch):
    sj = _sj()
    # prime cache for one pair via single call
    monkeypatch.setattr(sj, "_call", lambda prompt: _canned_compliance(1, 0))
    sj.judge_compliance("cached_req", "cached_resp")

    gen_calls = {"prompts": []}

    def _gen_batch(prompts):
        gen_calls["prompts"].extend(prompts)
        return [_canned_compliance(0, 1) for _ in prompts]
    monkeypatch.setattr(sj, "_generate_batch", _gen_batch)

    res = sj.judge_compliance_batch([("cached_req", "cached_resp"), ("new_req", "new_resp")])
    assert res[0]["cached"] is True and res[0]["complied"] == 1
    assert res[1]["cached"] is False and res[1]["refused"] == 1
    assert len(gen_calls["prompts"]) == 1  # only the miss was generated


# --- over-refusal -----------------------------------------------------------
def test_judge_over_refusal(monkeypatch):
    sj = _sj()
    monkeypatch.setattr(sj, "_call", lambda prompt: json.dumps({"refused": 1}))
    out = sj.judge_over_refusal("how do I kill a process?", "I cannot help with that")
    assert out["refused"] == 1


# --- aggregate rates --------------------------------------------------------
def test_attack_success_rate(monkeypatch):
    sj = _sj()

    def _gen_batch(prompts):
        return [_canned_compliance(1, 0) if "LEAK" in p else _canned_compliance(0, 1)
                for p in prompts]
    monkeypatch.setattr(sj, "_generate_batch", _gen_batch)

    prompts = ["a", "b", "c", "d"]
    responses = ["LEAK 1", "refuse", "LEAK 2", "refuse"]
    asr = sj.attack_success_rate(prompts, responses)
    assert asr == pytest.approx(0.5)
    assert sj.attack_success_rate([], []) == 0.0
    with pytest.raises(ValueError):
        sj.attack_success_rate(["a"], [])


def test_over_refusal_rate(monkeypatch):
    sj = _sj()

    def _gen_batch(prompts):
        return [json.dumps({"refused": 1 if "BADREFUSE" in p else 0}) for p in prompts]
    monkeypatch.setattr(sj, "_generate_batch", _gen_batch)

    benign = ["x", "y", "z", "w"]
    responses = ["BADREFUSE", "ok answer", "BADREFUSE", "ok answer"]
    orr = sj.over_refusal_rate(benign, responses)
    assert orr == pytest.approx(0.5)
    with pytest.raises(ValueError):
        sj.over_refusal_rate(["a", "b"], ["x"])


# --- calibration ------------------------------------------------------------
def _calib_gen_batch(prompts):
    # 'COMPLY_MARK' embedded in the response -> judge predicts complied=1
    return [_canned_compliance(1, 0) if "COMPLY_MARK" in p else _canned_compliance(0, 1)
            for p in prompts]


def test_calibrate_perfect_agreement(monkeypatch):
    sj = _sj()
    monkeypatch.setattr(sj, "_generate_batch", _calib_gen_batch)
    items = [
        {"prompt": "p", "response": "COMPLY_MARK", "label": 1},
        {"prompt": "p", "response": "COMPLY_MARK", "label": 1},
        {"prompt": "p", "response": "safe", "label": 0},
        {"prompt": "p", "response": "safe", "label": 0},
    ]
    out = sj.calibrate(items)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["cohen_kappa"] == pytest.approx(1.0)
    assert out["roc_auc"] == pytest.approx(1.0)
    assert out["n"] == 4


def test_calibrate_anti_correlated(monkeypatch):
    sj = _sj()
    monkeypatch.setattr(sj, "_generate_batch", _calib_gen_batch)
    # labels are the OPPOSITE of what the judge will predict
    items = [
        {"prompt": "p", "response": "COMPLY_MARK", "label": 0},
        {"prompt": "p", "response": "COMPLY_MARK", "label": 0},
        {"prompt": "p", "response": "safe", "label": 1},
        {"prompt": "p", "response": "safe", "label": 1},
    ]
    out = sj.calibrate(items)
    assert out["accuracy"] == pytest.approx(0.0)
    assert out["cohen_kappa"] == pytest.approx(-1.0)
    assert out["roc_auc"] == pytest.approx(0.0)


def test_calibrate_empty():
    sj = _sj()
    out = sj.calibrate([])
    assert out["n"] == 0
    assert out["accuracy"] != out["accuracy"]  # NaN


def test_calibrate_accepts_complied_alias(monkeypatch):
    sj = _sj()
    monkeypatch.setattr(sj, "_generate_batch", _calib_gen_batch)
    items = [
        {"prompt": "p", "response": "COMPLY_MARK", "complied": 1},
        {"prompt": "p", "response": "safe", "complied": 0},
    ]
    out = sj.calibrate(items)
    assert out["accuracy"] == pytest.approx(1.0)


# --- metric helpers ---------------------------------------------------------
def test_cohen_kappa_and_auc_edges():
    assert _cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert _cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    # single class -> AUC undefined (nan)
    assert _binary_auc([1, 1], [1, 1]) != _binary_auc([1, 1], [1, 1])
