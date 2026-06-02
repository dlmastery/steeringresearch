"""Fully-mocked unit checks for judge.py (no network, fast).

Every test monkeypatches the HTTP layer (``GeminiJudge._http_post``) or
``time.sleep`` — NO real Gemini call is ever made (the live key's credits are
depleted anyway). Cache dirs are redirected to ``tmp_path`` so tests are
hermetic and order-independent.
"""

from __future__ import annotations

import json

import pytest

from steering import judge as J
from steering.judge import (
    GeminiJudge,
    JudgeUnavailable,
    calibration_agreement,
    make_judge_or_none,
)


def _canned(behavior: int, coherence: int) -> str:
    """A canned candidate-text payload (what Gemini's parts[0].text contains)."""
    return json.dumps({"behavior": behavior, "coherence": coherence})


def _make(tmp_path, **kw) -> GeminiJudge:
    """Construct a judge with a fake key and a tmp cache dir."""
    kw.setdefault("api_key", "fake-key-for-tests")
    kw.setdefault("cache_dir", tmp_path / "cache")
    return GeminiJudge(**kw)


# ---------------------------------------------------------------------------
# Parsing / normalization.
# ---------------------------------------------------------------------------
def test_score_parses_and_normalizes(tmp_path, monkeypatch):
    judge = _make(tmp_path)
    monkeypatch.setattr(judge, "_http_post", lambda prompt: _canned(8, 9))

    out = judge.score("the sea rolled in waves", "ocean", "evokes the ocean")
    assert out["behavior"] == 8.0
    assert out["coherence"] == 9.0
    assert out["cached"] is False
    assert out["model"] == judge.model
    assert out["raw"] == _canned(8, 9)

    # behavior_efficacy normalizes the 0..10 behavior to 0..1.
    monkeypatch.setattr(judge, "_http_post", lambda prompt: _canned(7, 5))
    eff = judge.behavior_efficacy("x", "ocean2", "evokes the ocean")
    assert eff == pytest.approx(0.7)


def test_score_clamps_out_of_range(tmp_path, monkeypatch):
    judge = _make(tmp_path)
    monkeypatch.setattr(judge, "_http_post", lambda prompt: _canned(99, -4))
    out = judge.score("t", "b", "d")
    assert out["behavior"] == 10.0
    assert out["coherence"] == 0.0


# ---------------------------------------------------------------------------
# Caching.
# ---------------------------------------------------------------------------
def test_cache_hit_skips_api(tmp_path, monkeypatch):
    judge = _make(tmp_path)
    calls = {"n": 0}

    def _post(prompt):
        calls["n"] += 1
        return _canned(6, 7)

    monkeypatch.setattr(judge, "_http_post", _post)

    first = judge.score("same text", "behaviorA", "descA")
    assert first["cached"] is False
    assert calls["n"] == 1

    second = judge.score("same text", "behaviorA", "descA")
    assert second["cached"] is True
    assert second["behavior"] == 6.0
    assert second["coherence"] == 7.0
    assert calls["n"] == 1  # NO second API call


def test_cache_key_changes_with_inputs(tmp_path):
    judge = _make(tmp_path)
    k_text = judge._cache_key("text one", "b", "d")
    k_text2 = judge._cache_key("text two", "b", "d")
    k_beh = judge._cache_key("text one", "b2", "d")
    k_desc = judge._cache_key("text one", "b", "d2")
    assert len({k_text, k_text2, k_beh, k_desc}) == 4


def test_cache_key_deterministic_sha256(tmp_path):
    judge = _make(tmp_path)
    k1 = judge._cache_key("abc", "name", "desc")
    k2 = judge._cache_key("abc", "name", "desc")
    assert k1 == k2
    assert len(k1) == 64
    assert all(c in "0123456789abcdef" for c in k1)


# ---------------------------------------------------------------------------
# 429 / backoff path.
# ---------------------------------------------------------------------------
def _raise_http(code: int):
    import urllib.error

    def _post(prompt):
        raise urllib.error.HTTPError(
            url="https://x", code=code, msg="boom", hdrs=None, fp=None
        )

    return _post


def test_persistent_429_raises_after_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(J.time, "sleep", lambda *_a, **_k: None)  # instant
    judge = _make(tmp_path, max_retries=4)

    sleeps = {"n": 0}
    monkeypatch.setattr(J.time, "sleep", lambda *_a, **_k: sleeps.__setitem__("n", sleeps["n"] + 1))
    monkeypatch.setattr(judge, "_http_post", _raise_http(429))

    with pytest.raises(JudgeUnavailable) as exc:
        judge.score("t", "b", "d")
    assert "credits depleted" in str(exc.value) or "429" in str(exc.value)
    # 4 attempts → 3 backoff sleeps before final raise.
    assert sleeps["n"] == 3


def test_5xx_retries_then_succeeds(tmp_path, monkeypatch):
    import urllib.error

    monkeypatch.setattr(J.time, "sleep", lambda *_a, **_k: None)
    judge = _make(tmp_path, max_retries=4)

    state = {"n": 0}

    def _post(prompt):
        state["n"] += 1
        if state["n"] < 2:
            raise urllib.error.HTTPError("https://x", 503, "unavailable", None, None)
        return _canned(5, 5)

    monkeypatch.setattr(judge, "_http_post", _post)
    out = judge.score("t", "b", "d")
    assert out["behavior"] == 5.0
    assert state["n"] == 2


def test_4xx_auth_no_retry(tmp_path, monkeypatch):
    slept = {"n": 0}
    monkeypatch.setattr(J.time, "sleep", lambda *_a, **_k: slept.__setitem__("n", slept["n"] + 1))
    judge = _make(tmp_path, max_retries=4)
    monkeypatch.setattr(judge, "_http_post", _raise_http(403))

    with pytest.raises(JudgeUnavailable):
        judge.score("t", "b", "d")
    assert slept["n"] == 0  # 4xx auth errors are not retried


def test_network_error_is_unavailable(tmp_path, monkeypatch):
    import urllib.error

    judge = _make(tmp_path)

    def _post(prompt):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(judge, "_http_post", _post)
    with pytest.raises(JudgeUnavailable):
        judge.score("t", "b", "d")


# ---------------------------------------------------------------------------
# No-key path.
# ---------------------------------------------------------------------------
def test_make_judge_or_none_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Point the module's repo root at an empty dir so no .gemini_key is found.
    monkeypatch.setattr(J, "_REPO_ROOT", tmp_path)
    assert make_judge_or_none(cache_dir=tmp_path / "c") is None


def test_score_without_key_raises_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(J, "_REPO_ROOT", tmp_path)
    judge = GeminiJudge(cache_dir=tmp_path / "c")  # no key resolvable
    with pytest.raises(JudgeUnavailable):
        judge.score("t", "b", "d")


def test_gemini_key_file_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(J, "_REPO_ROOT", tmp_path)
    (tmp_path / ".gemini_key").write_text("filekey123", encoding="utf-8")
    judge = make_judge_or_none(cache_dir=tmp_path / "c")
    assert judge is not None  # key resolved from the gitignored file


# ---------------------------------------------------------------------------
# Malformed judge JSON.
# ---------------------------------------------------------------------------
def test_malformed_json_raises_unavailable(tmp_path, monkeypatch):
    judge = _make(tmp_path)
    monkeypatch.setattr(judge, "_http_post", lambda prompt: "not json at all")
    with pytest.raises(JudgeUnavailable):
        judge.score("t", "b", "d")


def test_missing_fields_raises_unavailable(tmp_path, monkeypatch):
    judge = _make(tmp_path)
    monkeypatch.setattr(judge, "_http_post", lambda prompt: json.dumps({"behavior": 5}))
    with pytest.raises(JudgeUnavailable):
        judge.score("t", "b", "d")


def test_non_numeric_score_raises_unavailable(tmp_path, monkeypatch):
    judge = _make(tmp_path)
    monkeypatch.setattr(
        judge, "_http_post", lambda prompt: json.dumps({"behavior": "high", "coherence": 5})
    )
    with pytest.raises(JudgeUnavailable):
        judge.score("t", "b", "d")


# ---------------------------------------------------------------------------
# Batch.
# ---------------------------------------------------------------------------
def test_score_batch(tmp_path, monkeypatch):
    judge = _make(tmp_path)
    monkeypatch.setattr(judge, "_http_post", lambda prompt: _canned(4, 6))
    items = [("t1", "b1", "d1"), ("t2", "b2", "d2")]
    results = judge.score_batch(items)
    assert len(results) == 2
    assert all(r["behavior"] == 4.0 and r["coherence"] == 6.0 for r in results)


# ---------------------------------------------------------------------------
# Calibration agreement.
# ---------------------------------------------------------------------------
def test_calibration_perfectly_correlated():
    j = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = [2.0, 4.0, 6.0, 8.0, 10.0]  # exact linear relationship
    agg = calibration_agreement(j, r)
    assert agg["pearson"] == pytest.approx(1.0)
    assert agg["spearman"] == pytest.approx(1.0)
    assert agg["n"] == 5


def test_calibration_anti_correlated():
    j = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = [5.0, 4.0, 3.0, 2.0, 1.0]
    agg = calibration_agreement(j, r)
    assert agg["pearson"] == pytest.approx(-1.0)
    assert agg["spearman"] == pytest.approx(-1.0)


def test_calibration_mae_and_length_guard():
    j = [1.0, 2.0, 3.0]
    r = [1.0, 2.0, 5.0]
    agg = calibration_agreement(j, r)
    assert agg["mae"] == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError):
        calibration_agreement([1.0, 2.0], [1.0])


def test_calibration_constant_input_is_nan():
    agg = calibration_agreement([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])
    assert agg["pearson"] != agg["pearson"]  # NaN


# ---------------------------------------------------------------------------
# Rubric prompt sanity (no network).
# ---------------------------------------------------------------------------
def test_rubric_prompt_contains_axes_and_text():
    p = J.build_rubric_prompt("HELLO WORLD", "formality", "uses formal register")
    assert "BEHAVIOR" in p and "COHERENCE" in p
    assert "formality" in p and "uses formal register" in p
    assert "HELLO WORLD" in p
    assert '{"behavior": <int 0-10>, "coherence": <int 0-10>}' in p


def test_rubric_version_in_cache_key(tmp_path, monkeypatch):
    judge = _make(tmp_path)
    before = judge._cache_key("text", "b", "d")
    monkeypatch.setattr(J, "RUBRIC_VERSION", "v999")
    after = judge._cache_key("text", "b", "d")
    assert before != after  # rubric version participates in the key
