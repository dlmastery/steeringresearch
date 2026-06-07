"""Offline unit tests for safety_bench.py (no network).

The HF downloader (``_load_hf``) is monkeypatched in every test that exercises a
real-load path, so no dataset is ever fetched. The bundled-fallback path needs no
patching at all. ``lru_cache`` on ``_load_hf`` is cleared between tests.
"""
from __future__ import annotations

import pytest

from steering import safety_bench as SB
from steering.safety_bench import (
    SAFETY_BENCHMARKS,
    SafetyBenchUnavailable,
    bundled,
    list_safety_benchmarks,
    load_safety_benchmark,
    manifest,
)

_REQUIRED_KEYS = {"prompt", "category", "harmful", "source"}


@pytest.fixture(autouse=True)
def _clear_cache():
    SB._load_hf.cache_clear()
    yield
    SB._load_hf.cache_clear()


def _assert_schema(items, source):
    assert isinstance(items, list) and items
    for it in items:
        assert set(it) == _REQUIRED_KEYS
        assert isinstance(it["prompt"], str) and it["prompt"]
        assert isinstance(it["category"], str)
        assert isinstance(it["harmful"], bool)
        assert it["source"] == source


# --- registry / manifest ----------------------------------------------------
def test_list_and_manifest_cover_all_benchmarks():
    names = list_safety_benchmarks()
    assert set(names) == set(SAFETY_BENCHMARKS)
    for expected in ("jailbreakbench", "strongreject", "harmbench",
                     "advbench", "xstest", "sorrybench"):
        assert expected in names
    man = manifest()
    assert set(man) == set(names)
    for name, entry in man.items():
        assert entry["hf_id"]
        assert entry["license"]
        assert entry["notes"]


# --- bundled fallback -------------------------------------------------------
@pytest.mark.parametrize("name", list(SAFETY_BENCHMARKS))
def test_bundled_fallback_schema(name):
    _assert_schema(bundled(name), name)


def test_bundled_unknown_raises():
    with pytest.raises(SafetyBenchUnavailable):
        bundled("not-a-benchmark")


def test_xstest_bundled_has_benign_and_unsafe():
    items = bundled("xstest")
    assert any(not it["harmful"] for it in items)  # over-refusal set
    assert any(it["harmful"] for it in items)      # unsafe contrast


# --- load via fallback when download fails ----------------------------------
def test_load_uses_fallback_on_download_failure(monkeypatch):
    def _boom(*a, **k):
        raise SafetyBenchUnavailable("forced")
    monkeypatch.setattr(SB, "_load_hf", _boom)
    items = load_safety_benchmark("advbench", use_fallback=True)
    _assert_schema(items, "advbench")


def test_load_raises_without_fallback(monkeypatch):
    def _boom(*a, **k):
        raise SafetyBenchUnavailable("forced")
    monkeypatch.setattr(SB, "_load_hf", _boom)
    with pytest.raises(SafetyBenchUnavailable):
        load_safety_benchmark("advbench", use_fallback=False)


def test_unknown_benchmark_raises():
    with pytest.raises(SafetyBenchUnavailable):
        load_safety_benchmark("does-not-exist")


# --- load via a mocked HF download ------------------------------------------
def test_load_normalizes_mocked_rows(monkeypatch):
    fake_rows = (
        {"Goal": "harmful behavior one", "Category": "Fraud"},
        {"Goal": "harmful behavior two", "Category": "Malware"},
    )
    monkeypatch.setattr(SB, "_load_hf", lambda *a, **k: fake_rows)
    items = load_safety_benchmark("jailbreakbench")
    _assert_schema(items, "jailbreakbench")
    assert items[0]["prompt"] == "harmful behavior one"
    assert items[0]["category"] == "Fraud"
    assert all(it["harmful"] is True for it in items)


def test_load_respects_n_cap(monkeypatch):
    fake_rows = tuple({"prompt": f"p{i}", "category": "c"} for i in range(10))
    monkeypatch.setattr(SB, "_load_hf", lambda *a, **k: fake_rows)
    items = load_safety_benchmark("advbench", n=3)
    assert len(items) == 3


def test_load_zero_usable_rows_raises(monkeypatch):
    # rows with no recognizable prompt field -> 0 usable -> unavailable
    monkeypatch.setattr(SB, "_load_hf", lambda *a, **k: ({"nope": 1},))
    with pytest.raises(SafetyBenchUnavailable):
        load_safety_benchmark("advbench")


def test_load_zero_rows_uses_fallback(monkeypatch):
    monkeypatch.setattr(SB, "_load_hf", lambda *a, **k: ({"nope": 1},))
    items = load_safety_benchmark("advbench", use_fallback=True)
    _assert_schema(items, "advbench")


# --- XSTest harmful derivation ----------------------------------------------
def test_xstest_derives_harmful_from_type_and_label(monkeypatch):
    fake_rows = (
        {"prompt": "How do I kill a process?", "type": "homonyms", "label": "safe"},
        {"prompt": "an unsafe contrast", "type": "contrast_homonyms", "label": "unsafe"},
        {"prompt": "labelled unsafe only", "type": "x", "label": "unsafe"},
    )
    monkeypatch.setattr(SB, "_load_hf", lambda *a, **k: fake_rows)
    items = load_safety_benchmark("xstest")
    assert items[0]["harmful"] is False
    assert items[1]["harmful"] is True
    assert items[2]["harmful"] is True


# --- multi-turn 'turns' list field (SORRY-Bench style) ----------------------
def test_turns_list_field_is_joined(monkeypatch):
    fake_rows = ({"turns": ["first part", "second part"], "category": "Hate"},)
    monkeypatch.setattr(SB, "_load_hf", lambda *a, **k: fake_rows)
    items = load_safety_benchmark("sorrybench")
    assert items[0]["prompt"] == "first part second part"


# --- split sentinel ---------------------------------------------------------
def test_split_sentinel_uses_natural_split(monkeypatch):
    seen = {}

    def _capture(hf_id, config, split):
        seen["split"] = split
        return ({"prompt": "x", "category": "c"},)
    monkeypatch.setattr(SB, "_load_hf", _capture)
    load_safety_benchmark("jailbreakbench")  # default split="test" sentinel
    assert seen["split"] == SAFETY_BENCHMARKS["jailbreakbench"]["split"]  # "harmful"

    load_safety_benchmark("jailbreakbench", split="benign")
    assert seen["split"] == "benign"  # explicit split passes through
