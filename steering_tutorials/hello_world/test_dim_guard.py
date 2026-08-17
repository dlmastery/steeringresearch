"""test_dim_guard.py — the 1152-vs-2560 guard, proved without loading a model.

Gemma-3-1B's residual stream is 1152 numbers wide; Gemma-3-4B's is 2560. A probe
trained on one cannot score the other, and the failure this lesson used to hit
was the raw torch matmul error ("mat1 and mat2 shapes cannot be multiplied
(1x2560 and 1152x128)") raised four frames deep, naming neither model.

These tests build FAKE activation matrices at both widths — no Gemma, no CUDA,
no HF download — and assert that every entry point refuses the mismatch loudly
and by name, and that a 1B and a 4B run cannot collide on a filename.

Run either way:
    python -m steering_tutorials.hello_world.test_dim_guard
    pytest steering_tutorials/hello_world/test_dim_guard.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .probe import (MLPProbe, ProbeDimensionMismatch, Scaler, check_feature_width,
                    load_probe, predict_proba, save_probe)

DIM_1B = 1152
DIM_4B = 2560
MODEL_1B = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
MODEL_4B = "DavidAU/gemma-3-4b-it-heretic-uncensored-abliterated-Extreme"


def _fake_probe(in_dim: int, model_id: str):
    """A randomly-initialised probe + scaler at ``in_dim``, as if just trained."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, in_dim)).astype(np.float32)
    probe = MLPProbe(in_dim, C.HIDDEN1, C.HIDDEN2, C.DROPOUT).eval()
    probe.meta = {"model_id": model_id, "in_dim": in_dim}
    return probe, Scaler.fit(x)


def _fake_activations(n: int, dim: int) -> np.ndarray:
    return np.random.default_rng(1).standard_normal((n, dim)).astype(np.float32)


def test_predict_proba_rejects_wrong_width():
    """A 1B probe fed 4B activations raises ProbeDimensionMismatch, not a matmul."""
    probe, scaler = _fake_probe(DIM_1B, MODEL_1B)
    try:
        predict_proba(probe, scaler, _fake_activations(4, DIM_4B))
    except ProbeDimensionMismatch as exc:
        msg = str(exc)
        assert "1152" in msg and "2560" in msg, msg
        assert MODEL_1B in msg, "the message must name the model the probe came from"
        assert "train_probe" in msg, "the message must say how to fix it"
        assert msg.isascii(), "no unicode: Windows cp1252 consoles crash on it"
    else:
        raise AssertionError("mismatch was NOT caught -- the guard is dead")


def test_matching_width_still_works():
    """The guard must not fire on the happy path."""
    probe, scaler = _fake_probe(DIM_1B, MODEL_1B)
    probs = predict_proba(probe, scaler, _fake_activations(4, DIM_1B))
    assert probs.shape == (4,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_forward_and_scaler_guard_both_fire():
    """The two lower-level entry points guard independently of predict_proba."""
    probe, scaler = _fake_probe(DIM_4B, MODEL_4B)
    for call in (
        lambda: probe(torch.zeros(2, DIM_1B)),
        lambda: scaler.transform(_fake_activations(2, DIM_1B)),
    ):
        try:
            call()
        except ProbeDimensionMismatch:
            pass
        else:
            raise AssertionError("a low-level entry point let the mismatch through")


def test_load_probe_expectations():
    """load_probe(expect_dim=...) fails at LOAD time, before any forward pass."""
    probe, scaler = _fake_probe(DIM_1B, MODEL_1B)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.pt"
        save_probe(path, probe, scaler, meta={"model_id": MODEL_1B, "layer": 12})

        ok, _, meta = load_probe(path, expect_dim=DIM_1B, expect_model_id=MODEL_1B)
        assert ok.in_dim == DIM_1B and meta["model_id"] == MODEL_1B
        assert meta["in_dim"] == DIM_1B, "save_probe must stamp the width into meta"

        for kwargs in ({"expect_dim": DIM_4B}, {"expect_model_id": MODEL_4B}):
            try:
                load_probe(path, **kwargs)
            except ProbeDimensionMismatch:
                pass
            else:
                raise AssertionError(f"load_probe accepted {kwargs}")


def test_save_probe_refuses_cross_model_overwrite():
    """Writing a 4B probe over a 1B probe.pt is refused, not silently done."""
    probe_1b, scaler_1b = _fake_probe(DIM_1B, MODEL_1B)
    probe_4b, scaler_4b = _fake_probe(DIM_4B, MODEL_4B)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.pt"
        save_probe(path, probe_1b, scaler_1b, meta={"model_id": MODEL_1B, "layer": 12})
        try:
            save_probe(path, probe_4b, scaler_4b, meta={"model_id": MODEL_4B, "layer": 12})
        except ProbeDimensionMismatch as exc:
            assert MODEL_1B in str(exc) and MODEL_4B in str(exc)
        else:
            raise AssertionError("probe.pt was clobbered by a different model")
        # ...and the 1B checkpoint on disk is untouched.
        assert load_probe(path)[0].in_dim == DIM_1B


def test_artifact_names_are_model_separated():
    """1B and 4B artifacts cannot land on the same filename."""
    assert C.model_tag(MODEL_1B) == "", "the default 1B keeps the historical names"
    assert C.model_tag(MODEL_4B) == "4b"
    assert C.probe_path_for(MODEL_1B).name == "probe.pt"
    assert C.probe_path_for(MODEL_4B).name == "probe_4b.pt"
    for stem, ext in (("probe", ".pt"), ("features", ".npz"), ("metrics", ".json")):
        a = C.artifact_path_for(stem, ext, MODEL_1B)
        b = C.artifact_path_for(stem, ext, MODEL_4B)
        assert a != b, f"{stem}{ext} collides across models"


def test_feature_cache_stamp_guard():
    """A cache stamped with another model is an error, not a silent cache miss."""
    from .model_utils import ActivationWidthError, check_cache_model

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "features.npz"
        np.savez(path, X=_fake_activations(6, DIM_1B), y=np.zeros(6, dtype=np.int64),
                 layer=12, model_id=MODEL_1B, hidden=DIM_1B)
        with np.load(path, allow_pickle=True) as cache:  # close before cleanup (Windows)
            check_cache_model(cache, path, MODEL_1B)  # matching model: no complaint
            try:
                check_cache_model(cache, path, MODEL_4B)
            except ActivationWidthError as exc:
                assert "1152" in str(exc) and MODEL_4B in str(exc)
            else:
                raise AssertionError("a 1B feature cache was accepted for a 4B run")


def test_check_feature_width_helper():
    check_feature_width(DIM_4B, DIM_4B)  # no raise
    try:
        check_feature_width(DIM_1B, DIM_4B, {"model_id": MODEL_1B}, where="unit")
    except ProbeDimensionMismatch as exc:
        assert "[unit]" in str(exc)
    else:
        raise AssertionError("check_feature_width did not fire")


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - a self-test reports, not raises
            failures += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed "
          f"(CPU only, no model loaded, model_tag='{C.MODEL_TAG or 'default-1b'}')")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":  # python -m steering_tutorials.hello_world.test_dim_guard
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
