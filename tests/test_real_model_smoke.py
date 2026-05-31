"""Skippable real-model smoke test — the regression guard for the Triton fix.

This loads a LOCAL real Gemma (if present at models/google/gemma-3-270m-it),
extracts a steering vector, and runs a tiny STEERED greedy generation. On
Windows, Gemma-3 generation otherwise trips torch.compile -> TorchInductor ->
"Cannot find a working triton installation"; `_force_eager_generation` (applied
in model.load_model) must keep this eager so generation succeeds.

It is SKIPPED unless the local weights exist, so the offline test suite stays
green with no network and no model.
"""

from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GEMMA_DIR = _REPO_ROOT / "models" / "google" / "gemma-3-270m-it"
_HAS_GEMMA = (_GEMMA_DIR / "config.json").exists()

pytestmark = pytest.mark.skipif(
    not _HAS_GEMMA,
    reason=f"local Gemma not found at {_GEMMA_DIR}; real-model smoke test skipped",
)


@pytest.mark.real_model
def test_gemma_steered_generation_does_not_crash():
    from steering.eval import generate_responses, generation_behavior_scorer
    from steering.extract import best_layer, extract_bank
    from steering.model import load_model, num_layers

    # Load the local Gemma WITHOUT bitsandbytes (quant='none' -> bf16) so the
    # test runs even where bitsandbytes is unavailable. CPU fallback is fine.
    model, tokenizer = load_model(str(_GEMMA_DIR), quant="none")

    # The Triton fix: generation_config must be set to a dynamic (eager) cache.
    gen_cfg = getattr(model, "generation_config", None)
    assert gen_cfg is not None
    assert getattr(gen_cfg, "cache_implementation", None) == "dynamic", \
        "generation must use the dynamic (uncompiled) cache to avoid Triton"

    pairs = [
        ("the ocean waves crash on the shore", "the desert sand is dry"),
        ("a deep blue sea full of coral", "a barren rocky mountain"),
    ]
    nl = num_layers(model)
    layer = min(best_layer(extract_bank(model, tokenizer, pairs)), nl - 1)
    vec_np = extract_bank(model, tokenizer, pairs)[layer]["diffmean"]
    vector = torch.tensor(vec_np, dtype=torch.float32)

    # 1) A tiny STEERED generation must complete without raising (the crash this
    #    test guards against was raised here on Windows).
    responses = generate_responses(
        model, tokenizer, ["Tell me about your day."],
        layer=layer, vector=vector,
        steering_kwargs={"operation": "add", "alpha": 4.0},
        max_new_tokens=8,
    )
    assert len(responses) == 1
    assert isinstance(responses[0], str)

    # 2) The non-circular generation behavior scorer must run the REAL generation
    #    path (scorer == "generation", not the FakeLM projection fallback).
    out = generation_behavior_scorer(
        model, tokenizer, layer, vector,
        prompts=["Describe the view from the window."],
        pairs=pairs,
        steering_kwargs={"operation": "add", "alpha": 4.0},
        max_new_tokens=8,
    )
    assert out["scorer"] == "generation"
    assert 0.0 <= out["score"] <= 1.0
