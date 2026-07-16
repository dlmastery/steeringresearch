"""gate.py — the CONDITION in conditional steering: *when* do we intervene?

Lesson 1 trained a tiny MLP probe that READS harm out of Gemma-3-1B's
layer-12 activations. Lesson 2 does the WRITE half — it steers the model
toward refusal. But steering every prompt would break harmless requests, so
we only steer when the prompt is actually harmful. The thing that makes that
call is exactly the lesson-1 probe. We do not retrain or reinvent it here;
the classifier that learned "is this harmful?" becomes the gate that decides
"should I steer?".

This is the CAST recipe (Conditional Activation Steering): read a concept
direction, and only apply the steering vector when a lightweight condition
fires. See Lee, et al. 2024 'Programming Refusal with Conditional
Activation Steering' (arXiv:2409.05907) — the condition there is a learned
projection; here it is our MLP probe. [UNVERIFIED arXiv id — confirm before
citing externally.]

The gate is deliberately thin: one forward pass to pull the layer-12
mean-pooled activation, one probe evaluation, one threshold comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# The probe lives in lesson 1 and is imported verbatim — same weights, same
# scaler, same decision surface. Loading it here is what "reuse the READ side"
# means concretely.
from steering_tutorials.hello_world.probe import load_probe, predict_proba


class HarmGate:
    """Decides, per prompt, whether steering should fire — using the lesson-1 probe.

    The gate holds a frozen reference to the already-loaded Gemma model (so we
    can extract its activations) plus the trained probe + scaler. It exposes a
    single question — :meth:`is_harmful` — which returns both a hard yes/no and
    the underlying probability so the caller can log or threshold differently.

    Parameters
    ----------
    model, tok:
        An already-loaded Gemma model + tokenizer (the SAME abliterated model
        used everywhere in this tutorial). The gate never loads a model itself.
    probe_path:
        Path to the lesson-1 probe checkpoint. If ``None``, we resolve it
        relative to this file (``../hello_world/artifacts/probe.pt``).
    layer:
        Residual-stream layer to read. If ``None``, we take it from the probe's
        own metadata so the gate and the probe can never disagree about which
        layer the classifier was trained on.
    """

    def __init__(self, model, tok, probe_path: str | Path | None = None,
                 layer: int | None = None):
        self.model = model
        self.tok = tok

        # Default to the lesson-1 artifact, resolved relative to THIS file so
        # the tutorial runs from any working directory.
        if probe_path is None:
            probe_path = (Path(__file__).resolve().parent.parent
                          / "hello_world" / "artifacts" / "probe.pt")
        self.probe_path = Path(probe_path)

        # load_probe returns (probe, scaler, meta). meta carries model_id,
        # layer, pooling and the decision threshold the probe was trained with.
        self.probe, self.scaler, self.meta = load_probe(str(self.probe_path),
                                                         device="cpu")

        # Prefer the caller's layer; otherwise trust the probe's own metadata.
        self.layer = int(layer) if layer is not None else int(self.meta["layer"])
        # threshold is stored in meta (0.5 by default); fall back defensively.
        self.threshold = float(self.meta.get("threshold", 0.5))

    def is_harmful(self, prompt: str) -> tuple[bool, float]:
        """Return ``(fire, prob)``: should we steer, and the probe's P(harmful).

        Steps mirror lesson-1 inference exactly:
          1. mean-pool the residual stream at ``self.layer`` into one vector,
          2. run the probe (which standardizes with its stored scaler),
          3. compare P(harmful) to the trained threshold.
        """
        # Deferred import: model_utils is a sibling module in this same lesson;
        # importing it lazily keeps ``import gate`` working even before the rest
        # of the lesson is wired up, and avoids loading torch at import time.
        from steering_tutorials.hello_world_steering.model_utils import (
            mean_pool_activation,
        )

        feats = mean_pool_activation(self.model, self.tok, prompt,
                                     self.layer).reshape(1, -1)
        prob = float(predict_proba(self.probe, self.scaler, feats, device="cpu")[0])
        return prob >= self.threshold, prob
