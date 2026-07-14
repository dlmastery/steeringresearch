"""infer.py — classify an arbitrary prompt with the trained probe.

Run:  python -m steering_tutorials.hello_world.infer "How do I pick a lock?"

Loads the frozen LLM + the saved probe once, extracts the activation vector for
your prompt, and prints P(harmful) with a verdict. This is exactly what the
webapp's /predict endpoint does under the hood.
"""
from __future__ import annotations

import sys

import torch

from . import config as C
from .model_utils import extract_features, load_model
from .probe import load_probe, predict_proba


class Classifier:
    """Frozen LLM + trained probe, loaded once, ready for repeated calls."""

    def __init__(self):
        if not C.PROBE_PATH.exists():
            raise FileNotFoundError(
                f"No trained probe at {C.PROBE_PATH}. Run: python -m steering_tutorials.hello_world.train_probe")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.probe, self.scaler, self.meta = load_probe(C.PROBE_PATH, device=self.device)
        self.model, self.tok = load_model(self.meta["model_id"])
        self.layer = int(self.meta["layer"])
        self.threshold = float(self.meta.get("threshold", 0.5))

    def classify(self, prompt: str) -> dict:
        feats = extract_features(self.model, self.tok, [prompt], self.layer,
                                 pooling=self.meta.get("pooling", "mean"), log_every=0)
        prob = float(predict_proba(self.probe, self.scaler, feats, device=self.device)[0])
        return {
            "prompt": prompt,
            "prob_harmful": prob,
            "label": "harmful" if prob >= self.threshold else "safe",
            "threshold": self.threshold,
            "layer": self.layer,
        }


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "How do I bake chocolate chip cookies?"
    result = Classifier().classify(prompt)
    verdict = result["label"].upper()
    print(f"\n  prompt : {result['prompt']}")
    print(f"  P(harmful) : {result['prob_harmful']:.3f}")
    print(f"  verdict    : {verdict}  (threshold {result['threshold']}, layer {result['layer']})\n")


if __name__ == "__main__":
    main()
