"""monitor.py -- the activation monitor: cognitive-element detectors + a rule.

This is the mechanical core of the GAVEL lesson and it is deliberately
MODEL-FREE: everything here operates on already-extracted activation vectors
(numpy arrays). Reading activations out of Gemma lives in run_gavel.py under
``main()``; that keeps this module importable and unit-testable on a CPU-only box
with no model download.

Three pieces, bottom-up:

  1. CEDetector -- one "cognitive element". A diff-of-means direction for a harm
     concept plus a calibrated threshold ``tau``. Its score of an activation ``h``
     is ``(h - benign_mean) . direction``; it FIRES when the score exceeds tau.
     ``tau`` is set on benign activations so only a chosen fraction (TARGET_FPR)
     of benign prompts trip it -- the paper's "configurable precision" knob.

  2. Rule -- an interpretable PREDICATE over which CEs fired. ``any_of`` (block if
     any CE fires), ``all_of`` (block only if all fire), ``at_least(k)``. The
     rule is data-independent: you can swap it without touching the detectors,
     which is exactly the reconfigurable-safeguard property GAVEL is about.

  3. GavelMonitor -- holds the CE library + a rule and turns an activation into a
     decision ``{block, scores, fired, triggered_by, reason}``. The per-CE scores
     make every block auditable: you can name *which* cognitive element tripped
     and by how much.

Method + naming follow the verified paper (arXiv:2601.19768): activations are
represented as fine-grained, interpretable cognitive elements, and safety is a
predicate rule composed over them, checked at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Cognitive element -- one calibrated diff-of-means detector.
# --------------------------------------------------------------------------- #
@dataclass
class CEDetector:
    """One cognitive element: a unit direction + a benign offset + a threshold.

    score(h) = (h - benign_mean) . direction   (a scalar; higher = more like the
    concept). ``fires(h)`` is ``score(h) > tau``. ``tau`` is calibrated on benign
    activations so only ``target_fpr`` of them trip this CE.
    """

    name: str
    direction: np.ndarray          # unit vector [hidden]
    benign_mean: np.ndarray        # [hidden] -- the contrast origin
    tau: float
    target_fpr: float              # the benign FPR tau was calibrated to
    n_pos: int                     # concept examples that built the direction

    def score(self, h: np.ndarray) -> float:
        return float((h - self.benign_mean) @ self.direction)

    def score_batch(self, H: np.ndarray) -> np.ndarray:
        """Scores for an ``[n, hidden]`` matrix -> ``[n]`` float array."""
        return (H - self.benign_mean) @ self.direction

    def fires(self, h: np.ndarray) -> bool:
        return self.score(h) > self.tau


def build_ce_detector(
    name: str,
    concept_acts: np.ndarray,
    benign_acts: np.ndarray,
    target_fpr: float = 0.05,
) -> CEDetector:
    """Build one CE from concept vs. benign activations, calibrating tau.

    direction = unit(mean(concept) - mean(benign)). We subtract the benign mean at
    score time, so benign activations score ~0 on average and the concept cloud
    scores positive. ``tau`` is the ``(1 - target_fpr)`` quantile of the benign
    scores, i.e. set so only ``target_fpr`` of benign prompts fire this CE.
    """
    concept_acts = np.asarray(concept_acts, dtype=np.float64)
    benign_acts = np.asarray(benign_acts, dtype=np.float64)
    mu_c = concept_acts.mean(axis=0)
    mu_b = benign_acts.mean(axis=0)
    d = mu_c - mu_b
    norm = np.linalg.norm(d)
    direction = d / norm if norm > 0 else d

    benign_scores = (benign_acts - mu_b) @ direction
    # (1 - fpr) quantile: only `target_fpr` of benign scores exceed tau.
    tau = float(np.quantile(benign_scores, 1.0 - target_fpr))
    return CEDetector(
        name=name,
        direction=direction.astype(np.float32),
        benign_mean=mu_b.astype(np.float32),
        tau=tau,
        target_fpr=float(target_fpr),
        n_pos=int(concept_acts.shape[0]),
    )


# --------------------------------------------------------------------------- #
# 2. Rule -- an interpretable predicate over which CEs fired.
# --------------------------------------------------------------------------- #
@dataclass
class Rule:
    """A named predicate over the set of fired CE names -> block? (bool).

    The predicate takes the mapping ``{ce_name: fired_bool}`` and returns True to
    BLOCK. Constructors below cover the common compositions; you can also pass any
    callable for a bespoke rule (e.g. lambda f: f['violence'] and f['payment']).
    """

    name: str
    predicate: Callable[[dict[str, bool]], bool]

    def evaluate(self, fired: dict[str, bool]) -> bool:
        return bool(self.predicate(fired))

    # -- common compositions --------------------------------------------------
    @classmethod
    def any_of(cls, names: list[str]) -> "Rule":
        names = list(names)
        return cls(f"any_of({', '.join(names)})",
                   lambda f: any(f.get(n, False) for n in names))

    @classmethod
    def all_of(cls, names: list[str]) -> "Rule":
        names = list(names)
        return cls(f"all_of({', '.join(names)})",
                   lambda f: bool(names) and all(f.get(n, False) for n in names))

    @classmethod
    def at_least(cls, k: int, names: list[str]) -> "Rule":
        names = list(names)
        return cls(f"at_least_{k}({', '.join(names)})",
                   lambda f: sum(bool(f.get(n, False)) for n in names) >= k)


# --------------------------------------------------------------------------- #
# 3. GavelMonitor -- the CE library + a rule -> a decision.
# --------------------------------------------------------------------------- #
class GavelMonitor:
    """A library of cognitive elements plus a rule; scores an activation -> block?

    ``decide(h)`` returns an auditable record: every CE's score and whether it
    fired, the rule's verdict, and a human-readable reason naming the CE(s) that
    tripped. That transparency is a first-class property of the approach -- a block
    is never a black box.
    """

    def __init__(self, detectors: list[CEDetector], rule: Rule):
        if not detectors:
            raise ValueError("GavelMonitor needs at least one cognitive element")
        self.detectors = list(detectors)
        self.rule = rule

    @property
    def ce_names(self) -> list[str]:
        return [d.name for d in self.detectors]

    def decide(self, h: np.ndarray) -> dict:
        h = np.asarray(h, dtype=np.float64)
        scores = {d.name: d.score(h) for d in self.detectors}
        fired = {d.name: (scores[d.name] > d.tau) for d in self.detectors}
        block = self.rule.evaluate(fired)
        triggered_by = [n for n, f in fired.items() if f]
        if block:
            reason = "blocked by rule %s; CE(s) fired: %s" % (
                self.rule.name, ", ".join(triggered_by) or "(rule-level)")
        else:
            reason = "passed: rule %s not satisfied" % self.rule.name
        return {"block": block, "scores": scores, "fired": fired,
                "triggered_by": triggered_by, "reason": reason}

    def block_mask(self, H: np.ndarray) -> np.ndarray:
        """Vectorized block decisions for an ``[n, hidden]`` matrix -> ``[n]`` bool."""
        H = np.asarray(H, dtype=np.float64)
        fired_cols = {d.name: (d.score_batch(H) > d.tau) for d in self.detectors}
        out = np.zeros(H.shape[0], dtype=bool)
        for i in range(H.shape[0]):
            fired = {n: bool(col[i]) for n, col in fired_cols.items()}
            out[i] = self.rule.evaluate(fired)
        return out

    def firing_rates(self, H: np.ndarray) -> dict[str, float]:
        """Per-CE fraction of rows in ``H`` that trip each cognitive element."""
        H = np.asarray(H, dtype=np.float64)
        n = max(1, H.shape[0])
        return {d.name: float((d.score_batch(H) > d.tau).mean()) if H.size else 0.0
                for d in self.detectors}

    # -- (de)serialization: directions + taus + benign means ------------------
    def save(self, path) -> None:
        np.savez(
            path,
            names=np.array(self.ce_names, dtype=object),
            directions=np.stack([d.direction for d in self.detectors]),
            benign_means=np.stack([d.benign_mean for d in self.detectors]),
            taus=np.array([d.tau for d in self.detectors], dtype=np.float64),
            target_fprs=np.array([d.target_fpr for d in self.detectors], dtype=np.float64),
            n_pos=np.array([d.n_pos for d in self.detectors], dtype=np.int64),
            rule_name=np.array(self.rule.name, dtype=object),
        )

    @classmethod
    def load(cls, path, rule: Rule) -> "GavelMonitor":
        z = np.load(path, allow_pickle=True)
        detectors = [
            CEDetector(
                name=str(z["names"][i]),
                direction=z["directions"][i].astype(np.float32),
                benign_mean=z["benign_means"][i].astype(np.float32),
                tau=float(z["taus"][i]),
                target_fpr=float(z["target_fprs"][i]),
                n_pos=int(z["n_pos"][i]),
            )
            for i in range(len(z["names"]))
        ]
        return cls(detectors, rule)


def make_rule(spec: str, ce_names: list[str]) -> Rule:
    """Translate a config RULE string into a Rule over the given CE names."""
    if spec == "any_of":
        return Rule.any_of(ce_names)
    if spec == "at_least_2":
        return Rule.at_least(2, ce_names)
    if spec == "all_of":
        return Rule.all_of(ce_names)
    raise ValueError("unknown RULE spec %r (want any_of / at_least_2 / all_of)" % spec)


# --------------------------------------------------------------------------- #
# CPU self-test -- NO model. Synthetic clouds verify the detector + rule math.
# Run: python -m steering_tutorials.gavel.monitor
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(0)
    dim = 32
    n = 400

    # Benign cloud at the origin; two concept clouds offset along distinct axes.
    benign = rng.normal(0, 1, size=(n, dim))
    ax_v = np.zeros(dim); ax_v[0] = 1.0          # "violence" axis
    ax_h = np.zeros(dim); ax_h[1] = 1.0          # "hate" axis
    violence = rng.normal(0, 1, size=(n, dim)) + 4.0 * ax_v
    hate = rng.normal(0, 1, size=(n, dim)) + 4.0 * ax_h

    target_fpr = 0.05
    ce_v = build_ce_detector("violence", violence, benign, target_fpr)
    ce_h = build_ce_detector("hate", hate, benign, target_fpr)

    # (a) each CE separates its own concept from benign.
    assert ce_v.score_batch(violence).mean() > ce_v.score_batch(benign).mean()
    assert ce_h.score_batch(hate).mean() > ce_h.score_batch(benign).mean()

    # (b) tau calibration: benign firing rate ~ target_fpr (within sampling noise).
    benign_fpr_v = float((ce_v.score_batch(benign) > ce_v.tau).mean())
    assert abs(benign_fpr_v - target_fpr) < 0.04, f"benign FPR off: {benign_fpr_v}"

    # (c) any_of rule blocks a violent prompt, passes a benign one.
    mon = GavelMonitor([ce_v, ce_h], Rule.any_of(["violence", "hate"]))
    dec_v = mon.decide(violence[0])
    dec_b = mon.decide(benign[0])
    assert dec_v["block"] and "violence" in dec_v["triggered_by"], dec_v
    # A benign vector should usually pass; check the aggregate block rate is low.
    block_rate_benign = float(mon.block_mask(benign).mean())
    assert block_rate_benign < 0.15, f"benign block rate too high: {block_rate_benign}"
    block_rate_violence = float(mon.block_mask(violence).mean())
    assert block_rate_violence > 0.90, f"violence block rate too low: {block_rate_violence}"

    # (d) all_of is strictly stricter than any_of: a violence-only vector that
    #     trips only the violence CE is BLOCKED by any_of but PASSED by all_of.
    strict = GavelMonitor([ce_v, ce_h], Rule.all_of(["violence", "hate"]))
    assert mon.decide(violence[0])["block"]
    assert not strict.decide(violence[0])["block"], "all_of should not fire on one CE"

    # (e) save/load round-trips the decision.
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.npz")
        mon.save(p)
        mon2 = GavelMonitor.load(p, Rule.any_of(["violence", "hate"]))
        assert mon2.decide(violence[0])["block"] == dec_v["block"]

    print("[self-test] OK -- CE calibration, any_of/all_of rules, and save/load all "
          "behave as expected (no model needed).")


if __name__ == "__main__":
    _self_test()
