"""inseparability.py -- turn the impossibility theorem into a MEASURED number.

REFERENCE (WebFetch-VERIFIED 2026-08-29)
----------------------------------------
Pant, Lohani, Kumar, "On the Inseparability of Instructions and Data in
Shared-Embedding Sequence Models", arXiv:2606.27567, 25 Jun 2026 (cs.CR).
Abstract, verbatim: "We prove this is not a coincidence: in shared-embedding
architectures that lack enforced control-data separation, perfect
prompt-injection prevention is mathematically impossible."

WHY THIS FILE EXISTS
--------------------
An impossibility proof is usually where a reader stops, because there is nothing
to run. But the theorem's first result is not an abstraction -- it is an
EQUALITY with an estimable right-hand side. Bayes-optimal error in recovering
"trusted vs untrusted" from a shared embedding is

    err* = (1 - TV(P_trusted, P_untrusted)) / 2

so the bound on ANY provenance classifier -- current, future, or perfect -- is a
property of the two token distributions, and we can measure it on our own
corpora rather than cite it. That is the whole point of this module: it converts
"perfect prevention is impossible" into "on THIS data, no detector can do better
than X, and here is X".

HOW TO READ THE NUMBER
----------------------
  err* near 0.5  -> the two channels are near-indistinguishable; a provenance
                    detector is near-useless no matter how good it is
  err* near 0.0  -> the distributions barely overlap and detection is easy
                    (which, on natural text, should make you suspect a confound
                    -- see below)
TV is estimated on a FINITE sample and is upward-biased: with few samples and a
large vocabulary, two draws from the SAME distribution look far apart. So a
naive estimate makes the problem look EASIER than it is. This module therefore
always reports a same-distribution CONTROL (split one corpus in half and measure
TV between the halves). The control is the floor; only TV above it is signal.
Reporting TV without that floor would overstate separability every time.

WHAT THIS DOES NOT SHOW
-----------------------
A low err* on unigrams does NOT mean injection detection is solved -- it means
these particular surface distributions differ, which an attacker who controls
encoding can change (the theorem's third result: finite training cannot certify
invariance over infinite semantic-equivalence classes). Measuring separability
under paraphrase/obfuscation is the honest follow-up, not this.

CPU-only. No model. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import math
import re
from collections import Counter

__all__ = ["total_variation", "bayes_error", "estimate_provenance_bound",
           "tokenize"]

_TOK = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list:
    """Deliberately crude unigrams: the point is a DISTRIBUTION, not parsing."""
    return _TOK.findall(str(text).lower())


def _dist(docs) -> Counter:
    c = Counter()
    for d in docs:
        c.update(tokenize(d))
    return c


def total_variation(p: Counter, q: Counter) -> float:
    """TV(P, Q) = 1/2 * sum_x |P(x) - Q(x)| over the union of supports."""
    np_, nq = sum(p.values()), sum(q.values())
    if not np_ or not nq:
        return float("nan")
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) / np_ - q.get(k, 0) / nq) for k in keys)


def bayes_error(tv: float) -> float:
    """The theorem's Result 1: err* = (1 - TV) / 2."""
    return (1.0 - float(tv)) / 2.0


def estimate_provenance_bound(trusted, untrusted, seed: int = 0) -> dict:
    """Measure TV and the implied Bayes floor, WITH the same-distribution control.

    Returns the raw estimate, the control floor, and the excess. Quote the
    excess, never the raw TV: on a finite sample the raw number is inflated by
    exactly the amount the control measures.
    """
    import random

    trusted, untrusted = list(trusted), list(untrusted)
    P, Q = _dist(trusted), _dist(untrusted)
    tv = total_variation(P, Q)

    # CONTROL: split each corpus in half and measure TV between the halves.
    # Same distribution by construction, so whatever TV appears here is
    # finite-sample bias, not separability.
    rng = random.Random(seed)
    ctrl = []
    for docs in (trusted, untrusted):
        d = list(docs)
        rng.shuffle(d)
        h = len(d) // 2
        if h >= 1:
            ctrl.append(total_variation(_dist(d[:h]), _dist(d[h:])))
    floor = sum(ctrl) / len(ctrl) if ctrl else float("nan")

    out = {
        "n_trusted": len(trusted), "n_untrusted": len(untrusted),
        "vocab_trusted": len(P), "vocab_untrusted": len(Q),
        "tv_raw": round(tv, 4),
        "tv_same_distribution_floor": round(floor, 4),
        "tv_excess_over_floor": round(tv - floor, 4) if floor == floor else None,
        "bayes_error_raw": round(bayes_error(tv), 4),
        "bayes_error_at_floor": round(bayes_error(floor), 4) if floor == floor else None,
        "reference": "arXiv:2606.27567 Result 1: err* = (1 - TV)/2",
    }
    out["reading"] = (
        "A provenance detector on THESE unigram distributions cannot beat "
        "%.1f%% error; the same-distribution control already reaches %.1f%%, so "
        "only the %.4f TV above the floor is real separability."
        % (100 * out["bayes_error_raw"],
           100 * (out["bayes_error_at_floor"] or float("nan")),
           out["tv_excess_over_floor"] if out["tv_excess_over_floor"] is not None else float("nan")))
    return out


def _self_test() -> None:
    import random

    rng = random.Random(0)

    # 1. identical distributions -> TV ~ 0 -> err* ~ 0.5 (detection impossible)
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    a = [" ".join(rng.choice(words) for _ in range(40)) for _ in range(300)]
    b = [" ".join(rng.choice(words) for _ in range(40)) for _ in range(300)]
    r = estimate_provenance_bound(a, b)
    assert r["bayes_error_raw"] > 0.45, r
    print("OK  identical channels: TV %.4f -> Bayes error %.4f (near 0.5 = "
          "undetectable, as the theorem says)"
          % (r["tv_raw"], r["bayes_error_raw"]))

    # 2. disjoint vocabularies -> TV ~ 1 -> err* ~ 0
    c = [" ".join(rng.choice(["zeta", "eta", "theta"]) for _ in range(40))
         for _ in range(300)]
    r2 = estimate_provenance_bound(a, c)
    assert r2["bayes_error_raw"] < 0.05, r2
    print("OK  disjoint channels: TV %.4f -> Bayes error %.4f (detection easy)"
          % (r2["tv_raw"], r2["bayes_error_raw"]))

    # 3. the finite-sample floor is REAL and non-zero
    assert r["tv_same_distribution_floor"] > 0.0
    assert r["tv_excess_over_floor"] < 0.02
    print("OK  same-distribution FLOOR is %.4f, not 0 -- a raw TV read without "
          "it would claim separability that is pure sample noise"
          % r["tv_same_distribution_floor"])

    # 4. the floor shrinks as n grows (it is bias, and bias vanishes)
    small = [" ".join(rng.choice(words) for _ in range(10)) for _ in range(20)]
    rs = estimate_provenance_bound(small, small[::-1])
    assert rs["tv_same_distribution_floor"] > r["tv_same_distribution_floor"]
    print("OK  the floor is LARGER at n=20 (%.4f) than n=300 (%.4f) -- it is "
          "finite-sample bias, so a small corpus overstates separability"
          % (rs["tv_same_distribution_floor"], r["tv_same_distribution_floor"]))

    assert abs(bayes_error(0.0) - 0.5) < 1e-12
    assert abs(bayes_error(1.0) - 0.0) < 1e-12
    print("OK  err* = (1-TV)/2 endpoints: TV=0 -> 0.5, TV=1 -> 0.0")
    print("")
    print("OK -- inseparability.py: the theorem's bound is measurable, and its "
          "finite-sample floor is reported beside it.")


if __name__ == "__main__":
    _self_test()
