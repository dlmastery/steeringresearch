"""stats.py — the statistical RIGOR toolkit (the winner contract, made real).

CLAUDE.md §7 binds every "winner / beats baseline / significant" claim to a
four-part contract. The machinery existed only on paper; this module is the real
thing, pure and offline:

  - ``paired_wilcoxon``   : paired Wilcoxon signed-rank (method vs baseline / seed).
  - ``paired_sign_test``  : exact binomial sign test on per-item paired deltas
                            (the correctly-specified replicate test when the
                            replicate is a heterogeneous concept/item, not a seed).
  - ``directional_consistency`` : fraction of items moving in the hypothesised sign.
  - ``bootstrap_ci``      : percentile bootstrap CI on the mean paired delta.
  - ``holm_bonferroni``   : Holm step-down across a sweep family.
  - ``seed_noise_band``   : empirical k-sigma seed band (DERIVED, not assumed).
  - ``ordinal_gate``      : the strict worst-vs-best ordinal gate — VALID ONLY for
                            a matched SINGLE-condition design (same item, repeated
                            seeds). Mis-specified across heterogeneous items.
  - ``verdict``           : sign-aware label in
                            {EXTERNAL-READY, DIRECTIONAL, NULL, NEGATIVE}.
  - ``min_meaningful_effect`` / ``power_note`` : the effect-size registry + a rough
                            power note so screening (n<7) cannot make p-claims.
  - ``rigor_report``      : runs the whole contract and emits ``external_ready``.

Two reviewer-sourced corrections live here:
  * Holm is no longer VACUOUS — ``external_ready`` requires Holm to be APPLIED
    (a real family supplied + this method rejected), surfaced as
    ``legs["holm_applied"]``. No family ⇒ not external-ready.
  * The replicate test for the concept/item-as-replicate design is the PAIRED
    per-item sign test + directional consistency, NOT worst-vs-best across
    heterogeneous items (which the strict ``ordinal_gate`` performs and which is
    only meaningful for a matched single-condition design).

scipy is a pinned dependency (requirements.txt / pyproject.toml), so
``paired_wilcoxon`` uses ``scipy.stats.wilcoxon`` when importable and otherwise
falls back to a self-contained numpy implementation (normal approximation with
continuity + tie correction) so the toolkit never hard-depends on scipy at call
time.

House style mirrors geometry.py / extract.py: ``from __future__ import
annotations``, typed, seeded determinism, numpy float arithmetic.
"""

from __future__ import annotations

from math import comb
from typing import Optional, Sequence, Union

import numpy as np

# Public functions accept either a python sequence of floats or a numpy array;
# every one immediately coerces via np.asarray, so both are interchangeable.
ArrayLike1D = Union[Sequence[float], np.ndarray]

try:  # scipy is a pinned dep; degrade gracefully if a stripped env lacks it.
    from scipy import stats as _scipy_stats

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only in scipy-less envs
    _scipy_stats = None  # type: ignore[assignment]
    _HAVE_SCIPY = False


def _wilcoxon_numpy(diffs: np.ndarray) -> tuple[float, float]:
    """Self-contained paired Wilcoxon signed-rank (statistic, p_value).

    Drops zero differences (Wilcoxon convention), ranks |diffs| with average
    ranks for ties, sums the positive-sign ranks as the statistic W+, and uses
    the normal approximation with continuity correction and a tie correction to
    the variance for the two-sided p-value. Matches scipy's default
    (mode="auto" → "approx" for the tie/zero case) closely on textbook inputs.
    """
    diffs = diffs[diffs != 0]
    n = diffs.size
    if n == 0:
        return 0.0, 1.0
    abs_d = np.abs(diffs)
    order = np.argsort(abs_d, kind="mergesort")
    sorted_abs = abs_d[order]
    # Average ranks (1-based) with tie handling.
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # mean of ranks i+1..j+1
        ranks[i : j + 1] = avg
        i = j + 1
    signed = np.sign(diffs[order])
    w_plus = float(ranks[signed > 0].sum())
    w_minus = float(ranks[signed < 0].sum())
    statistic = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    # Tie correction to the variance.
    _, counts = np.unique(sorted_abs, return_counts=True)
    tie_term = float((counts**3 - counts).sum())
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return statistic, 1.0
    # Continuity correction toward the mean.
    z = (w_plus - mean_w)
    z = (z - np.sign(z) * 0.5) / np.sqrt(var_w)
    # Two-sided p via the standard normal survival function.
    p = 2.0 * _norm_sf(abs(z))
    p = float(min(1.0, max(0.0, p)))
    return statistic, p


def _norm_sf(x: float) -> float:
    """Survival function of the standard normal (1 - CDF) via erfc, no scipy."""
    from math import erfc, sqrt

    return 0.5 * erfc(x / sqrt(2.0))


def paired_wilcoxon(a: ArrayLike1D, b: ArrayLike1D) -> dict:
    """Paired Wilcoxon signed-rank test on two equal-length samples.

    a, b : equal-length per-seed measurements (e.g. method vs baseline). The test
    is on the paired differences a - b; the null is a symmetric-about-zero
    difference distribution.

    Returns {"statistic", "p_value", "n"} where ``n`` is the number of NON-ZERO
    paired differences actually used. Uses scipy when available, else the numpy
    fallback (same convention).
    """
    a_arr = np.asarray(a, dtype=np.float64).reshape(-1)
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired_wilcoxon needs equal-length inputs, got {a_arr.shape} vs {b_arr.shape}")
    diffs = a_arr - b_arr
    n_nonzero = int(np.count_nonzero(diffs))

    if n_nonzero == 0:
        # No signal at all: degenerate, p=1.0.
        return {"statistic": 0.0, "p_value": 1.0, "n": 0}

    if _HAVE_SCIPY:
        # zero_method="wilcox" drops zero diffs (matches the fallback); the
        # normal approximation keeps parity with small/tied samples.
        res = _scipy_stats.wilcoxon(
            a_arr, b_arr, zero_method="wilcox", correction=True, mode="approx"
        )
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n": n_nonzero}

    statistic, p = _wilcoxon_numpy(diffs)
    return {"statistic": float(statistic), "p_value": float(p), "n": n_nonzero}


def bootstrap_ci(
    deltas: ArrayLike1D,
    *,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI on the MEAN of the paired deltas.

    deltas      : per-seed paired differences (method - baseline).
    n_resamples : bootstrap resample count (≥10k per the rigor contract).
    ci          : central interval mass (0.95 ⇒ 2.5/97.5 percentiles).
    seed        : RNG seed (deterministic).

    Returns {"mean", "lo", "hi"}. "CI excludes 0" ⇔ lo and hi share a sign.
    """
    d = np.asarray(deltas, dtype=np.float64).reshape(-1)
    n = d.size
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one delta")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = d[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot_means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def holm_bonferroni(pvalues: ArrayLike1D, *, alpha: float = 0.05) -> dict:
    """Holm step-down multiple-comparison correction.

    Sort p-values ascending; the i-th smallest (0-based) is compared against
    alpha / (m - i). Once one fails to reject, all larger p-values also fail
    (step-down monotonicity). Adjusted p-values are the running maximum of
    (m - i) * p, clipped to 1.0, mapped back to the INPUT order.

    pvalues : the family of raw p-values (one per sweep comparison).
    alpha   : family-wise error rate.

    Returns {"reject": list[bool], "adjusted": list[float], "order": list[int]}
    all aligned to the INPUT order; "order" is the ascending-p sort order.
    """
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    m = p.size
    if m == 0:
        return {"reject": [], "adjusted": [], "order": []}
    order = np.argsort(p, kind="mergesort")  # ascending p
    reject_sorted = np.zeros(m, dtype=bool)
    adjusted_sorted = np.empty(m, dtype=np.float64)

    running_max = 0.0
    still_rejecting = True
    for rank, idx in enumerate(order):
        factor = m - rank
        adj = min(1.0, factor * float(p[idx]))
        running_max = max(running_max, adj)  # enforce monotone adjusted p
        adjusted_sorted[rank] = running_max
        if still_rejecting and float(p[idx]) <= alpha / factor:
            reject_sorted[rank] = True
        else:
            still_rejecting = False
            reject_sorted[rank] = False

    # Map sorted results back to input order.
    reject = [False] * m
    adjusted = [1.0] * m
    for rank, idx in enumerate(order):
        reject[int(idx)] = bool(reject_sorted[rank])
        adjusted[int(idx)] = float(adjusted_sorted[rank])
    return {"reject": reject, "adjusted": adjusted, "order": [int(i) for i in order]}


def seed_noise_band(values: ArrayLike1D, *, k: float = 2.0) -> dict:
    """Empirical k-sigma band over same-condition seeds (DERIVED, not assumed).

    The project bans rule-of-thumb noise thresholds: the seed band must be the
    measured 2σ of repeated same-condition runs. Uses the sample standard
    deviation (ddof=1) so a single seed yields std=0 / a degenerate band.

    values : same-condition per-seed metric values.
    k      : band half-width in standard deviations (default 2.0 ⇒ ±2σ).

    Returns {"mean", "std", "band": (lo, hi)}.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("seed_noise_band needs at least one value")
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return {"mean": mean, "std": std, "band": (mean - k * std, mean + k * std)}


def ordinal_gate(eval_values: ArrayLike1D, baseline_values: ArrayLike1D) -> dict:
    """Strict worst(eval) > best(baseline) ordinal gate.

    VALID ONLY for a MATCHED SINGLE-CONDITION design — the same item/concept run
    under repeated seeds, so the two arrays are exchangeable draws from one
    condition. There it is the hardest, most defensible win criterion: the WORST
    evaluation seed still beats the BEST baseline seed (no overlap at all).

    MIS-SPECIFIED for the concept/item-as-replicate design: different items have
    different intrinsic baseline difficulty, so comparing the worst item under
    the method to the best (different) item under the baseline is not a like-for-
    like comparison. Use ``paired_sign_test`` + ``directional_consistency`` on the
    per-item deltas instead (that is what ``rigor_report`` now does for its gate).
    Larger-is-better is assumed (the composite is constructed that way).

    Returns {"passes", "worst_eval", "best_baseline"}.
    """
    e = np.asarray(eval_values, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline_values, dtype=np.float64).reshape(-1)
    if e.size == 0 or b.size == 0:
        raise ValueError("ordinal_gate needs non-empty eval and baseline arrays")
    worst_eval = float(e.min())
    best_baseline = float(b.max())
    return {
        "passes": bool(worst_eval > best_baseline),
        "worst_eval": worst_eval,
        "best_baseline": best_baseline,
    }


def paired_sign_test(a: ArrayLike1D, b: ArrayLike1D) -> dict:
    """Exact two-sided binomial sign test on paired per-item data.

    The correctly-specified replicate test for the concept/item-as-replicate
    design: each item contributes ONE paired delta a-b; under the null the sign
    of that delta is a fair coin. Ties (zero deltas) are dropped (the standard
    sign-test convention), and the two-sided p-value is

        p = min(1, 2 * sum_{i=0}^{k} C(n, i) * 0.5^n),   k = min(#pos, #neg)

    which is distribution-free (no normality, no exchangeability across the
    heterogeneous items — unlike the strict ``ordinal_gate``).

    a, b : equal-length per-item measurements (method vs baseline).

    Returns {"statistic", "p_value", "n", "n_pos", "n_neg"} where ``statistic``
    is the number of positive deltas and ``n`` the count of NON-ZERO deltas.
    """
    a_arr = np.asarray(a, dtype=np.float64).reshape(-1)
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"paired_sign_test needs equal-length inputs, got {a_arr.shape} vs {b_arr.shape}"
        )
    diffs = a_arr - b_arr
    n_pos = int(np.count_nonzero(diffs > 0))
    n_neg = int(np.count_nonzero(diffs < 0))
    n = n_pos + n_neg
    if n == 0:
        return {"statistic": 0, "p_value": 1.0, "n": 0, "n_pos": 0, "n_neg": 0}
    k = min(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5**n)
    p = float(min(1.0, 2.0 * tail))
    return {"statistic": n_pos, "p_value": p, "n": n, "n_pos": n_pos, "n_neg": n_neg}


def directional_consistency(deltas: ArrayLike1D, hypothesized_sign: int = 1) -> float:
    """Fraction of items whose delta matches the hypothesised sign (0..1).

    deltas           : per-item method-minus-baseline differences.
    hypothesized_sign: +1 if larger-is-better (the composite default), -1 if the
                       hypothesis predicts a DECREASE.

    A zero delta matches neither sign and so lowers the consistency (no change is
    not directional evidence). Returns 0.0 for an empty input.
    """
    d = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return 0.0
    want = 1 if hypothesized_sign >= 0 else -1
    return float(np.count_nonzero(np.sign(d) == want) / d.size)


# ---------------------------------------------------------------------------
# Effect-size registry + power note (screening must not make p-claims).
# ---------------------------------------------------------------------------
# The smallest delta worth calling a win, per metric — a method that "wins" by a
# sub-threshold margin is NUMEROLOGY (CLAUDE.md §7), not a result. Registry so the
# floor is one documented edit, not a magic number scattered across call sites.
MIN_MEANINGFUL_EFFECT: dict[str, float] = {
    "composite": 0.02,           # composite units (0..1-ish scale)
    "behavior_efficacy": 0.05,   # concept-rate / judge score
    "mmlu_drop_pp": 0.02,        # 2 pp of capability
    "dppl_norm": 0.05,           # 5% relative PPL move
    "compliance_rate": 0.01,     # any real safety leak matters
    "harmless_refusal_rate": 0.05,
}


def min_meaningful_effect(metric: str, default: float = 0.0) -> float:
    """Look up the minimum meaningful effect for ``metric`` (``default`` if unset)."""
    return MIN_MEANINGFUL_EFFECT.get(metric, default)


def power_note(n: int, effect: float, sd: float) -> dict:
    """Rough power / minimum-achievable-p note so screening (n<7) can't claim p.

    Two cheap, honest sanity numbers (NOT a substitute for a real power analysis):

      * ``min_achievable_p`` — the smallest two-sided exact sign-test p attainable
        at this ``n`` (every item moving the same way): ``min(1, 2 * 0.5**n)``.
        If that already exceeds 0.05 the design is *structurally* underpowered:
        no arrangement of the data can reach significance, so a p-claim is
        impossible regardless of the effect. (n=4 ⇒ 0.125; n=5 ⇒ 0.0625; n=6 ⇒
        0.03125; n=7 ⇒ 0.015625 — the floor for the project's n≥7 rule.)
      * ``approx_power`` — a normal-approximation power for a one-sample mean at
        alpha=0.05 two-sided given ``effect`` and per-item ``sd``: with
        z = effect/(sd/sqrt(n)), power ≈ Φ(z - 1.96) (clipped to [0,1]).

    ``is_screening`` mirrors CLAUDE.md §7: n<7 is SCREENING and must not carry a
    significance claim. Returns the numbers + a one-line human ``note``.
    """
    n = int(n)
    is_screening = n < 7
    min_p = float(min(1.0, 2.0 * (0.5**n))) if n >= 1 else 1.0
    can_reach = min_p <= 0.05
    if sd > 0 and n >= 1:
        z = float(effect) / (float(sd) / np.sqrt(n))
        approx_power = float(min(1.0, max(0.0, 1.0 - _norm_sf(abs(z) - 1.96))))
    else:
        approx_power = 0.0
    if not can_reach:
        note = (
            f"n={n}: structurally underpowered — minimum attainable two-sided "
            f"p is {min_p:.4f} > 0.05; NO p<0.05 claim is possible (SCREENING only)."
        )
    elif is_screening:
        note = (
            f"n={n}: SCREENING (n<7). p<0.05 is attainable (min p={min_p:.4f}) but "
            f"the §7 evaluation floor is n≥7; treat any p as provisional."
        )
    else:
        note = (
            f"n={n}: EVALUATION-eligible. min attainable p={min_p:.4f}; "
            f"approx power≈{approx_power:.2f} at effect={effect:g}, sd={sd:g}."
        )
    return {
        "n": n,
        "is_screening": is_screening,
        "min_achievable_p": min_p,
        "can_reach_p05": can_reach,
        "approx_power": approx_power,
        "note": note,
    }


def verdict(
    deltas: ArrayLike1D,
    *,
    family_pvalues: Optional[ArrayLike1D] = None,
    mme: float = 0.0,
    hypothesized_sign: int = 1,
) -> dict:
    """Sign-AWARE verdict label over per-item paired deltas (method - baseline).

    Returns a label in {EXTERNAL-READY, DIRECTIONAL, NULL, NEGATIVE}. A reviewer
    caught a SIGNIFICANT result in the WRONG direction being mislabelled
    "DIRECTIONAL"; this helper makes the sign first-class:

      * NEGATIVE        — significant (paired Wilcoxon p<0.05) but the mean delta
                          points AGAINST ``hypothesized_sign`` (a real regression).
      * EXTERNAL-READY  — ALL of: n≥7 (the §7 evaluation floor) AND significant AND
                          bootstrap CI excludes 0 AND mean-delta sign ==
                          ``hypothesized_sign`` AND |mean delta| ≥ ``mme`` AND Holm
                          APPLIED (a real family supplied + this method rejected).
      * DIRECTIONAL     — points the hypothesised way with SOME evidence
                          (significant, or CI excludes 0, or >half the items
                          consistent) but not the full external bar.
      * NULL            — no directional signal.

    deltas        : per-item paired differences.
    family_pvalues: the sweep family for the Holm leg (this method's Wilcoxon p
                    should be a member); without it EXTERNAL-READY is impossible.
    mme           : minimum meaningful effect on |mean delta| (see
                    ``min_meaningful_effect``); a sub-threshold "win" is NUMEROLOGY.
    hypothesized_sign: +1 larger-is-better (default), -1 if a decrease is predicted.
    """
    d = np.asarray(deltas, dtype=np.float64).reshape(-1)
    n = d.size
    if n == 0:
        raise ValueError("verdict needs at least one delta")
    zeros = np.zeros_like(d)
    wil = paired_wilcoxon(d, zeros)
    boot = bootstrap_ci(d)
    mean = float(d.mean())
    sign = int(np.sign(mean))
    want = 1 if hypothesized_sign >= 0 else -1

    significant = wil["p_value"] < 0.05
    ci_excludes_zero = (boot["lo"] > 0.0 and boot["hi"] > 0.0) or (
        boot["lo"] < 0.0 and boot["hi"] < 0.0
    )
    big_enough = abs(mean) >= float(mme)
    consistency = directional_consistency(d, want)
    enough_n = n >= 7

    holm_applied = False
    holm_rejected = False
    if family_pvalues is not None:
        fam = np.asarray(family_pvalues, dtype=np.float64).reshape(-1)
        if fam.size:
            holm_applied = True
            holm = holm_bonferroni(fam)
            this_idx = int(np.argmin(np.abs(fam - wil["p_value"])))
            holm_rejected = bool(holm["reject"][this_idx])
    holm_ok = holm_applied and holm_rejected

    if significant and sign != 0 and sign != want:
        label = "NEGATIVE"
    elif (
        enough_n
        and significant
        and ci_excludes_zero
        and sign == want
        and big_enough
        and holm_ok
    ):
        label = "EXTERNAL-READY"
    elif sign == want and (significant or ci_excludes_zero or consistency > 0.5):
        label = "DIRECTIONAL"
    else:
        label = "NULL"

    return {
        "label": label,
        "n": int(n),
        "mean_delta": mean,
        "sign": sign,
        "hypothesized_sign": want,
        "wilcoxon": wil,
        "bootstrap_ci": boot,
        "significant": significant,
        "ci_excludes_zero": ci_excludes_zero,
        "directional_consistency": consistency,
        "mme": float(mme),
        "big_enough": big_enough,
        "enough_n": enough_n,
        "holm_applied": holm_applied,
        "holm_rejected": holm_rejected,
    }


def rigor_report(
    method_seeds: ArrayLike1D,
    baseline_seeds: ArrayLike1D,
    *,
    family_pvalues: Optional[ArrayLike1D] = None,
    hypothesized_sign: int = 1,
) -> dict:
    """The winner contract in one call (CLAUDE.md §7) — reviewer-corrected.

    Runs, on the paired per-item deltas (method - baseline):
      * paired Wilcoxon signed-rank (magnitude-aware significance),
      * percentile bootstrap CI on the mean delta,
      * the PAIRED per-item SIGN TEST + directional consistency — the correctly
        specified replicate test for the concept/item-as-replicate design,
      * and, when ``family_pvalues`` is supplied, the Holm step-down.

    TWO reviewer fixes vs the old implementation:

      1. HOLM IS NO LONGER VACUOUS. The old code set ``holm_rejected=True`` when
         no family was supplied, so a single un-corrected comparison could be
         "external-ready". Now ``legs["holm_applied"]`` is True only when a real,
         non-empty family is supplied, and ``external_ready`` REQUIRES Holm to be
         applied AND this method rejected. No family ⇒ not external-ready.

      2. THE GATE IS THE PAIRED SIGN TEST, NOT WORST-VS-BEST. The strict
         ``ordinal_gate`` (worst eval > best baseline) is mis-specified across
         heterogeneous items; ``external_ready`` now uses the per-item sign test
         (p<0.05) AND full directional consistency. ``ordinal_gate`` is still
         REPORTED (for the matched single-condition reader) but does not gate.

    method_seeds, baseline_seeds : PAIRED per-item (or per-seed) measurements,
        same length and order. ``hypothesized_sign`` is +1 (larger-is-better,
        the composite default) or -1.

    ``external_ready`` is True ONLY when ALL of:
      (1) Wilcoxon p < 0.05,
      (2) bootstrap CI excludes 0,
      (3) paired sign test p < 0.05 AND directional consistency == 1.0,
      (4) n ≥ 7 (the §7 evaluation floor),
      (5) Holm APPLIED (real family supplied) AND this method Holm-rejected.
    """
    a = np.asarray(method_seeds, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline_seeds, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"rigor_report needs paired equal-length inputs, got {a.shape} vs {b.shape}")
    deltas = a - b
    n = int(a.size)
    want = 1 if hypothesized_sign >= 0 else -1

    wilcoxon = paired_wilcoxon(a, b)
    boot = bootstrap_ci(deltas)
    gate = ordinal_gate(a, b)               # reported only (matched-design reader)
    sign = paired_sign_test(a, b)
    consistency = directional_consistency(deltas, want)

    # Leg (2): the bootstrap CI excludes zero ⇔ lo and hi are on the same side.
    ci_excludes_zero = (boot["lo"] > 0.0 and boot["hi"] > 0.0) or (
        boot["lo"] < 0.0 and boot["hi"] < 0.0
    )
    # Leg (1): Wilcoxon significant.
    wilcoxon_sig = wilcoxon["p_value"] < 0.05
    # Leg (3): the correctly-specified per-item gate (sign test + full consistency).
    sign_test_sig = sign["p_value"] < 0.05
    fully_consistent = consistency >= 1.0 - 1e-12
    paired_gate_passes = bool(sign_test_sig and fully_consistent)
    # Leg (4): the §7 evaluation floor.
    enough_n = n >= 7
    # Leg (3, legacy/reported): the strict ordinal gate.
    gate_passes = bool(gate["passes"])

    # Leg (5): Holm — MEANINGFUL only (the vacuous-pass bug is fixed here).
    holm: Optional[dict] = None
    holm_applied = False
    holm_rejected = False
    if family_pvalues is not None:
        fam = np.asarray(family_pvalues, dtype=np.float64).reshape(-1)
        if fam.size:
            holm_applied = True
            holm = holm_bonferroni(family_pvalues)
            # Identify THIS method's entry as the family member closest to its
            # own Wilcoxon p-value (the family is built from per-method p-values).
            this_idx = int(np.argmin(np.abs(fam - wilcoxon["p_value"])))
            holm_rejected = bool(holm["reject"][this_idx])

    external_ready = bool(
        wilcoxon_sig
        and ci_excludes_zero
        and paired_gate_passes
        and enough_n
        and holm_applied
        and holm_rejected
    )

    return {
        "wilcoxon": wilcoxon,
        "bootstrap_ci": boot,
        "ordinal_gate": gate,
        "paired_sign_test": sign,
        "directional_consistency": consistency,
        "holm": holm,
        "n": n,
        "legs": {
            "wilcoxon_significant": wilcoxon_sig,
            "ci_excludes_zero": ci_excludes_zero,
            "sign_test_significant": sign_test_sig,
            "fully_consistent": fully_consistent,
            "paired_gate_passes": paired_gate_passes,
            "ordinal_gate_passes": gate_passes,
            "enough_seeds": enough_n,
            "holm_applied": holm_applied,
            "holm_rejected": holm_rejected,
        },
        "external_ready": external_ready,
    }
