"""stats.py — the statistical RIGOR toolkit (the winner contract, made real).

CLAUDE.md §7 binds every "winner / beats baseline / significant" claim to a
four-part contract. The machinery existed only on paper; this module is the real
thing, pure and offline:

  - ``paired_wilcoxon``   : paired Wilcoxon signed-rank (method vs baseline / seed).
  - ``bootstrap_ci``      : percentile bootstrap CI on the mean paired delta.
  - ``holm_bonferroni``   : Holm step-down across a sweep family.
  - ``seed_noise_band``   : empirical k-sigma seed band (DERIVED, not assumed).
  - ``ordinal_gate``      : the strict EXTERNAL-READY ordinal gate
                            (worst eval seed > best baseline seed).
  - ``rigor_report``      : runs the whole contract and emits ``external_ready``.

scipy is a pinned dependency (requirements.txt / pyproject.toml), so
``paired_wilcoxon`` uses ``scipy.stats.wilcoxon`` when importable and otherwise
falls back to a self-contained numpy implementation (normal approximation with
continuity + tie correction) so the toolkit never hard-depends on scipy at call
time.

House style mirrors geometry.py / extract.py: ``from __future__ import
annotations``, typed, seeded determinism, numpy float arithmetic.
"""

from __future__ import annotations

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
    """Strict EXTERNAL-READY ordinal gate: worst(eval) > best(baseline).

    The hardest, most defensible win criterion in the project: the WORST
    evaluation seed must still beat the BEST baseline seed (no overlap of the
    two seed distributions at all). Larger-is-better is assumed (the composite
    is constructed that way).

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


def rigor_report(
    method_seeds: ArrayLike1D,
    baseline_seeds: ArrayLike1D,
    *,
    family_pvalues: Optional[ArrayLike1D] = None,
) -> dict:
    """The four-part winner contract in one call (CLAUDE.md §7).

    Runs paired Wilcoxon + bootstrap CI on the paired deltas (method - baseline)
    + the ordinal gate, and — when ``family_pvalues`` is supplied (this method's
    raw p-value must be one of them) — the Holm step-down across the sweep family.

    method_seeds, baseline_seeds : PAIRED per-seed measurements (same length,
        same seed order). bootstrap/Wilcoxon operate on the per-seed deltas.
    family_pvalues : optional family of raw p-values for the Holm leg. This
        method's Wilcoxon p-value should appear in the family; the Holm leg is
        satisfied iff the closest-matching family entry is Holm-rejected.

    ``external_ready`` is True ONLY when ALL of:
      (1) Wilcoxon p < 0.05,
      (2) bootstrap CI excludes 0,
      (3) ordinal gate passes,
      (4) IF a family is given, this method is Holm-rejected.
    """
    a = np.asarray(method_seeds, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline_seeds, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"rigor_report needs paired equal-length inputs, got {a.shape} vs {b.shape}")
    deltas = a - b

    wilcoxon = paired_wilcoxon(a, b)
    boot = bootstrap_ci(deltas)
    gate = ordinal_gate(a, b)

    # Leg (2): the bootstrap CI excludes zero ⇔ lo and hi are on the same side.
    ci_excludes_zero = (boot["lo"] > 0.0 and boot["hi"] > 0.0) or (
        boot["lo"] < 0.0 and boot["hi"] < 0.0
    )
    # Leg (1): Wilcoxon significant.
    wilcoxon_sig = wilcoxon["p_value"] < 0.05
    # Leg (3): ordinal gate.
    gate_passes = bool(gate["passes"])

    holm: Optional[dict] = None
    holm_rejected = True  # vacuously satisfied when no family is supplied
    if family_pvalues is not None:
        holm = holm_bonferroni(family_pvalues)
        fam = np.asarray(family_pvalues, dtype=np.float64).reshape(-1)
        if fam.size:
            # Identify THIS method's entry as the family member closest to its
            # own Wilcoxon p-value (the family is built from per-method p-values).
            this_idx = int(np.argmin(np.abs(fam - wilcoxon["p_value"])))
            holm_rejected = bool(holm["reject"][this_idx])
        else:
            holm_rejected = False

    external_ready = bool(
        wilcoxon_sig and ci_excludes_zero and gate_passes and holm_rejected
    )

    return {
        "wilcoxon": wilcoxon,
        "bootstrap_ci": boot,
        "ordinal_gate": gate,
        "holm": holm,
        "legs": {
            "wilcoxon_significant": wilcoxon_sig,
            "ci_excludes_zero": ci_excludes_zero,
            "ordinal_gate_passes": gate_passes,
            "holm_rejected": holm_rejected,
        },
        "external_ready": external_ready,
    }
