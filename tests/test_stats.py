"""Unit checks for stats.py — the four-part winner contract.

  - paired_wilcoxon matches a textbook example (+ a SciPy cross-check if present);
  - bootstrap_ci excludes 0 on a clear positive delta, includes 0 on noise;
  - holm_bonferroni matches a hand-computed step-down;
  - ordinal_gate true/false cases;
  - rigor_report.external_ready is True on a clean win and False when ANY single
    leg fails (each leg tested independently).
"""

import numpy as np

from steering.stats import (
    bootstrap_ci,
    directional_consistency,
    holm_bonferroni,
    min_meaningful_effect,
    ordinal_gate,
    paired_sign_test,
    paired_wilcoxon,
    power_note,
    rigor_report,
    seed_noise_band,
    verdict,
)

try:
    from scipy import stats as _scipy_stats

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# paired_wilcoxon
# --------------------------------------------------------------------------- #
def test_wilcoxon_textbook_example():
    # Classic worked example: differences with a known W statistic.
    # b is all-zero so diffs == a; signed-rank statistic min(W+, W-).
    a = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0, 8.0]
    b = [0.0] * len(a)
    res = paired_wilcoxon(a, b)
    # |diffs| ranks: 1->1, 2->2, 3->3, 4->4, 5->5, 6->6, 7->7, 8->8.
    # negatives are -2 and -5 -> W- = 2 + 5 = 7 ; W+ = 36 - 7 = 29 ; stat = 7.
    assert res["statistic"] == 7.0
    assert res["n"] == 8
    assert 0.0 <= res["p_value"] <= 1.0


def test_wilcoxon_scipy_crosscheck():
    if not _HAVE_SCIPY:  # pragma: no cover
        return
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 1.0, size=15)
    b = rng.normal(0.0, 1.0, size=15)
    res = paired_wilcoxon(a, b)
    ref = _scipy_stats.wilcoxon(a, b, zero_method="wilcox", correction=True, mode="approx")
    assert abs(res["statistic"] - float(ref.statistic)) < 1e-9
    assert abs(res["p_value"] - float(ref.pvalue)) < 1e-9


def test_wilcoxon_all_zero_diffs_is_degenerate():
    res = paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert res["n"] == 0
    assert res["p_value"] == 1.0


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_positive_delta_excludes_zero():
    deltas = [0.8, 1.0, 1.2, 0.9, 1.1, 1.0, 0.95]
    res = bootstrap_ci(deltas, n_resamples=10000, seed=0)
    assert res["lo"] > 0.0, res
    assert res["mean"] > 0.0


def test_bootstrap_ci_zero_mean_noise_includes_zero():
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.0, 1.0, size=30)
    res = bootstrap_ci(deltas, n_resamples=10000, seed=0)
    assert res["lo"] < 0.0 < res["hi"], res


# --------------------------------------------------------------------------- #
# holm_bonferroni
# --------------------------------------------------------------------------- #
def test_holm_hand_computed():
    # p = [0.01, 0.04, 0.03], m=3, alpha=0.05.
    # sorted: 0.01 (idx0) vs 0.05/3=0.0167 -> reject
    #         0.03 (idx2) vs 0.05/2=0.025  -> 0.03 > 0.025 -> FAIL, stop
    #         0.04 (idx1) -> fail
    res = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    assert res["reject"] == [True, False, False]
    assert res["order"] == [0, 2, 1]
    # adjusted: idx0 -> 3*0.01=0.03 ; idx2 -> max(0.03, 2*0.03=0.06)=0.06 ;
    #           idx1 -> max(0.06, 1*0.04=0.04)=0.06
    adj = res["adjusted"]
    assert abs(adj[0] - 0.03) < 1e-9
    assert abs(adj[2] - 0.06) < 1e-9
    assert abs(adj[1] - 0.06) < 1e-9


def test_holm_all_reject():
    res = holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05)
    assert res["reject"] == [True, True, True]


def test_holm_empty():
    res = holm_bonferroni([], alpha=0.05)
    assert res == {"reject": [], "adjusted": [], "order": []}


# --------------------------------------------------------------------------- #
# seed_noise_band
# --------------------------------------------------------------------------- #
def test_seed_noise_band_two_sigma():
    values = [1.0, 1.0, 1.0, 1.0, 2.0]
    res = seed_noise_band(values, k=2.0)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    assert abs(res["mean"] - mean) < 1e-9
    assert abs(res["std"] - std) < 1e-9
    assert abs(res["band"][0] - (mean - 2 * std)) < 1e-9
    assert abs(res["band"][1] - (mean + 2 * std)) < 1e-9


def test_seed_noise_band_single_seed_is_degenerate():
    res = seed_noise_band([3.0])
    assert res["std"] == 0.0
    assert res["band"] == (3.0, 3.0)


# --------------------------------------------------------------------------- #
# ordinal_gate
# --------------------------------------------------------------------------- #
def test_ordinal_gate_pass():
    res = ordinal_gate([0.8, 0.9, 0.85], [0.5, 0.6, 0.55])
    assert res["passes"] is True
    assert res["worst_eval"] == 0.8
    assert res["best_baseline"] == 0.6


def test_ordinal_gate_fail_on_overlap():
    res = ordinal_gate([0.8, 0.55, 0.9], [0.5, 0.6, 0.55])
    # worst eval 0.55 is NOT > best baseline 0.6
    assert res["passes"] is False


# --------------------------------------------------------------------------- #
# rigor_report — the contract
# --------------------------------------------------------------------------- #
def _clean_win():
    # Method clearly and consistently above baseline, no overlap.
    method = [0.80, 0.82, 0.81, 0.83, 0.79, 0.84, 0.80]
    baseline = [0.50, 0.52, 0.49, 0.51, 0.48, 0.53, 0.50]
    return method, baseline


def test_rigor_report_external_ready_clean_win():
    method, baseline = _clean_win()
    fam = [0.001, 0.2, 0.3]  # this method's p is the smallest
    res = rigor_report(method, baseline, family_pvalues=fam)
    assert res["external_ready"] is True
    legs = res["legs"]
    assert legs["wilcoxon_significant"]
    assert legs["ci_excludes_zero"]
    assert legs["paired_gate_passes"]      # sign test + full directional consistency
    assert legs["ordinal_gate_passes"]     # still reported for matched-design reader
    assert legs["enough_seeds"]            # n>=7 evaluation floor
    assert legs["holm_applied"]            # a real family was supplied
    assert legs["holm_rejected"]


def test_rigor_report_holm_not_vacuous_without_family():
    # HOLM-VACUOUS BUG FIX: Wilcoxon + CI + paired-gate all pass, but with NO
    # family the Holm leg was NOT applied -> cannot be external-ready.
    method, baseline = _clean_win()
    res = rigor_report(method, baseline)  # no family
    assert res["holm"] is None
    assert res["legs"]["holm_applied"] is False
    assert res["legs"]["wilcoxon_significant"]  # the other legs DID pass
    assert res["legs"]["ci_excludes_zero"]
    assert res["legs"]["paired_gate_passes"]
    assert res["external_ready"] is False, "no family => Holm not applied => not ready"


def test_rigor_report_holm_applied_and_rejecting_is_external_ready():
    # And it flips to True once a real family is supplied and Holm rejects.
    method, baseline = _clean_win()
    res = rigor_report(method, baseline, family_pvalues=[0.001, 0.2, 0.3])
    assert res["legs"]["holm_applied"] is True
    assert res["legs"]["holm_rejected"] is True
    assert res["external_ready"] is True


def test_rigor_report_fails_when_paired_gate_fails():
    # One item REGRESSES (delta -0.05): sign test p=0.125 and consistency<1.0, so
    # the correctly-specified per-item gate fails even though most items improve.
    method = [0.80, 0.55, 0.82, 0.81, 0.79, 0.83, 0.80]
    baseline = [0.50, 0.60, 0.49, 0.51, 0.48, 0.53, 0.50]
    res = rigor_report(method, baseline, family_pvalues=[0.001, 0.2, 0.3])
    assert res["legs"]["paired_gate_passes"] is False
    assert res["legs"]["ordinal_gate_passes"] is False  # also overlaps
    assert res["directional_consistency"] < 1.0
    assert res["external_ready"] is False


def test_rigor_report_fails_when_wilcoxon_not_significant():
    # Tiny, noisy, sign-mixed deltas -> not significant; also CI includes 0.
    rng = np.random.default_rng(7)
    baseline = rng.normal(0.5, 0.1, size=7)
    method = baseline + rng.normal(0.0, 0.1, size=7)  # zero-mean perturbation
    res = rigor_report(list(method), list(baseline))
    assert res["legs"]["wilcoxon_significant"] is False
    assert res["external_ready"] is False


def test_rigor_report_fails_when_ci_includes_zero():
    # Construct a case where the gate could pass-ish but CI straddles 0.
    rng = np.random.default_rng(3)
    baseline = rng.normal(0.5, 0.2, size=12)
    method = baseline + rng.normal(0.0, 0.3, size=12)
    res = rigor_report(list(method), list(baseline))
    # zero-mean delta -> bootstrap CI must include 0
    assert res["legs"]["ci_excludes_zero"] is False
    assert res["external_ready"] is False


def test_rigor_report_fails_when_holm_rejects_other_but_not_this():
    # Method wins on its own legs, but in the family its p-value is NOT
    # Holm-rejected (a stricter-corrected sibling steals the budget).
    method, baseline = _clean_win()
    res_solo = rigor_report(method, baseline)
    this_p = res_solo["wilcoxon"]["p_value"]
    # Family where this method's p fails Holm: pair it with a tiny sibling and
    # set alpha-budget so this one is not rejected. this_p ~ 0.016 for n=7.
    # Family = [this_p, this_p_clone] makes m=2; smallest vs 0.025, next vs 0.05.
    # Use a family that pushes this method out: a much smaller sibling + a large
    # third so this method's rank-2 threshold 0.05/2 = 0.025; if this_p < 0.025
    # it would still reject. Instead inflate the family size so the threshold
    # tightens below this_p.
    fam = [0.0001, this_p, this_p + 1e-9, 0.9, 0.95]  # m=5
    res = rigor_report(method, baseline, family_pvalues=fam)
    # With m=5, this method's rank threshold may fall below this_p -> not rejected.
    if not res["legs"]["holm_rejected"]:
        assert res["external_ready"] is False
    else:
        # If still rejected, at least confirm the other legs held (sanity).
        assert res["external_ready"] is True


# --------------------------------------------------------------------------- #
# paired_sign_test (the concept/item-as-replicate test)
# --------------------------------------------------------------------------- #
def test_paired_sign_test_all_positive():
    # 7 items all improve -> n_pos=7, two-sided p = 2 * 0.5**7 = 0.015625.
    a = [0.8, 0.9, 0.7, 0.85, 0.6, 0.95, 0.75]
    b = [0.5, 0.4, 0.3, 0.45, 0.2, 0.55, 0.35]
    res = paired_sign_test(a, b)
    assert res["n"] == 7
    assert res["n_pos"] == 7 and res["n_neg"] == 0
    assert abs(res["p_value"] - 2 * 0.5**7) < 1e-12
    assert res["p_value"] < 0.05


def test_paired_sign_test_drops_ties_and_is_two_sided():
    # deltas: +,+,0,-  -> one tie dropped, n=3, k=min(2,1)=1.
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 1.0, 3.0, 5.0]
    res = paired_sign_test(a, b)
    assert res["n"] == 3  # the zero delta is dropped
    assert res["n_pos"] == 2 and res["n_neg"] == 1
    # two-sided p = 2 * (C(3,0)+C(3,1)) * 0.5**3 = 2 * (1+3)/8 = 1.0
    assert abs(res["p_value"] - 1.0) < 1e-12


def test_paired_sign_test_all_ties_degenerate():
    res = paired_sign_test([1.0, 2.0], [1.0, 2.0])
    assert res["n"] == 0
    assert res["p_value"] == 1.0


# --------------------------------------------------------------------------- #
# directional_consistency
# --------------------------------------------------------------------------- #
def test_directional_consistency_known_arrays():
    # 3 of 4 positive -> 0.75 for hypothesised +1.
    assert abs(directional_consistency([0.1, 0.2, -0.3, 0.4]) - 0.75) < 1e-12
    # all positive -> 1.0
    assert directional_consistency([0.1, 0.2, 0.3]) == 1.0
    # hypothesised NEGATIVE flips the count.
    assert abs(directional_consistency([-0.1, -0.2, 0.3], hypothesized_sign=-1) - (2 / 3)) < 1e-12
    # a zero delta matches neither sign.
    assert abs(directional_consistency([0.0, 0.5]) - 0.5) < 1e-12
    assert directional_consistency([]) == 0.0


# --------------------------------------------------------------------------- #
# verdict — sign-aware labelling
# --------------------------------------------------------------------------- #
def test_verdict_external_ready_positive_significant():
    deltas = [0.30, 0.32, 0.31, 0.33, 0.29, 0.34, 0.30]  # n=7, all positive
    res = verdict(deltas, family_pvalues=[0.001, 0.5, 0.6], mme=0.05)
    assert res["label"] == "EXTERNAL-READY"
    assert res["significant"] and res["ci_excludes_zero"]
    assert res["holm_applied"] and res["holm_rejected"]


def test_verdict_significant_negative_is_NEGATIVE_not_directional():
    # The reviewer's catch: a significant result in the WRONG direction.
    deltas = [-0.30, -0.32, -0.31, -0.33, -0.29, -0.34, -0.30]
    res = verdict(deltas, family_pvalues=[0.001, 0.5, 0.6], mme=0.05)
    assert res["significant"] is True
    assert res["label"] == "NEGATIVE"
    assert res["label"] != "DIRECTIONAL"


def test_verdict_no_family_cannot_be_external_ready():
    deltas = [0.30, 0.32, 0.31, 0.33, 0.29, 0.34, 0.30]
    res = verdict(deltas, mme=0.05)  # no family -> Holm not applied
    assert res["holm_applied"] is False
    assert res["label"] != "EXTERNAL-READY"
    assert res["label"] == "DIRECTIONAL"


def test_verdict_small_n_cannot_be_external_ready():
    # n<7 is SCREENING -> even a clean positive family cannot be EXTERNAL-READY.
    deltas = [0.30, 0.32, 0.31, 0.33, 0.29]  # n=5
    res = verdict(deltas, family_pvalues=[0.001, 0.5, 0.6], mme=0.05)
    assert res["enough_n"] is False
    assert res["label"] != "EXTERNAL-READY"
    assert res["label"] == "DIRECTIONAL"


def test_verdict_sub_mme_effect_not_external_ready():
    # Significant + holm + n>=7 + right sign, but the effect is below the minimum
    # meaningful effect -> NUMEROLOGY, downgraded to DIRECTIONAL.
    deltas = [0.030, 0.031, 0.029, 0.032, 0.028, 0.033, 0.030]
    res = verdict(deltas, family_pvalues=[0.001, 0.5, 0.6], mme=0.10)
    assert res["big_enough"] is False
    assert res["label"] != "EXTERNAL-READY"


def test_verdict_zero_mean_is_null():
    rng = np.random.default_rng(11)
    deltas = rng.normal(0.0, 0.1, size=8)
    res = verdict(list(deltas), family_pvalues=[0.4, 0.5, 0.6])
    assert res["label"] == "NULL"


# --------------------------------------------------------------------------- #
# min_meaningful_effect + power_note
# --------------------------------------------------------------------------- #
def test_min_meaningful_effect_registry():
    assert min_meaningful_effect("composite") == 0.02
    assert min_meaningful_effect("behavior_efficacy") == 0.05
    # unknown metric -> default
    assert min_meaningful_effect("nope") == 0.0
    assert min_meaningful_effect("nope", default=0.07) == 0.07


def test_power_note_small_n_cannot_reach_p05():
    note = power_note(4, effect=0.3, sd=0.1)
    assert note["is_screening"] is True
    assert note["can_reach_p05"] is False  # min p = 2*0.5**4 = 0.125 > 0.05
    assert abs(note["min_achievable_p"] - 0.125) < 1e-12
    assert "underpowered" in note["note"]


def test_power_note_n7_is_evaluation_eligible():
    note = power_note(7, effect=0.3, sd=0.1)
    assert note["is_screening"] is False
    assert note["can_reach_p05"] is True
    assert abs(note["min_achievable_p"] - 2 * 0.5**7) < 1e-12
    assert 0.0 <= note["approx_power"] <= 1.0
