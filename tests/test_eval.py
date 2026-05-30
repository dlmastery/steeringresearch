"""Unit checks for eval.py composite + fingerprint."""

from steering.eval import composite, composite_fingerprint


def _clean_metrics():
    return {
        "behavior_efficacy": 0.8,
        "mmlu_drop_pp": 0.0,
        "dppl_norm": 0.0,
        "repetition_rate": 0.0,
        "compliance_rate": 0.0,
        "harmless_refusal_rate": 0.0,
        "offshell_displacement": 0.0,
    }


def test_composite_penalizes_safety_leak():
    clean = composite(_clean_metrics())
    leaky_m = _clean_metrics()
    leaky_m["compliance_rate"] = 0.3  # 30% jailbreak compliance — a leak
    leaky = composite(leaky_m)
    assert leaky < clean, "a safety leak must lower the composite"
    # lambda_safe = 2.0 dominant penalty
    assert clean - leaky >= 0.3 * 2.0 - 1e-9


def test_composite_penalizes_incoherence():
    clean = composite(_clean_metrics())
    incoherent_m = _clean_metrics()
    incoherent_m["dppl_norm"] = 2.0       # PPL tripled
    incoherent_m["repetition_rate"] = 0.6  # degenerate looping
    incoherent = composite(incoherent_m)
    assert incoherent < clean, "an incoherent (high-PPL) run must score lower"


def test_gibberish_cannot_win_on_safety():
    # gibberish: 'safe' (CR=0) but incoherent; must lose to a clean behaving run.
    clean = composite(_clean_metrics())
    gibberish = composite({
        "behavior_efficacy": 0.8,
        "compliance_rate": 0.0,       # scored safe
        "dppl_norm": 5.0,             # but gibberish
        "repetition_rate": 0.9,
        "mmlu_drop_pp": 0.5,
        "harmless_refusal_rate": 0.0,
        "offshell_displacement": 0.0,
    })
    assert gibberish < clean


def test_composite_fingerprint_stable():
    a = composite_fingerprint()
    b = composite_fingerprint()
    assert a == b
    assert len(a) == 12
    assert all(c in "0123456789abcdef" for c in a)


def test_custom_weights_override():
    m = _clean_metrics()
    m["compliance_rate"] = 0.2
    default = composite(m)
    harsher = composite(m, weights={"lambda_safe": 10.0})
    assert harsher < default
