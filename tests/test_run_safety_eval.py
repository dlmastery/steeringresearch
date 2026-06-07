"""Offline end-to-end test for scripts/run_safety_eval.py.

Drives the headline safety-eval driver's ``run()`` with an injected
``FakeResidualLM`` + ``StubSafetyJudge`` so the FULL pipeline executes with NO
GPU, NO network, NO git, and WITHOUT appending to the real experiment ledger
(``no_log=True`` + ``dry_run=True``). Asserts the driver produces a Pareto table
and per-metric verdicts, and that the composite it reports/logs is the REAL
fingerprinted composite (``eval.composite``), not a raw mean of the axes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Import the driver module by path (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "run_safety_eval", ROOT / "scripts" / "run_safety_eval.py"
)
assert _SPEC and _SPEC.loader
rse = importlib.util.module_from_spec(_SPEC)
sys.modules["run_safety_eval"] = rse
_SPEC.loader.exec_module(rse)

from steering.eval import composite, composite_fingerprint  # noqa: E402
from steering.fakelm import FakeResidualLM  # noqa: E402
from steering.model import _FakeTokenizer  # noqa: E402

_ALLOWED_VERDICTS = {"EXTERNAL-READY", "DIRECTIONAL", "NULL", "NEGATIVE"}


def _fake_model_tok():
    fake = FakeResidualLM(vocab_size=64, dim=32, n_layers=8, hidden=64, seed=0)
    return fake, _FakeTokenizer(fake.vocab_size)


def _cfg():
    return rse.SafetyEvalConfig(
        model="fake",
        benchmarks=("jailbreakbench", "xstest"),
        n=3,
        alpha=6.0,
        seeds=(0,),
        layer_condition=3,
        layer_write=5,
        max_new_tokens=4,
        dry_run=True,
        no_log=True,
    )


def _ledger_size() -> int:
    p = ROOT / "autoresearch_results" / "experiment_log.jsonl"
    return p.stat().st_size if p.exists() else 0


def test_dry_run_produces_pareto_and_verdicts():
    model, tok = _fake_model_tok()
    before = _ledger_size()
    camp = rse.run(_cfg(), model=model, tok=tok, judge=rse.StubSafetyJudge())

    # --- Pareto table -------------------------------------------------------
    pareto = camp["pareto"]
    assert isinstance(pareto, list) and len(pareto) >= 2
    methods = {r["method"] for r in pareto}
    assert "cast_method" in methods and "no_steer" in methods
    for r in pareto:
        assert 0.0 <= r["asr"] <= 1.0
        assert 0.0 <= r["over_refusal"] <= 1.0
        assert isinstance(r["pareto_optimal"], bool)
    # exactly the rendered table, with header + the method row.
    assert "ASR" in camp["pareto_text"] and "cast_method" in camp["pareto_text"]
    # at least one row is Pareto-optimal (the frontier is never empty).
    assert any(r["pareto_optimal"] for r in pareto)

    # --- per-metric verdicts -----------------------------------------------
    verdicts = camp["verdicts"]
    assert "harmful_refusal" in verdicts
    for name, v in verdicts.items():
        assert v["verdict"] in _ALLOWED_VERDICTS
        assert "wilcoxon_p" in v and "external_ready" in v
        assert isinstance(v["external_ready"], bool)

    # --- the composite is the REAL composite, not a raw mean ---------------
    metrics = camp["method_metrics"]
    expected = round(composite(metrics), 4)
    assert camp["method_composite"] == expected
    assert camp["composite_fingerprint"] == composite_fingerprint()
    # The composite must NOT be the arithmetic mean of the axis values.
    raw_mean = sum(metrics.values()) / len(metrics)
    assert abs(camp["method_composite"] - round(raw_mean, 4)) > 1e-9 or metrics["behavior_efficacy"] == 0.0

    # --- did NOT touch the real ledger -------------------------------------
    assert _ledger_size() == before
    assert "logged_experiment_num" not in camp  # no_log path never logs


def test_composite_breakdown_sums_to_composite():
    model, tok = _fake_model_tok()
    camp = rse.run(_cfg(), model=model, tok=tok, judge=rse.StubSafetyJudge())
    bd = camp["composite_breakdown"]
    parts = sum(v for k, v in bd.items() if k != "composite")
    assert abs(parts - bd["composite"]) < 1e-3
    assert abs(bd["composite"] - camp["method_composite"]) < 1e-3


def test_stub_judge_respects_refusal_text():
    j = rse.StubSafetyJudge()
    # An explicit refusal always scores as refused (complied=0) regardless of hash.
    res = j.judge_compliance_batch([("harmful req", "I can't help with that request.")])
    assert res[0]["complied"] == 0 and res[0]["refused"] == 1
    asr = j.attack_success_rate(["q"], ["I cannot help with that."])
    assert asr == 0.0


def test_intents_registry_loads_and_builds():
    intents = rse.load_intents()
    names = {i["name"] for i in intents}
    assert {"weapons", "cyber", "self_harm", "illicit_goods", "harassment"} <= names
    model, tok = _fake_model_tok()
    built = rse.build_intents(
        model, tok, intents, layer_condition=3, layer_write=5, target_fpr=0.05
    )
    assert built.names and len(built.cast_specs) == len(built.names)
    assert built.composed_safety.shape[0] == model.dim


def test_main_cli_dry_run_returns_zero():
    rc = rse.main(["--dry-run", "--no-log", "--n", "2", "--max-new-tokens", "3",
                   "--benchmarks", "jailbreakbench", "--layer-condition", "3",
                   "--layer-write", "5"])
    assert rc == 0
