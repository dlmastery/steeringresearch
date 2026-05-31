"""Integration check: a Rung-0 --model fake run writes valid artifacts and the
JSONL stays append-only with auto-incrementing experiment_num."""

import json
from pathlib import Path

import steering.runner as runner


def _run(tmp_path: Path, tag: str):
    """Run one fake experiment with RESULTS_DIR pointed at tmp_path."""
    orig = runner.RESULTS_DIR
    runner.RESULTS_DIR = tmp_path
    try:
        return runner.run_single_experiment(
            model_name="fake",
            rung=0,
            layer=2,
            alpha=4.0,
            operation="add",
            source="diffmean",
            behavior="ocean",
            seed=0,
            description=f"Rung-0 plumbing run {tag}",
            tag=tag,
        )
    finally:
        runner.RESULTS_DIR = orig


def test_rung0_run_writes_artifacts(tmp_path):
    _run(tmp_path, "smoke-1")

    log_path = tmp_path / "experiment_log.jsonl"
    best_path = tmp_path / "best_config.json"
    ann_path = tmp_path / "reasoning_annotations.json"

    assert log_path.exists(), "experiment_log.jsonl must be written"
    assert best_path.exists(), "best_config.json must be written (first run is champion)"
    assert ann_path.exists(), "reasoning_annotations.json must be written"

    # Valid JSONL row with the required fields.
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["experiment_num"] == 1
    for key in ("composite", "behavior_efficacy", "capability_retention",
                "perplexity", "compliance_rate", "selectivity_gap",
                "offshell_displacement", "composite_fingerprint", "tier", "n_seeds"):
        assert key in row, f"row missing required field {key}"
    assert row["tier"] == "SCREENING" and row["n_seeds"] == 1

    # best_config matches the single run.
    best = json.loads(best_path.read_text(encoding="utf-8"))
    assert best["experiment_num"] == 1
    assert best["status"] == "KEEP"

    # Reasoning skeleton: pre-run fields are TODO-REWRITE (runner refuses to fabricate),
    # post-run verdict + learning are filled.
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    a1 = ann["1"]
    assert a1["_needs_rewrite"] is True
    assert a1["diagnosis"].startswith("TODO-REWRITE")
    assert a1["citations"].startswith("TODO-REWRITE")
    assert a1["hypothesis"].startswith("TODO-REWRITE")
    assert a1["prediction"].startswith("TODO-REWRITE")
    assert a1["verdict"] and "composite" in a1["verdict"]
    assert a1["learning"] and "Behavior" in a1["learning"]


def test_second_run_autoincrements_and_appends(tmp_path):
    _run(tmp_path, "smoke-1")
    _run(tmp_path, "smoke-2")

    log_path = tmp_path / "experiment_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, "JSONL must be append-only (2 rows after 2 runs)"

    nums = [json.loads(line)["experiment_num"] for line in lines]
    assert nums == [1, 2], "experiment_num must auto-increment"

    # running.json must be cleared after each run (transient signal).
    assert not (tmp_path / "running.json").exists()

    # reasoning annotations has both experiments.
    ann = json.loads((tmp_path / "reasoning_annotations.json").read_text(encoding="utf-8"))
    assert "1" in ann and "2" in ann
