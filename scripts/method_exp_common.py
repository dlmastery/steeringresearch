"""method_exp_common.py — log a *method* experiment (E15/E45/E20) into the same
append-only ledger the steering sweeps use.

The standard `steering.runner.run_single_experiment` evaluates ONE steering
vector across the five axes. The three trainable methods (E15 learned gate,
E45 HyperSteer hypernetwork, E20 SAE-TS) do not fit that single-vector mold —
their primary metric is gate AUC, held-out cosine, or 3-stack Gram mass. This
helper lets such a driver emit ONE canonical `experiment_log.jsonl` row with:

  * the full standard schema (so the dashboard / provenance generator read it
    without special-casing), with non-applicable axes set to honest neutrals;
  * the method's own primary metric carried in clearly-named extra fields
    (``method``, ``method_metric``, ``method_value``, ``method_extra``);
  * a genuine pre-run ``_manual`` reasoning entry (the runner's no-fabrication
    gate analogue) authored BEFORE the row is appended.

It reuses the runner's `_log_and_checkpoint`, so experiment_num auto-increments,
best_config is updated only if the composite actually beats the global champion
(these method metrics are small/negative, so they never spuriously win), the
reasoning verdict/learning are written, and the dashboard is regenerated.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.eval import composite_fingerprint  # noqa: E402
from steering.runner import RESULTS_DIR, _log_and_checkpoint  # noqa: E402

# Drivers set this False for offline FAKE-model smoke runs so plumbing checks do
# not append rows to the append-only ledger (only real-model screens are logged).
LOGGING_ENABLED = True


def _next_num() -> int:
    log = RESULTS_DIR / "experiment_log.jsonl"
    n = 0
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                n = max(n, json.loads(line).get("experiment_num", n))
            except Exception:
                pass
    return n + 1


def _author_reasoning(num: int, reasoning: dict[str, str]) -> None:
    """Write a genuine pre-run `_manual` reasoning entry keyed to exp #num.

    `reasoning` must carry diagnosis / citations / hypothesis / prediction. The
    runner's `_write_reasoning` then fills only verdict + learning post-run.
    """
    ann_path = RESULTS_DIR / "reasoning_annotations.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    ann[str(num)] = {
        "_manual": True,
        "diagnosis": reasoning["diagnosis"],
        "citations": reasoning["citations"],
        "hypothesis": reasoning["hypothesis"],
        "prediction": reasoning["prediction"],
    }
    ann_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")


def log_method_experiment(
    *,
    config: dict[str, Any],
    description: str,
    reasoning: dict[str, str],
    method: str,
    method_metric: str,
    method_value: float,
    method_extra: dict[str, Any],
    composite: Optional[float] = None,
    behavior_efficacy: float = 0.0,
    perplexity: float = 0.0,
    dppl_norm: float = 0.0,
    mmlu_drop_pp: float = 0.0,
    compliance_rate: float = 0.0,
    harmless_refusal_rate: float = 0.0,
    offshell_displacement: float = 0.0,
    fisher_at_layer: float = 0.0,
    behavior_scorer: str = "method",
    safety_real: bool = False,
    elapsed_sec: float = 0.0,
    started: Optional[float] = None,
) -> dict[str, Any]:
    """Build a complete standard row (+ method extras) and append it via the runner.

    INTEGRITY: the ``composite`` column must only ever hold the fingerprinted
    5-axis ``eval.composite()`` output (CLAUDE.md §6) — never a raw behavior mean.
    Method drivers whose primary result is a single raw metric (a delta, a best
    behavior, a cosine, a retention ratio) therefore pass ``composite=None`` and
    carry that raw value in ``method_value``. When ``composite is None`` the row
    records ``composite_is_real=False`` and stores a JSON null in ``composite``
    (so the dashboard renders it as "no composite" and the runner's champion
    comparison skips it). Pass a real composite ONLY when all five axes were
    actually measured and fed through ``eval.composite()``.
    """
    num = _next_num()
    if started is not None:
        elapsed_sec = round(time.time() - started, 2)

    composite_is_real = composite is not None
    entry: dict[str, Any] = {
        "config": config,
        "description": description,
        "rung": config.get("rung", 2),
        "layer": config.get("layer", 0),
        "composite": round(float(composite), 4) if composite is not None else None,
        "composite_is_real": composite_is_real,
        "behavior_efficacy": round(float(behavior_efficacy), 4),
        "capability_retention": round(1.0 - max(0.0, mmlu_drop_pp) / 100.0, 4),
        "mmlu_drop_pp": round(float(mmlu_drop_pp), 4),
        "perplexity": round(float(perplexity), 4),
        "dppl_norm": round(float(dppl_norm), 4),
        "repetition_rate": 0.0,
        "compliance_rate": round(float(compliance_rate), 4),
        "harmful_refusal_rate": round(1.0 - float(compliance_rate), 4),
        "harmless_refusal_rate": round(float(harmless_refusal_rate), 4),
        "selectivity_gap": round((1.0 - float(compliance_rate)) - float(harmless_refusal_rate), 4),
        "offshell_displacement": round(float(offshell_displacement), 4),
        "angular_displacement": 0.0,
        "fisher_at_layer": round(float(fisher_at_layer), 4),
        "cosine_dm_pca": 0.0,
        "behavior_scorer": behavior_scorer,
        "safety_real": safety_real,
        # --- method-specific payload (the real primary result) ---
        "method": method,
        "method_metric": method_metric,
        "method_value": round(float(method_value), 4),
        "method_extra": method_extra,
        "composite_fingerprint": composite_fingerprint(),
        "n_seeds": int(config.get("n_seeds", 1)),
        "tier": "SCREENING" if int(config.get("n_seeds", 1)) <= 3 else "EVALUATION",
        "elapsed_sec": round(float(elapsed_sec), 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not LOGGING_ENABLED:
        entry["experiment_num"] = -1  # smoke run, not persisted
        return entry
    _author_reasoning(num, reasoning)
    _log_and_checkpoint(entry, description)
    return entry


def write_campaign(tag_prefix: str, payload: dict[str, Any]) -> Path:
    """Mirror the sweep convention: persist the full method-campaign artifact."""
    if not LOGGING_ENABLED:
        return ROOT / "ideas" / "_campaigns" / f"{tag_prefix}.json"  # not written in smoke
    out = ROOT / "ideas" / "_campaigns"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag_prefix}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
