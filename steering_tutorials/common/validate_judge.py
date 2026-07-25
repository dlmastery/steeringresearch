"""common/validate_judge.py — measure whether the course's judge can actually judge.

Nearly every generation lesson in this course reports a judge-scored number
(refusal rate, compliance rate, gibberish rate). None of those numbers means
anything until you know the judge can TELL THE CLASSES APART. This script is how
you find out: score a set of responses you have already labelled by hand, and
report ROC-AUC of the judge's score against your labels.

It runs BOTH readouts on the SAME items so the comparison is clean:

  hard        ``Judge.verdict``        -> one word; scored 1.0 if REFUSAL else 0.0
  continuous  ``Judge.refusal_score``  -> p(REFUSAL) under the judge's own
                                          next-token distribution, one forward pass

Only the readout differs — same model, same rubric, same items, same labels — so
any AUC difference is attributable to the readout and nothing else.

WHAT THE SIBLING RESEARCH HARNESS MEASURED (see ../JUDGE_VALIDITY.md)
---------------------------------------------------------------------
Qwen3-4B-Instruct-2507, 300 labelled AxBench concept items:
  integer/argmax readout  ROC-AUC 0.665   (1208 s)
  continuous readout      ROC-AUC 0.751   ( 94 s, 12.8x faster)
The continuous readout is better and faster. It is NOT a rescue: 0.751 is still
below the 0.85 usability bar that program pre-registered. Expect the same shape
here — a better readout cannot make a small judge know something it does not know.

FAILURE HANDLING (this is load-bearing)
---------------------------------------
A judge call that cannot produce a usable score is SKIPPED and COUNTED. It is
never defaulted to a plausible-looking verdict. The sibling harness carried a bug
where an unparseable judge reply fell through to "REFUSED", which handed every
broken call a free safety pass and silently inflated the measured refusal rate.
A validation script that inherits that bug validates nothing.

USAGE
-----
Build a labels file first: take a lesson's generations, hand-label each one, and
write JSONL with one object per line::

    {"prompt": "...", "response": "...", "label": 1}   # 1 = this reply REFUSED
    {"prompt": "...", "response": "...", "label": 0}   # 0 = it complied

Then (GPU; loads the judge model)::

    set STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct
    python -m steering_tutorials.common.validate_judge --labels my_labels.jsonl

Aim for at least ~100 labelled items with both classes well represented; fewer
than that and the AUC's own confidence interval is wider than the effects you are
trying to measure. Plumbing check with no model at all::

    python -m steering_tutorials.common.validate_judge --self-test

Output goes to stdout and to ``--out`` (default
``steering_tutorials/common/artifacts/judge_card.json``) — that file IS your
lesson's judge card. Cite it next to every judge-scored number you report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "artifacts" / "judge_card.json"

# The bar the sibling program pre-registered. A judge below this cannot resolve
# small effects; results scored with it inherit an instrument caveat.
AUC_GATE = 0.85


def load_labels(path: str | Path) -> list[dict]:
    """Read a JSONL of ``{"prompt", "response", "label"}``; validate hard.

    Every row must carry all three fields and a label in {0, 1}. We refuse to
    guess a missing label — a validation set with invented labels measures the
    invention, not the judge.
    """
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} is not valid JSON: {exc}") from exc
            missing = [k for k in ("prompt", "response", "label") if k not in obj]
            if missing:
                raise ValueError(f"{path}:{i} missing field(s) {missing}")
            if int(obj["label"]) not in (0, 1):
                raise ValueError(f"{path}:{i} label must be 0 or 1, got {obj['label']!r}")
            rows.append({"prompt": str(obj["prompt"]),
                         "response": str(obj["response"]),
                         "label": int(obj["label"])})
    if not rows:
        raise ValueError(f"{path} contained no rows")
    return rows


def roc_auc(y: "np.ndarray", score: "np.ndarray") -> float:
    """ROC-AUC with correct tie handling (the rank/Mann-Whitney form).

    Ties matter enormously here: the HARD readout produces only two distinct
    score values, so every within-class pair is tied. A naive implementation that
    mishandles ties would flatter the hard readout and hide the very effect this
    script exists to expose.
    """
    from scipy.stats import rankdata

    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC needs both classes present in the labels")
    ranks = rankdata(score)  # average ranks => ties handled
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score_items(
    rows: list[dict],
    hard_fn: Callable[[str, str], str],
    cont_fn: Callable[[str, str], float],
    log_every: int = 25,
) -> dict[str, Any]:
    """Score every row with both readouts; SKIP + COUNT failures, never default.

    Returns aligned arrays for whichever items BOTH readouts scored, so the two
    AUCs are computed on an identical item set and are directly comparable.
    """
    y: list[int] = []
    hard: list[float] = []
    cont: list[float] = []
    failures: list[dict] = []

    t_hard = t_cont = 0.0
    for i, row in enumerate(rows, 1):
        try:
            t0 = time.time()
            verdict = hard_fn(row["prompt"], row["response"])
            t_hard += time.time() - t0

            t0 = time.time()
            p_refusal = cont_fn(row["prompt"], row["response"])
            t_cont += time.time() - t0
        except Exception as exc:
            # Skip and count. Do NOT substitute a verdict — a defaulted score is
            # indistinguishable from a real one downstream, which is the bug.
            failures.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [skip {i}] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        y.append(row["label"])
        hard.append(1.0 if str(verdict).upper() == "REFUSAL" else 0.0)
        cont.append(float(p_refusal))
        if log_every and i % log_every == 0:
            print(f"  {i}/{len(rows)} scored", file=sys.stderr)

    return {
        "y": np.asarray(y, dtype=int),
        "hard": np.asarray(hard, dtype=float),
        "cont": np.asarray(cont, dtype=float),
        "failures": failures,
        "hard_seconds": round(t_hard, 1),
        "cont_seconds": round(t_cont, 1),
    }


def build_card(scored: dict[str, Any], n_input: int, judge_id: str,
               labels_path: str) -> dict:
    """Assemble the judge card from scored arrays. Pure; no I/O, no model."""
    y, hard, cont = scored["y"], scored["hard"], scored["cont"]
    auc_hard = roc_auc(y, hard)
    auc_cont = roc_auc(y, cont)
    return {
        "judge_model": judge_id,
        "labels_file": labels_path,
        "n_input": int(n_input),
        "n_scored": int(len(y)),
        "n_skipped": len(scored["failures"]),
        "n_positive": int((y == 1).sum()),
        "roc_auc_hard": auc_hard,
        "roc_auc_continuous": auc_cont,
        "delta_continuous_minus_hard": auc_cont - auc_hard,
        "distinct_values_hard": int(len(np.unique(np.round(hard, 6)))),
        "distinct_values_continuous": int(len(np.unique(np.round(cont, 6)))),
        "seconds_hard": scored["hard_seconds"],
        "seconds_continuous": scored["cont_seconds"],
        "gate": AUC_GATE,
        "passes_gate_hard": bool(auc_hard >= AUC_GATE),
        "passes_gate_continuous": bool(auc_cont >= AUC_GATE),
        "failures": scored["failures"][:20],
        "note": ("Failures are SKIPPED and COUNTED, never defaulted to a verdict. "
                 "A judge below the gate cannot resolve small effects; every number "
                 "scored with it carries an instrument caveat "
                 "(see steering_tutorials/JUDGE_VALIDITY.md)."),
    }


def print_card(card: dict) -> None:
    print("\n=== JUDGE CARD ===")
    print(f"  judge model     : {card['judge_model']}")
    print(f"  items           : {card['n_scored']} scored / {card['n_input']} input "
          f"({card['n_skipped']} skipped, {card['n_positive']} positive)")
    print(f"  ROC-AUC hard    : {card['roc_auc_hard']:.4f}   "
          f"({card['distinct_values_hard']} distinct values, "
          f"{card['seconds_hard']}s)")
    print(f"  ROC-AUC continuous: {card['roc_auc_continuous']:.4f}   "
          f"({card['distinct_values_continuous']} distinct values, "
          f"{card['seconds_continuous']}s)")
    print(f"  delta (cont - hard): {card['delta_continuous_minus_hard']:+.4f}")
    print(f"  GATE >= {card['gate']} -> hard "
          f"{'PASS' if card['passes_gate_hard'] else 'FAIL'} / continuous "
          f"{'PASS' if card['passes_gate_continuous'] else 'FAIL'}")
    if not card["passes_gate_continuous"]:
        print("  READ THIS: the judge did NOT clear the gate. Every judge-scored")
        print("  number in your lesson carries an instrument caveat. A better")
        print("  readout improves the judge; it does not rescue it.")


def _self_test() -> int:
    """CPU-only plumbing check — no model, no download, no GPU.

    Uses stub readouts so the arithmetic (AUC, tie handling, skip-not-default) is
    exercised end to end. This validates the SCRIPT, never the judge.
    """
    rows = [{"prompt": f"p{i}", "response": f"r{i}", "label": int(i % 2 == 0)}
            for i in range(20)]
    rows.append({"prompt": "boom", "response": "boom", "label": 1})  # will fail

    def hard_fn(p: str, _r: str) -> str:
        if p == "boom":
            raise RuntimeError("simulated judge failure")
        # Hard readout: right on 8/10 positives, wrong on 2 -> coarse, only 2 values.
        return "REFUSAL" if int(p[1:]) % 2 == 0 and int(p[1:]) < 16 else "COMPLIANCE"

    def cont_fn(p: str, _r: str) -> float:
        if p == "boom":
            raise RuntimeError("simulated judge failure")
        i = int(p[1:])
        return 0.9 - 0.01 * i if i % 2 == 0 else 0.4 - 0.01 * i

    scored = score_items(rows, hard_fn, cont_fn, log_every=0)
    card = build_card(scored, n_input=len(rows), judge_id="stub", labels_path="<self-test>")

    assert card["n_scored"] == 20, "the failing item must be skipped, not scored"
    assert card["n_skipped"] == 1, "the failing item must be COUNTED"
    assert card["distinct_values_hard"] == 2, "hard readout is 2-valued by construction"
    assert card["distinct_values_continuous"] == 20, "continuous readout must be graded"
    assert card["roc_auc_continuous"] > card["roc_auc_hard"], (
        "on this stub the graded score must out-rank the quantized one")
    # Tie handling sanity: a constant score must give exactly 0.5, not 1.0.
    y = np.array([1, 1, 0, 0])
    assert abs(roc_auc(y, np.array([1.0, 1.0, 1.0, 1.0])) - 0.5) < 1e-9
    assert abs(roc_auc(y, np.array([1.0, 1.0, 0.0, 0.0])) - 1.0) < 1e-9

    print_card(card)
    print("\nvalidate_judge.py self-test OK "
          "(plumbing only — this says NOTHING about any real judge).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", help="JSONL of {prompt, response, label} (label 1 = REFUSAL)")
    ap.add_argument("--model", default=None,
                    help="target model id to load as the judge when STEER_JUDGE_MODEL "
                         "is unset (defaults to the lesson-2 config model)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-only plumbing check; loads no model")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.labels:
        ap.error("--labels is required (or use --self-test)")

    rows = load_labels(args.labels)
    if args.limit:
        rows = rows[: args.limit]
    n_pos = sum(r["label"] for r in rows)
    print(f"[validate_judge] {len(rows)} labelled items ({n_pos} positive / "
          f"{len(rows) - n_pos} negative)")
    if len(rows) < 100:
        print("[validate_judge] WARNING: fewer than 100 items. The AUC's own "
              "confidence interval will be wide — treat the result as provisional.")

    # Import late so --self-test never touches torch/transformers.
    from steering_tutorials.hello_world_steering.judge import Judge

    judge_id = os.environ.get("STEER_JUDGE_MODEL", "").strip()
    if judge_id:
        # Judge.__init__ loads the off-family judge itself and ignores these.
        model, tok = None, None
    else:
        from steering_tutorials.hello_world_steering import config as cfg
        from steering_tutorials.hello_world_steering.model_utils import load_model
        model_id = args.model or cfg.MODEL_ID
        print(f"[validate_judge] STEER_JUDGE_MODEL unset -> SELF-JUDGE with "
              f"{model_id}. This is the weak configuration; prefer an off-family "
              f"judge for anything you intend to report.")
        model, tok = load_model(model_id)

    judge = Judge(model, tok)
    scored = score_items(rows, judge.verdict, judge.refusal_score)
    card = build_card(scored, n_input=len(rows), judge_id=judge.judge_id,
                      labels_path=str(args.labels))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print_card(card)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
