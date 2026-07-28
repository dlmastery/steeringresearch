"""length_ap.py -- the missing cell: what does a LENGTH-ONLY ranker score on AP?

README section 9.2 prices every localizer against the directionless length bar on
ROC-AUC, but the lesson HEADLINES Average Precision at the 5% base rate, and AP and
AUC are not commensurable -- AP's chance floor is the base rate (0.05), AUC's is 0.5.
So "sparse AP 0.568 vs a 0.675 length AUC" was a bare-number comparison, flagged in
the README as the one open cell.

This closes it. Rank the SAME sampled repositories by raw trace character length
alone (both sign conventions, keeping the better one -- the directionless fold from
CONFOUND_DISCIPLINE.md rule 4), and report AP on exactly the metric the headline uses.

CPU only. No model, no embeddings -- length is a `len()` call.

    python steering_tutorials/meerkat/length_ap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from steering_tutorials.meerkat import config as C  # noqa: E402
from steering_tutorials.meerkat import data  # noqa: E402

OUT = HERE / "artifacts" / "length_ap.json"


def _boot_ci(y, s, n=2000, seed=0):
    """Percentile bootstrap 95% CI on AP -- same protocol as run_meerkat._ap_ci."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:      # a resample with one class has no AP
            continue
        vals.append(average_precision_score(y[idx], s[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    pool = data.load_trace_pool()
    print(f"[pool] {len(pool['traces'])} traces "
          f"({sum(pool['labels'])} attack / {len(pool['labels']) - sum(pool['labels'])} benign)")

    out = {"note": "length-only ranker, the missing AP cell from README 9.2",
           "n_repos": int(C.N_REPOS), "repo_size": int(C.REPO_SIZE), "regimes": {}}

    for rate in (C.BASE_RATE, 0.5):
        ys, ss = [], []
        for r in range(C.N_REPOS):
            repo = data.sample_repository(pool, C.REPO_SIZE, rate, seed=C.SEED + r)
            ys += list(repo["labels"])
            # NEGATIVE length: benign traces are the LONG ones here (121.7 vs 104.5
            # chars), so "shorter => attack" is the tell. We score both signs below
            # and keep the stronger, which is the directionless fold.
            ss += [-len(t) for t in repo["traces"]]
        y = np.asarray(ys)
        s = np.asarray(ss, dtype=float)

        ap_short, ap_long = average_precision_score(y, s), average_precision_score(y, -s)
        flip = ap_long > ap_short                       # which sign is the real tell
        best = -s if flip else s
        block = {
            "base_rate": float(rate),
            "n_scored": int(len(y)),
            "positives": int(y.sum()),
            "ap_shorter_is_attack": float(ap_short),
            "ap_longer_is_attack": float(ap_long),
            "ap_directionless": float(max(ap_short, ap_long)),
            "ap_ci": _boot_ci(y, best),
            "roc_auc_directionless": float(max(roc_auc_score(y, s), roc_auc_score(y, -s))),
            "ap_chance_floor": float(y.mean()),         # AP's floor IS the base rate
        }
        out["regimes"][f"{rate}"] = block
        print(f"[rate {rate:.2f}] length-only AP = {block['ap_directionless']:.4f} "
              f"CI={block['ap_ci']} | chance floor {block['ap_chance_floor']:.4f} | "
              f"AUC {block['roc_auc_directionless']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
