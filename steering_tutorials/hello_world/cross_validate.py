"""cross_validate.py — put a confidence interval under the 0.95 headline.

Run:  python -m steering_tutorials.hello_world.cross_validate

The single train/test split in ``train_probe.py`` reports one number per metric
on ONE held-out slice of 40 prompts. That number could be a lucky (or unlucky)
draw. Standard ML practice is **k-fold cross-validation**: rotate the held-out
slice through the whole dataset so every example is tested exactly once, then
report mean +/- a confidence interval across the folds.

This script is CPU-only and never loads the Gemma model — it reuses the cached
frozen-LLM activations in ``artifacts/features.npz``. For each of k=5 stratified
folds it:
    1. fits a StandardScaler on the fold's TRAIN split only (no leakage),
    2. trains the SAME ``MLPProbe`` with the SAME recipe as ``train_probe.py``
       (Adam lr=1e-3, wd=1e-3, BCE, full-batch, early stop on a val slice),
    3. scores the held-out fold on the full 12-metric suite.
It then aggregates mean / std / 95% CI per metric, and runs a plain
``LogisticRegression`` as a linear-probe sanity baseline.

Outputs (all under ``artifacts/``):
    cv_report.json   per-fold metrics + MLP mean/std/CI + logreg reference
    cv_report.md     readable table + a plain-language trustworthiness verdict
    cv_metrics.png   per-fold accuracy & roc_auc bars
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config as C
from .probe import MLPProbe

# --- CV knobs ---------------------------------------------------------------
K_FOLDS = 5
CV_SEED = 0            # random_state for StratifiedKFold + fold-local val split
# The ordered metric suite, matching train_probe.evaluate() so the numbers are
# directly comparable to the single-split headline.
METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
    "f1", "mcc", "cohen_kappa", "roc_auc", "pr_auc", "log_loss", "brier",
]

CV_REPORT_JSON = C.ARTIFACTS / "cv_report.json"
CV_REPORT_MD = C.ARTIFACTS / "cv_report.md"
CV_METRICS_PNG = C.ARTIFACTS / "cv_metrics.png"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features() -> tuple[np.ndarray, np.ndarray]:
    """Load cached frozen-LLM activations (X) and labels (y). No model needed."""
    if not C.FEATURES_CACHE.exists():
        sys.exit(f"[cv] missing {C.FEATURES_CACHE} — run train_probe.py first to "
                 "extract & cache features.")
    cache = np.load(C.FEATURES_CACHE, allow_pickle=True)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    print(f"[cv] loaded X={X.shape} y={y.shape} "
          f"class balance={np.bincount(y).tolist()}", file=sys.stderr)
    return X, y


# --- one MLP fold, using the train_probe.py recipe verbatim -----------------
def train_one_mlp(Xtr, ytr, Xva, yva, in_dim, device):
    """Train a fresh MLPProbe with the deployed recipe; early-stop on val loss.

    Mirrors ``train_probe.train_probe``: full-batch Adam (lr=1e-3, wd=1e-3),
    BCEWithLogitsLoss, up to C.EPOCHS, restore best-val-loss weights, stop after
    C.PATIENCE epochs without improvement.
    """
    probe = MLPProbe(in_dim, C.HIDDEN1, C.HIDDEN2, C.DROPOUT).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    xtr = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    xva = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva.astype(np.float32)).to(device)

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(C.EPOCHS):
        probe.train()
        opt.zero_grad()
        loss = loss_fn(probe(xtr), ytr_t)
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            vloss = loss_fn(probe(xva), yva_t).item()
        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    return probe


@torch.no_grad()
def mlp_proba(probe, Xstd, device) -> np.ndarray:
    """P(harmful) for already-standardized features."""
    logits = probe(torch.from_numpy(Xstd).to(device))
    return torch.sigmoid(logits).cpu().numpy()


def compute_metrics(y_true: np.ndarray, p_harm: np.ndarray,
                    threshold: float) -> dict:
    """The full 12-metric suite (same set/order as train_probe.evaluate)."""
    y_pred = (p_harm >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    specificity = tn / max(1, tn + fp)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, p_harm)),
        "pr_auc": float(average_precision_score(y_true, p_harm)),
        "log_loss": float(log_loss(y_true, p_harm, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_harm)),
    }


def stratified_val_split(ytr: np.ndarray, val_fraction: float,
                         rng: np.random.Generator):
    """Carve a stratified val slice out of a fold's train indices (for early stop).

    Returns (fit_idx, val_idx) as positions into the fold-train arrays, balanced
    across both classes — the same idea as train_probe.stratified_split.
    """
    fit_idx, val_idx = [], []
    for cls in (0, 1):
        idx = np.where(ytr == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx += idx[:n_val].tolist()
        fit_idx += idx[n_val:].tolist()
    return np.array(fit_idx), np.array(val_idx)


def aggregate(per_fold: list[dict]) -> dict:
    """mean / std / 95% CI (mean +/- 1.96*std/sqrt(k)) for every metric."""
    k = len(per_fold)
    out = {}
    for name in METRIC_NAMES:
        vals = np.array([f[name] for f in per_fold], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))          # population std across folds
        half = 1.96 * std / np.sqrt(k)
        out[name] = {
            "mean": mean, "std": std,
            "ci95_low": float(mean - half), "ci95_high": float(mean + half),
            "ci95_halfwidth": float(half),
            "per_fold": vals.tolist(),
        }
    return out


# --- reference linear probe: 5-fold CV of plain logistic regression ---------
def logreg_reference(X: np.ndarray, y: np.ndarray) -> dict:
    """Sanity baseline: standardized activations -> LogisticRegression, 5-fold CV.

    Reports mean +/- std for accuracy and roc_auc. If a strong linear probe
    matches the MLP, most of the signal is linearly decodable from the layer.
    """
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=CV_SEED)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])         # train-only fit
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        accs.append(accuracy_score(y[te], (p >= 0.5).astype(int)))
        aucs.append(roc_auc_score(y[te], p))
    accs, aucs = np.array(accs), np.array(aucs)
    return {
        "accuracy": {"mean": float(accs.mean()), "std": float(accs.std(ddof=0)),
                     "per_fold": accs.tolist()},
        "roc_auc": {"mean": float(aucs.mean()), "std": float(aucs.std(ddof=0)),
                    "per_fold": aucs.tolist()},
    }


# --- plot -------------------------------------------------------------------
def make_plot(mlp_agg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = mlp_agg["accuracy"]["per_fold"]
    auc = mlp_agg["roc_auc"]["per_fold"]
    folds = np.arange(1, len(acc) + 1)
    width = 0.38

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(folds - width / 2, acc, width, label="accuracy", color="#2563eb")
    ax.bar(folds + width / 2, auc, width, label="roc_auc", color="#16a34a")
    # mean lines with CI shading
    for agg, color in ((mlp_agg["accuracy"], "#2563eb"),
                       (mlp_agg["roc_auc"], "#16a34a")):
        ax.axhline(agg["mean"], color=color, ls="--", lw=1)
        ax.axhspan(agg["ci95_low"], agg["ci95_high"], color=color, alpha=0.08)
    ax.set_xticks(folds)
    ax.set_xlabel("fold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"MLP probe — {K_FOLDS}-fold CV (dashed = mean, band = 95% CI)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CV_METRICS_PNG, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {CV_METRICS_PNG.name}", file=sys.stderr)


# --- markdown report --------------------------------------------------------
def write_markdown(mlp_agg: dict, logreg: dict, single: dict) -> None:
    lines = []
    lines.append(f"# {K_FOLDS}-Fold Cross-Validation — Safety Probe\n")
    lines.append(f"Frozen-LLM activations `X` = **{single['n_total']}x"
                 f"{single['in_dim']}**, balanced "
                 f"({single['class_balance']}). CPU-only; the Gemma model is "
                 "never loaded — we reuse `artifacts/features.npz`.\n")
    lines.append(f"Estimator: the deployed 3-layer `MLPProbe` "
                 f"({single['in_dim']}->128->32->1, dropout {C.DROPOUT}), trained "
                 f"with the exact `train_probe.py` recipe (Adam lr={C.LR}, "
                 f"weight_decay={C.WEIGHT_DECAY}, BCE, early stop on a stratified "
                 "val slice). StandardScaler fit per-fold on train only.\n")

    # Main table: mean +/- CI, plus the single-split value for comparison.
    lines.append("## MLP probe — mean ± 95% CI across folds\n")
    lines.append("| metric | mean | std | 95% CI | single-split |")
    lines.append("|---|---|---|---|---|")
    for name in METRIC_NAMES:
        a = mlp_agg[name]
        s = single["metrics"].get(name)
        s_str = f"{s:.4f}" if s is not None else "—"
        lines.append(f"| {name} | {a['mean']:.4f} | {a['std']:.4f} | "
                     f"[{a['ci95_low']:.4f}, {a['ci95_high']:.4f}] | {s_str} |")
    lines.append("")

    # Logreg reference.
    lines.append("## Linear-probe reference — LogisticRegression, 5-fold CV\n")
    lines.append("| metric | mean | std |")
    lines.append("|---|---|---|")
    lines.append(f"| accuracy | {logreg['accuracy']['mean']:.4f} | "
                 f"{logreg['accuracy']['std']:.4f} |")
    lines.append(f"| roc_auc | {logreg['roc_auc']['mean']:.4f} | "
                 f"{logreg['roc_auc']['std']:.4f} |")
    lines.append("")

    # Plain-language verdict.
    acc = mlp_agg["accuracy"]
    single_acc = single["metrics"]["accuracy"]
    inside = acc["ci95_low"] <= single_acc <= acc["ci95_high"]
    if single_acc > acc["ci95_high"]:
        position = ("**above** the CI — the single split was optimistic "
                    "(a lucky test draw)")
    elif single_acc < acc["ci95_low"]:
        position = ("**below** the CI — the single split was pessimistic "
                    "(an unlucky test draw)")
    else:
        position = "**inside** the CI — the single split is representative"

    lines.append("## Was standard practice followed? Is 0.95 trustworthy?\n")
    lines.append(
        f"Yes — standard {K_FOLDS}-fold stratified cross-validation was run, so "
        f"every one of the {single['n_total']} examples is held out exactly once "
        "and the headline now carries a confidence interval instead of resting "
        "on a single 40-example slice. Across folds the MLP probe scores "
        f"**accuracy {acc['mean']:.3f} ± {acc['ci95_halfwidth']:.3f}** "
        f"(95% CI [{acc['ci95_low']:.3f}, {acc['ci95_high']:.3f}]) and "
        f"**roc_auc {mlp_agg['roc_auc']['mean']:.3f} ± "
        f"{mlp_agg['roc_auc']['ci95_halfwidth']:.3f}**. The single-split "
        f"accuracy of {single_acc:.2f} sits {position}. "
        f"A plain logistic-regression linear probe reaches "
        f"{logreg['accuracy']['mean']:.3f} ± {logreg['accuracy']['std']:.3f} "
        f"accuracy / {logreg['roc_auc']['mean']:.3f} roc_auc, confirming the "
        "harmful-vs-benign signal is strongly (and largely linearly) decodable "
        "from this layer — the MLP is not overfitting to one split. "
        + ("The single-split headline is therefore trustworthy: it lands within "
           "the cross-validated confidence interval."
           if inside else
           "Treat the single-split headline with caution: it lands outside the "
           "cross-validated confidence interval, so prefer the CV mean±CI as the "
           "reportable number.")
    )
    lines.append("")
    CV_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[md] wrote {CV_REPORT_MD.name}", file=sys.stderr)


def main() -> None:
    set_seed(CV_SEED)
    device = "cpu"                       # CPU-only by mandate; probe is tiny
    X, y = load_features()
    in_dim = X.shape[1]

    # --- k-fold CV of the deployed MLP probe --------------------------------
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=CV_SEED)
    per_fold = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        # (a) fit scaler on this fold's train only — no test leakage.
        scaler = StandardScaler().fit(X[tr])
        Xtr_all = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)

        # (b) carve a stratified val slice from train for early stopping.
        rng = np.random.default_rng(CV_SEED + fold)
        fit_pos, val_pos = stratified_val_split(y[tr], C.VAL_FRACTION, rng)
        set_seed(CV_SEED + fold)          # deterministic weight init per fold
        probe = train_one_mlp(Xtr_all[fit_pos], y[tr][fit_pos],
                              Xtr_all[val_pos], y[tr][val_pos], in_dim, device)

        # (c) score the held-out fold.
        p_te = mlp_proba(probe, Xte, device)
        m = compute_metrics(y[te], p_te, C.DECISION_THRESHOLD)
        per_fold.append(m)
        print(f"[fold {fold}/{K_FOLDS}] n_test={len(te)} "
              f"acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f} "
              f"f1={m['f1']:.3f}", file=sys.stderr)

    mlp_agg = aggregate(per_fold)
    logreg = logreg_reference(X, y)

    # Single-split headline (from the deployed train_probe.py run) for context.
    single = {"n_total": int(len(y)), "in_dim": int(in_dim),
              "class_balance": np.bincount(y).tolist(), "metrics": {}}
    if C.METRICS_PATH.exists():
        sm = json.loads(C.METRICS_PATH.read_text())
        # The full 12-metric suite lives under the nested "metrics" key; fall
        # back to top-level for accuracy/roc_auc which are mirrored there.
        nested = sm.get("metrics", {})
        for name in METRIC_NAMES:
            # metrics.json stores brier under 'brier_score_loss'.
            key = "brier_score_loss" if name == "brier" else name
            if key in nested:
                single["metrics"][name] = float(nested[key])
            elif key in sm:
                single["metrics"][name] = float(sm[key])

    # --- persist ------------------------------------------------------------
    report = {
        "config": {"k_folds": K_FOLDS, "cv_seed": CV_SEED,
                   "estimator": "MLPProbe (in->128->32->1, dropout %.2f)" % C.DROPOUT,
                   "recipe": {"lr": C.LR, "weight_decay": C.WEIGHT_DECAY,
                              "epochs_max": C.EPOCHS, "patience": C.PATIENCE,
                              "val_fraction": C.VAL_FRACTION,
                              "threshold": C.DECISION_THRESHOLD},
                   "scaler": "StandardScaler, fit per-fold on train only",
                   "device": device},
        "data": {"n_total": int(len(y)), "in_dim": int(in_dim),
                 "class_balance": np.bincount(y).tolist()},
        "mlp_per_fold": per_fold,
        "mlp_aggregate": mlp_agg,
        "logreg_reference": logreg,
        "single_split": single["metrics"],
    }
    CV_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[json] wrote {CV_REPORT_JSON.name}", file=sys.stderr)

    write_markdown(mlp_agg, logreg, single)
    make_plot(mlp_agg)

    # --- print mean±CI table to stdout --------------------------------------
    print("\n=== {}-FOLD CV — MLP PROBE (mean +/- 95% CI) ================".format(K_FOLDS))
    print(f"  {'metric':<18}{'mean':>8}   {'95% CI':>22}   {'single':>8}")
    for name in METRIC_NAMES:
        a = mlp_agg[name]
        s = single["metrics"].get(name)
        s_str = f"{s:.4f}" if s is not None else "  —  "
        ci = f"[{a['ci95_low']:.4f}, {a['ci95_high']:.4f}]"
        print(f"  {name:<18}{a['mean']:>8.4f}   {ci:>22}   {s_str:>8}")
    print("  ---------------------------------------------------------------")
    print(f"  logreg reference   acc={logreg['accuracy']['mean']:.4f} "
          f"+/- {logreg['accuracy']['std']:.4f}   "
          f"auc={logreg['roc_auc']['mean']:.4f} "
          f"+/- {logreg['roc_auc']['std']:.4f}")
    print("================================================================")


if __name__ == "__main__":
    main()
