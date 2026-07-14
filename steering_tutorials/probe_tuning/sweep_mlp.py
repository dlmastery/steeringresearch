"""sweep_mlp.py — does a better probe head beat the deployed default?

Run:  python -m steering_tutorials.probe_tuning.sweep_mlp
      python -m steering_tutorials.probe_tuning.sweep_mlp --features ../hello_world/artifacts/features_large.npz

This is lesson 2 (`probe_tuning`): the model-selection search kept OUT of the
minimal `hello_world` lesson. The deployed head (`hello_world/probe.py: MLPProbe`)
is 1152->128->32->1, dropout 0.30, trained with Adam (lr=1e-3, weight_decay=1e-3),
BCE, early stopping (see `hello_world/cross_validate.py`). Its 5-fold CV score is
the number to beat: accuracy 0.870 +/- 0.026, roc_auc 0.941 +/- 0.014. This script
asks whether ANY nearby head — wider, narrower, a 2-layer variant, different
dropout / lr / weight-decay — meaningfully improves on that, using the SAME train
recipe and the SAME 5-fold protocol.

Cross-lesson read (allowed): the frozen-LLM activations are lesson 1's cache at
`../hello_world/artifacts/features.npz`, and the train recipe / probe geometry are
imported from `steering_tutorials.hello_world`. We WRITE only into this lesson's
own `probe_tuning/artifacts/` — nothing is written back into hello_world.

Honesty guard-rails baked in:
    * Every config is scored by StratifiedKFold(k=5, shuffle, random_state=0) with
      the StandardScaler fit per-fold on TRAIN ONLY. Selection is by CROSS-
      VALIDATION mean roc_auc — the held-out TEST set is NEVER consulted here, so
      there is no test-set peeking.
    * With only ~200 rows and ~23 configs, the "winner" of a sweep is very likely
      to be CV noise. We therefore only call a config a real improvement if it
      beats the default by MORE THAN ONE default-CV std (the noise band), and we
      say so plainly in the verdict.

CPU-only by mandate; the Gemma model is never loaded — we reuse the cached
frozen-LLM activations.

Outputs (all under this lesson's ``probe_tuning/artifacts/``):
    sweep_mlp.json   every config's mean/std CV accuracy+roc_auc, winner, default
    sweep_mlp.md     ranked table + honest verdict
    sweep_mlp.png    CV roc_auc across configs (sorted)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Cross-lesson import (allowed): reuse lesson 1's train recipe / probe geometry.
from steering_tutorials.hello_world import config as C

# --- this lesson's own paths (we never write into hello_world) --------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
# Lesson 1's cached activations — read only, resolved relative to this script.
DEFAULT_FEATURES = (ROOT / ".." / "hello_world" / "artifacts" / "features.npz").resolve()

# --- sweep knobs ------------------------------------------------------------
K_FOLDS = 5
CV_SEED = 0                       # random_state for StratifiedKFold + fold seeds
DECISION_THRESHOLD = C.DECISION_THRESHOLD

SWEEP_JSON = ARTIFACTS / "sweep_mlp.json"
SWEEP_MD = ARTIFACTS / "sweep_mlp.md"
SWEEP_PNG = ARTIFACTS / "sweep_mlp.png"

# The deployed default head — the row every candidate must beat.
DEFAULT = {"hidden1": 128, "hidden2": 32, "dropout": 0.30,
           "lr": 1e-3, "weight_decay": 1e-3}


# --- a flexible MLP head (2- or 3-layer) mirroring hello_world.probe.MLPProbe
class SweepMLP(nn.Module):
    """MLP head for the sweep. ``hidden2=None`` gives a 2-layer variant.

    3-layer (hidden2 set):  in -> h1 -> h2 -> 1   (ReLU + Dropout between layers)
    2-layer (hidden2 None): in -> h1 -> 1         (ReLU + Dropout after h1)

    The 3-layer path is architecturally identical to the deployed
    ``hello_world.probe.MLPProbe`` so a matching config reproduces it exactly.
    """

    def __init__(self, in_dim: int, h1: int, h2: int | None, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout)]
        if h2 is not None:
            layers += [nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h2, 1)]
        else:
            layers += [nn.Linear(h1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [batch] logits


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cached frozen-LLM activations (X) and labels (y). No model needed."""
    if not path.exists():
        sys.exit(f"[sweep] missing {path} — run hello_world/train_probe.py first to "
                 "extract & cache features, or pass --features to an existing .npz.")
    cache = np.load(path, allow_pickle=True)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    print(f"[sweep] loaded X={X.shape} y={y.shape} "
          f"class balance={np.bincount(y).tolist()} from {path}",
          file=sys.stderr)
    return X, y


def stratified_val_split(ytr: np.ndarray, val_fraction: float,
                         rng: np.random.Generator):
    """Carve a stratified val slice out of a fold's train indices (early stop).

    Same idea as ``hello_world.cross_validate.stratified_val_split`` — balanced
    across both classes so the early-stop signal is not skewed by imbalance.
    """
    fit_idx, val_idx = [], []
    for cls in (0, 1):
        idx = np.where(ytr == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx += idx[:n_val].tolist()
        fit_idx += idx[n_val:].tolist()
    return np.array(fit_idx), np.array(val_idx)


def train_one(Xtr, ytr, Xva, yva, in_dim, cfg, device):
    """Train one SweepMLP with the deployed recipe; early-stop on val loss.

    Mirrors ``hello_world.cross_validate.train_one_mlp`` exactly except the head
    geometry and (dropout, lr, weight_decay) come from ``cfg``: full-batch Adam,
    BCE, up to C.EPOCHS, restore best-val-loss weights, stop after C.PATIENCE
    stalls.
    """
    probe = SweepMLP(in_dim, cfg["hidden1"], cfg["hidden2"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
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
            best_val = vloss
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    return probe


@torch.no_grad()
def proba(probe, Xstd, device) -> np.ndarray:
    """P(harmful) for already-standardized features."""
    return torch.sigmoid(probe(torch.from_numpy(Xstd).to(device))).cpu().numpy()


def cross_validate_config(X, y, in_dim, cfg, device):
    """5-fold stratified CV of one config; return per-fold acc + roc_auc.

    Scaler fit per-fold on train only (no leakage). A stratified val slice is
    carved from each fold's train split for early stopping — the held-out fold
    is used ONLY to score, never to select hyperparameters or stop training.
    """
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=CV_SEED)
    accs, aucs = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler().fit(X[tr])            # train-only fit
        Xtr_all = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)

        rng = np.random.default_rng(CV_SEED + fold)
        fit_pos, val_pos = stratified_val_split(y[tr], C.VAL_FRACTION, rng)
        set_seed(CV_SEED + fold)                        # deterministic init per fold
        probe = train_one(Xtr_all[fit_pos], y[tr][fit_pos],
                          Xtr_all[val_pos], y[tr][val_pos], in_dim, cfg, device)

        p = proba(probe, Xte, device)
        accs.append(accuracy_score(y[te], (p >= DECISION_THRESHOLD).astype(int)))
        aucs.append(roc_auc_score(y[te], p))
    accs, aucs = np.array(accs), np.array(aucs)
    return {
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std(ddof=0)),
        "roc_auc_mean": float(aucs.mean()), "roc_auc_std": float(aucs.std(ddof=0)),
        "accuracy_per_fold": accs.tolist(), "roc_auc_per_fold": aucs.tolist(),
    }


# --- the grid ---------------------------------------------------------------
def build_grid() -> list[dict]:
    """A bounded, deduplicated grid (<=40 configs) around the deployed default.

    Two structured sub-sweeps rather than a full 144-cell product (which would
    over-explore a 200-row set):
      1. ARCHITECTURE: hidden1 {256,128,64} x hidden2 {64,32,16,None}, at the
         default (dropout 0.30, lr 1e-3, wd 1e-3) — 12 heads incl. the default
         geometry and four 2-layer variants.
      2. REGULARIZATION: on the deployed (128,32) geometry, dropout {0.2,0.3,0.5}
         x lr {1e-3,3e-4} x weight_decay {1e-3,1e-4} — 12 cells incl. the default.
    Deduped -> ~23 configs. Keeps the whole sweep to a couple of minutes on CPU.
    """
    grid: list[dict] = []
    seen: set[tuple] = set()

    def add(hidden1, hidden2, dropout, lr, wd):
        key = (hidden1, hidden2, dropout, lr, wd)
        if key in seen:
            return
        seen.add(key)
        grid.append({"hidden1": hidden1, "hidden2": hidden2, "dropout": dropout,
                     "lr": lr, "weight_decay": wd})

    # 1) architecture sweep at the default regularization
    for h1 in (256, 128, 64):
        for h2 in (64, 32, 16, None):
            add(h1, h2, DEFAULT["dropout"], DEFAULT["lr"], DEFAULT["weight_decay"])

    # 2) regularization sweep on the deployed (128, 32) geometry
    for dropout in (0.2, 0.3, 0.5):
        for lr in (1e-3, 3e-4):
            for wd in (1e-3, 1e-4):
                add(128, 32, dropout, lr, wd)

    return grid


def config_label(cfg: dict) -> str:
    """Compact human label, e.g. '128->32 d0.3 lr1e-3 wd1e-3' (None -> 2-layer)."""
    arch = f"{cfg['hidden1']}->{cfg['hidden2']}" if cfg["hidden2"] is not None \
        else f"{cfg['hidden1']}->(2-layer)"
    return f"{arch} d{cfg['dropout']} lr{cfg['lr']:g} wd{cfg['weight_decay']:g}"


def is_default(cfg: dict) -> bool:
    return all(cfg[k] == DEFAULT[k] for k in DEFAULT)


# --- plot -------------------------------------------------------------------
def make_plot(results: list[dict], default_idx: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # sort by CV roc_auc ascending so the best sits on the right
    order = sorted(range(len(results)), key=lambda i: results[i]["roc_auc_mean"])
    aucs = [results[i]["roc_auc_mean"] for i in order]
    errs = [results[i]["roc_auc_std"] for i in order]
    # colour: default = orange, current best = green, rest = blue
    best_pos_in_order = len(order) - 1
    colors = []
    for rank, i in enumerate(order):
        if i == default_idx:
            colors.append("#f59e0b")            # default
        elif rank == best_pos_in_order:
            colors.append("#16a34a")            # sweep winner
        else:
            colors.append("#2563eb")
    xs = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(order)), 4.4))
    ax.bar(xs, aucs, yerr=errs, capsize=2, color=colors)
    default_auc = results[default_idx]["roc_auc_mean"]
    ax.axhline(default_auc, color="#f59e0b", ls="--", lw=1,
               label=f"default roc_auc = {default_auc:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([config_label(results[i]["config"]) for i in order],
                       rotation=90, fontsize=6)
    lo = min(aucs) - max(errs) - 0.01
    ax.set_ylim(max(0.0, lo), 1.005)
    ax.set_ylabel("CV roc_auc (mean +/- std across 5 folds)")
    ax.set_title("MLP head sweep — 5-fold CV roc_auc (orange=default, green=best)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(SWEEP_PNG, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {SWEEP_PNG.name}", file=sys.stderr)


# --- markdown report --------------------------------------------------------
def write_markdown(ranked: list[dict], default_row: dict, winner: dict,
                   n_total: int, in_dim: int, class_balance: list[int],
                   beats: bool, margin: float, noise_band: float) -> None:
    lines: list[str] = []
    lines.append("# MLP Head Hyperparameter Sweep — Safety Probe\n")
    lines.append("Lesson 2 (`probe_tuning`): the model-selection search kept out of "
                 "the minimal `hello_world` lesson.\n")
    lines.append(f"Frozen-LLM activations `X` = **{n_total}x{in_dim}**, balanced "
                 f"({class_balance}), read from lesson 1's cached "
                 "`../hello_world/artifacts/features.npz`. CPU-only; the Gemma model "
                 "is never loaded.\n")
    lines.append(f"**{len(ranked)} configs** scored by StratifiedKFold(k={K_FOLDS}, "
                 f"shuffle, random_state={CV_SEED}), StandardScaler fit per-fold on "
                 "train only, deployed train recipe (Adam, BCE, early stop on a "
                 "stratified val slice). **Selection is by cross-validation mean "
                 "roc_auc — the held-out test set is never consulted (no test-set "
                 "peeking).**\n")

    lines.append("## Top configs by CV roc_auc\n")
    lines.append("| rank | config | CV roc_auc | CV accuracy | note |")
    lines.append("|---|---|---|---|---|")
    for rank, r in enumerate(ranked[:8], start=1):
        note = "**default**" if is_default(r["config"]) else ""
        if rank == 1 and not is_default(r["config"]):
            note = (note + " sweep-winner").strip()
        lines.append(
            f"| {rank} | `{config_label(r['config'])}` | "
            f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | "
            f"{r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f} | {note} |")
    lines.append("")

    lines.append("## Baseline — deployed default\n")
    lines.append(f"`{config_label(default_row['config'])}` (rank "
                 f"{default_row['rank']} of {len(ranked)}): "
                 f"**roc_auc {default_row['roc_auc_mean']:.4f} +/- "
                 f"{default_row['roc_auc_std']:.4f}**, accuracy "
                 f"{default_row['accuracy_mean']:.4f} +/- "
                 f"{default_row['accuracy_std']:.4f}.\n")

    lines.append("## Verdict — does anything meaningfully beat the default?\n")
    winner_label = config_label(winner["config"])
    if is_default(winner["config"]):
        lines.append(
            "The **deployed default is itself the top-ranked config** by CV "
            f"roc_auc ({winner['roc_auc_mean']:.4f}). No swept head improves on "
            "it. Keep the default.\n")
    elif beats:
        lines.append(
            f"The best swept head `{winner_label}` scores "
            f"{winner['roc_auc_mean']:.4f} roc_auc — **{margin:+.4f}** vs the "
            f"default's {default_row['roc_auc_mean']:.4f}. That margin exceeds the "
            f"default's own CV noise band (1 std = {noise_band:.4f}). This is a "
            "*candidate* improvement, but on a "
            f"{n_total}-row set with {len(ranked)} configs the sweep can still "
            "manufacture a lucky winner. **Recommendation: do not swap the default "
            "on this evidence alone** — confirm the candidate at n>=7 seeds with a "
            "paired test (per the project's rigor floor) before deploying it.\n")
    else:
        lines.append(
            f"The best swept head `{winner_label}` scores "
            f"{winner['roc_auc_mean']:.4f} roc_auc — only **{margin:+.4f}** vs the "
            f"default's {default_row['roc_auc_mean']:.4f}, which is **within the "
            f"default's CV noise band** (1 std = {noise_band:.4f}). No config beats "
            "the default by more than the fold-to-fold noise. On a "
            f"{n_total}-row set with {len(ranked)} configs, the top of the "
            "leaderboard is dominated by CV noise. **Recommendation: keep the "
            "simple deployed default (128->32, dropout 0.30, lr 1e-3, wd 1e-3) — "
            "it is already near-optimal for this data.**\n")

    SWEEP_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[md] wrote {SWEEP_MD.name}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(DEFAULT_FEATURES),
                    help="path to the cached activations .npz (default: lesson 1's "
                         "../hello_world/artifacts/features.npz)")
    args = ap.parse_args()

    device = "cpu"                              # CPU-only by mandate; heads are tiny
    X, y = load_features(Path(args.features).resolve())
    in_dim = X.shape[1]
    class_balance = np.bincount(y).tolist()

    grid = build_grid()
    print(f"[sweep] {len(grid)} configs x {K_FOLDS} folds on CPU...", file=sys.stderr)

    results: list[dict] = []
    default_idx = -1
    for i, cfg in enumerate(grid):
        cv = cross_validate_config(X, y, in_dim, cfg, device)
        row = {"config": cfg, **cv}
        results.append(row)
        if is_default(cfg):
            default_idx = i
        tag = "  <-- default" if is_default(cfg) else ""
        print(f"[{i + 1:>2}/{len(grid)}] {config_label(cfg):<34} "
              f"auc={cv['roc_auc_mean']:.4f}+/-{cv['roc_auc_std']:.4f} "
              f"acc={cv['accuracy_mean']:.4f}{tag}", file=sys.stderr)

    if default_idx < 0:                          # safety: default must be in-grid
        sys.exit("[sweep] internal error: default config missing from grid")

    # rank by CV roc_auc (primary), accuracy as tiebreak
    order = sorted(range(len(results)),
                   key=lambda i: (results[i]["roc_auc_mean"],
                                  results[i]["accuracy_mean"]), reverse=True)
    ranked = [results[i] for i in order]
    rank_of = {id(results[i]): r for r, i in enumerate(order, start=1)}

    default_row = dict(results[default_idx])
    default_row["rank"] = rank_of[id(results[default_idx])]
    winner = ranked[0]

    # "beats" = winner exceeds the default by MORE THAN the default's 1-std noise
    noise_band = results[default_idx]["roc_auc_std"]
    margin = winner["roc_auc_mean"] - results[default_idx]["roc_auc_mean"]
    beats = (not is_default(winner["config"])) and (margin > noise_band)

    # --- persist ------------------------------------------------------------
    report = {
        "protocol": {
            "k_folds": K_FOLDS, "cv_seed": CV_SEED,
            "selection": "cross-validation mean roc_auc (no test-set peeking)",
            "scaler": "StandardScaler, fit per-fold on train only",
            "recipe": {"optimizer": "Adam", "loss": "BCEWithLogits",
                       "epochs_max": C.EPOCHS, "patience": C.PATIENCE,
                       "val_fraction": C.VAL_FRACTION,
                       "threshold": DECISION_THRESHOLD},
            "device": device, "features": str(Path(args.features).resolve()),
        },
        "data": {"n_total": int(len(y)), "in_dim": int(in_dim),
                 "class_balance": class_balance},
        "default": {"config": DEFAULT, "rank": default_row["rank"],
                    "roc_auc_mean": results[default_idx]["roc_auc_mean"],
                    "roc_auc_std": results[default_idx]["roc_auc_std"],
                    "accuracy_mean": results[default_idx]["accuracy_mean"],
                    "accuracy_std": results[default_idx]["accuracy_std"]},
        "winner": {"config": winner["config"],
                   "roc_auc_mean": winner["roc_auc_mean"],
                   "roc_auc_std": winner["roc_auc_std"],
                   "accuracy_mean": winner["accuracy_mean"],
                   "accuracy_std": winner["accuracy_std"]},
        "beats_default_by_more_than_1std": bool(beats),
        "margin_vs_default_roc_auc": float(margin),
        "default_roc_auc_noise_band_1std": float(noise_band),
        "results": [
            {"rank": rank_of[id(r)], "config": r["config"], **{k: v
             for k, v in r.items() if k != "config"}}
            for r in ranked
        ],
    }
    SWEEP_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[json] wrote {SWEEP_JSON.name}", file=sys.stderr)

    write_markdown(ranked, default_row, winner, len(y), in_dim, class_balance,
                   beats, margin, noise_band)
    make_plot(results, default_idx)

    # --- print ranked table + verdict to stdout -----------------------------
    print("\n=== MLP HEAD SWEEP — {}-FOLD CV (ranked by roc_auc) ============"
          .format(K_FOLDS))
    print(f"  {'#':>2}  {'config':<34}{'roc_auc':>18}{'accuracy':>18}")
    for r in ranked:
        tag = "  <-- default" if is_default(r["config"]) else ""
        auc = f"{r['roc_auc_mean']:.4f}+/-{r['roc_auc_std']:.4f}"
        acc = f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}"
        print(f"  {rank_of[id(r)]:>2}  {config_label(r['config']):<34}"
              f"{auc:>18}{acc:>18}{tag}")
    print("  ---------------------------------------------------------------")
    print(f"  default rank {default_row['rank']}/{len(ranked)} | "
          f"winner margin {margin:+.4f} vs default noise band "
          f"{noise_band:.4f} (1 std)")
    if is_default(winner["config"]):
        print("  VERDICT: the deployed DEFAULT is the top-ranked config. Keep it.")
    elif beats:
        print("  VERDICT: best head beats default by > 1 std -- a CANDIDATE only.")
        print("           Confirm at n>=7 seeds with a paired test before swapping.")
    else:
        print("  VERDICT: no config beats the default beyond CV noise. KEEP the")
        print("           simple default (128->32, d0.30, lr1e-3, wd1e-3).")
    print("================================================================")


if __name__ == "__main__":
    main()
