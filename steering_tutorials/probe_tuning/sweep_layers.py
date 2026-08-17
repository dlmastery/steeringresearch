"""sweep_layers.py — WHICH layer, and WHICH pooling, should the probe read?

Run (CPU-only, no model is loaded):
    python -m steering_tutorials.probe_tuning.sweep_layers
    PT_LAYERS=every2 PT_POOLINGS=mean,last PT_FOLDS=5 \
        python -m steering_tutorials.probe_tuning.sweep_layers

This closes the "KNOWN GAP" the README has been carrying: lesson 1's layer 12 +
mean pooling is an INHERITED DEFAULT that no sweep in this repository has ever
validated. This script validates it — or refuses to.

Two axes, swept together:
  (a) LAYER   — every residual-stream layer present in the cache.
  (b) POOLING — mean vs last-token vs max over token positions, plus a multi-layer
                WINDOW (concatenate w adjacent layers) at each pooling.
Sweeping only (a) at fixed mean pooling would be a trap. Hu, Wang, Lim, Gao & Chen,
2 May 2026, 'Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories
for Robust Jailbreak Detection' (arXiv:2605.02958), Section 5.4 (Ablation Study),
verbatim: "Mean pooling dilutes this strong 'needle' into the vast 'haystack' of
irrelevant background noise." That line is BODY TEXT, not in the abstract. Their
Table 2 caption makes GLOBAL MAX-POOLING the paper's own sparsity-aware default
("Mean Pooling: Replaces the sparsity-aware Global Max-Pooling with Global Average
Pooling"), so `max` is a first-class cell here, not an extra.

  (c) THE BENIGN ARM. Appendix D of the same paper, verbatim: "sequence-level mean
      aggregation can achieve high recall on several attack sets but yields
      near-random XSTest AUROC." So a sweep scored ONLY on harmful-vs-benign would
      rank mean pooling well and never see it fail on the over-refusal
      distribution - the sweep would reproduce the documented blind spot as a
      measurement artifact. Every cell is therefore ALSO scored zero-shot on XSTest
      (cached as group 1). That arm is REPORTED-ONLY: selecting on it would be OOD
      test-set peeking. If the cache has no XSTest rows, the JSON records
      `benign_arm: "ABSENT"` and the run warns that its AUCs are attack-only.

INPUT — the cache written by ``extract_layers.py`` (the GPU half):
    artifacts/layer_features_<tag>.npz   H[n_prompts, n_layers, n_poolings, hidden]
This script loads no model; it is the same CPU model-selection job as
``sweep_mlp.py`` and it REUSES that file's CV machinery (``train_one`` /
``stratified_val_split`` / ``proba`` / ``DEFAULT``) so the layer sweep and the head
sweep are scored by the identical protocol and are comparable. The one local piece
is ``cross_validate_two_arm``, which is ``sweep_mlp.cross_validate_config``'s fold
loop plus the extra zero-shot XSTest scoring; ``--selftest`` proves the two agree
exactly when the benign arm is absent, so "identical protocol" is checkable rather
than merely asserted.

Discipline (identical to sweep_mlp.py, restated because it is the whole point):
  * Selection is by StratifiedKFold CV mean ROC-AUC. The StandardScaler is fit
    per fold on TRAIN ONLY. Lesson 1's held-out test slice is never consulted.
    NO TEST-SET PEEKING.
  * A cell only "wins" if it beats the DEPLOYED cell (layer 12, mean pooling) by
    more than the deployed cell's own fold-to-fold std - the CV noise band. With
    dozens of cells the top of the leaderboard is mostly luck.
  * Even a cell that clears the band is a CANDIDATE, not a deployment: the rigor
    floor wants n>=7 seeds with a paired test first.

Outputs (per-config paths, under ``probe_tuning/artifacts/``):
    sweep_layers_<tag>.json   every cell's CV mean/std AUC + accuracy, winner,
                              deployed cell, margin, noise band, shuffle control
    sweep_layers_<tag>.md     ranked table + honest verdict
    sweep_layers_<tag>.png    AUC vs layer, one line per pooling
The JSON is rewritten after EVERY cell, so a crash or a reap still leaves data.
ASCII-only stdout (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Cross-lesson / same-lesson reuse: the deployed head + the exact CV protocol.
from steering_tutorials.hello_world import config as C
from steering_tutorials.probe_tuning import sweep_mlp as SM

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# --- env caps so a run fits in one foreground window -------------------------
ENV_CACHE = os.environ.get("PT_CACHE", "")               # explicit cache path
ENV_LAYERS = os.environ.get("PT_LAYERS", "all")          # subset of cached layers
ENV_POOLINGS = os.environ.get("PT_POOLINGS", "all")      # subset of cached poolings
ENV_WINDOWS = os.environ.get("PT_WINDOWS", "1,3")        # 1 = single layer
ENV_FOLDS = int(os.environ.get("PT_FOLDS", "5"))
ENV_SEED = int(os.environ.get("PT_SEED", "0"))
ENV_N = int(os.environ.get("PT_N", "0"))                 # 0 = use all cached rows
ENV_SHUFFLE = os.environ.get("PT_SHUFFLE_CONTROL", "1") == "1"


GROUP_MAIN = 0
GROUP_XSTEST = 1
# A cell is flagged with the SALO Appendix-D signature when it looks strong on the
# attack arm yet lands near chance on the over-refusal arm.
APPENDIX_D_MAIN_MIN = 0.80
APPENDIX_D_XSTEST_MAX = 0.60


# --------------------------------------------------------------------------- #
# CV: the main arm (selection) + the XSTest arm (reported only)
# --------------------------------------------------------------------------- #
def cross_validate_two_arm(X, y, in_dim, cfg, device, X_ood=None, y_ood=None):
    """``sweep_mlp.cross_validate_config``'s fold loop, plus zero-shot XSTest.

    Every line that touches the MAIN arm is the same as ``sweep_mlp``: the same
    StratifiedKFold(K_FOLDS, shuffle, CV_SEED), the same per-fold train-only
    StandardScaler, the same stratified val slice for early stopping, the same
    ``train_one``. Run ``--selftest`` to check that claim: with ``X_ood=None`` this
    must reproduce ``sweep_mlp.cross_validate_config`` exactly.

    The addition: each fold's probe ALSO scores the whole XSTest arm, standardized
    with that fold's own train-only scaler. That is a zero-shot transfer number -
    XSTest is never trained on and never used to select a cell (selecting on it
    would be OOD test-set peeking). It exists to catch what SALO Appendix D
    documents: mean aggregation can post high attack recall while its XSTest AUROC
    is near random.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    has_ood = X_ood is not None and len(X_ood) > 0 and len(np.unique(y_ood)) > 1
    skf = StratifiedKFold(n_splits=SM.K_FOLDS, shuffle=True, random_state=SM.CV_SEED)
    accs, aucs, ood_aucs = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler().fit(X[tr])            # train-only fit
        Xtr_all = scaler.transform(X[tr]).astype(np.float32)
        Xte = scaler.transform(X[te]).astype(np.float32)

        rng = np.random.default_rng(SM.CV_SEED + fold)
        fit_pos, val_pos = SM.stratified_val_split(y[tr], C.VAL_FRACTION, rng)
        SM.set_seed(SM.CV_SEED + fold)                  # deterministic init per fold
        probe = SM.train_one(Xtr_all[fit_pos], y[tr][fit_pos],
                             Xtr_all[val_pos], y[tr][val_pos], in_dim, cfg, device)

        p = SM.proba(probe, Xte, device)
        accs.append(accuracy_score(y[te], (p >= SM.DECISION_THRESHOLD).astype(int)))
        aucs.append(roc_auc_score(y[te], p))

        if has_ood:
            Xo = scaler.transform(X_ood).astype(np.float32)
            ood_aucs.append(roc_auc_score(y_ood, SM.proba(probe, Xo, device)))

    accs, aucs = np.array(accs), np.array(aucs)
    out = {
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std(ddof=0)),
        "roc_auc_mean": float(aucs.mean()), "roc_auc_std": float(aucs.std(ddof=0)),
        "accuracy_per_fold": accs.tolist(), "roc_auc_per_fold": aucs.tolist(),
    }
    if has_ood:
        o = np.array(ood_aucs)
        out.update({"xstest_roc_auc_mean": float(o.mean()),
                    "xstest_roc_auc_std": float(o.std(ddof=0)),
                    "xstest_roc_auc_per_fold": o.tolist()})
    return out


def selftest() -> int:
    """Prove cross_validate_two_arm == sweep_mlp.cross_validate_config (no OOD)."""
    rng = np.random.default_rng(0)
    n, d = 80, 16
    y = np.array([1] * 40 + [0] * 40, dtype=np.int64)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, :2] += y[:, None] * 1.2
    SM.K_FOLDS, SM.CV_SEED = 3, 0
    ref = SM.cross_validate_config(X, y, d, SM.DEFAULT, "cpu")
    got = cross_validate_two_arm(X, y, d, SM.DEFAULT, "cpu")
    ok = True
    for k in ("roc_auc_mean", "roc_auc_std", "accuracy_mean", "accuracy_std"):
        same = abs(ref[k] - got[k]) < 1e-12
        ok &= same
        print(f"  {k:<16} sweep_mlp={ref[k]:.10f} two_arm={got[k]:.10f} "
              f"{'OK' if same else 'MISMATCH'}")
    print("SELFTEST", "PASS - the two arms share sweep_mlp's exact protocol" if ok
          else "FAIL - the layer sweep is NOT running sweep_mlp's protocol")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def find_cache(explicit: str) -> Path:
    """Resolve the layer-feature cache, or exit with the command that builds it."""
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            sys.exit(f"[sweep_layers] no cache at {path}")
        return path
    hits = sorted(ARTIFACTS.glob("layer_features_*.npz"))
    hits = [h for h in hits if not h.name.endswith(".partial.npz")]
    if not hits:
        sys.exit(
            "[sweep_layers] no layer-feature cache found in "
            f"{ARTIFACTS}.\n"
            "  This sweep needs activations from EVERY layer; lesson 1's "
            "features.npz holds layer 12 only.\n"
            "  Build it first (GPU, foreground):\n"
            "    python -m steering_tutorials.probe_tuning.extract_layers")
    if len(hits) > 1:
        print(f"[sweep_layers] {len(hits)} caches present; using {hits[-1].name} "
              "(pass --cache to choose)", file=sys.stderr)
    return hits[-1]


def load_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    H = z["H"]
    y = z["y"].astype(np.int64)
    layers = z["layers"].astype(int).tolist()
    poolings = [str(p) for p in z["poolings"].tolist()]
    # Anchor assertions: a cache whose header disagrees with its arrays is not
    # evidence, and must fail loudly rather than produce a plausible sweep.
    if H.shape != (len(y), len(layers), len(poolings), meta["hidden"]):
        sys.exit(f"[sweep_layers] cache shape {H.shape} disagrees with its own "
                 f"header (n={len(y)}, layers={len(layers)}, "
                 f"poolings={len(poolings)}, hidden={meta['hidden']}) - refusing "
                 "to sweep an inconsistent artifact")
    if meta.get("layers") != layers or meta.get("poolings") != poolings:
        sys.exit("[sweep_layers] cache header layers/poolings disagree with the "
                 "stored arrays - refusing to sweep")

    # ``group`` splits the main (selection) arm from the XSTest (reported) arm. A
    # cache written before the benign arm existed simply has no group column; we
    # treat it as all-main and let the ABSENT path downstream say so loudly, rather
    # than inventing an arm that was never extracted.
    if "group" in z.files:
        group = z["group"].astype(np.int64)
    else:
        group = np.full(len(y), GROUP_MAIN, dtype=np.int64)
        meta.setdefault("benign_arm", {
            "status": "ABSENT",
            "reason": "cache predates the benign arm (no 'group' column)",
            "consequence": "every per-cell AUC is ATTACK-ONLY and cannot detect the "
                           "SALO Appendix-D failure mode"})
    n_ood = int((group == GROUP_XSTEST).sum())
    print(f"[sweep_layers] cache {path.name}: H={H.shape} "
          f"main={int((group == GROUP_MAIN).sum())} xstest={n_ood} "
          f"layers={layers} poolings={poolings} "
          f"fingerprint={meta.get('fingerprint')}", file=sys.stderr)
    return {"H": H, "y": y, "group": group, "layers": layers, "poolings": poolings,
            "meta": meta, "path": path}


def subset_layers(spec: str, cached: list[int]) -> list[int]:
    """Restrict the swept layers to a subset of what the cache actually holds."""
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return list(cached)
    if spec.startswith("every"):
        step = max(1, int(spec[len("every"):] or "1"))
        return cached[::step]
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        want = set(range(int(lo), int(hi) + 1))
    else:
        want = {int(t) for t in spec.split(",") if t.strip()}
    out = [l for l in cached if l in want]
    missing = sorted(want - set(cached))
    if missing:
        print(f"[sweep_layers] requested layers not in cache, skipped: {missing}",
              file=sys.stderr)
    if not out:
        sys.exit(f"[sweep_layers] layer spec '{spec}' selects nothing from {cached}")
    return out


# --------------------------------------------------------------------------- #
# cells
# --------------------------------------------------------------------------- #
def build_cells(layers: list[int], poolings: list[str],
                windows: list[int]) -> list[dict]:
    """One cell per (pooling, window, center layer), de-duplicated.

    window=1 is the plain single-layer cell. window=w>1 concatenates the w
    adjacent CACHED layers centered on the cell's layer (clipped at the ends), so
    a window cell has ``w * hidden`` input features. Windows are the cheap test of
    the SALO worry: if one layer's needle is real but narrow, a window that spans
    it should not lose to the single layer, while a diluting pooling should.
    """
    cells: list[dict] = []
    seen: set[tuple] = set()
    for pooling in poolings:
        for w in windows:
            for center in layers:
                if w <= 1:
                    span = [center]
                else:
                    pos = layers.index(center)
                    half = w // 2
                    lo = max(0, min(pos - half, len(layers) - w))
                    span = layers[lo:lo + w]
                    if len(span) < w:            # cache too small for this window
                        continue
                key = (pooling, tuple(span))
                if key in seen:
                    continue
                seen.add(key)
                cells.append({"pooling": pooling, "window": len(span),
                              "center_layer": int(center),
                              "layers": [int(l) for l in span]})
    return cells


def cell_label(cell: dict) -> str:
    if cell["window"] == 1:
        return f"L{cell['layers'][0]:02d} {cell['pooling']}"
    return (f"L{cell['layers'][0]:02d}-{cell['layers'][-1]:02d} "
            f"{cell['pooling']} (w{cell['window']})")


def cell_matrix(H: np.ndarray, layers: list[int], poolings: list[str],
                cell: dict) -> np.ndarray:
    """Slice the cache down to this cell's design matrix [n, window*hidden]."""
    pi = poolings.index(cell["pooling"])
    idx = [layers.index(l) for l in cell["layers"]]
    X = H[:, idx, pi, :]                      # [n, window, hidden]
    return np.ascontiguousarray(X.reshape(X.shape[0], -1)).astype(np.float32)


def deployed_cell(cells: list[dict], layers: list[int]) -> tuple[dict, bool]:
    """The cell lesson 1 actually ships: LAYER=12, POOLING=mean, single layer.

    Returns (cell, substituted). If the deployed layer is absent from the cache
    (a subsampled extraction), the nearest cached layer stands in and the JSON
    says so - a silently substituted baseline is how a sweep flatters itself.
    """
    want_pooling = str(C.POOLING)
    singles = [c for c in cells if c["window"] == 1 and c["pooling"] == want_pooling]
    if not singles:
        sys.exit(f"[sweep_layers] the deployed pooling '{want_pooling}' is not in "
                 "this sweep - cannot price any cell against the deployed default")
    exact = [c for c in singles if c["layers"][0] == int(C.LAYER)]
    if exact:
        return exact[0], False
    nearest = min(singles, key=lambda c: abs(c["layers"][0] - int(C.LAYER)))
    return nearest, True


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_plot(png: Path, results: list[dict], dep_idx: int, poolings: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = {"mean": "#2563eb", "last": "#16a34a", "max": "#a855f7"}
    for pooling in poolings:
        rows = [r for r in results
                if r["cell"]["pooling"] == pooling and r["cell"]["window"] == 1]
        if not rows:
            continue
        rows.sort(key=lambda r: r["cell"]["center_layer"])
        xs = [r["cell"]["center_layer"] for r in rows]
        ys = [r["roc_auc_mean"] for r in rows]
        es = [r["roc_auc_std"] for r in rows]
        col = colors.get(pooling, "#64748b")
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=3.5, lw=1.4, capsize=2,
                    color=col, label=f"{pooling}-pool (single layer)")
    win_rows = [r for r in results if r["cell"]["window"] > 1]
    if win_rows:
        ax.scatter([r["cell"]["center_layer"] for r in win_rows],
                   [r["roc_auc_mean"] for r in win_rows],
                   marker="s", s=22, facecolors="none", edgecolors="#ef4444",
                   label="multi-layer window")
    dep = results[dep_idx]
    ax.axhline(dep["roc_auc_mean"], color="#f59e0b", ls="--", lw=1.2,
               label=f"deployed {cell_label(dep['cell'])} = "
                     f"{dep['roc_auc_mean']:.3f}")
    ax.axhspan(dep["roc_auc_mean"] - dep["roc_auc_std"],
               dep["roc_auc_mean"] + dep["roc_auc_std"],
               color="#f59e0b", alpha=0.12,
               label=f"deployed 1-std noise band ({dep['roc_auc_std']:.3f})")
    ax.axhline(0.5, color="#94a3b8", ls=":", lw=1, label="chance")
    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel("CV roc_auc (mean +/- std across folds)")
    ax.set_title("Layer x pooling sweep - which activation should the probe read?")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {png.name}", file=sys.stderr)


def write_markdown(md: Path, ranked: list[dict], dep_row: dict, winner: dict,
                   payload: dict) -> None:
    d = payload["data"]
    p = payload["protocol"]
    lines: list[str] = []
    lines.append("# Layer x Pooling Sweep - where should the safety probe read?\n")
    lines.append("Lesson 2 (`probe_tuning`). This is the sweep the README used to "
                 "list as a KNOWN GAP: lesson 1's **layer 12, mean-pooled** is an "
                 "inherited default, and until this ran, nothing in this repository "
                 "had tested it.\n")
    lines.append(f"Activations: `{Path(payload['cache']['path']).name}` - "
                 f"**{d['n_total']} prompts** ({d['n_per_class']}/class, balance "
                 f"{d['class_balance']}) through frozen "
                 f"`{payload['cache']['model_id']}`, "
                 f"{len(p['layers_swept'])} layers x {len(p['poolings_swept'])} "
                 f"poolings x windows {p['windows']} = **{len(ranked)} cells**.\n")
    lines.append(f"Each cell scored by StratifiedKFold(k={p['k_folds']}, shuffle, "
                 f"random_state={p['cv_seed']}) with the StandardScaler fit per-fold "
                 "on train only, using the DEPLOYED head and train recipe "
                 f"(`{p['head']}`). **Selection is by cross-validation mean roc_auc; "
                 "the held-out test set is never consulted.**\n")

    has_ood = payload["appendix_d_flag_rule"]["evaluable"]
    ba = payload["benign_arm_detail"]
    lines.append("## The benign / over-refusal arm\n")
    if has_ood:
        lines.append(f"**PRESENT** — {ba.get('dataset')}, n={ba.get('n')}, balance "
                     f"{ba.get('class_balance')}, loaded via "
                     f"`{ba.get('loader')}`. It is **reported-only**: every cell is "
                     "scored on it zero-shot per CV fold, and it is **never** used "
                     "to select a cell (that would be OOD test-set peeking).\n")
        lines.append("This arm exists because of arXiv:2605.02958 Appendix D — "
                     "*\"sequence-level mean aggregation can achieve high recall on "
                     "several attack sets but yields near-random XSTest AUROC\"*. "
                     "Without it, a mean-pooling cell could top this leaderboard "
                     "while being broken on over-refusal, and the sweep would never "
                     "know.\n")
    else:
        lines.append(f"**ABSENT** — {ba.get('reason')}.\n")
        lines.append("> **WARNING: every AUC on this page is ATTACK-ONLY.** "
                     "arXiv:2605.02958 Appendix D reports that *\"sequence-level "
                     "mean aggregation can achieve high recall on several attack "
                     "sets but yields near-random XSTest AUROC\"*. This run cannot "
                     "detect that failure mode, so **no pooling conclusion drawn "
                     "here is safe**. Re-extract with the XSTest arm first.\n")

    lines.append("## Top cells by CV roc_auc\n")
    xs_col = " XSTest roc_auc |" if has_ood else ""
    lines.append(f"| rank | cell | in_dim | CV roc_auc |{xs_col} CV accuracy | note |")
    lines.append("|---|---|---|---|---|---|" + ("---|" if has_ood else ""))
    for rank, r in enumerate(ranked[:12], start=1):
        note = "**deployed**" if r["is_deployed"] else ""
        if rank == 1 and not r["is_deployed"]:
            note = (note + " sweep-winner").strip()
        if r.get("appendix_d_flag"):
            note = (note + " **APPENDIX-D FLAG**").strip()
        xs = (f" {r['xstest_roc_auc_mean']:.4f} +/- {r['xstest_roc_auc_std']:.4f} |"
              if has_ood else "")
        lines.append(f"| {rank} | `{cell_label(r['cell'])}` | {r['in_dim']} | "
                     f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} |{xs} "
                     f"{r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f} | "
                     f"{note} |")
    lines.append("")

    flagged = payload.get("appendix_d_flagged_cells") or []
    if has_ood:
        lines.append("## Appendix-D check — strong on attacks, near chance on "
                     "over-refusal\n")
        lines.append(f"Flag rule: main roc_auc >= {APPENDIX_D_MAIN_MIN} **and** "
                     f"XSTest roc_auc <= {APPENDIX_D_XSTEST_MAX}.\n")
        if flagged:
            lines.append("| cell | main roc_auc | XSTest roc_auc |")
            lines.append("|---|---|---|")
            for f in flagged:
                lines.append(f"| `{f['label']}` | {f['roc_auc_mean']:.4f} | "
                             f"{f['xstest_roc_auc_mean']:.4f} |")
            lines.append("")
            lines.append(f"**{len(flagged)} cell(s) reproduce the Appendix-D "
                         "signature**: they look strong on harmful-vs-benign and "
                         "land near chance on over-refusal. A sweep without the "
                         "benign arm would have ranked them on the first number "
                         "alone.\n")
        else:
            lines.append("**No cell trips the flag** on this data — the Appendix-D "
                         "failure mode did not reproduce here. That is a real "
                         "negative result, and it is only sayable BECAUSE the "
                         "benign arm was measured.\n")

    lines.append("## Baseline - the deployed cell\n")
    sub = payload["deployed"]["substituted_for_missing_layer"]
    lines.append(f"`{cell_label(dep_row['cell'])}` (rank {dep_row['rank']} of "
                 f"{len(ranked)}): **roc_auc {dep_row['roc_auc_mean']:.4f} +/- "
                 f"{dep_row['roc_auc_std']:.4f}**, accuracy "
                 f"{dep_row['accuracy_mean']:.4f} +/- "
                 f"{dep_row['accuracy_std']:.4f}."
                 + (f" NOTE: lesson 1's layer {C.LAYER} is NOT in this cache, so the "
                    "nearest cached layer stands in as the baseline - the margin "
                    "below is against a substitute.\n" if sub else "\n"))

    sh = payload.get("shuffle_control")
    if sh:
        lines.append("## Shuffle control\n")
        lines.append(f"The winning cell re-scored with PERMUTED labels: roc_auc "
                     f"**{sh['roc_auc_mean']:.4f} +/- {sh['roc_auc_std']:.4f}** "
                     f"(chance = 0.5). {'Passes' if sh['passes'] else 'FAILS'} - "
                     "a pipeline that scores above chance on shuffled labels is "
                     "leaking, and every number above would be void.\n")

    lines.append("## Verdict - is layer 12 + mean pooling the right read?\n")
    margin = payload["margin_vs_deployed_roc_auc"]
    band = payload["deployed_roc_auc_noise_band_1std"]
    if winner["is_deployed"]:
        lines.append(f"The **deployed cell is itself the top-ranked cell** "
                     f"({winner['roc_auc_mean']:.4f}). Lesson 1's inherited layer "
                     "12 + mean pooling is vindicated by measurement, not just by "
                     "inheritance. Keep it.\n")
    elif payload["beats_deployed_by_more_than_1std"]:
        lines.append(f"The best cell `{cell_label(winner['cell'])}` scores "
                     f"{winner['roc_auc_mean']:.4f} - **{margin:+.4f}** vs the "
                     f"deployed {dep_row['roc_auc_mean']:.4f}, which EXCEEDS the "
                     f"deployed cell's own CV noise band (1 std = {band:.4f}). "
                     "That makes it a **CANDIDATE, not a deployment**: with "
                     f"{len(ranked)} cells the leaderboard can still manufacture a "
                     "lucky winner. Confirm at n>=7 seeds with a paired test before "
                     "changing lesson 1's LAYER/POOLING.\n")
    else:
        lines.append(f"The best cell `{cell_label(winner['cell'])}` scores "
                     f"{winner['roc_auc_mean']:.4f} - only **{margin:+.4f}** vs the "
                     f"deployed {dep_row['roc_auc_mean']:.4f}, **inside** the "
                     f"deployed cell's noise band (1 std = {band:.4f}). No layer and "
                     "no pooling beats the inherited default by more than "
                     "fold-to-fold noise. The honest reading is NOT 'layer 12 is "
                     "optimal' but '**the probe is insensitive to this choice over "
                     "the swept range**' - which is itself the answer to the "
                     "question the README left open.\n")

    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[md] wrote {md.name}", file=sys.stderr)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep layer x pooling for the probe.")
    ap.add_argument("--cache", default=ENV_CACHE, help="path to layer_features_*.npz")
    ap.add_argument("--layers", default=ENV_LAYERS, help="subset spec (PT_LAYERS)")
    ap.add_argument("--poolings", default=ENV_POOLINGS, help="subset (PT_POOLINGS)")
    ap.add_argument("--windows", default=ENV_WINDOWS,
                    help="comma list of multi-layer window sizes (PT_WINDOWS)")
    ap.add_argument("--folds", type=int, default=ENV_FOLDS)
    ap.add_argument("--seed", type=int, default=ENV_SEED)
    ap.add_argument("--n", type=int, default=ENV_N,
                    help="cap TOTAL rows, class-balanced (PT_N; 0 = all cached)")
    ap.add_argument("--no-shuffle-control", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="check cross_validate_two_arm == sweep_mlp's protocol, "
                         "then exit (no cache needed)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    cache = load_cache(find_cache(args.cache))
    meta = cache["meta"]
    group = cache["group"]

    # --- split the two arms --------------------------------------------------
    main_rows = np.where(group == GROUP_MAIN)[0]
    ood_rows = np.where(group == GROUP_XSTEST)[0]
    H, y = cache["H"][main_rows], cache["y"][main_rows]
    H_ood, y_ood = cache["H"][ood_rows], cache["y"][ood_rows]
    benign_arm = dict(meta.get("benign_arm") or {"status": "ABSENT",
                                                 "reason": "not recorded in cache"})
    has_ood = len(ood_rows) > 0 and len(np.unique(y_ood)) > 1
    if not has_ood:
        benign_arm.setdefault("status", "ABSENT")
        benign_arm["status"] = "ABSENT"
        benign_arm.setdefault("reason", "no usable XSTest rows in the cache")
        benign_arm["consequence"] = (
            "every per-cell AUC below is ATTACK-ONLY and cannot detect the SALO "
            "Appendix-D failure mode (high attack recall with near-random XSTest "
            "AUROC)")
        print("\n" + "!" * 70, file=sys.stderr)
        print("[sweep_layers] WARNING - BENIGN ARM ABSENT.", file=sys.stderr)
        print("  Every per-cell AUC in this run is ATTACK-ONLY (harmful vs benign).",
              file=sys.stderr)
        print("  SALO (arXiv:2605.02958) Appendix D: 'sequence-level mean "
              "aggregation can", file=sys.stderr)
        print("  achieve high recall on several attack sets but yields near-random "
              "XSTest", file=sys.stderr)
        print("  AUROC.' This run CANNOT see that failure, so a mean-pooling cell "
              "may rank", file=sys.stderr)
        print("  well here and still be broken on over-refusal. Re-extract with the "
              "XSTest", file=sys.stderr)
        print("  arm before drawing any pooling conclusion.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        H_ood, y_ood = None, None

    if args.n and args.n < len(y):
        per = args.n // 2
        keep = np.concatenate([np.where(y == 1)[0][:per], np.where(y == 0)[0][:per]])
        keep.sort()
        H, y = H[keep], y[keep]
        print(f"[sweep_layers] ROW CAP: {len(y)} rows - screening tier only",
              file=sys.stderr)

    layers = subset_layers(args.layers, cache["layers"])
    poolings = ([p for p in cache["poolings"]] if args.poolings.strip().lower() == "all"
                else [p.strip() for p in args.poolings.split(",") if p.strip()])
    unknown = [p for p in poolings if p not in cache["poolings"]]
    if unknown:
        sys.exit(f"[sweep_layers] poolings {unknown} are not in the cache "
                 f"{cache['poolings']} - re-run extract_layers with PT_POOLINGS")
    windows = sorted({int(w) for w in args.windows.split(",") if w.strip()})

    cells = build_cells(layers, poolings, windows)
    dep_cell, substituted = deployed_cell(cells, layers)

    # The CV protocol is sweep_mlp's, verbatim; these two module knobs are the
    # only thing this sweep varies about it, and both are recorded in the JSON.
    SM.K_FOLDS = int(args.folds)
    SM.CV_SEED = int(args.seed)

    tag = cache["path"].stem.replace("layer_features_", "")
    out_json = ARTIFACTS / f"sweep_layers_{tag}.json"
    out_md = ARTIFACTS / f"sweep_layers_{tag}.md"
    out_png = ARTIFACTS / f"sweep_layers_{tag}.png"

    payload = {
        "cache": {"path": str(cache["path"]), "model_id": meta["model_id"],
                  "fingerprint": meta.get("fingerprint"),
                  "dataset": meta.get("dataset"),
                  "provenance": meta.get("provenance", {})},
        "protocol": {
            "k_folds": SM.K_FOLDS, "cv_seed": SM.CV_SEED,
            "selection": "cross-validation mean roc_auc (no test-set peeking)",
            "scaler": "StandardScaler, fit per-fold on train only",
            "head": SM.config_label(SM.DEFAULT),
            "head_config": SM.DEFAULT,
            "recipe": {"optimizer": "Adam", "loss": "BCEWithLogits",
                       "epochs_max": C.EPOCHS, "patience": C.PATIENCE,
                       "val_fraction": C.VAL_FRACTION,
                       "threshold": SM.DECISION_THRESHOLD},
            "device": "cpu",
            "layers_swept": layers, "poolings_swept": poolings, "windows": windows,
            "n_cells": len(cells),
        },
        "data": {"n_total": int(len(y)), "n_per_class": int(min(np.bincount(y))),
                 "class_balance": np.bincount(y).tolist(),
                 "hidden": int(meta["hidden"]),
                 "tier": ("EVALUATION-eligible n" if min(np.bincount(y)) >= 500
                          else "SCREENING (below the >=500/class rubric)")},
        # Either the XSTest provenance dict, or ABSENT with the consequence spelled
        # out. Never omitted - a missing benign arm must be visible in the artifact.
        "benign_arm": benign_arm if has_ood else "ABSENT",
        "benign_arm_detail": benign_arm,
        "appendix_d_flag_rule": {
            "source": "arXiv:2605.02958 Appendix D",
            "quote": "sequence-level mean aggregation can achieve high recall on "
                     "several attack sets but yields near-random XSTest AUROC",
            "flagged_when": f"main roc_auc >= {APPENDIX_D_MAIN_MIN} AND xstest "
                            f"roc_auc <= {APPENDIX_D_XSTEST_MAX}",
            "evaluable": bool(has_ood),
        },
        "deployed": {"cell": dep_cell,
                     "from": f"hello_world.config LAYER={C.LAYER} POOLING={C.POOLING}",
                     "substituted_for_missing_layer": bool(substituted)},
        "results": [],
        "status": "running",
    }
    write_json(out_json, payload)          # data on disk before any compute

    print(f"[sweep_layers] {len(cells)} cells x {SM.K_FOLDS} folds on CPU "
          f"(n={len(y)}) - this is the slow part; results stream into "
          f"{out_json.name} after every cell", file=sys.stderr)

    results: list[dict] = []
    dep_idx = -1
    for i, cell in enumerate(cells):
        X = cell_matrix(H, cache["layers"], cache["poolings"], cell)
        Xo = (cell_matrix(H_ood, cache["layers"], cache["poolings"], cell)
              if has_ood else None)
        cv = cross_validate_two_arm(X, y, X.shape[1], SM.DEFAULT, "cpu", Xo, y_ood)
        row = {"cell": cell, "label": cell_label(cell), "in_dim": int(X.shape[1]),
               "is_deployed": cell == dep_cell, **cv}
        # The Appendix-D signature: strong on attacks, near chance on over-refusal.
        row["appendix_d_flag"] = bool(
            has_ood
            and cv["roc_auc_mean"] >= APPENDIX_D_MAIN_MIN
            and cv["xstest_roc_auc_mean"] <= APPENDIX_D_XSTEST_MAX)
        results.append(row)
        if row["is_deployed"]:
            dep_idx = i
        payload["results"] = results       # rewritten every cell: a reap keeps data
        write_json(out_json, payload)
        tail = "  <-- deployed" if row["is_deployed"] else ""
        ood_txt = (f" xstest={cv['xstest_roc_auc_mean']:.4f}" if has_ood
                   else " xstest=ABSENT")
        if row["appendix_d_flag"]:
            tail = "  [APPENDIX-D FLAG]" + tail
        print(f"[{i + 1:>3}/{len(cells)}] {cell_label(cell):<28} "
              f"auc={cv['roc_auc_mean']:.4f}+/-{cv['roc_auc_std']:.4f}"
              f"{ood_txt} acc={cv['accuracy_mean']:.4f}{tail}", file=sys.stderr)

    if dep_idx < 0:                        # anchor: the baseline must be in-sweep
        sys.exit("[sweep_layers] internal error: deployed cell missing from results")

    order = sorted(range(len(results)),
                   key=lambda i: (results[i]["roc_auc_mean"],
                                  results[i]["accuracy_mean"]), reverse=True)
    ranked = [results[i] for i in order]
    for rank, i in enumerate(order, start=1):
        results[i]["rank"] = rank

    dep_row = results[dep_idx]
    winner = ranked[0]
    band = dep_row["roc_auc_std"]
    margin = winner["roc_auc_mean"] - dep_row["roc_auc_mean"]
    beats = (not winner["is_deployed"]) and (margin > band)

    # --- shuffle control on the winning cell --------------------------------
    if ENV_SHUFFLE and not args.no_shuffle_control:
        rng = np.random.default_rng(SM.CV_SEED + 1000)
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        Xw = cell_matrix(H, cache["layers"], cache["poolings"], winner["cell"])
        sh = cross_validate_two_arm(Xw, y_shuf, Xw.shape[1], SM.DEFAULT, "cpu")
        payload["shuffle_control"] = {
            "cell": winner["cell"], "label": cell_label(winner["cell"]),
            "roc_auc_mean": sh["roc_auc_mean"], "roc_auc_std": sh["roc_auc_std"],
            "passes": bool(abs(sh["roc_auc_mean"] - 0.5) <= 0.10),
            "note": "labels permuted; roc_auc must collapse to ~0.5 or the whole "
                    "sweep is leaking",
        }

    payload.update({
        "winner": {"cell": winner["cell"], "label": cell_label(winner["cell"]),
                   "in_dim": winner["in_dim"],
                   "roc_auc_mean": winner["roc_auc_mean"],
                   "roc_auc_std": winner["roc_auc_std"],
                   "accuracy_mean": winner["accuracy_mean"],
                   "accuracy_std": winner["accuracy_std"],
                   "xstest_roc_auc_mean": winner.get("xstest_roc_auc_mean"),
                   "xstest_roc_auc_std": winner.get("xstest_roc_auc_std"),
                   "appendix_d_flag": winner.get("appendix_d_flag")},
        "appendix_d_flagged_cells": [
            {"label": r["label"], "roc_auc_mean": r["roc_auc_mean"],
             "xstest_roc_auc_mean": r.get("xstest_roc_auc_mean")}
            for r in results if r.get("appendix_d_flag")],
        "deployed_result": {"rank": dep_row["rank"],
                            "roc_auc_mean": dep_row["roc_auc_mean"],
                            "roc_auc_std": dep_row["roc_auc_std"],
                            "accuracy_mean": dep_row["accuracy_mean"],
                            "accuracy_std": dep_row["accuracy_std"]},
        "beats_deployed_by_more_than_1std": bool(beats),
        "margin_vs_deployed_roc_auc": float(margin),
        "deployed_roc_auc_noise_band_1std": float(band),
        "results": ranked,
        "status": "complete",
    })
    write_json(out_json, payload)          # JSON complete BEFORE md/png/print
    print(f"[json] wrote {out_json.name}", file=sys.stderr)

    write_markdown(out_md, ranked, dep_row, winner, payload)
    make_plot(out_png, results, dep_idx, poolings)

    # --- ascii-only summary --------------------------------------------------
    print("\n=== LAYER x POOLING SWEEP - {}-FOLD CV (ranked by roc_auc) ====="
          .format(SM.K_FOLDS))
    print(f"  {'#':>3}  {'cell':<28}{'in_dim':>8}{'roc_auc':>18}{'xstest':>10}"
          f"{'accuracy':>18}")
    for r in ranked[:20]:
        tail = "  <-- deployed" if r["is_deployed"] else ""
        if r.get("appendix_d_flag"):
            tail = "  [APPENDIX-D FLAG]" + tail
        auc = f"{r['roc_auc_mean']:.4f}+/-{r['roc_auc_std']:.4f}"
        xs = (f"{r['xstest_roc_auc_mean']:.4f}" if has_ood else "ABSENT")
        acc = f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}"
        print(f"  {r['rank']:>3}  {r['label']:<28}{r['in_dim']:>8}"
              f"{auc:>18}{xs:>10}{acc:>18}{tail}")
    if len(ranked) > 20:
        print(f"  ... {len(ranked) - 20} more cells in {out_md.name}")
    print("  ---------------------------------------------------------------")
    print(f"  n={payload['data']['n_total']} "
          f"({payload['data']['n_per_class']}/class) | tier: "
          f"{payload['data']['tier']}")
    print(f"  deployed {dep_row['label']} rank {dep_row['rank']}/{len(ranked)} | "
          f"winner margin {margin:+.4f} vs noise band {band:.4f} (1 std)")
    sh = payload.get("shuffle_control")
    if sh:
        print(f"  shuffle control on the winner: roc_auc {sh['roc_auc_mean']:.4f} "
              f"({'PASS' if sh['passes'] else 'FAIL'} vs chance 0.5)")
    if has_ood:
        n_flag = len(payload.get("appendix_d_flagged_cells") or [])
        print(f"  benign arm: XSTest n={len(y_ood)} (reported-only) | "
              f"Appendix-D flagged cells: {n_flag}")
    else:
        print("  benign arm: ABSENT -- every AUC above is ATTACK-ONLY and cannot")
        print("              detect the SALO Appendix-D failure mode.")
    if winner["is_deployed"]:
        print("  VERDICT: the DEPLOYED layer+pooling is the top cell. Keep it.")
    elif beats:
        print("  VERDICT: a cell beats the deployed read by > 1 std -- CANDIDATE.")
        print("           Confirm at n>=7 seeds with a paired test before changing")
        print("           hello_world.config LAYER / POOLING.")
    else:
        print("  VERDICT: nothing beats the deployed read beyond CV noise. The probe")
        print("           is INSENSITIVE to layer/pooling over the swept range.")
    print("================================================================")


if __name__ == "__main__":
    main()
