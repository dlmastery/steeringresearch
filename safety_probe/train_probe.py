"""train_probe.py — the whole training lifecycle, top to bottom.

Run:  python -m safety_probe.train_probe

Steps (each is a numbered section below):
    1. Load the safety dataset (harmful vs. safe prompts).
    2. Load the frozen LLM and extract one activation vector per prompt
       (cached to artifacts/features.npz so re-runs are instant).
    3. Split into train / val / test (stratified, fixed seed).
    4. Standardize features (fit scaler on train only).
    5. Train the 3-layer MLP probe with early stopping on the val loss.
    6. Evaluate on the held-out test set (accuracy, precision, recall, F1, AUC).
    7. Save the probe, a metrics.json, and three dashboard plots.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data import load_safety_dataset
from .model_utils import extract_features, hidden_size, load_model, num_layers
from .probe import MLPProbe, Scaler, predict_proba, save_probe


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Step 2 helper: extract-or-load cached features -------------------------
def get_features(prompts: list[str], labels: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (X, y, layer). Uses artifacts/features.npz if it matches this config."""
    if C.FEATURES_CACHE.exists():
        cache = np.load(C.FEATURES_CACHE, allow_pickle=True)
        if (int(cache["layer"]) == C.LAYER
                and str(cache["model_id"]) == C.MODEL_ID
                and cache["X"].shape[0] == len(prompts)):
            print(f"[features] cache hit {C.FEATURES_CACHE.name} "
                  f"X={cache['X'].shape}", file=sys.stderr)
            return cache["X"], cache["y"], int(cache["layer"])

    model, tok = load_model(C.MODEL_ID)
    layer = max(0, min(C.LAYER, num_layers(model) - 1))
    X = extract_features(model, tok, prompts, layer, pooling=C.POOLING)
    y = np.asarray(labels, dtype=np.int64)
    np.savez(C.FEATURES_CACHE, X=X, y=y, layer=layer,
             model_id=C.MODEL_ID, hidden=hidden_size(model))
    # free VRAM — we only need X from here on
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return X, y, layer


# --- Step 3 helper: stratified split ----------------------------------------
def stratified_split(y: np.ndarray, rng: np.random.Generator):
    """Return index arrays (train, val, test) balanced across both classes."""
    train_idx, val_idx, test_idx = [], [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * C.TEST_FRACTION))
        n_val = int(round(n * C.VAL_FRACTION))
        test_idx += idx[:n_test].tolist()
        val_idx += idx[n_test:n_test + n_val].tolist()
        train_idx += idx[n_test + n_val:].tolist()
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# --- Step 5: train loop -----------------------------------------------------
def train_probe(Xtr, ytr, Xva, yva, in_dim, device):
    probe = MLPProbe(in_dim, C.HIDDEN1, C.HIDDEN2, C.DROPOUT).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    xtr = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    xva = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva.astype(np.float32)).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(C.EPOCHS):
        probe.train()
        opt.zero_grad()
        loss = loss_fn(probe(xtr), ytr_t)
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            vloss = loss_fn(probe(xva), yva_t).item()
            vacc = ((torch.sigmoid(probe(xva)) > 0.5).float() == yva_t).float().mean().item()
        history["train_loss"].append(loss.item())
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)

        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= C.PATIENCE:
                print(f"[train] early stop at epoch {epoch} (best val loss {best_val:.4f})",
                      file=sys.stderr)
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe, history


# --- Step 6: metrics --------------------------------------------------------
def evaluate(y_true: np.ndarray, p_harm: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                  precision_score, recall_score, roc_auc_score,
                                  roc_curve)

    y_pred = (p_harm >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN,FP],[FN,TP]]
    fpr, tpr, _ = roc_curve(y_true, p_harm)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_harm)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "n_test": int(len(y_true)),
    }


# --- Step 7: plots ----------------------------------------------------------
def make_plots(metrics: dict, history: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC curve
    roc = metrics["roc_curve"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(roc["fpr"], roc["tpr"], color="#2563eb", lw=2,
            label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — held-out test"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(C.ROC_PNG, dpi=110); plt.close(fig)

    # Training history
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.plot(history["train_loss"], label="train loss", color="#2563eb")
    ax.plot(history["val_loss"], label="val loss", color="#dc2626")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.set_title("Training history")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(C.HISTORY_PNG, dpi=110); plt.close(fig)

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["safe", "harmful"]); ax.set_yticklabels(["safe", "harmful"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout(); fig.savefig(C.CONFUSION_PNG, dpi=110); plt.close(fig)
    print(f"[plots] wrote {C.ROC_PNG.name}, {C.HISTORY_PNG.name}, {C.CONFUSION_PNG.name}",
          file=sys.stderr)


def main() -> None:
    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) data
    prompts, labels = load_safety_dataset()

    # 2) features (cached)
    X, y, layer = get_features(prompts, labels)
    in_dim = X.shape[1]

    # 3) split
    rng = np.random.default_rng(C.SEED)
    tr, va, te = stratified_split(y, rng)
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}", file=sys.stderr)

    # 4) standardize (fit on train only — no test leakage)
    scaler = Scaler.fit(X[tr])
    Xtr, Xva, Xte = scaler.transform(X[tr]), scaler.transform(X[va]), scaler.transform(X[te])

    # 5) train
    probe, history = train_probe(Xtr, y[tr], Xva, y[va], in_dim, device)

    # 6) evaluate on held-out test
    p_test = predict_proba(probe, scaler, X[te], device=device)  # note: X[te] raw; predict_proba re-scales
    metrics = evaluate(y[te], p_test, C.DECISION_THRESHOLD)
    metrics.update({
        "model_id": C.MODEL_ID, "layer": int(layer), "pooling": C.POOLING,
        "hidden_dim": int(in_dim), "threshold": C.DECISION_THRESHOLD,
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "dataset": "JailbreakBench harmful vs. benign",
        "history": history,
    })
    # a few concrete test examples for the dashboard
    examples = []
    for idx, p, prob in zip(te, [prompts[i] for i in te], p_test):
        examples.append({"prompt": prompts[idx][:160], "true": int(y[idx]),
                         "prob_harmful": float(prob)})
    examples.sort(key=lambda e: e["prob_harmful"], reverse=True)
    metrics["examples"] = examples

    # 7) persist
    save_probe(C.PROBE_PATH, probe, scaler,
               meta={"model_id": C.MODEL_ID, "layer": int(layer),
                     "pooling": C.POOLING, "threshold": C.DECISION_THRESHOLD})
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    make_plots(metrics, history)

    print("\n=== TEST-SET RESULTS =========================================")
    print(f"  accuracy   {metrics['accuracy']:.3f}")
    print(f"  precision  {metrics['precision']:.3f}")
    print(f"  recall     {metrics['recall']:.3f}")
    print(f"  f1         {metrics['f1']:.3f}")
    print(f"  roc_auc    {metrics['roc_auc']:.3f}")
    print(f"  confusion  {metrics['confusion_matrix']}  [[TN,FP],[FN,TP]]")
    print(f"  saved      {C.PROBE_PATH.name}, {C.METRICS_PATH.name}")
    print("==============================================================")


if __name__ == "__main__":
    main()
