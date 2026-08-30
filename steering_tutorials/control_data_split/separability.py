"""separability.py -- the parts of ASIDE that reproduce WITHOUT any training.

WHAT THIS MODULE IS FOR
------------------------
`aside.py` gives the rotation (orthogonal, norm-preserving, role-masked).
`inseparability.py` gives the surface-level Bayes floor (unigram TV, no model).
This module is the BRIDGE to the model's residual stream: given ACTIVATIONS
already extracted at one or more layers, plus the out-of-band role label from
`data.py`, it answers the two questions the paper's Figure/ablation makes with
no fine-tuning at all:

  1. LAYER-WISE LINEAR-PROBE SEPARABILITY -- can a linear classifier tell
     instruction tokens/documents from data tokens/documents at layer L? The
     paper reports ASIDE is ~100%% separable from the embedding layer onward
     while vanilla only becomes separable at a later layer (`layer_separability_
     sweep` + `probe_auc`).
  2. THE COSINE TRAJECTORY -- cosine similarity, layer by layer, between the
     ROTATED-forward hidden state and the VANILLA-forward hidden state of the
     SAME document (`cosine_trajectory`). The paper reports this starts near 0
     at the embedding layer and RISES toward ~0.7-0.8 by mid-depth -- i.e. later
     layers partially, but never fully, undo the rotation.
  3. THE ISE CONTRAST -- the SAME two measurements, run on `aside.
     learned_offset_baseline` instead of the rotation. The paper reports ISE
     collapses toward vanilla (cosine > 0.9); a rotation's displacement is
     INPUT-DEPENDENT (aside.py's own self-test) so no single downstream bias
     can cancel it the way it can cancel a translation.

THE GPU BOUNDARY -- READ BEFORE CALLING extract_hidden_states
----------------------------------------------------------------
Every function above is pure numpy/sklearn and takes ALREADY-EXTRACTED
activations as plain arrays, so it is fully unit-tested here on SYNTHETIC data
(`_self_test`, gated by `CDS_SELFTEST=1`) with NO model, NO GPU, NO network.
`extract_hidden_states` is the one function that needs a real loaded model and
tokenizer; it is written in full (lazy `torch` import) but is NOT called by this
module's self-test or by any CPU-only agent building this package. It is the
function the lead runs on GPU to turn a `data.load_role_corpus()` corpus into
the `activations_by_layer` dict every function above consumes.

CPU-only unless `extract_hidden_states` is explicitly called with a real model.
ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import numpy as np

from steering_tutorials.control_data_split.aside import (
    DEFAULT_ANGLE, learned_offset_baseline, rotate_vectors,
)
from steering_tutorials.control_data_split.inseparability import (
    estimate_provenance_bound,
)

__all__ = [
    "probe_auc", "layer_separability_sweep", "cosine_trajectory",
    "diff_of_means_offset", "provenance_floor", "extract_hidden_states",
]


# --- 1. linear-probe separability ---------------------------------------------
def probe_auc(X, y, seed: int = 0, test_size: float = 0.3) -> dict:
    """Held-out ROC-AUC of a logistic-regression probe for role label `y` on
    activations `X` (one row per document). This is the whole "is it linearly
    separable" question, made a number: 0.5 = chance, 1.0 = perfectly separable.

    Raises if `y` has fewer than 2 classes -- a probe cannot be fit or scored on
    one class, and silently returning 0.5 in that case would look like a real
    (if uninformative) measurement rather than a malformed call.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X has %d rows but y has %d" % (X.shape[0], y.shape[0]))
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("probe_auc needs BOTH roles present, got only %r"
                         % classes.tolist())

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    return {
        "auc": auc, "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "n_pos_test": int(int(yte.sum())), "n_neg_test": int(len(yte) - int(yte.sum())),
    }


def layer_separability_sweep(activations_by_layer: dict, y, seed: int = 0) -> dict:
    """`probe_auc` at every layer in `activations_by_layer` (dict[int -> [n,d]]).

    Returns dict[layer -> probe_auc(...) result], sorted-key iteration order
    preserved as insertion order of the input dict's sorted keys, so a caller
    printing this in a loop gets the layers in ascending order.
    """
    out = {}
    for layer in sorted(activations_by_layer):
        out[layer] = probe_auc(activations_by_layer[layer], y, seed=seed)
    return out


# --- 2/3. the cosine trajectory (rotation AND the ISE contrast share this) ----
def cosine_trajectory(rotated_by_layer: dict, vanilla_by_layer: dict) -> dict:
    """Per-layer mean cosine similarity between a PAIRED, ROW-ALIGNED pair of
    hidden-state dicts -- e.g. (rotated-forward, vanilla-forward) for the
    rotation condition, or (offset-forward, vanilla-forward) for the ISE
    contrast. Both dicts must share row order (same documents, same order).

    Returns dict[layer -> {"cosine_mean", "cosine_std", "n"}] over the layers
    present in BOTH dicts (a layer extracted for one condition but not the
    other is silently skipped, not padded with a fabricated number).
    """
    common_layers = sorted(set(rotated_by_layer) & set(vanilla_by_layer))
    if not common_layers:
        raise ValueError("no overlapping layers between the two hidden-state dicts")
    out = {}
    for layer in common_layers:
        R = np.asarray(rotated_by_layer[layer], dtype=np.float64)
        V = np.asarray(vanilla_by_layer[layer], dtype=np.float64)
        if R.shape != V.shape:
            raise ValueError("layer %r shape mismatch: %r vs %r"
                             % (layer, R.shape, V.shape))
        num = np.sum(R * V, axis=1)
        den = np.linalg.norm(R, axis=1) * np.linalg.norm(V, axis=1)
        den_safe = np.where(den == 0, np.nan, den)
        cos = num / den_safe
        out[layer] = {
            "cosine_mean": float(np.nanmean(cos)),
            "cosine_std": float(np.nanstd(cos)),
            "n": int(R.shape[0]),
        }
    return out


# --- helper: a training-free stand-in for the "learned" ISE offset -----------
def diff_of_means_offset(embeddings, is_data) -> np.ndarray:
    """DATA-role mean minus INSTRUCTION-role mean at one layer -- the natural
    training-free stand-in for ISE's learned offset (`aside.
    learned_offset_baseline` needs a vector; this is where it comes from
    without an SFT run). Feed the result to `extract_hidden_states(mode=
    "offset", offset=...)` as the ISE-contrast arm.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    m = np.asarray(is_data, dtype=bool)
    if m.shape[0] != X.shape[0]:
        raise ValueError("embeddings has %d rows but is_data has %d"
                         % (X.shape[0], m.shape[0]))
    if not m.any() or m.all():
        raise ValueError("need BOTH roles present to form a difference-of-means offset")
    return X[m].mean(axis=0) - X[~m].mean(axis=0)


# --- bridge to the surface-level (no-model) floor -----------------------------
def provenance_floor(instruction_texts, data_texts, seed: int = 0) -> dict:
    """Thin wrapper around `inseparability.estimate_provenance_bound` for this
    lesson's own two text pools -- the SURFACE-level (no model, no activations)
    separability floor any activation-level probe_auc should be read against.
    A layer-L probe beating this floor by little is not telling you much that
    unigram counts didn't already tell you.
    """
    return estimate_provenance_bound(instruction_texts, data_texts, seed=seed)


# --- the GPU boundary ---------------------------------------------------------
def extract_hidden_states(model, tokenizer, texts, is_data_mask, layers,
                          mode: str = "vanilla", theta: float = DEFAULT_ANGLE,
                          offset=None, batch_size: int = 16, device=None,
                          max_length: int = 512) -> dict:
    """Run REAL forward passes and mean-pool per-layer activations. NEEDS A
    LOADED MODEL -- not called by this package's CPU self-test.

    Parameters
    ----------
    model, tokenizer : a loaded HF causal LM (e.g. Gemma-3-1B) + its tokenizer.
    texts : list[str], one document per row (see `data.load_role_corpus`).
    is_data_mask : list[bool], aligned with `texts` -- the out-of-band role
        label. Every token of a DATA document is rotated/offset together (this
        lesson's role label is per-DOCUMENT, not per-token like the paper's
        instruction/data template positions -- a simplification worth stating
        plainly in the README, not hiding).
    layers : iterable[int] -- `hidden_states` indices to keep. 0 = the
        embedding-layer output (pre-block); 1..num_hidden_layers = post-block.
    mode : "vanilla"  -- no intervention (the baseline forward).
           "rotate"   -- `aside.rotate_embeddings` applied to DATA rows' token
                         embeddings before the rest of the forward pass.
           "offset"   -- `aside.learned_offset_baseline` applied instead
                         (`offset` is required, e.g. from `diff_of_means_offset`
                         computed on a prior vanilla layer-0 pass).

    Returns
    -------
    dict[int layer -> np.ndarray[len(texts), hidden_size]], each row the
    attention-mask-weighted mean over that document's non-pad tokens (the same
    pooling convention as `hello_world`'s layer-12 probe).
    """
    if mode not in ("vanilla", "rotate", "offset"):
        raise ValueError("mode must be one of vanilla/rotate/offset, got %r" % mode)
    if mode == "offset" and offset is None:
        raise ValueError(
            "mode='offset' needs an `offset` vector -- build one with "
            "diff_of_means_offset() on a prior vanilla-mode layer-0 extraction.")
    is_data_mask = list(is_data_mask)
    if len(is_data_mask) != len(texts):
        raise ValueError("texts has %d rows but is_data_mask has %d"
                         % (len(texts), len(is_data_mask)))

    import torch  # lazy: this module must import fine with no torch installed

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    emb_layer = model.get_input_embeddings()

    # Closed over by the hook: which rows in the CURRENT batch are DATA-role,
    # broadcast to every real (non-pad) token position of that row.
    _hook_state = {"mask": None}

    def _intervene_hook(_module, _inputs, output):
        tok_mask = _hook_state["mask"]
        if tok_mask is None or not bool(tok_mask.any()):
            return output
        flat = output.reshape(-1, output.shape[-1])
        flat_mask = tok_mask.reshape(-1)
        sub = flat[flat_mask].detach().to("cpu").float().numpy().astype(np.float64)
        all_true = np.ones(sub.shape[0], dtype=bool)
        if mode == "rotate":
            done = rotate_vectors(sub, theta)
        else:  # "offset"
            done = learned_offset_baseline(sub, all_true, offset)
        replacement = torch.as_tensor(done, dtype=output.dtype, device=output.device)
        new_flat = flat.clone()
        new_flat[flat_mask] = replacement
        return new_flat.reshape(output.shape)

    handle = emb_layer.register_forward_hook(_intervene_hook) if mode != "vanilla" else None

    out_by_layer = {l: [] for l in layers}
    try:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_is_data = is_data_mask[start:start + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            enc = {k: v.to(dev) for k, v in enc.items()}
            tok_mask = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
            for i, is_data in enumerate(batch_is_data):
                if is_data:
                    tok_mask[i, :] = enc["attention_mask"][i].bool()
            _hook_state["mask"] = tok_mask
            with torch.no_grad():
                # Use the BASE transformer, not the causal-LM wrapper. The
                # lm_head projects every position to the 262k-token vocabulary
                # -- [batch, seq, 262144] is ~4 GiB at batch 16 / seq 512, and
                # it OOMed the angle sweep on a 16 GB card. We only ever read
                # hidden_states, so the logits are pure waste.
                base = getattr(model, "model", model)
                out = base(**enc, output_hidden_states=True)
            attn = enc["attention_mask"].unsqueeze(-1).float()
            denom = attn.sum(1).clamp(min=1.0)
            for l in layers:
                hs = out.hidden_states[l]                     # [batch, tokens, dim]
                pooled = (hs * attn).sum(1) / denom
                out_by_layer[l].append(pooled.detach().to("cpu").float().numpy()
                                       .astype(np.float64))
    finally:
        _hook_state["mask"] = None
        if handle is not None:
            handle.remove()

    return {l: np.concatenate(v, axis=0) for l, v in out_by_layer.items()}


def _self_test() -> None:
    """SYNTHETIC self-test (CDS_SELFTEST=1): no model, no GPU, no network.

    Verifies the two comparative claims aside.py's own docstring makes, using
    the probe/cosine machinery above rather than asserting them by hand:
      * a rotation applied to synthetic embeddings IS linearly separable, and
      * an additive offset is separable TOO, but ROW-CENTERING (subtracting
        each row's own scalar mean across dimensions -- the single-bias-term
        cancellation aside.py describes) removes the offset's separability
        while leaving the rotation's intact.
    """
    rng = np.random.default_rng(0)
    n, dim = 200, 16
    half = n // 2
    is_data = np.array([False] * half + [True] * half)

    # A real token-embedding manifold is NOT isotropic noise around the origin
    # -- it sits at a substantial, structured, non-zero location. That anisotropy
    # is what the rotation actually needs: for an ISOTROPIC zero-mean Gaussian,
    # rotating a sample by ANY orthogonal matrix leaves its distribution
    # UNCHANGED (a rotationally-symmetric law is rotation-invariant), so a probe
    # could not separate rotated from vanilla no matter how real ASIDE is. `base`
    # stands in for the manifold's location; since aside.py's own self-test shows
    # cos(v, Rv) = 0 for EVERY v at theta=pi/2, R @ base is exactly orthogonal to
    # base, so rotating a base-centered class moves its mean somewhere the
    # vanilla class's mean structurally is not.
    base = np.full(dim, 2.0)
    X = base + rng.normal(scale=0.5, size=(n, dim))

    # 1. ROTATION corpus: DATA rows rotated at pi/2, INSTRUCTION rows untouched.
    X_rot = X.copy()
    X_rot[is_data] = rotate_vectors(X[is_data], DEFAULT_ANGLE)
    r_rot = probe_auc(X_rot, is_data, seed=0)
    assert r_rot["auc"] > 0.95, r_rot
    print("OK  rotated-vs-vanilla is linearly separable BY CONSTRUCTION (AUC %.3f)"
          % r_rot["auc"])

    # 2. OFFSET corpus: DATA rows get a shift UNIFORM across dimensions (c * ones)
    #    -- the exact form a single additive bias can cancel.
    c = 4.0
    offset = np.full(dim, c)
    X_off = X.copy()
    X_off[is_data] = X[is_data] + offset
    r_off = probe_auc(X_off, is_data, seed=0)
    assert r_off["auc"] > 0.95, r_off
    print("OK  offset-vs-vanilla is ALSO linearly separable (AUC %.3f)" % r_off["auc"])

    # 3. Row-centering: row_mean(x + c*1) = row_mean(x) + c, so subtracting each
    #    row's own scalar mean removes a UNIFORM shift exactly -- one bias term
    #    cancels a translation, per aside.py's docstring.
    def _row_center(A):
        return A - A.mean(axis=1, keepdims=True)

    r_off_centered = probe_auc(_row_center(X_off), is_data, seed=0)
    assert r_off_centered["auc"] < 0.65, r_off_centered
    print("OK  centering CANCELS the offset's separability (AUC %.3f -> %.3f)"
          % (r_off["auc"], r_off_centered["auc"]))

    # 4. The rotation's displacement is INPUT-DEPENDENT (aside.py's own
    #    self-test), not a constant shift, so the SAME centering does not touch it.
    r_rot_centered = probe_auc(_row_center(X_rot), is_data, seed=0)
    assert r_rot_centered["auc"] > 0.9, r_rot_centered
    print("OK  centering does NOT cancel the rotation (AUC stays %.3f) -- a "
          "change of basis is not a bias a downstream layer can subtract away"
          % r_rot_centered["auc"])

    # 5. layer_separability_sweep across synthetic "layers"
    acts = {0: X, 4: X_rot, 8: X_rot}
    sweep = layer_separability_sweep(acts, is_data, seed=0)
    assert sweep[0]["auc"] < sweep[4]["auc"], sweep
    print("OK  layer_separability_sweep: layer 0 (vanilla, AUC %.3f) < layer 4 "
          "(rotated, AUC %.3f)" % (sweep[0]["auc"], sweep[4]["auc"]))

    # 6. cosine_trajectory: at theta=pi/2 a rotated row is EXACTLY orthogonal to
    #    its vanilla self (aside.py's own self-test), while an additive offset on
    #    a zero-mean X generically leaves a positive-cosine relationship
    #    (E[dot(x, x+o)] = E||x||^2 > 0 regardless of the offset's magnitude or
    #    direction). So the rotation condition's mean cosine must be the lower one.
    cos_rot = cosine_trajectory({0: X_rot}, {0: X})
    cos_off = cosine_trajectory({0: X_off}, {0: X})
    assert cos_rot[0]["cosine_mean"] < cos_off[0]["cosine_mean"], (cos_rot, cos_off)
    print("OK  cosine_trajectory: rotation moves direction far more than an "
          "offset (mean cos %.3f vs %.3f) -- matches the paper's ISE-collapses-"
          "toward-vanilla contrast" % (cos_rot[0]["cosine_mean"], cos_off[0]["cosine_mean"]))

    # 7. diff_of_means_offset recovers the TRUE offset direction on the offset corpus.
    est = diff_of_means_offset(X_off, is_data)
    true_dir = offset / np.linalg.norm(offset)
    est_dir = est / np.linalg.norm(est)
    cos_to_true = float(np.dot(true_dir, est_dir))
    assert cos_to_true > 0.99, cos_to_true
    print("OK  diff_of_means_offset recovers the true offset direction (cos=%.4f)"
          % cos_to_true)

    # 8. probe_auc refuses a single-class y rather than returning a fake number.
    try:
        probe_auc(X, np.zeros(n, dtype=bool), seed=0)
    except ValueError as exc:
        assert "BOTH roles" in str(exc)
        print("OK  probe_auc refuses a single-class label instead of guessing 0.5")
    else:
        raise AssertionError("probe_auc accepted a single-class y")

    print("")
    print("OK -- separability.py: probe/cosine machinery verified on SYNTHETIC "
          "activations. extract_hidden_states is intentionally UNTESTED here -- "
          "it needs a real model and is the function the lead runs on GPU.")


if __name__ == "__main__":
    import os

    if os.environ.get("CDS_SELFTEST") == "1":
        _self_test()
    else:
        print("Set CDS_SELFTEST=1 to run the synthetic self-test (no model, "
              "no GPU, no network).")
