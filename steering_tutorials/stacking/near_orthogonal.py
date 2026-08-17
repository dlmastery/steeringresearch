"""near_orthogonal.py — the THIRD clause of the CLAUDE.md sec.9 decision rule.

The sec.9 rule has three clauses. This lesson measured two of them:

    different SITE ................................... STACK    (run_stacking, run_hillclimb)
    same SITE + same DIRECTION + different OP ........ COMPETE  (run_stacking, run_hillclimb)
    near-orthogonal DIRECTIONS ....................... STACK **UNTIL THE NORM
                                                       BUDGET IS SPENT**   <- HERE

``run_hillclimb`` touches the third clause with prior **C**, but only at its
degenerate endpoint: C is *exactly* orthogonal (cos = 3e-08), it is added once,
and nothing in that ladder ever asks the clause's actual question — *how near is
"near", and when do you stop adding?* A clause with a stopping condition in it
cannot be tested by a ladder that never stops.

This module supplies the two things that were missing:

1. **A measured cosine axis.** :func:`rotate_toward` builds a direction at an
   EXACT, chosen cosine to the refusal direction, so "near-orthogonal" becomes a
   dial rather than an adjective. Sweeping it from 0.0 to 1.0 walks continuously
   from the STACK clause (orthogonal directions) into the COMPETE clause (same
   direction) and shows where — if anywhere — the boundary sits. Every cosine is
   re-measured from the vectors that were actually injected
   (:func:`gram_cosines`), never assumed from the construction.

2. **A norm-budget account that says when to stop.** For unit directions
   ``u_1..u_k`` injected at ONE site with equal relative step ``alpha``, the
   first-order composed displacement is

       ||dh|| / ||h||  =  alpha * sqrt( 1^T G 1 ),    G = pairwise cosine matrix

   (:func:`predicted_budget`). That single expression IS the stack-vs-compete
   mechanism in closed form: k parallel directions cost ``k*alpha`` (they
   double-count, the COMPETE clause), k orthogonal ones cost only
   ``sqrt(k)*alpha`` (they stack), and near-orthogonal ones sit between. So the
   budget ceiling converts directly into a CAPACITY — how many more directions
   fit (:func:`orthogonal_capacity`) — which is the "until the norm budget is
   spent" half of the clause, made arithmetic.

The additive ladder built on top (:func:`replay_ladder`) adds exactly ONE
direction per rung and is **revertible**: a rung that fails its pre-registered
gate is DROPPED (the ladder reverts to the last kept prior set and tries the next
candidate) rather than carried forward, and a rung that breaks the budget STOPS
the ladder. The forbidden "everything on" hybrid is therefore unreachable by
construction — the ladder can only ever hold priors that passed their own gate.

References
----------
  Panickssery/Rimsky et al. 2023, 'Steering Llama 2 via Contrastive Activation
    Addition' (arXiv:2312.06681) — the additive edit each injector applies.
  corpus/steering-stackable-vs-competing-analysis.md sec.1-4 — the site /
    direction / norm-budget taxonomy this module operationalises.
  corpus/steering-first-principles-v2-with-PSR-and-rogue-scalpel.md — the N5
    cumulative-||dh||/||h|| leading indicator used as the budget.

No model is loaded at import time. Run the CPU self-test with
``python -m steering_tutorials.stacking.near_orthogonal``.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- #
# Pre-registered constants for the near-orthogonality band. A direction pair is
# "near-orthogonal" (sec.9 STACK clause) only below NEAR_ORTHO_MAX_COS; at or
# above it the pair is treated as the SAME direction (sec.9 COMPETE clause) and
# is refused admission to the ladder. The value is deliberately far below the
# 0.95 "same direction" bar the field often uses, because HC-S measured that
# cos=0.966 directions are NOT interchangeable (+3.33 PPL) — if 0.95 is not
# enough to call two directions the same, it is certainly not enough to call two
# directions different.
# --------------------------------------------------------------------------- #
NEAR_ORTHO_MAX_COS = 0.35


# --------------------------------------------------------------------------- #
# 1. Directions at a CHOSEN cosine — "near-orthogonal" as a dial
# --------------------------------------------------------------------------- #
def rotate_toward(v_unit: np.ndarray, u_unit: np.ndarray,
                  cos_target: float) -> np.ndarray:
    """Unit direction ``w`` with ``dot(w, v_unit) == cos_target`` exactly.

    ``w = cos_target * v + sqrt(1 - cos_target^2) * u`` with ``u`` orthonormal to
    ``v``. Because ``|v| = |u| = 1`` and ``v.u = 0``, ``|w| = 1`` and ``w.v =
    cos_target`` identically — no optimisation, no approximation.

    Parameters
    ----------
    v_unit, u_unit : np.ndarray ``[hidden]``
        Unit vectors; ``u_unit`` must be orthogonal to ``v_unit`` (asserted).
    cos_target : float in ``[-1, 1]``
        The cosine the returned direction will have with ``v_unit``.
    """
    v = np.asarray(v_unit, dtype=np.float64).reshape(-1)
    u = np.asarray(u_unit, dtype=np.float64).reshape(-1)
    v = v / np.linalg.norm(v)
    u = u - (u @ v) * v
    nu = np.linalg.norm(u)
    if nu < 1e-8:
        raise ValueError("u_unit is parallel to v_unit - no rotation plane")
    u = u / nu
    if not -1.0 <= float(cos_target) <= 1.0:
        raise ValueError(f"cos_target {cos_target} outside [-1, 1]")
    c = float(cos_target)
    s = math.sqrt(max(0.0, 1.0 - c * c))
    w = c * v + s * u
    w = w / np.linalg.norm(w)
    assert abs(float(w @ v) - c) < 1e-6, "rotate_toward missed its cosine"
    return w.astype(np.float32)


def cosine_family(v_unit: np.ndarray, u_unit: np.ndarray,
                  cosines: "list[float]") -> "list[dict]":
    """One direction per requested cosine, each carrying its MEASURED cosine.

    Returns ``[{"cos_target": float, "cos_measured": float, "vector": ndarray},
    ...]``. ``cos_measured`` is re-derived from the returned vector, so the
    report never quotes the cosine it *asked* for in place of the one it got.
    """
    v = np.asarray(v_unit, dtype=np.float32).reshape(-1)
    out = []
    for c in cosines:
        w = rotate_toward(v, u_unit, c)
        out.append({"cos_target": float(c),
                    "cos_measured": float(np.dot(w.astype(np.float64),
                                                 v.astype(np.float64))),
                    "vector": w})
    return out


def orthonormal_complement_basis(acts: np.ndarray, v_unit: np.ndarray,
                                 k: int) -> np.ndarray:
    """``k`` orthonormal directions spanning real activation variance, all ⟂ ``v``.

    The refusal axis is projected out of the centred activations, then the top
    ``k`` right-singular vectors of what remains are Gram-Schmidt'd against
    ``v_unit`` and against each other. Each returned row is therefore (a) a
    direction the model's activations actually vary along — not a random draw —
    and (b) exactly orthogonal to the refusal axis and to its siblings.

    These are the raw material for the ladder: rotating each of them a little way
    back toward ``v`` (:func:`rotate_toward`) yields a family of *near*-orthogonal
    directions whose full cosine geometry is known and checkable.

    Same honesty caveat as ``hillclimb.orthogonal_direction``: this pool carries
    ONE labelled contrast, so these are ORTHOGONALITY CONTROLS, not second
    concepts, and must be reported as such.
    """
    X = np.asarray(acts, dtype=np.float64)
    v = np.asarray(v_unit, dtype=np.float64).reshape(-1)
    v = v / np.linalg.norm(v)
    if k < 1:
        raise ValueError("k must be >= 1")

    X = X - X.mean(axis=0, keepdims=True)
    X = X - np.outer(X @ v, v)
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)

    basis: list[np.ndarray] = []
    for row in vt:
        c = row - (row @ v) * v
        for b in basis:
            c = c - (c @ b) * b
        n = np.linalg.norm(c)
        if n < 1e-6:                     # numerically exhausted; skip
            continue
        basis.append(c / n)
        if len(basis) == k:
            break
    if len(basis) < k:
        raise ValueError(f"only {len(basis)} independent directions available, need {k}")
    B = np.vstack(basis)
    assert np.abs(B @ v).max() < 1e-6, "complement basis not orthogonal to v"
    return B.astype(np.float32)


def gram_cosines(vectors: "list[np.ndarray]") -> np.ndarray:
    """Pairwise cosine matrix of the vectors AS INJECTED (unit-normalised copy).

    This is the measurement, not the construction: whatever the directions were
    built to be, the ladder reports the cosines of the arrays it actually handed
    to the steering hooks.
    """
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    M = np.vstack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors])
    M = M / np.linalg.norm(M, axis=1, keepdims=True).clip(1e-12)
    return M @ M.T


def max_abs_offdiag(gram: np.ndarray) -> float:
    """Largest |cosine| between two DIFFERENT directions (0.0 for k < 2)."""
    g = np.asarray(gram, dtype=np.float64)
    if g.shape[0] < 2:
        return 0.0
    off = g.copy()
    np.fill_diagonal(off, 0.0)
    return float(np.abs(off).max())


# --------------------------------------------------------------------------- #
# 2. The norm-budget account — "stack until the budget is spent", in arithmetic
# --------------------------------------------------------------------------- #
def predicted_budget(gram: np.ndarray, alpha: float) -> float:
    """First-order composed displacement ratio ``||dh||/||h||`` for the stack.

    For unit directions ``u_i`` injected at ONE site, each with relative step
    ``alpha`` (``h <- h + alpha*||h||*u_i``), the composed delta to first order is
    ``alpha*||h||*sum_i u_i``, so

        ||dh|| / ||h||  =  alpha * sqrt( sum_ij cos(u_i, u_j) )
                        =  alpha * sqrt( 1^T G 1 ).

    Limits worth reading off:
      * ``G = J`` (all parallel)   -> ``alpha * k``        the COMPETE cost
      * ``G = I`` (all orthogonal) -> ``alpha * sqrt(k)``  the STACK cost
    so the ratio of the two, ``sqrt(k)``, is exactly what orthogonality buys.

    FIRST-ORDER: the hooks fire sequentially, so each one's ``||h||`` reference
    is already nudged by the previous one — an O(alpha^2) correction this formula
    drops. The runner always reports the MEASURED budget beside this prediction;
    the gap between them is data, not error.
    """
    g = np.asarray(gram, dtype=np.float64)
    if g.size == 0:
        return 0.0
    total = float(np.ones(g.shape[0]) @ g @ np.ones(g.shape[0]))
    return float(alpha) * math.sqrt(max(0.0, total))


def orthogonal_capacity(alpha: float, ceiling: float,
                        spent: float = 0.0) -> int:
    """How many MORE exactly-orthogonal ``alpha``-steps fit under ``ceiling``.

    Orthogonal additions accumulate in quadrature, so after ``k`` more steps the
    budget is ``sqrt(spent^2 + k*alpha^2)``; requiring that to stay ``<= ceiling``
    gives ``k = floor((ceiling^2 - spent^2) / alpha^2)``. Returns 0 once the
    budget is already spent. This is the clause's stopping condition stated
    before any generation is run — a prediction, checkable against where the
    measured ladder actually stops.
    """
    a = float(alpha)
    if a <= 0:
        raise ValueError("alpha must be > 0")
    head = float(ceiling) ** 2 - float(spent) ** 2
    if head <= 0:
        return 0
    return int(math.floor(head / (a * a)))


def admit_direction(candidate: np.ndarray, admitted: "list[np.ndarray]",
                    alpha: float, ceiling: float,
                    max_cos: float = NEAR_ORTHO_MAX_COS) -> dict:
    """Pre-flight admission test for one candidate direction. NO model needed.

    Two pre-registered reasons to refuse, both from sec.9:

    ``NOT_NEAR_ORTHOGONAL``
        the candidate's largest |cosine| against an already-admitted direction is
        >= ``max_cos``. Above that bar this is the *same direction* case, which
        sec.9 classifies as COMPETE — it does not belong in a stack ladder.
    ``BUDGET_EXCEEDED``
        the predicted budget WITH the candidate exceeds ``ceiling``. The budget is
        spent; sec.9 says stop.

    Returns a diagnostics dict (``admit``, ``reason``, cosines, predicted budgets
    before/after, remaining capacity) so a refusal is always explained by the
    numbers that caused it.
    """
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    cand = cand / np.linalg.norm(cand).clip(1e-12)
    cos_vs = [float(cand @ (np.asarray(a_, dtype=np.float64).reshape(-1)
                            / np.linalg.norm(a_).clip(1e-12)))
              for a_ in admitted]
    max_c = max((abs(c) for c in cos_vs), default=0.0)

    g_before = gram_cosines(list(admitted))
    g_after = gram_cosines(list(admitted) + [cand])
    b_before = predicted_budget(g_before, alpha)
    b_after = predicted_budget(g_after, alpha)

    reason = None
    if max_c >= max_cos:
        reason = "NOT_NEAR_ORTHOGONAL"
    elif b_after > ceiling:
        reason = "BUDGET_EXCEEDED"
    return {
        "admit": reason is None,
        "reason": reason,
        "cos_vs_admitted": cos_vs,
        "max_abs_cos": max_c,
        "max_cos_bar": float(max_cos),
        "predicted_budget_before": b_before,
        "predicted_budget_after": b_after,
        "budget_ceiling": float(ceiling),
        "orthogonal_capacity_left": orthogonal_capacity(alpha, ceiling, b_after),
    }


# --------------------------------------------------------------------------- #
# 3. The revertible additive ladder
# --------------------------------------------------------------------------- #
#: The pre-registered stop/revert rules, in the order they are evaluated. Kept as
#: data (not buried in an ``if``) so the report can print the rule it applied.
STOP_RULES = {
    "BUDGET": "measured N5 budget > budget_ceiling -> DROP the rung and STOP "
              "(more directions can only spend more)",
    "COHERENCE": "harmful gibberish rose > coherence_tol vs the last KEPT rung "
                 "-> DROP the rung, revert, try the next candidate",
    "COMPETE": "harmful refusal fell > compete_tol vs the last KEPT rung -> the "
               "added direction competes rather than stacks -> DROP, revert",
    "SELECTIVITY": "benign refusal rose > selectivity_tol vs the last KEPT rung "
                   "-> over-refusal on harmless prompts -> DROP, revert",
}

#: Defaults for :func:`replay_ladder`. Pre-registered in the module that runs it.
LADDER_PARAMS = {
    "budget_ceiling": 0.20,
    "coherence_tol": 0.05,
    "compete_tol": 0.0,
    "selectivity_tol": 0.10,
    "max_consecutive_drops": 2,
}


def _get(cell: "dict | None", arm: str, field: str):
    if not cell:
        return None
    a = cell.get(arm)
    return None if not a else a.get(field)


def replay_ladder(cells: dict, order: "list[str]", params: dict) -> dict:
    """Apply the KEEP/DROP/STOP rules to recorded rung cells. PURE — no model.

    ``cells`` maps a rung key to ``{"harmful": {...}, "benign": {...} | None,
    "norm_budget": float, "added": str, ...}``; ``order`` is the base rung
    followed by the candidate rungs in the order they were measured.

    The walk is **revertible**: every comparison is against the last rung that
    was KEPT, so a dropped rung leaves no trace in the reference state — the next
    candidate is judged as if the failed one had never been added. That is what
    "each rung is revertible: if a rung fails, it is dropped, not carried" means
    operationally.

    Returns ``{"rows": [...], "kept": [...], "stopped_at": key|None,
    "binding_constraint": str|None, "params": params}``. A rung whose cell is
    missing (never measured, e.g. the run stopped early) is reported
    ``verdict="NOT_MEASURED"`` — never silently skipped and never imputed.
    """
    p = {**LADDER_PARAMS, **(params or {})}
    rows: list[dict] = []
    kept: list[str] = []
    ref_key = order[0] if order else None
    stopped_at = None
    binding = None
    consecutive_drops = 0

    for i, key in enumerate(order):
        cell = cells.get(key)
        row = {
            "rung": key,
            "added": (cell or {}).get("added"),
            "priors": (cell or {}).get("priors"),
            "cos_to_refusal": (cell or {}).get("cos_to_refusal"),
            "max_abs_cos_within_stack": (cell or {}).get("max_abs_cos_within_stack"),
            "predicted_budget": (cell or {}).get("predicted_budget"),
            "norm_budget": (cell or {}).get("norm_budget"),
            "refusal_rate": _get(cell, "harmful", "refusal_rate"),
            "gibberish_rate": _get(cell, "harmful", "gibberish_rate"),
            "benign_refusal_rate": _get(cell, "benign", "refusal_rate"),
            "benign_gibberish_rate": _get(cell, "benign", "gibberish_rate"),
            "reference_rung": None if i == 0 else ref_key,
        }
        if stopped_at is not None:
            row["verdict"] = "NOT_EVALUATED"
            row["failed_rules"] = []
            rows.append(row)
            continue
        if cell is None:
            row["verdict"] = "NOT_MEASURED"
            row["failed_rules"] = []
            rows.append(row)
            continue
        if i == 0:
            row["verdict"] = "BASE"
            row["failed_rules"] = []
            kept.append(key)
            ref_key = key
            rows.append(row)
            continue

        ref = cells.get(ref_key)
        d_ref = _get(cell, "harmful", "refusal_rate")
        r_ref = _get(ref, "harmful", "refusal_rate")
        d_gib = _get(cell, "harmful", "gibberish_rate")
        r_gib = _get(ref, "harmful", "gibberish_rate")
        d_ben = _get(cell, "benign", "refusal_rate")
        r_ben = _get(ref, "benign", "refusal_rate")

        row["marginal_refusal"] = (None if d_ref is None or r_ref is None
                                   else float(d_ref - r_ref))
        row["marginal_gibberish"] = (None if d_gib is None or r_gib is None
                                     else float(d_gib - r_gib))
        row["marginal_benign_refusal"] = (None if d_ben is None or r_ben is None
                                          else float(d_ben - r_ben))

        failed: list[str] = []
        budget = cell.get("norm_budget")
        if budget is not None and budget > p["budget_ceiling"]:
            failed.append("BUDGET")
        if row["marginal_gibberish"] is not None and \
                row["marginal_gibberish"] > p["coherence_tol"]:
            failed.append("COHERENCE")
        if row["marginal_refusal"] is not None and \
                row["marginal_refusal"] < -p["compete_tol"]:
            failed.append("COMPETE")
        if row["marginal_benign_refusal"] is not None and \
                row["marginal_benign_refusal"] > p["selectivity_tol"]:
            failed.append("SELECTIVITY")

        row["failed_rules"] = failed
        row["verdict"] = "KEEP" if not failed else "DROP"
        rows.append(row)

        if not failed:
            kept.append(key)
            ref_key = key
            consecutive_drops = 0
        else:
            consecutive_drops += 1
            if "BUDGET" in failed:
                stopped_at, binding = key, "BUDGET"
            elif consecutive_drops >= p["max_consecutive_drops"]:
                stopped_at, binding = key, "CONSECUTIVE_DROPS"

    if binding is None:
        # Deliberately narrow: this function only sees MEASURED rungs. A ladder
        # can also be stopped before measurement by the pre-flight admission
        # test, which the caller layers on top (see run_near_orthogonal).
        binding = "NONE (no MEASURED rung triggered a stop rule)"
    return {"rows": rows, "kept": kept, "stopped_at": stopped_at,
            "binding_constraint": binding, "params": p, "stop_rules": STOP_RULES}


# --------------------------------------------------------------------------- #
# CPU self-test — no model, no GPU, no network.
# Run: python -m steering_tutorials.stacking.near_orthogonal
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # ---- (a) rotate_toward hits its cosine EXACTLY, at every requested value --
    v = rng.normal(size=64).astype(np.float32)
    v /= np.linalg.norm(v)
    u = rng.normal(size=64).astype(np.float32)
    u -= (u @ v) * v
    u /= np.linalg.norm(u)
    for c in (0.0, 0.05, 0.2, 0.5, 0.75, 0.95, 1.0):
        w = rotate_toward(v, u, c)
        assert abs(float(w @ v) - c) < 1e-6, f"cosine {c} not achieved"
        assert abs(float(np.linalg.norm(w)) - 1.0) < 1e-6, "w not unit"
    fam = cosine_family(v, u, [0.0, 0.4, 0.9])
    assert [round(f["cos_measured"], 6) for f in fam] == [0.0, 0.4, 0.9]

    # ---- (b) the complement basis is orthonormal AND orthogonal to v ---------
    acts = rng.normal(size=(200, 64)).astype(np.float32) + 4.0 * np.outer(
        rng.normal(size=200), v)                 # refusal-dominated on purpose
    B = orthonormal_complement_basis(acts, v, k=4)
    assert B.shape == (4, 64)
    assert np.abs(B.astype(np.float64) @ v.astype(np.float64)).max() < 1e-6
    G = gram_cosines(list(B))
    assert np.abs(G - np.eye(4)).max() < 1e-5, "complement basis not orthonormal"
    assert max_abs_offdiag(G) < 1e-5

    # ---- (c) the budget account matches a DIRECT numeric sum -----------------
    # Build k near-orthogonal directions at cos=0.2 to v and confirm that
    # alpha*sqrt(1^T G 1) equals ||alpha * sum u_i|| computed the slow way.
    alpha = 0.08
    ws = [rotate_toward(v, B[i], 0.2) for i in range(4)]
    G4 = gram_cosines(ws)
    direct = float(np.linalg.norm(alpha * np.sum(
        np.vstack([w.astype(np.float64) for w in ws]), axis=0)))
    # 1e-6, not 1e-9: the directions are stored float32 (as the hooks receive
    # them) so the two routes differ by float32 rounding, not by algebra.
    assert abs(predicted_budget(G4, alpha) - direct) < 1e-6, "budget formula wrong"
    # The two textbook limits.
    assert abs(predicted_budget(np.ones((4, 4)), alpha) - 4 * alpha) < 1e-9
    assert abs(predicted_budget(np.eye(4), alpha) - math.sqrt(4) * alpha) < 1e-9
    # cos=0.2 siblings cost MORE than orthogonal ones but far less than parallel.
    assert math.sqrt(4) * alpha < predicted_budget(G4, alpha) < 4 * alpha

    # ---- (d) capacity is the ceiling restated as a count ---------------------
    assert orthogonal_capacity(0.08, 0.20, 0.0) == 6      # floor(0.04/0.0064)
    assert orthogonal_capacity(0.08, 0.20, 0.20) == 0     # already spent
    assert orthogonal_capacity(0.08, 0.20, 0.30) == 0     # over-spent, never < 0

    # ---- (e) admission refuses for the right reason, with numbers ------------
    ok = admit_direction(ws[1], [ws[0]], alpha, ceiling=0.20)
    assert ok["admit"] and ok["reason"] is None
    near_dup = rotate_toward(v, B[0], 0.99)
    bad = admit_direction(near_dup, [rotate_toward(v, B[0], 0.98)], alpha, 0.20)
    assert bad["admit"] is False and bad["reason"] == "NOT_NEAR_ORTHOGONAL"
    broke = admit_direction(ws[1], [ws[0]], alpha, ceiling=0.05)
    assert broke["admit"] is False and broke["reason"] == "BUDGET_EXCEEDED"

    # ---- (f) the ladder is REVERTIBLE: a dropped rung is not carried ---------
    def cell(ref, gib, ben_ref=0.10, budget=0.10, added="w"):
        return {"added": added, "norm_budget": budget,
                "harmful": {"refusal_rate": ref, "gibberish_rate": gib},
                "benign": {"refusal_rate": ben_ref, "gibberish_rate": 0.2}}

    cells = {
        "L0": cell(0.30, 0.40),
        "L1": cell(0.36, 0.42),          # KEEP  (refusal up, coherence fine)
        "L2": cell(0.20, 0.60),          # DROP  (coherence +0.18, refusal -0.16)
        "L3": cell(0.38, 0.43),          # KEEP  — compared to L1, NOT to L2
    }
    out = replay_ladder(cells, ["L0", "L1", "L2", "L3"], {})
    by = {r["rung"]: r for r in out["rows"]}
    assert by["L1"]["verdict"] == "KEEP"
    assert by["L2"]["verdict"] == "DROP"
    assert set(by["L2"]["failed_rules"]) == {"COHERENCE", "COMPETE"}
    assert by["L3"]["reference_rung"] == "L1", "ladder did not revert to L1"
    assert abs(by["L3"]["marginal_refusal"] - 0.02) < 1e-9   # 0.38 - 0.36, not - 0.20
    assert out["kept"] == ["L0", "L1", "L3"]

    # ---- (g) a budget break STOPS the ladder; later rungs are NOT_EVALUATED --
    cells2 = {"L0": cell(0.30, 0.40, budget=0.08),
              "L1": cell(0.35, 0.41, budget=0.25),      # over the 0.20 ceiling
              "L2": cell(0.40, 0.41, budget=0.30)}
    out2 = replay_ladder(cells2, ["L0", "L1", "L2"], {})
    by2 = {r["rung"]: r for r in out2["rows"]}
    assert by2["L1"]["verdict"] == "DROP" and "BUDGET" in by2["L1"]["failed_rules"]
    assert by2["L2"]["verdict"] == "NOT_EVALUATED"
    assert out2["stopped_at"] == "L1" and out2["binding_constraint"] == "BUDGET"

    # ---- (h) over-refusal on BENIGN prompts drops a rung on its own ----------
    cells3 = {"L0": cell(0.30, 0.40, ben_ref=0.10),
              "L1": cell(0.40, 0.40, ben_ref=0.45)}     # +0.35 benign refusal
    by3 = {r["rung"]: r for r in replay_ladder(cells3, ["L0", "L1"], {})["rows"]}
    assert by3["L1"]["verdict"] == "DROP"
    assert by3["L1"]["failed_rules"] == ["SELECTIVITY"]

    # ---- (i) a missing cell is reported, never imputed -----------------------
    by4 = {r["rung"]: r for r in
           replay_ladder({"L0": cell(0.3, 0.4)}, ["L0", "L1"], {})["rows"]}
    assert by4["L1"]["verdict"] == "NOT_MEASURED"
    assert by4["L1"]["refusal_rate"] is None

    print("[self-test] OK - rotate_toward hits exact cosines; complement basis "
          "orthonormal; budget formula == direct norm and matches the k / sqrt(k) "
          "limits; capacity arithmetic sound; admission refuses with a reason; "
          "ladder reverts on DROP, stops on BUDGET, and never imputes a missing cell.")


if __name__ == "__main__":
    _self_test()
