"""run_nonident.py — the payoff: low cosine, similar effect => non-identifiable.

Pipeline
--------
1. BUILD.  Load the abliterated Gemma. Read activations from N_EXTRACT harmful +
   benign prompts and build the K candidate directions (``vectors.py``). Save
   them and their pairwise cosine matrix.

2. STEER + JUDGE.  On a DISJOINT held-out set of harmful prompts, steer with each
   candidate at the SAME matched relative alpha and have the judge label each
   output REFUSAL / COMPLIANCE / GIBBERISH. Because every candidate is
   unit-normalized and steering is norm-relative, all candidates get an
   equal-magnitude nudge — so any difference in refusal rate is due to
   DIRECTION, not strength. Also record the unsteered baseline.

3. REPORT.  Cross-tabulate (pairwise cosine) against (per-direction refusal
   rate). The headline statistic: the MINIMUM pairwise cosine among the
   directions that are "effective" (reach >= EFFECTIVE_FRACTION of the best
   refusal rate). If two directions with cosine well below 1.0 both steer, the
   refusal direction is non-identifiable. Save results.json + a two-panel PNG.

Everything model-touching lives under ``main()`` so ``import run_nonident`` is a
no-op (safe for tests / a webapp).

RESULTS SCHEMA (kept in sync with README)
-----------------------------------------
{
  "model_id": str, "layer": int, "matched_alpha": float,
  "n_extract": int, "n_eval": int,
  "names": [str, ...],
  "cosine_matrix": [[float, ...], ...],           # K x K, order = names
  "directions": [{"name": str, "recipe": str, "pooling": str}, ...],
  "baseline": {"refusal_rate": float, "compliance_rate": float,
               "gibberish_rate": float, "n": int},
  "per_direction": [
      {"name": str, "refusal_rate": float, "compliance_rate": float,
       "gibberish_rate": float, "n": int}, ...
  ],
  "nonident": {"best_refusal": float, "effective_threshold": float,
               "effective_names": [str, ...], "min_cosine_effective": float,
               "refusal_spread": float, "verdict": str},
  "examples": [{"prompt": str, "direction": str,
                "baseline_response": str, "steered_response": str,
                "baseline_verdict": str, "steered_verdict": str}, ...],
  "plots": {"nonident": "nonident_<key>.png"},

  # --- added 2026-08-22 with the alpha sweep. Every key ABOVE keeps its exact
  # meaning: per_direction / nonident are still the MATCHED_ALPHA cell, so the
  # registered headline is unchanged by the presence of other doses. ---------
  "alpha_sweep": {                       # keyed by alpha, e.g. "0.06"
    "<alpha>": {"alpha": float, "n": int, "is_headline": bool,
                "baseline": {...},       # alpha-free; the same object each time
                "per_direction": [ ...same shape as the top-level one... ],
                "nonident": { ...same shape as the top-level one... }}
  },
  "alpha_sweep_alphas": [float, ...],
  "alpha_sweep_note": str,
  "n_eval_requested": int, "n_eval_default": int, "capped": bool, "tier": str,
  "checkpoint_dir": str, "run_fingerprint": str,
  "partial": bool                        # True in the after-each-alpha writes
}

WHY THE SWEEP IS NOT THE HEADLINE
---------------------------------
``ALPHAS`` sat in config.py as DEAD CODE until 2026-08-22 -- the runner
referenced only ``MATCHED_ALPHA`` -- so the lesson was catalogued as "sweep
wired but never executed" when there was nothing to execute. Wiring it raises an
obvious temptation: report whichever dose looks best. That would be selecting the
dose on the outcome, so the sweep is written to a SEPARATE block and the
top-level numbers stay pinned to ``MATCHED_ALPHA``. What the sweep buys is the
falsifier's other half: if the family only coheres at one dose, that is a
knife-edge, and the third plot panel shows it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from . import config as C


# --------------------------------------------------------------------------- #
# Run size, alpha grid, and paths -- all resolved in config.py, read here.
#
# The cost is (1 baseline + K directions x len(SWEEP_ALPHAS)) x N_EVAL
# generations, each followed by a judge call. At the pre-registered size that is
# 150 + 6 x 3 x 150 = 2850 generations at 48 tokens. Two defences, because on
# this host a long job is reaped rather than finished (CLAUDE.md 18.5):
#
#   * NONIDENT_N_EVAL / NONIDENT_ALPHAS shrink the run into a window you have.
#   * every cell is CHECKPOINTED per prompt, so a reap costs one generation and
#     the next launch resumes where it stopped.
#
# Both are keyed into the artifact name, so a shrunken screening run can never
# land on the pre-registered run's path.
# --------------------------------------------------------------------------- #
N_EVAL = int(C.N_EVAL)
SWEEP_ALPHAS: list[float] = list(C.SWEEP_ALPHAS)
RESULTS_PATH: Path = C.RESULTS_PATH
PLOT_PATH: Path = C.PLOT_PATH
CKPT_DIR: Path = C.CKPT_DIR


# --------------------------------------------------------------------------- #
# Checkpointing — pure filesystem, no model. Unit-tested in __main__.
#
# The unit of work is a CELL: one (alpha, direction) pair, or the alpha-free
# baseline. Each cell is a JSONL file whose Nth line is the judged result for the
# Nth eval prompt, appended and flushed as soon as it exists. So the granularity
# of loss under a reap is ONE generation, not one cell and certainly not the run
# (CLAUDE.md 18.5: "checkpoint at the granularity of the most expensive
# irreversible step").
#
# Two things make the resume safe rather than merely convenient:
#
#   * a MANIFEST pins the run fingerprint (model, layer, sizes, seed, alpha grid,
#     the exact eval prompts, the judge id). Resuming across a change in any of
#     them would splice two experiments together and report the seam as a result,
#     so a mismatch REFUSES rather than warns.
#   * :func:`_load_cell` keeps only the contiguous 0..k prefix and drops a torn
#     final line. A process killed mid-write leaves half a line; a loader that
#     accepted it would resume from the wrong index and silently misalign every
#     verdict after it against its prompt.
# --------------------------------------------------------------------------- #
def cell_id(alpha: float | None, name: str) -> str:
    """Stable filename stem for one cell. ``None`` alpha == the baseline cell."""
    if alpha is None:
        return "baseline"
    return "%s__%s" % (C.alpha_tag([alpha]), name)


def _cell_path(ckpt_dir, alpha: float | None, name: str) -> Path:
    return Path(ckpt_dir) / ("cell_%s.jsonl" % cell_id(alpha, name))


def _load_cell(path) -> list[dict]:
    """Read a cell's completed records: the contiguous ``i = 0, 1, 2, ...`` run.

    Anything after the first gap or the first unparseable line is discarded, so
    a torn tail costs one generation instead of corrupting the alignment between
    verdicts and prompts.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                break                      # torn final write -- stop here.
            if not isinstance(rec, dict) or rec.get("i") != len(out):
                break                      # a gap: everything after it is unusable.
            out.append(rec)
    return out


def _append_cell(path, rec: dict) -> None:
    """Append one JSON record and force it to disk before generating the next.

    Flushed AND fsync'd: a checkpoint still sitting in the OS write buffer when
    the process is killed is not a checkpoint.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def check_manifest(ckpt_dir, manifest: dict) -> str:
    """Reconcile ``manifest`` with any checkpoint already in ``ckpt_dir``.

    Returns ``"fresh"`` (nothing on disk, manifest written) or ``"resume"``.
    Raises :class:`SystemExit` when a checkpoint exists under a DIFFERENT
    fingerprint -- naming both fingerprints and the one-line fix.
    """
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    mpath = d / "manifest.json"
    if not mpath.exists():
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return "fresh"

    prev = json.loads(mpath.read_text(encoding="utf-8"))
    if prev.get("fingerprint") == manifest.get("fingerprint"):
        return "resume"

    differing = sorted(k for k in set(prev) | set(manifest)
                       if k != "fingerprint" and prev.get(k) != manifest.get(k))
    raise SystemExit(
        "REFUSING TO RESUME [non_identifiability]: the checkpoint in %s was "
        "written under fingerprint %s but this run is %s.\n"
        "  Differing fields: %s\n"
        "  Resuming across that boundary would splice two different experiments "
        "into one artifact. Delete the checkpoint directory to start over, or "
        "restore the config that produced it."
        % (d, str(prev.get("fingerprint"))[:12],
           str(manifest.get("fingerprint"))[:12],
           ", ".join(differing) or "(none visible -- an eval-prompt or judge change)"))


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — unit-tested in __main__.
# --------------------------------------------------------------------------- #
def _rates(verdicts: list[str]) -> dict[str, float]:
    """Fraction of REFUSAL / COMPLIANCE / GIBBERISH among a verdict list."""
    n = max(1, len(verdicts))
    return {
        "refusal_rate": verdicts.count("REFUSAL") / n,
        "compliance_rate": verdicts.count("COMPLIANCE") / n,
        "gibberish_rate": verdicts.count("GIBBERISH") / n,
    }


def summarize_nonidentifiability(
    per_direction: list[dict],
    names: list[str],
    cosine: np.ndarray,
    effective_fraction: float,
    control_name: str = "random_in_pcspan",
) -> dict:
    """Compute the headline non-identifiability statistics.

    The claim is supported when several directions of LOW mutual cosine all reach
    a SIMILAR (high) refusal rate. We quantify that as:

      * ``best_refusal``          — the highest per-direction refusal rate.
      * ``effective_threshold``   — ``effective_fraction`` of ``best_refusal``.
      * ``effective_names``       — directions at/above that threshold, EXCLUDING
                                    the random control (whose job is to *fail*).
      * ``min_cosine_effective``  — the minimum pairwise cosine among those
                                    effective directions. LOW here + several
                                    effective directions == non-identifiable.
      * ``refusal_spread``        — max-min refusal rate among effective
                                    directions (small == "similar effect").

    ``verdict`` is a plain-language read, not a statistical claim (see caveats).
    """
    idx = {n: i for i, n in enumerate(names)}
    rate = {d["name"]: d["refusal_rate"] for d in per_direction}

    contrast = [d for d in per_direction if d["name"] != control_name]
    best_refusal = max((d["refusal_rate"] for d in contrast), default=0.0)
    threshold = effective_fraction * best_refusal

    effective = [d["name"] for d in contrast if d["refusal_rate"] >= threshold
                 and best_refusal > 0.0]

    # Minimum pairwise cosine among effective directions (off-diagonal only).
    min_cos = 1.0
    for a in range(len(effective)):
        for b in range(a + 1, len(effective)):
            c = float(cosine[idx[effective[a]], idx[effective[b]]])
            min_cos = min(min_cos, c)
    if len(effective) < 2:
        min_cos = float("nan")   # need >= 2 directions to talk about "same effect"

    eff_rates = [rate[n] for n in effective]
    spread = (max(eff_rates) - min(eff_rates)) if eff_rates else 0.0

    if len(effective) >= 2 and min_cos < 0.9:
        verdict = (f"SUPPORTED (screening): {len(effective)} directions with "
                   f"pairwise cosine down to {min_cos:.2f} reach within "
                   f"{effective_fraction:.0%} of the best refusal rate")
    elif len(effective) >= 2:
        verdict = ("MIXED (screening): several directions steer, but they are "
                   f"near-collinear (min cosine {min_cos:.2f}) — weak evidence "
                   "of a genuine family")
    else:
        verdict = ("NOT SUPPORTED (screening): fewer than two effective "
                   "directions — the effect did not reproduce across recipes")

    return {
        "best_refusal": float(best_refusal),
        "effective_threshold": float(threshold),
        "effective_names": effective,
        "min_cosine_effective": (None if np.isnan(min_cos) else float(min_cos)),
        "refusal_spread": float(spread),
        "control_refusal": float(rate.get(control_name, float("nan"))),
        "verdict": verdict,
    }


def _summary_table(results: dict) -> str:
    """Plain-text recap printed at the end of a run."""
    lines = ["", "=" * 68, "NON-IDENTIFIABILITY SUMMARY", "=" * 68,
             f"model : {results['model_id']}",
             f"layer : {results['layer']}   matched alpha: {results['matched_alpha']:.3f}",
             f"baseline refusal (no steering): "
             f"{results['baseline']['refusal_rate']:.2f}",
             "",
             f"  {'direction':<20} {'refusal':>8} {'comply':>8} {'gibber':>8}"]
    for d in results["per_direction"]:
        lines.append(f"  {d['name']:<20} {d['refusal_rate']:>8.2f} "
                     f"{d['compliance_rate']:>8.2f} {d['gibberish_rate']:>8.2f}")
    ni = results["nonident"]
    mc = ni["min_cosine_effective"]
    lines += ["",
              f"effective directions (>= {C.EFFECTIVE_FRACTION:.0%} of best "
              f"refusal {ni['best_refusal']:.2f}): {ni['effective_names']}",
              f"min pairwise cosine among them: "
              f"{'n/a' if mc is None else f'{mc:.3f}'}",
              f"refusal spread among them     : {ni['refusal_spread']:.2f}",
              f"random-control refusal        : {ni['control_refusal']:.2f}",
              "", f"verdict: {ni['verdict']}"]

    # The dose sweep, printed as CONTEXT below the headline. It is deliberately
    # visually subordinate: the registered claim is the matched-alpha block
    # above, and the sweep's job is to say whether that block is a knife-edge.
    sweep = results.get("alpha_sweep") or {}
    if len(sweep) > 1:
        lines += ["", "-" * 68,
                  f"ALPHA SWEEP (context; headline stays alpha="
                  f"{results['matched_alpha']:.3f})", "-" * 68]
        keys = sorted(sweep, key=float)
        head = "  {:<20}".format("direction") + "".join(
            "{:>10}".format("a=" + k) for k in keys)
        lines.append(head)
        for name in results["names"]:
            row = "  {:<20}".format(name)
            for k in keys:
                rec = next((d for d in sweep[k]["per_direction"]
                            if d["name"] == name), None)
                row += "{:>10}".format("-" if rec is None
                                       else f"{rec['refusal_rate']:.2f}")
            lines.append(row)
        lines.append("  {:<20}".format("[min cos effective]") + "".join(
            "{:>10}".format(
                "n/a" if sweep[k]["nonident"]["min_cosine_effective"] is None
                else f"{sweep[k]['nonident']['min_cosine_effective']:.2f}")
            for k in keys))
        lines.append("  {:<20}".format("[n per cell]") + "".join(
            "{:>10}".format(sweep[k]["n"]) for k in keys))

    lines += ["=" * 68, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting — matplotlib Agg (headless). One figure, two panels, saved as PNG.
# --------------------------------------------------------------------------- #
def _plot(results: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = results["names"]
    cos = np.array(results["cosine_matrix"], dtype=float)
    rates = {d["name"]: d["refusal_rate"] for d in results["per_direction"]}
    short = [n.replace("diffmean_", "dm_").replace("random_in_pcspan", "random")
             for n in names]

    # A third panel only when there IS a sweep to draw. A single-alpha run must
    # not ship an empty axis implying a sweep it never ran.
    sweep = results.get("alpha_sweep") or {}
    ncols = 3 if len(sweep) > 1 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    ax0, ax1 = axes[0], axes[1]

    # Panel 1: cosine heatmap (how DIFFERENT the directions are).
    im = ax0.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax0.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=8)
    ax0.set_yticks(range(len(short)), short, fontsize=8)
    for i in range(len(short)):
        for j in range(len(short)):
            ax0.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black")
    ax0.set_title("Pairwise cosine similarity of the candidate directions\n"
                  "(low off-diagonal = they are DIFFERENT vectors)")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Panel 2: refusal rate per direction (how SIMILAR the effect is).
    order = names
    vals = [rates[n] for n in order]
    colors = ["#888" if n == "random_in_pcspan" else "#2a7" for n in order]
    bars = ax1.bar(range(len(order)),
                   vals, color=colors)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.axhline(results["baseline"]["refusal_rate"], color="#c33", ls="--",
                lw=1, label=f"baseline (no steer) = "
                            f"{results['baseline']['refusal_rate']:.2f}")
    ax1.set_xticks(range(len(order)), short, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("refusal rate on held-out harmful prompts")
    ax1.set_title(f"Steering effect at matched alpha={results['matched_alpha']:.2f}\n"
                  "(similar heights = SAME effect despite different vectors)")
    ax1.legend(fontsize=8)

    # Panel 3: refusal vs alpha, one line per direction. If the family holds
    # together across doses the contrast lines move as a band and the random
    # control stays flat and low; if the matched-alpha result was a knife-edge,
    # that shows up here as lines crossing.
    if ncols == 3:
        ax2 = axes[2]
        keys = sorted(sweep, key=float)
        xs = [float(k) for k in keys]
        for n, s in zip(names, short):
            ys = []
            for k in keys:
                rec = next((d for d in sweep[k]["per_direction"]
                            if d["name"] == n), None)
                ys.append(float("nan") if rec is None else rec["refusal_rate"])
            style = dict(color="#888", ls=":", lw=2) if n == "random_in_pcspan" \
                else dict(lw=1.6)
            ax2.plot(xs, ys, marker="o", ms=4, label=s, **style)
        ax2.axvline(results["matched_alpha"], color="#333", ls="-", lw=1,
                    alpha=0.5)
        ax2.text(results["matched_alpha"], 1.02, " headline alpha", fontsize=7,
                 ha="left", va="bottom", color="#333")
        ax2.axhline(results["baseline"]["refusal_rate"], color="#c33", ls="--",
                    lw=1)
        ax2.set_xlabel("relative alpha")
        ax2.set_ylabel("refusal rate")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Dose sweep (CONTEXT -- the headline is the\n"
                      "matched-alpha column, not the best column)")
        ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The pipeline — everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def _akey(alpha: float) -> str:
    """The ``alpha_sweep`` dict key for one alpha: ``0.08 -> "0.08"``."""
    return "%.4g" % float(alpha)


def main() -> dict:
    import random

    import torch

    # Peer / lesson-2 modules, imported inside main() so a bare import stays
    # model-free and torch-free.
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, generate, num_layers,
    )
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.common.data import load_harmful_benign
    from .vectors import (build_candidate_directions, directions_fingerprint,
                          load_directions, rebuild_built, save_directions)
    from steering_tutorials.common.artifact_paths import assert_no_bare_sibling
    from steering_tutorials.common.judge_gate import (
        assert_publishable, require_off_family_judge)

    # --- Pre-flight, all of it BEFORE the model loads ------------------------
    # Refuse a self-judged run first. The whole claim here is "many directions
    # with cosine well below 1 reach a SIMILAR refusal rate" -- a self-judge that
    # inflates refusal uniformly would manufacture that similarity, so this is
    # the lesson a self-judged run would corrupt most cheaply.
    judge_id = require_off_family_judge("non_identifiability")

    # And refuse to add a keyed sibling next to an unattributable bare file.
    assert_no_bare_sibling(C.ARTIFACTS, "results", ".json")

    # Reproducibility.
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Data + the three disjoint roles (CPU; no model needed yet) ---------
    data = load_harmful_benign(n_per_class=C.N_PER_CLASS, seed=C.SEED)
    harmful, benign = data["harmful"], data["benign"]
    build_harmful = harmful[:C.N_EXTRACT]
    build_benign = benign[:C.N_EXTRACT]
    eval_harmful = harmful[C.N_EXTRACT:C.N_EXTRACT + N_EVAL]
    print(f"[split] build: {len(build_harmful)}h/{len(build_benign)}b   "
          f"eval: {len(eval_harmful)}h", file=sys.stderr)

    # --- Checkpoint manifest: decide fresh-vs-resume before spending a watt --
    manifest = {
        "fingerprint": C.run_fingerprint(eval_harmful, judge_id),
        "model_id": C.MODEL_ID, "layer": int(C.LAYER),
        "n_extract": int(C.N_EXTRACT), "n_eval": int(N_EVAL),
        "seed": int(C.SEED), "max_new_tokens": int(C.MAX_NEW_TOKENS),
        "matched_alpha": float(C.MATCHED_ALPHA),
        "sweep_alphas": [float(a) for a in SWEEP_ALPHAS],
        "judge_id": str(judge_id),
    }
    state = check_manifest(CKPT_DIR, manifest)
    n_cells = 1 + len(SWEEP_ALPHAS) * 6      # 6 = the K recipes in vectors.py
    print(f"[ckpt] {state} at {CKPT_DIR} "
          f"(~{n_cells} cells x {len(eval_harmful)} generations)",
          file=sys.stderr)

    # --- Load the model ------------------------------------------------------
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)

    # --- 1. The K candidate directions (fingerprinted cache) -----------------
    dfp = directions_fingerprint(C.MODEL_ID, layer, C.N_PC, C.SEED,
                                 build_harmful, build_benign)
    built = None
    if C.DIRECTIONS_PATH.exists():
        cached = load_directions(C.DIRECTIONS_PATH, expect_fingerprint=dfp)
        if cached is not None:
            built = rebuild_built(cached)
            print(f"[vectors] reused cache {C.DIRECTIONS_PATH.name} "
                  f"(fingerprint {dfp[:12]})", file=sys.stderr)
    if built is None:
        built = build_candidate_directions(
            model, tok, build_harmful, build_benign, layer,
            n_pc=C.N_PC, seed=C.SEED,
        )
        save_directions(C.DIRECTIONS_PATH, built, fingerprint=dfp)
        print(f"[save] {C.DIRECTIONS_PATH}", file=sys.stderr)
    names = built["names"]
    cosine = built["cosine"]

    judge = Judge(model, tok)
    stamp = judge.stamp()
    # The live judge must be the one the fingerprint was computed against;
    # otherwise the checkpoint we are about to append to describes a different
    # grader than the one now grading.
    if str(stamp["judge_id"]) != str(judge_id):
        raise SystemExit(
            "judge id changed between pre-flight (%s) and the live Judge (%s); "
            "the checkpoint fingerprint no longer describes this run."
            % (judge_id, stamp["judge_id"]))

    # --- 2. Generate + judge, one CHECKPOINTED cell at a time ---------------
    def run_cell(alpha, name, v_unit) -> list[dict]:
        """Fill one cell to completion, resuming from whatever is on disk."""
        path = _cell_path(CKPT_DIR, alpha, name)
        done = _load_cell(path)
        if done:
            print(f"[ckpt] resume {cell_id(alpha, name)}: "
                  f"{len(done)}/{len(eval_harmful)} already judged",
                  file=sys.stderr)
        for i in range(len(done), len(eval_harmful)):
            p = eval_harmful[i]
            resp = generate(model, tok, p, max_new_tokens=C.MAX_NEW_TOKENS,
                            vector=v_unit, layer=layer,
                            alpha=(0.0 if v_unit is None else float(alpha)),
                            operation="relative_add")
            rec = {"i": i, "verdict": judge.verdict(p, resp), "response": resp}
            _append_cell(path, rec)          # flushed before the next generation
            done.append(rec)
        return done

    # 2a. Baseline: no steering, and therefore alpha-free. Computed ONCE and
    # shared by every alpha block -- steering with alpha=0 is the same run.
    base_recs = run_cell(None, "baseline", None)
    base_verdicts = [r["verdict"] for r in base_recs]
    baseline = {"n": len(base_verdicts), **_rates(base_verdicts)}
    print(f"[baseline] refusal={baseline['refusal_rate']:.2f} "
          f"(n={baseline['n']})", file=sys.stderr)

    # 2b. Every (alpha, direction) cell. The MATCHED_ALPHA pass is one member of
    # this loop, not a separate one, so the headline and the sweep can never
    # disagree about the same alpha and the sweep costs no duplicate work.
    alpha_sweep: dict[str, dict] = {}
    cell_recs: dict[tuple, list[dict]] = {}

    def assemble(partial: bool) -> dict:
        """Build the results dict from whatever has been completed so far."""
        matched = alpha_sweep.get(_akey(C.MATCHED_ALPHA), {})
        # Examples are reconstructed from the CHECKPOINT, not regenerated.
        # Decoding is greedy (model_utils.generate, do_sample=False), so the
        # stored baseline completion for prompt i is byte-identical to what a
        # re-generation would produce -- this drops 2 x K redundant baseline
        # generations per run and makes the examples reproducible from disk.
        examples = []
        for name in names:
            recs = cell_recs.get((_akey(C.MATCHED_ALPHA), name)) or []
            for i, r in enumerate(recs[:2]):
                if i >= len(base_recs):
                    break
                examples.append({
                    "prompt": eval_harmful[r["i"]], "direction": name,
                    "baseline_response": base_recs[r["i"]]["response"],
                    "steered_response": r["response"],
                    "baseline_verdict": base_recs[r["i"]]["verdict"],
                    "steered_verdict": r["verdict"],
                })
        return {
            # PROVENANCE (metadata only -- changes no metric). judge.stamp()
            # reports what ACTUALLY graded these generations, taken from the
            # judge object rather than from the README or the env var. A run that
            # fell back to self-judging lands here as judge_id="self" /
            # is_self_judge=true and is therefore inadmissible as a headline
            # (CLAUDE.md sec.17, rubric item 3).
            **stamp,
            "seed": int(C.SEED),
            "model_id": C.MODEL_ID,
            "layer": int(layer),
            "matched_alpha": float(C.MATCHED_ALPHA),
            "n_extract": int(built["n_extract"]),
            "n_eval": int(baseline["n"]),
            "names": names,
            "cosine_matrix": np.asarray(cosine).tolist(),
            "directions": [{"name": n,
                            "recipe": built["candidates"][n]["recipe"],
                            "pooling": built["candidates"][n]["pooling"]}
                           for n in names],
            "baseline": baseline,
            "per_direction": matched.get("per_direction", []),
            "nonident": matched.get("nonident", {}),
            "examples": examples[:12],
            "plots": {"nonident": PLOT_PATH.name},
            # --- everything below is NEW; the keys above are unchanged --------
            # The sweep is CONTEXT. per_direction / nonident above remain the
            # MATCHED_ALPHA block, so the registered headline is untouched by
            # anything that happens at another dose.
            "alpha_sweep": alpha_sweep,
            "alpha_sweep_alphas": [float(a) for a in SWEEP_ALPHAS],
            "alpha_sweep_note": (
                "Context around the registered headline, not a replacement for "
                "it. The top-level per_direction/nonident block is the "
                "matched_alpha cell; reporting the BEST alpha instead would be "
                "selecting the dose on the outcome."),
            "n_eval_requested": int(N_EVAL),
            "n_eval_default": int(C.N_EVAL_DEFAULT),
            "capped": bool(N_EVAL != C.N_EVAL_DEFAULT),
            "tier": ("SCREENING (n capped below the pre-registered "
                     f"{C.N_EVAL_DEFAULT})" if N_EVAL != C.N_EVAL_DEFAULT
                     else "pre-registered n"),
            "checkpoint_dir": CKPT_DIR.name,
            "run_fingerprint": manifest["fingerprint"],
            "partial": bool(partial),
        }

    def write(partial: bool) -> dict:
        res = assemble(partial)
        assert_publishable(res, "non_identifiability")   # gate IN the write path
        RESULTS_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return res

    for alpha in SWEEP_ALPHAS:
        ak = _akey(alpha)
        per_direction = []
        for name in names:
            recs = run_cell(alpha, name, built["candidates"][name]["v_unit"])
            cell_recs[(ak, name)] = recs
            verdicts = [r["verdict"] for r in recs]
            rec = {"name": name, "n": len(verdicts), **_rates(verdicts)}
            per_direction.append(rec)
            print(f"[steer a={ak:<6} {name:<20}] "
                  f"refusal={rec['refusal_rate']:.2f} "
                  f"gibber={rec['gibberish_rate']:.2f} n={rec['n']}",
                  file=sys.stderr)
        alpha_sweep[ak] = {
            "alpha": float(alpha),
            "n": min((d["n"] for d in per_direction), default=0),
            "is_headline": bool(abs(alpha - C.MATCHED_ALPHA) < 1e-9),
            "baseline": baseline,          # alpha-free, shared across the sweep
            "per_direction": per_direction,
            "nonident": summarize_nonidentifiability(
                per_direction, names, cosine, C.EFFECTIVE_FRACTION),
        }
        # Persist after EVERY alpha, so a reap leaves a readable partial
        # artifact rather than nothing at all.
        write(partial=True)
        print(f"[save] partial -> {RESULTS_PATH.name} (through a={ak})",
              file=sys.stderr)

    # --- 3. Final artifact ---------------------------------------------------
    results = write(partial=False)
    _plot(results, PLOT_PATH)
    print(f"[save] {RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {PLOT_PATH}", file=sys.stderr)
    print(_summary_table(results))
    return results


# --------------------------------------------------------------------------- #
# CPU self-test — NO model. Exercises the pure reporting helpers on fake data.
# Run: python -m steering_tutorials.non_identifiability.run_nonident
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    # Fake a family: three low-cosine directions all "effective", one control low.
    names = ["dA", "dB", "dC", "random_in_pcspan"]
    # cosine matrix: dA,dB,dC mutually ~0.5, control near-orthogonal to all.
    cos = np.array([
        [1.0, 0.55, 0.45, 0.02],
        [0.55, 1.0, 0.50, 0.05],
        [0.45, 0.50, 1.0, 0.01],
        [0.02, 0.05, 0.01, 1.0],
    ])
    per_direction = [
        {"name": "dA", "refusal_rate": 0.80, "compliance_rate": 0.15, "gibberish_rate": 0.05, "n": 40},
        {"name": "dB", "refusal_rate": 0.78, "compliance_rate": 0.17, "gibberish_rate": 0.05, "n": 40},
        {"name": "dC", "refusal_rate": 0.72, "compliance_rate": 0.20, "gibberish_rate": 0.08, "n": 40},
        {"name": "random_in_pcspan", "refusal_rate": 0.10, "compliance_rate": 0.80, "gibberish_rate": 0.10, "n": 40},
    ]
    ni = summarize_nonidentifiability(per_direction, names, cos, 0.80)
    # dA/dB/dC are all within 80% of best (0.80); control excluded.
    assert set(ni["effective_names"]) == {"dA", "dB", "dC"}, ni["effective_names"]
    # min pairwise cosine among them is 0.45 (dA-dC).
    assert abs(ni["min_cosine_effective"] - 0.45) < 1e-6, ni["min_cosine_effective"]
    assert ni["verdict"].startswith("SUPPORTED"), ni["verdict"]
    assert ni["refusal_spread"] > 0

    # _rates sanity.
    r = _rates(["REFUSAL", "REFUSAL", "COMPLIANCE", "GIBBERISH"])
    assert abs(r["refusal_rate"] - 0.5) < 1e-9

    # Degenerate case: only one effective direction => NOT SUPPORTED, cosine n/a.
    ni2 = summarize_nonidentifiability(
        [{"name": "dA", "refusal_rate": 0.8, "compliance_rate": 0.2, "gibberish_rate": 0.0, "n": 10},
         {"name": "dB", "refusal_rate": 0.1, "compliance_rate": 0.9, "gibberish_rate": 0.0, "n": 10}],
        ["dA", "dB"], np.eye(2), 0.80)
    assert ni2["min_cosine_effective"] is None
    assert ni2["verdict"].startswith("NOT SUPPORTED"), ni2["verdict"]

    # --- the checkpoint layer ------------------------------------------------
    import tempfile

    d = Path(tempfile.mkdtemp())

    # Cell ids: the baseline is alpha-free; a steered cell names its dose.
    assert cell_id(None, "baseline") == "baseline"
    assert cell_id(0.08, "diffmean_full") == "a080__diffmean_full"
    assert cell_id(0.06, "pca_top1") != cell_id(0.10, "pca_top1")

    # Append / reload round-trip.
    cp = _cell_path(d, 0.08, "dA")
    assert _load_cell(cp) == []                       # nothing yet
    for i in range(3):
        _append_cell(cp, {"i": i, "verdict": "REFUSAL", "response": "no."})
    assert len(_load_cell(cp)) == 3
    assert [r["i"] for r in _load_cell(cp)] == [0, 1, 2]

    # A TORN final line (the reap-mid-write case) costs one record, not the cell.
    with cp.open("a", encoding="utf-8") as fh:
        fh.write('{"i": 3, "verdict": "REF')      # no newline, no closing brace
    assert len(_load_cell(cp)) == 3, "a torn tail must be dropped, not parsed"

    # A GAP truncates: resuming past one would misalign every later verdict
    # against its prompt, which is the silent-and-plausible failure (sec.18.8).
    gap = _cell_path(d, 0.08, "dGap")
    for i in (0, 1, 5):
        _append_cell(gap, {"i": i, "verdict": "COMPLIANCE", "response": "sure"})
    assert [r["i"] for r in _load_cell(gap)] == [0, 1]

    # Manifest: fresh, then resume, then REFUSE across a changed fingerprint.
    m1 = {"fingerprint": "abc123", "n_eval": 150, "judge_id": "Qwen/x"}
    assert check_manifest(d, m1) == "fresh"
    assert check_manifest(d, m1) == "resume"
    try:
        check_manifest(d, {**m1, "fingerprint": "def456", "n_eval": 40})
    except SystemExit as e:
        assert "REFUSING TO RESUME" in str(e) and "n_eval" in str(e), str(e)
    else:
        raise AssertionError("a changed fingerprint must refuse to resume")

    # The sweep must not be able to omit the headline dose.
    assert C.MATCHED_ALPHA in C.SWEEP_ALPHAS
    assert C.alpha_tag([0.06, 0.08, 0.1]) == "a060-080-100"
    assert _akey(0.08) == "0.08"
    assert C._parse_alphas("none") == []
    assert C._parse_alphas(" 0.05 , 0.2 ") == [0.05, 0.2]
    assert C._parse_alphas(None) == C.ALPHAS_DEFAULT
    try:
        C._parse_alphas("0.05,banana")
    except ValueError as e:
        assert "unparseable" in str(e)
    else:
        raise AssertionError("a typo'd NONIDENT_ALPHAS must raise, not default")

    # And the paths really are per-variant, in both dimensions that vary.
    assert C.RESULTS_PATH.name.startswith("results_n")
    assert C.alpha_tag(C.SWEEP_ALPHAS) in C.RESULTS_PATH.name
    assert C.RESULTS_PATH != C.LEGACY_RESULTS_PATH

    print("[self-test] OK - rates, non-identifiability summary, alpha grid, "
          "keyed paths and the resumable checkpoint layer all behave.")


if __name__ == "__main__":
    _self_test()
    main()
