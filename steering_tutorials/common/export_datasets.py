"""export_datasets.py -- materialise the SHARED data slices the course draws on.

Writes, via :mod:`dataset_export` (never bypassing its licence gate):

  1. ``harmful_benign_n500_s0``  -- the canonical >=500/class binary set from
     :func:`data.build_harmful_benign` (``n_per_class=500, seed=0``).
  2. ``concepts_150_s0``         -- the per-harm-category concept split from
     :func:`data.load_concepts` at its documented defaults
     (``n_per_concept=150, seed=0, split="all", min_available=100``).
  3. ``concepts_500_s0``         -- the SAME split at ``n_per_concept=500``,
     matching ``multi_intent``'s actual config
     (``config.py``'s ``MULTI_INTENT_N_PER_CONCEPT`` default "500"). Every
     kept concept's pool (sexual 388 / harassment 143 / violence 111) is
     BELOW 500, so this call draws the FULL pool per concept rather than
     capping at the request -- ``concepts_150_s0`` is a strict subset of it,
     not an independent sample.

ALL THREE slices are, in the CURRENT code, 100% ``lmsys/toxic-chat`` (cc-by-nc-4.0,
NON-COMMERCIAL). ``build_harmful_benign``'s harmful class is topped up from
JailbreakBench (mit) **only if** the toxic-chat toxic pool is smaller than
``n_per_class`` after dedup; at ``n_per_class=500`` the full toxic-chat toxic
pool (~693 unique) clears 500 on its own, so the default config's harmful
class is 100% toxic-chat and the JBB top-up path is simply not exercised. This
script does NOT assume that -- it reads ``header["per_source_counts_sampled"]``
and ``header["topup_log"]`` from the actual run and reports whatever the code
actually did, exactly as instructed.

Because the composition can in principle mix sources with DIFFERENT licences,
every row is tagged with its own ``_source`` / ``_licence`` (per-row
provenance, always checkable), while the single ``source=`` passed to
``export_slice`` is the MOST RESTRICTIVE licence actually present in the
slice, so the whole file inherits the strictest term rather than the most
convenient one.

CPU-only. No model, no GPU. Run with::

    C:/Users/evija/anaconda3/python.exe -m steering_tutorials.common.export_datasets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from . import data
    from . import dataset_export as DE
except ImportError:  # pragma: no cover - direct-script form
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent.parent))
    from steering_tutorials.common import data
    from steering_tutorials.common import dataset_export as DE

LESSON_DIR = Path(__file__).resolve().parent  # steering_tutorials/common/

# Base names as they appear in DE.REDISTRIBUTABLE -- the "@0124" toxic-chat
# release tag is data.py's own convenience suffix, not a licence-table key.
_LICENCE_BASE = {
    data.SRC_TOXICCHAT: "lmsys/toxic-chat",
    data.SRC_JBB: "JailbreakBench/JBB-Behaviors",
}


def _licence_of_row_source(row_source: str) -> str:
    base = _LICENCE_BASE.get(row_source, row_source.split("@")[0])
    return DE.licence_of(base)


def _strictest_source(row_sources: set[str]) -> str:
    """Pick the REDISTRIBUTABLE key whose licence is the most restrictive.

    Only cc-by-nc-4.0 (non-commercial) vs. mit/apache/cc-by-4.0 (permissive)
    appears among this course's sources, so "most restrictive" reduces to
    "prefer non-commercial if present". Fails loudly if an unmapped source
    shows up so a future new upstream can't silently pick the wrong bucket.
    """
    bases = {_LICENCE_BASE.get(s, s.split("@")[0]) for s in row_sources}
    nc = [b for b in bases if DE.licence_of(b) == "cc-by-nc-4.0"]
    if nc:
        return nc[0]
    if len(bases) == 1:
        return next(iter(bases))
    raise SystemExit(
        f"export_datasets: multiple non-NC sources {bases} and no single "
        "most-restrictive licence rule for them -- add one before exporting.")


def export_harmful_benign(n_per_class: int = 500, seed: int = 0) -> dict:
    rec = data.build_harmful_benign(n_per_class=n_per_class, seed=seed)
    header = rec["header"]

    rows = []
    for label, key in (("harmful", "harmful"), ("benign", "benign")):
        for r in rec[key]:
            rows.append({
                "label": label,
                "prompt": r["prompt"],
                "category": r["category"],
                "group_id": r["group_id"],
                "_source": r["source"],
                "_licence": _licence_of_row_source(r["source"]),
            })

    row_sources = {r["_source"] for r in rows}
    source = _strictest_source(row_sources)

    composition = {
        "harmful": dict(header["per_source_counts_sampled"]),
        "benign": {data.SRC_TOXICCHAT: header["n_benign"]},  # benign has no top-up path
        "topup_log": header["topup_log"],
    }

    notes = (
        "Composed of lmsys/toxic-chat (Lin et al. 2023, arXiv:2310.17389, "
        "cc-by-nc-4.0, NON-COMMERCIAL) as the primary/only source for BOTH "
        "classes at this n_per_class, plus a JailbreakBench/JBB-Behaviors "
        "(mit) length-matched top-up to the harmful class that only "
        "activates if the deduped toxic-chat toxic pool is smaller than "
        "n_per_class -- at n_per_class=500 that pool (~693 unique) already "
        "clears 500, so the JBB top-up did not fire in THIS run (see "
        "extra.composition.topup_log). Exported source= is the strictest "
        "licence actually present (cc-by-nc-4.0), so the whole slice "
        "carries the non-commercial notice regardless of which rows are "
        "JBB. Every row also carries its own _source/_licence."
    )

    man = DE.export_slice(
        LESSON_DIR, "harmful_benign_n500_s0", source, rows,
        split=header["split_used"], seed=seed, notes=notes,
        extra={"composition": composition, "header": header},
    )
    print(json.dumps(man, indent=1))
    return man


def export_concepts(n_per_concept: int = 150, seed: int = 0) -> dict:
    slice_name = f"concepts_{n_per_concept}_s0"
    con = data.load_concepts(n_per_concept=n_per_concept, seed=seed)

    rows = []
    for concept_name in con["concept_names"]:
        c = con["concepts"][concept_name]
        for split_name in ("exemplars", "steer", "eval"):
            for prompt in c[split_name]:
                rows.append({
                    "concept": concept_name,
                    "split": split_name,
                    "prompt": prompt,
                    "n_available": c["n_available"],
                    "_source": data.SRC_TOXICCHAT,
                    "_licence": DE.licence_of("lmsys/toxic-chat"),
                })
    for prompt in con["baseline"]:
        rows.append({
            "concept": "baseline", "split": "baseline", "prompt": prompt,
            "n_available": None,
            "_source": data.SRC_TOXICCHAT,
            "_licence": DE.licence_of("lmsys/toxic-chat"),
        })

    pool_sizes = {name: con["concepts"][name]["n_available"] for name in con["concept_names"]}
    capped_by_pool = n_per_concept >= max(pool_sizes.values())

    notes = (
        "Every row derives from lmsys/toxic-chat (Lin et al. 2023, "
        "arXiv:2310.17389, cc-by-nc-4.0, NON-COMMERCIAL) -- concept rows "
        "from toxic prompts folded to a coarse harm category via "
        "openai_moderation, baseline rows from non-toxic prompts. No other "
        "upstream source is used by load_concepts. concept_names are the "
        "ones that clear MIN_CONCEPT_AVAILABLE=100 available prompts; "
        "dropped concepts (with their pool sizes) and the zero-shot "
        "held_out concept are recorded in extra. requested n_per_concept="
        f"{n_per_concept} vs. per-concept pool sizes {pool_sizes} -- "
        + ("every kept concept's pool is SMALLER than the request, so this "
           "slice is effectively the FULL available pool per concept, not "
           "n_per_concept capped." if capped_by_pool else
           "n_per_concept is below at least one concept's pool, so this "
           "slice is a strict SUBSET of that concept's full pool.")
    )

    man = DE.export_slice(
        LESSON_DIR, slice_name, "lmsys/toxic-chat", rows, seed=seed,
        notes=notes,
        extra={
            "concept_names": con["concept_names"],
            "dropped": con["dropped"],
            "held_out": con["held_out"],
            "n_per_concept_requested": n_per_concept,
            "pool_sizes": pool_sizes,
            "capped_by_pool_not_request": capped_by_pool,
        },
    )
    print(json.dumps(man, indent=1))
    return man


def _verify(name: str, manifest: dict) -> None:
    """Re-read the written .jsonl.gz and confirm row count + fingerprint match."""
    import gzip

    path = LESSON_DIR / "datasets" / manifest["file"]
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    assert len(rows) == manifest["n_rows"], (
        f"{name}: row count mismatch {len(rows)} != {manifest['n_rows']}")
    fp = DE.slice_fingerprint(rows)
    assert fp == manifest["slice_fingerprint"], (
        f"{name}: fingerprint mismatch {fp} != {manifest['slice_fingerprint']}")
    size_mb = manifest["bytes_gz"] / (1024 * 1024)
    print(f"[verify] {name}: n_rows={len(rows)} fp={fp} "
          f"gz={manifest['bytes_gz']}B ({size_mb:.3f} MB) OK")
    if size_mb > 25:
        print(f"[export_datasets] STOP: {name} is {size_mb:.1f} MB gzipped, "
              "over the ~25 MB budget -- report before committing.",
              file=sys.stderr)


if __name__ == "__main__":
    m1 = export_harmful_benign(n_per_class=500, seed=0)
    m2 = export_concepts(n_per_concept=150, seed=0)
    # multi_intent's actual config (config.py MULTI_INTENT_N_PER_CONCEPT default
    # "500") exceeds every kept concept's pool (sexual 388 / harassment 143 /
    # violence 111), so its real load_concepts call draws the FULL pool per
    # concept, not 150. Ship that exact slice too so multi_intent's manifest
    # can point at data that matches its real config, not a strict subset of it.
    m3 = export_concepts(n_per_concept=500, seed=0)
    _verify("harmful_benign_n500_s0", m1)
    _verify("concepts_150_s0", m2)
    _verify("concepts_500_s0", m3)
    print("[export_datasets] OK -- all three slices written, verified round-trip.")
