"""export_datasets.py -- ship the exact policy-labelled corpus BIENCODER_GUARD used.

CPU-only, no model, no GPU. `data.load_corpus` / `data.load_cross_annotator` /
`data.load_ood_benchmark` / `data.load_heldout_split` are pure dataset builders
(network + local HF cache, no LLM, no encoder). Rebuilds at config.py defaults --
n_per_class=500, n_benign=3000, seed=0 -- which match `results_embeddinggemma.json`'s
`n_per_class`/`n_benign`/`seed` exactly.

MULTI-SOURCE, split by the corpus's own `sources` tag, per row:
  aegis / aegis_benign            -> nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (cc-by-4.0, REDISTRIBUTABLE)
  toxicchat / toxicchat_benign    -> lmsys/toxic-chat (cc-by-nc-4.0, REDISTRIBUTABLE)
  beavertails / beavertails_benign -> PKU-Alignment/BeaverTails -- NOT in
                                     `dataset_export.REDISTRIBUTABLE` or `.GATED`.
                                     Its licence has not been vetted in the shared
                                     spine (out of this script's scope to add), so
                                     per the spine's own refusal rule these rows are
                                     NOT exported -- a refetch manifest is written
                                     instead (see below). THIS IS THE LARGEST SINGLE
                                     SOURCE IN THE CORPUS (~4852/13065 rows, ~37%)
                                     and the achieved results.json's own
                                     `source_distribution` confirms it.
  wildguard / wildguard_benign     -> allenai/wildguardmix, GATED. The achieved corpus
                                     has ZERO wildguard rows (verified below), so
                                     nothing to export or manifest for this run.

Two extra transfer arms, both single-source:
  cross_annotator  Aegis 2.0 TEST split (nvidia/..., REDISTRIBUTABLE) -- exported.
  ood_benchmark    intrinsec-ai/cstm-bench (mit, REDISTRIBUTABLE) -- exported.
  heldout_split    BeaverTails 30k_test -- same unvetted source as above; NOT
                   exported, refetch manifest only.

Run: python -m steering_tutorials.biencoder_guard.export_datasets
"""
from __future__ import annotations

from pathlib import Path

from collections import Counter

from steering_tutorials.common import dataset_export as DE

from . import config as C

from . import data as D

_HERE = Path(__file__).resolve().parent

SRC_AEGIS = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
SRC_TOXICCHAT = "lmsys/toxic-chat"
SRC_BEAVERTAILS = "PKU-Alignment/BeaverTails"     # unvetted -- see module docstring
SRC_WILDGUARD = "allenai/wildguardmix"            # GATED
SRC_CSTM = "intrinsec-ai/cstm-bench"

_SOURCE_TO_HF = {
    "aegis": SRC_AEGIS, "aegis_benign": SRC_AEGIS,
    "toxicchat": SRC_TOXICCHAT, "toxicchat_benign": SRC_TOXICCHAT,
    "beavertails": SRC_BEAVERTAILS, "beavertails_benign": SRC_BEAVERTAILS,
    "wildguard": SRC_WILDGUARD, "wildguard_benign": SRC_WILDGUARD,
}


def _row_policies(y_row, policies) -> list:
    return [policies[i]["id"] for i, v in enumerate(y_row) if v > 0]


def _corpus_rows(corpus: dict, source_filter) -> list:
    """Rows whose source tag maps to `source_filter` (a set of tags to KEEP)."""
    rows = []
    for text, y_row, source, group in zip(corpus["texts"], corpus["Y"],
                                          corpus["sources"], corpus["groups"]):
        if source not in source_filter:
            continue
        rows.append({"text": text, "policies": _row_policies(y_row, corpus["policies"]),
                     "is_harmful": int(bool(_row_policies(y_row, corpus["policies"]))),
                     "group": int(group), "_source": _SOURCE_TO_HF[source],
                     "_source_tag": source})
    return rows


def main() -> None:
    manifests = []

    corpus = D.load_corpus(n_per_class=C.N_PER_CLASS, n_benign=C.N_BENIGN, seed=C.SEED)
    tag_counts = dict(Counter(corpus["sources"]))
    print("[export] main corpus source_distribution: %s" % tag_counts)

    aegis_tags = {"aegis", "aegis_benign"}
    toxicchat_tags = {"toxicchat", "toxicchat_benign"}
    beavertails_tags = {"beavertails", "beavertails_benign"}
    wildguard_tags = {"wildguard", "wildguard_benign"}

    for name_base, src, tags in (("aegis", SRC_AEGIS, aegis_tags),
                                 ("toxicchat", SRC_TOXICCHAT, toxicchat_tags)):
        rows = _corpus_rows(corpus, tags)
        if not rows:
            print("[export] %s: 0 rows in the achieved corpus -- skipping" % name_base)
            continue
        composition = dict(Counter(r["_source_tag"] for r in rows))
        name = "biencoder_%s_n%d_s%d" % (name_base, len(rows), C.SEED)
        man = DE.export_slice(
            _HERE, name, src, rows, split="train", seed=C.SEED,
            notes=("biencoder_guard main-corpus slice from %s (n_per_class=%d "
                   "n_benign=%d requested). Composition: %s."
                   % (src, C.N_PER_CLASS, C.N_BENIGN, composition)),
            extra={"composition": composition, "policies": corpus["policies"]})
        manifests.append((name, src, man["licence"], man["n_rows"], man["slice_fingerprint"]))
        print("[export] %s: n=%d fp=%s composition=%s"
              % (name, man["n_rows"], man["slice_fingerprint"], composition))

    bt_rows = _corpus_rows(corpus, beavertails_tags)
    if bt_rows:
        man = DE.write_refetch_manifest(
            _HERE, "biencoder_beavertails_main", SRC_BEAVERTAILS, n_rows=len(bt_rows),
            seed=C.SEED,
            loader_hint=("python -c \"from steering_tutorials.biencoder_guard import "
                        "data as D, config as C; D.load_corpus(n_per_class=%d, "
                        "n_benign=%d, seed=%d)\" -- then filter rows whose "
                        "corpus['sources'][i] in ('beavertails','beavertails_benign')"
                        % (C.N_PER_CLASS, C.N_BENIGN, C.SEED)),
            notes=("UNVETTED, not GATED: PKU-Alignment/BeaverTails is absent from "
                   "both dataset_export.REDISTRIBUTABLE and .GATED, so export_slice "
                   "refuses it (unchecked licence != permissive). This is the "
                   "LARGEST single source in the main corpus (%d/%d rows here, "
                   "~%.0f%%). FLAGGED for the licence owner of common/dataset_export.py "
                   "to read PKU-Alignment/BeaverTails' HF card and classify it -- out "
                   "of this script's scope to add. No row ids are recorded: "
                   "BeaverTails is streamed and has no stable per-row id in the "
                   "loader; the loader_hint reproduces the exact selection instead."
                   % (len(bt_rows), len(corpus["texts"]),
                      100.0 * len(bt_rows) / max(1, len(corpus["texts"])))))
        print("[export] biencoder_beavertails_main: NOT exported (unvetted licence); "
              "refetch manifest written, n=%d" % len(bt_rows))
    else:
        print("[export] beavertails: 0 rows in the achieved corpus")

    wg_rows = _corpus_rows(corpus, wildguard_tags)
    print("[export] wildguard: %d rows in the achieved corpus (expected 0, gated + "
          "no HF token on this host)" % len(wg_rows))
    if wg_rows:
        DE.write_refetch_manifest(
            _HERE, "biencoder_wildguard_main", SRC_WILDGUARD, n_rows=len(wg_rows),
            seed=C.SEED,
            loader_hint="D.load_corpus(...) then filter sources in "
                       "('wildguard','wildguard_benign')",
            notes="GATED (allenai/wildguardmix). Unexpectedly non-zero this run.")

    cross = D.load_cross_annotator(seed=C.SEED)
    if cross.get("n", 0) > 0:
        rows = []
        for text, y_row, group in zip(cross["texts"], cross["Y"], cross["groups"]):
            rows.append({"text": text, "policies": _row_policies(y_row, cross["policies"]),
                        "is_harmful": int(bool(_row_policies(y_row, cross["policies"]))),
                        "group": int(group), "_source": SRC_AEGIS})
        name = "biencoder_cross_annotator_n%d_s%d" % (len(rows), C.SEED)
        man = DE.export_slice(
            _HERE, name, SRC_AEGIS, rows, split=cross["source"].split("/")[-1], seed=C.SEED,
            notes="biencoder_guard cross_annotator transfer arm: Aegis 2.0 test split.")
        manifests.append((name, SRC_AEGIS, man["licence"], man["n_rows"],
                          man["slice_fingerprint"]))
        print("[export] %s: n=%d fp=%s" % (name, man["n_rows"], man["slice_fingerprint"]))
    else:
        print("[export] cross_annotator: 0 rows (load failed) -- nothing to export")

    ood = D.load_ood_benchmark(seed=C.SEED)
    if ood.get("n", 0) > 0:
        rows = []
        for text, y_row, group in zip(ood["texts"], ood["Y"], ood["groups"]):
            rows.append({"text": text, "policies": _row_policies(y_row, ood["policies"]),
                        "is_harmful": int(bool(_row_policies(y_row, ood["policies"]))),
                        "group": int(group), "_source": SRC_CSTM})
        name = "biencoder_ood_cstm_n%d_s%d" % (len(rows), C.SEED)
        man = DE.export_slice(
            _HERE, name, SRC_CSTM, rows, split="+".join(C.CSTM_SPLITS), seed=C.SEED,
            notes=("biencoder_guard ood_benchmark transfer arm: CSTM-Bench, scenario-"
                   "level jailbreak label. n=%d. Screening-tier (n~%d)."
                   % (len(rows), len(rows))))
        manifests.append((name, SRC_CSTM, man["licence"], man["n_rows"],
                          man["slice_fingerprint"]))
        print("[export] %s: n=%d fp=%s" % (name, man["n_rows"], man["slice_fingerprint"]))
    else:
        print("[export] ood_benchmark: 0 rows (load failed) -- nothing to export")

    heldout = D.load_heldout_split(seed=C.SEED)
    if heldout.get("n", 0) > 0:
        DE.write_refetch_manifest(
            _HERE, "biencoder_heldout_split", SRC_BEAVERTAILS, n_rows=heldout["n"],
            seed=C.SEED,
            loader_hint=("python -c \"from steering_tutorials.biencoder_guard import "
                        "data as D; D.load_heldout_split(seed=%d)\"" % C.SEED),
            notes=("biencoder_guard heldout_split transfer arm: BeaverTails 30k_test. "
                   "Same unvetted-licence situation as the main corpus's beavertails "
                   "rows -- NOT exported, refetch manifest only."))
        print("[export] biencoder_heldout_split: NOT exported (unvetted licence); "
              "refetch manifest written, n=%d" % heldout["n"])
    else:
        print("[export] heldout_split: 0 rows (load failed)")

    print("\n[export] biencoder_guard: %d exported slice(s)." % len(manifests))
    for name, src, lic, n, fp in manifests:
        print("  %-32s %-42s %-14s n=%-5d fp=%s" % (name, src, lic, n, fp))


if __name__ == "__main__":
    main()
