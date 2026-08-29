"""export_datasets.py -- ship the exact trace pool MEERKAT clustered on.

CPU-only, no model, no GPU. `data.load_trace_pool` / `data.load_ood_cstm` are pure
dataset builders (network + local HF cache only, no LLM, no clustering). Exports the
POOL the lesson draws sparse "repositories" from via `sample_repository` -- the
repository itself is a resample of this same pool (no separate upstream rows), so the
pool is what needs to ship.

CONFIG USED: `results_bge.json` and `results_embeddinggemma.json` both report
n_attack=500, n_benign=500, seed=0 -- the current config.py default and what this
script rebuilds. `results_minilm.json` is a STALE legacy run at n=200/class (predates
the >=500/class floor); it is NOT separately exported since re-running the CURRENT
loader at n=200 would not reproduce it bit-for-bit (the pool-build order depends on
the requested n), and the current default (500) is what two of the three encoder arms
actually used -- flagged to the team lead as a discrepancy, not silently reconciled.

MULTI-SOURCE: positives ("attack") from SafeMTData/SafeMTData (mit); benign
("ultrachat") from HuggingFaceH4/ultrachat_200k (mit), length-matched to the attack
sub-step window. OOD is intrinsec-ai/cstm-bench (mit), both splits.

Run: python -m steering_tutorials.meerkat.export_datasets
"""
from __future__ import annotations

from pathlib import Path

from collections import Counter

from steering_tutorials.common import dataset_export as DE

from . import config as C

from . import data as D

_HERE = Path(__file__).resolve().parent

SRC_ATTACK = "SafeMTData/SafeMTData"
SRC_ULTRACHAT = "HuggingFaceH4/ultrachat_200k"
SRC_OOD = "intrinsec-ai/cstm-bench"

_SOURCE_TO_HF = {"attack": SRC_ATTACK, "ultrachat": SRC_ULTRACHAT}


def _pool_rows(pool: dict) -> list:
    rows = []
    for trace, label, group, source in zip(pool["traces"], pool["labels"],
                                            pool["groups"], pool["sources"]):
        rows.append({"text": trace, "label": int(label), "group": group,
                     "_source": _SOURCE_TO_HF[source], "_source_tag": source})
    return rows


def _ood_rows(pool: dict) -> list:
    rows = []
    for trace, label, group in zip(pool["traces"], pool["labels"], pool["groups"]):
        rows.append({"text": trace, "label": int(label), "group": group,
                     "_source": SRC_OOD, "_source_tag": "cstm"})
    return rows


def main() -> None:
    manifests = []

    pool = D.load_trace_pool(n_attack=C.N_ATTACK, n_benign=C.N_BENIGN, seed=C.SEED)
    rows = _pool_rows(pool)
    composition = dict(Counter(r["_source_tag"] for r in rows))
    n_attack = sum(pool["labels"])
    name = "meerkat_pool_natt%d_nben%d_s%d" % (n_attack, len(rows) - n_attack, C.SEED)
    man = DE.export_slice(
        _HERE, name, SRC_ATTACK, rows, split="pool", seed=C.SEED,
        notes=("meerkat trace pool (the >=500/class reservoir sample_repository draws "
               "sparse repositories from). n_attack=%d n_benign=%d. Composition: %s."
               % (n_attack, len(rows) - n_attack, composition)),
        extra={"composition": composition, "repo_size": C.REPO_SIZE,
              "n_repos": C.N_REPOS, "base_rate": C.BASE_RATE})
    manifests.append((name, SRC_ATTACK, man["licence"], man["n_rows"],
                      man["slice_fingerprint"]))
    print("[export] %s: n=%d fp=%s composition=%s"
          % (name, man["n_rows"], man["slice_fingerprint"], composition))

    ood = D.load_ood_cstm(seed=C.SEED)
    ood_rows = _ood_rows(ood)
    n_att_ood = sum(ood["labels"])
    ood_name = "meerkat_ood_n%d_s%d" % (len(ood_rows), C.SEED)
    man = DE.export_slice(
        _HERE, ood_name, SRC_OOD, ood_rows, split="dilution+cross_session", seed=C.SEED,
        notes=("meerkat OOD arm (CSTM-Bench, both splits, whole-scenario traces). "
               "n=%d attack=%d benign=%d. Screening-tier (n~108)."
               % (len(ood_rows), n_att_ood, len(ood_rows) - n_att_ood)))
    manifests.append((ood_name, SRC_OOD, man["licence"], man["n_rows"],
                      man["slice_fingerprint"]))
    print("[export] %s: n=%d fp=%s" % (ood_name, man["n_rows"], man["slice_fingerprint"]))

    print("\n[export] meerkat: %d slice(s) written." % len(manifests))
    for name, src, lic, n, fp in manifests:
        print("  %-30s %-24s %-6s n=%-4d fp=%s" % (name, src, lic, n, fp))


if __name__ == "__main__":
    main()
