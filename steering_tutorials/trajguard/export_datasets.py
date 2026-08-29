"""export_datasets.py -- ship the exact prompt slices TRAJGUARD trained/tested on.

CPU-only, no model, no GPU. Rebuilds the deterministic prompt selections via this
lesson's own CPU-only functions (`data.select_prompts`, `ood.select_ood_prompts`) --
never the token TRAJECTORIES themselves (those are Gemma activations captured by
running the model; the licensed thing here is the upstream PROMPT TEXT, which is
what `export_slice` fingerprints).

Substrates actually run (per `artifacts/trajectory_meta_*.json`): "overt" (n=500/class,
seed=0) and "disguised" (n=181/class, seed=0, POOL-LIMITED per config.py). Both draw
from `lmsys/toxic-chat` (cc-by-nc-4.0, REDISTRIBUTABLE). The OOD arm
(`jackhhao/jailbreak-classification`, apache-2.0, REDISTRIBUTABLE) is IDENTICAL for
both substrates (same n=274/class, seed=0, length-cap quantile 0.9 -- confirmed by
diffing `trajectory_meta_ood_overt.json` and `trajectory_meta_ood_disguised.json`),
so it is exported once, not twice.

Run: python -m steering_tutorials.trajguard.export_datasets
"""
from __future__ import annotations

from pathlib import Path

import sys

from steering_tutorials.common import dataset_export as DE

from . import config as C

from . import data as D
from . import ood as O

_HERE = Path(__file__).resolve().parent

SRC_TOXICCHAT = "lmsys/toxic-chat"   # data.py tags rows "lmsys/toxic-chat@0124"; the
                                     # export gate keys off the plain HF repo id.


def _toxicchat_rows(sel: dict, substrate: str) -> list:
    rows = []
    for r in sel["harmful"]:
        rows.append({"text": r["prompt"], "label": 1, "group_id": r["group_id"],
                     "category": r["category"], "substrate": substrate,
                     "_source": SRC_TOXICCHAT})
    for r in sel["benign"]:
        rows.append({"text": r["prompt"], "label": 0, "group_id": r["group_id"],
                     "category": r["category"], "substrate": substrate,
                     "_source": SRC_TOXICCHAT})
    return rows


def _ood_rows(sel: dict) -> list:
    rows = []
    for r in sel["positive"]:
        rows.append({"text": r["prompt"], "label": 1, "group_id": r["group_id"],
                     "_source": O.SRC_OOD})
    for r in sel["negative"]:
        rows.append({"text": r["prompt"], "label": 0, "group_id": r["group_id"],
                     "_source": O.SRC_OOD})
    return rows


def main() -> None:
    manifests = []

    for substrate, n in (("overt", 500), ("disguised", 181)):
        sel = D.select_prompts(substrate=substrate, n_per_class=n, seed=C.SEED)
        rows = _toxicchat_rows(sel, substrate)
        name = "toxicchat_%s_n%d_s%d" % (substrate, sel["header"]["n_harmful"], C.SEED)
        man = DE.export_slice(
            _HERE, name, SRC_TOXICCHAT, rows, split="all", seed=C.SEED,
            notes=("TrajGuard %s substrate: harmful=%s definition, benign=toxicity==0 "
                   "AND jailbreaking==0 length-matched. n_harmful=%d n_benign=%d "
                   "pool_limited=%s (config.py POOL_MEASURED=%s)."
                   % (substrate, sel["header"]["substrate_definition"],
                      sel["header"]["n_harmful"], sel["header"]["n_benign"],
                      sel["header"]["pool_limited"], C.POOL_MEASURED)),
            extra={"substrate_header": sel["header"]})
        manifests.append((name, SRC_TOXICCHAT, man["licence"], man["n_rows"],
                          man["slice_fingerprint"]))
        print("[export] %s: n=%d fp=%s" % (name, man["n_rows"], man["slice_fingerprint"]))

    ood_sel = O.select_ood_prompts(n_per_class=C.OOD_N_PER_CLASS, seed=C.SEED,
                                    length_cap_quantile=C.OOD_LENGTH_CAP_QUANTILE)
    ood_rows = _ood_rows(ood_sel)
    ood_name = "jailbreakclass_ood_n%d_s%d" % (ood_sel["header"]["n_positive"], C.SEED)
    man = DE.export_slice(
        _HERE, ood_name, O.SRC_OOD, ood_rows, split="default", seed=C.SEED,
        notes=("TrajGuard OOD arm, shared verbatim by BOTH substrates (overt + "
               "disguised) per matching trajectory_meta_ood_*.json config snapshots. "
               "n_positive=%d n_negative=%d length_cap_quantile=%.2f pool_limited=%s."
               % (ood_sel["header"]["n_positive"], ood_sel["header"]["n_negative"],
                  ood_sel["header"]["length_cap_quantile"],
                  ood_sel["header"]["pool_limited"])),
        extra={"ood_header": ood_sel["header"]})
    manifests.append((ood_name, O.SRC_OOD, man["licence"], man["n_rows"],
                      man["slice_fingerprint"]))
    print("[export] %s: n=%d fp=%s" % (ood_name, man["n_rows"], man["slice_fingerprint"]))

    print("\n[export] trajguard: %d slice(s) written." % len(manifests))
    for name, src, lic, n, fp in manifests:
        print("  %-32s %-42s %-14s n=%-4d fp=%s" % (name, src, lic, n, fp))


if __name__ == "__main__":
    main()
