"""export_datasets.py -- ship the exact multi-trajectory samples CROSS_TRAJECTORY used.

CPU-only, no model, no GPU. `data.load_dataset` / `data.load_ood_cstm` are pure
dataset builders (network + local HF cache only, no LLM); this rebuilds the same
deterministic samples the lesson trained/tested on, at the config defaults
`results_embeddinggemma.json` confirms were actually used: N_POS=N_NEG=500, K=5,
seed=0, both conditions ("easy" and "hard"), plus the CSTM-Bench OOD arm
(ood_selection="uniform", ood_k=5).

Each row here is one SAMPLE = an unordered set of K trajectory strings, not a single
prompt (see data.py's module docstring). MULTI-SOURCE per condition:
  easy  positives  SafeMTData/SafeMTData (mit)           source tag "attack"
        negatives  HuggingFaceH4/ultrachat_200k (mit)    source tag "ultrachat"
  hard  positives  SafeMTData/SafeMTData (mit)           source tag "attack"
        negatives  SafeMTData/SafeMTData (mit)           source tag "attack_prefix"
        (a DISJOINT-group prefix half of the same attack pool -- see data.py)
  ood   intrinsec-ai/cstm-bench (mit), both splits        source tag "cstm"
Both conditions' non-OOD sources are already 'mit', so there is no licence tension in
picking one as "the" declared source; SafeMTData/SafeMTData is used since it is the
harmful class's origin, with the per-source row composition recorded in `extra`.

tom-gibbs Semi-Benign (`load_tom_gibbs_semi_benign`) is NOT exported: grepping
`results_embeddinggemma.json` and `run_cross_trajectory.py` for "tomgibbs"/"semi_benign"
finds no hits -- this lesson's actual run never used it, only multiturn_jailbreak's
alternate NEG_SOURCE does (and that lesson's own actual run also used "ultrachat",
never tomgibbs -- see multiturn_jailbreak/export_datasets.py).

Run: python -m steering_tutorials.cross_trajectory.export_datasets
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

_SOURCE_TO_HF = {"attack": SRC_ATTACK, "ultrachat": SRC_ULTRACHAT,
                 "attack_prefix": SRC_ATTACK}


def _condition_rows(built: dict) -> list:
    rows = []
    for sample, label, group, source in zip(built["samples"], built["labels"],
                                             built["groups"], built["sources"]):
        rows.append({"trajectories": sample, "label": int(label), "group": group,
                     "_source": _SOURCE_TO_HF[source], "_source_tag": source})
    return rows


def _ood_rows(built: dict) -> list:
    rows = []
    for sample, label, group in zip(built["samples"], built["labels"], built["groups"]):
        rows.append({"trajectories": sample, "label": int(label), "group": group,
                     "_source": SRC_OOD, "_source_tag": "cstm"})
    return rows


def main() -> None:
    manifests = []

    for condition in ("easy", "hard"):
        built = D.load_dataset(n_pos=C.N_POS, n_neg=C.N_NEG, k=C.K_TRAJ,
                               condition=condition, seed=C.SEED)
        rows = _condition_rows(built)
        composition = dict(Counter(r["_source_tag"] for r in rows))
        name = "crosstraj_%s_n%d_k%d_s%d" % (condition, built["meta"]["n_pos"],
                                             C.K_TRAJ, C.SEED)
        man = DE.export_slice(
            _HERE, name, SRC_ATTACK, rows, split=condition, seed=C.SEED,
            notes=("cross_trajectory %s condition: n_pos=%d n_neg=%d k=%d. Positives "
                   "are SafeMTData/SafeMTData; negatives are %s. Composition: %s."
                   % (condition, built["meta"]["n_pos"], built["meta"]["n_neg"],
                      C.K_TRAJ,
                      "HuggingFaceH4/ultrachat_200k" if condition == "easy"
                      else "SafeMTData/SafeMTData (disjoint-group prefix half)",
                      composition)),
            extra={"composition": composition, "meta": built["meta"]})
        manifests.append((name, SRC_ATTACK, man["licence"], man["n_rows"],
                          man["slice_fingerprint"]))
        print("[export] %s: n=%d fp=%s composition=%s"
              % (name, man["n_rows"], man["slice_fingerprint"], composition))

    ood_built = D.load_ood_cstm(k=C.OOD_K, seed=C.SEED, how=C.OOD_SELECT)
    ood_rows = _ood_rows(ood_built)
    ood_name = "crosstraj_ood_n%d_k%d_s%d" % (ood_built["meta"]["n_scenarios"],
                                              C.OOD_K, C.SEED)
    man = DE.export_slice(
        _HERE, ood_name, SRC_OOD, ood_rows, split="dilution+cross_session", seed=C.SEED,
        notes=("cross_trajectory OOD arm (CSTM-Bench, both splits). n_scenarios=%d "
               "n_attack=%d n_benign=%d selection=%s k=%d. Screening-tier: n~108."
               % (ood_built["meta"]["n_scenarios"], ood_built["meta"]["n_attack"],
                  ood_built["meta"]["n_benign"], C.OOD_SELECT, C.OOD_K)),
        extra={"meta": ood_built["meta"]})
    manifests.append((ood_name, SRC_OOD, man["licence"], man["n_rows"],
                      man["slice_fingerprint"]))
    print("[export] %s: n=%d fp=%s" % (ood_name, man["n_rows"], man["slice_fingerprint"]))

    print("\n[export] cross_trajectory: %d slice(s) written." % len(manifests))
    for name, src, lic, n, fp in manifests:
        print("  %-28s %-24s %-6s n=%-4d fp=%s" % (name, src, lic, n, fp))


if __name__ == "__main__":
    main()
