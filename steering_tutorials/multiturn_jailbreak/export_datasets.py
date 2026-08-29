"""export_datasets.py -- ship the exact multi-turn conversations MULTITURN_JAILBREAK
used.

CPU-only, no model, no GPU. `data.load_dataset` / `data.load_ood` are pure dataset
builders (network + local HF cache only, no LLM). Rebuilds at the config.py defaults,
which match `results_embgemma.json`'s `run_config` exactly: n_pos=n_neg=600,
min_turns=3, max_turns=8, hard_window=4, seed=0, both conditions ("easy"/"hard"),
OOD splits cross_session+dilution, ood_window=4.

Each row is one CONVERSATION = a list of user-turn strings, label 1=attack/
0=benign. MULTI-SOURCE:
  easy  positives "attack"    SafeMTData/SafeMTData (mit)
        negatives neg_source  the run_config's actual `neg_source` -- "ultrachat"
                              (HuggingFaceH4/ultrachat_200k, mit) for the run that
                              produced results_embgemma.json/results_gemma.json.
                              MJ_NEG_SOURCE=tomgibbs is a config OPTION
                              (tom-gibbs/multi-turn_jailbreak_attack_datasets, mit)
                              never exercised by the shipped results -- not exported.
  hard  positives "attack_full"    SafeMTData/SafeMTData (last W turns, incl. payload)
        negatives "attack_prefix"  SafeMTData/SafeMTData (first W turns of a
                                   DIFFERENT, disjoint-group attack -- benign lead-up)
  ood   cstm-bench/<split>/<class>  intrinsec-ai/cstm-bench (mit), both splits

Run: python -m steering_tutorials.multiturn_jailbreak.export_datasets
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
SRC_TOMGIBBS = "tom-gibbs/multi-turn_jailbreak_attack_datasets"
SRC_OOD = "intrinsec-ai/cstm-bench"


def _hf_source_for(tag: str) -> str:
    # "easy" tags positives "attack"; "hard" tags them "attack_full" (last W turns,
    # includes the payload) vs "attack_prefix" (first W turns of a DIFFERENT attack,
    # a disjoint-group benign lead-up) -- see data._build_hard. All three are the
    # same upstream dataset.
    if tag in ("attack", "attack_full", "attack_prefix"):
        return SRC_ATTACK
    if tag == "ultrachat":
        return SRC_ULTRACHAT
    if tag.startswith("tomgibbs"):
        return SRC_TOMGIBBS
    raise ValueError("unrecognised source tag %r" % tag)


def _condition_rows(built: dict) -> list:
    rows = []
    for conv, label, group_key, category, source in zip(
            built["conversations"], built["labels"], built["group_keys"],
            built["categories"], built["sources"]):
        rows.append({"turns": conv, "label": int(label), "group_key": group_key,
                     "category": category, "_source": _hf_source_for(source),
                     "_source_tag": source})
    return rows


def _ood_rows(built: dict) -> list:
    rows = []
    for conv, label, group_key, category, source in zip(
            built["conversations"], built["labels"], built["group_keys"],
            built["categories"], built["sources"]):
        rows.append({"turns": conv, "label": int(label), "group_key": group_key,
                     "category": category, "_source": SRC_OOD, "_source_tag": source})
    return rows


def main() -> None:
    manifests = []

    for condition in ("easy", "hard"):
        built = D.load_dataset(n_pos=C.N_POS, n_neg=C.N_NEG,
                               min_turns=C.MIN_USER_TURNS, max_turns=C.MAX_USER_TURNS,
                               seed=C.SEED, condition=condition)
        rows = _condition_rows(built)
        composition = dict(Counter(r["_source_tag"] for r in rows))
        m = built["meta"]
        name = "mtjb_%s_npos%d_nneg%d_s%d" % (condition, m["n_pos_achieved"],
                                              m["n_neg_achieved"], C.SEED)
        man = DE.export_slice(
            _HERE, name, SRC_ATTACK, rows, split=condition, seed=C.SEED,
            notes=("multiturn_jailbreak %s condition: n_pos=%d n_neg=%d "
                   "neg_source=%s distinct_groups=%d. Composition: %s."
                   % (condition, m["n_pos_achieved"], m["n_neg_achieved"],
                      m["neg_source"], m["n_distinct_groups"], composition)),
            extra={"composition": composition, "meta": m})
        manifests.append((name, SRC_ATTACK, man["licence"], man["n_rows"],
                          man["slice_fingerprint"]))
        print("[export] %s: n=%d fp=%s composition=%s"
              % (name, man["n_rows"], man["slice_fingerprint"], composition))

    ood = D.load_ood(splits=C.OOD_SPLITS, window=C.OOD_WINDOW,
                     min_turns=C.MIN_USER_TURNS)
    ood_rows = _ood_rows(ood)
    m = ood["meta"]
    ood_name = "mtjb_ood_n%d_s%d" % (m["n"], C.SEED)
    man = DE.export_slice(
        _HERE, ood_name, SRC_OOD, ood_rows, split="+".join(C.OOD_SPLITS), seed=C.SEED,
        notes=("multiturn_jailbreak OOD arm (CSTM-Bench, splits=%s, window=%d). "
               "n=%d pos=%d neg=%d scenarios=%d. %s"
               % (C.OOD_SPLITS, C.OOD_WINDOW, m["n"], m["n_pos_achieved"],
                  m["n_neg_achieved"], m["n_distinct_groups"],
                  m["rule1_exempt_reason"])),
        extra={"meta": m})
    manifests.append((ood_name, SRC_OOD, man["licence"], man["n_rows"],
                      man["slice_fingerprint"]))
    print("[export] %s: n=%d fp=%s" % (ood_name, man["n_rows"], man["slice_fingerprint"]))

    print("\n[export] multiturn_jailbreak: %d slice(s) written." % len(manifests))
    for name, src, lic, n, fp in manifests:
        print("  %-30s %-24s %-6s n=%-4d fp=%s" % (name, src, lic, n, fp))


if __name__ == "__main__":
    main()
