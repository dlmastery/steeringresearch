"""export_datasets.py -- ship the three prompt sets hello_world actually trained
and evaluated on.

hello_world is DELIBERATELY standalone (config.py's own docstring) and does not
import `steering_tutorials.common`, so unlike most of this dispatch it owns
distinct data of its own, not a shared slice. Three sources, three exports:

  1. `data.py`     -- JailbreakBench (Chao et al. 2024, arXiv:2404.01318): the
                      100 harmful + 100 benign starter set `train_probe.py`'s
                      SMOKE arm trains on. Small by the >=500/class rubric
                      (CLAUDE.md Sec.17 rubric item 1) -- exported as-is and
                      labelled as the smoke/starter set, never a headline.
  2. `data_large.py` (DATASETS.md) -- lmsys/toxic-chat, the >=500/class-target
                      REAL training set (374/class achieved -- toxic-chat's own
                      deduped toxic pool is the ceiling, per rubric item 2; see
                      artifacts/large_prompts.json header). This is what
                      `train_large.py` actually trains the reported probe on.
  3. `data_hard.py` (HARD_DATASETS.md) -- the adversarial OOD eval set:
                      JailBreakV-28K (obfuscated harmful, label 1) paired with
                      XSTest safe prompts (superficially-alarming benign,
                      label 0). Two upstream sources in one file, so it is
                      exported as TWO slices (one per source), which is also
                      the natural licence unit -- JailBreakV-28K is MIT, XSTest
                      is CC-BY-4.0, so a single mixed "source" string would be
                      wrong for `export_slice`'s licence gate.

Sets 2 and 3 are read from the artifacts ALREADY on disk
(`artifacts/large_prompts.json`, `artifacts/hard_prompts.json`) rather than
re-fetched: those are the literal rows `train_large.py` / the OOD eval already
ran against, each carrying its own seed/n_per_class header, so reading them is
more faithful than re-deriving a fresh sample that might drift from what was
actually measured. Set 1 has no such cache (data.py returns a plain list), so
it is loaded via its own loader -- deterministic, unseeded, the full matched
100+100.

CPU-only, no model, no GPU. Only network use is `hf_hub_download`'s cache
lookup for JailbreakBench, and every dataset here is already in the local HF
hub cache from prior runs, so nothing here re-downloads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steering_tutorials.common.dataset_export import (
    REDISTRIBUTABLE, export_slice, licence_of,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def export_jbb() -> dict:
    """Set 1: JailbreakBench starter/smoke set (data.py, 100+100, unseeded)."""
    from steering_tutorials.hello_world.data import load_safety_dataset

    prompts, labels = load_safety_dataset()
    rows = [{"prompt": p, "label": int(y)} for p, y in zip(prompts, labels)]
    man = export_slice(
        HERE, "jbb_starter_n100", "JailbreakBench/JBB-Behaviors", rows,
        split="harmful-behaviors+benign-behaviors",
        notes=(
            "The exact 100 harmful (label 1) + 100 benign (label 0) prompts "
            "data.py.load_safety_dataset() returns -- the SMOKE/starter arm "
            "train_probe.py trains on by default. 100/class is BELOW the "
            ">=500/class rubric (CLAUDE.md Sec.17 item 1); never quote this "
            "as a headline number, see data_large.py for the real training "
            "set. Deterministic and unseeded: the full matched CSV pair, no "
            "sampling."
        ),
        extra={"n_harmful": labels.count(1), "n_benign": labels.count(0),
               "role": "smoke/starter set, cross-checked against >=500/class rubric"},
    )
    print("[export] jbb_starter_n100: %d rows, %.2f KB gz, fp=%s"
          % (man["n_rows"], man["bytes_gz"] / 1e3, man["slice_fingerprint"]))
    return man


def export_large() -> dict:
    """Set 2: the toxic-chat >=500/class-target training set (data_large.py)."""
    src = ARTIFACTS / "large_prompts.json"
    blob = json.loads(src.read_text(encoding="utf-8"))
    header, rows = blob["header"], blob["rows"]
    n = header["n_per_class_actual"]
    seed = header["seed"]

    man = export_slice(
        HERE, "toxicchat_n%d_s%d" % (n, seed), "lmsys/toxic-chat", rows,
        split=header["split_used"], seed=seed,
        notes=(
            "The literal rows train_large.py trained the reported probe on, "
            "read from the already-committed artifacts/large_prompts.json "
            "(not re-sampled). Requested %d/class, ACHIEVED %d/class -- "
            "toxic-chat's own deduped toxic pool (%d) is the ceiling, per "
            "DATASETS.md's principled-sampling writeup (rubric item 2: "
            "pool-limited concept lessons must say so honestly). Natural "
            "toxic base rate %.4f, %d exact/near-dup rows dropped, %d "
            "ambiguous (label appeared under both classes) dropped."
            % (header["n_per_class_requested"], n, header["harmful_pool"],
               header["natural_base_rate"]["toxic_fraction"],
               header["n_dropped_duplicates"], header["n_dropped_ambiguous"])
        ),
        extra={"large_prompts_header": header},
    )
    print("[export] toxicchat_n%d_s%d: %d rows, %.2f KB gz, fp=%s"
          % (n, seed, man["n_rows"], man["bytes_gz"] / 1e3, man["slice_fingerprint"]))
    return man


def export_hard() -> list[dict]:
    """Set 3: the adversarial OOD eval set (data_hard.py), split by source.

    JailBreakV-28K/JailBreakV-28K (harmful, label 1) and Paul/XSTest (benign,
    label 0) are two different upstream sources under two different licences,
    so they ship as two slices. BOTH are checked against
    dataset_export.REDISTRIBUTABLE before writing anything: as of when this was
    written neither is in that dict yet (verified separately via
    HfApi.dataset_info: JailBreakV-28K/JailBreakV-28K=mit,
    Paul/XSTest=cc-by-4.0, both ungated) -- ds-common owns dataset_export.py
    and was asked to add them. This function checks first and SKIPS (loudly,
    not silently) rather than crashing the whole export run if they are not
    yet vetted, so re-running this script after the spine is updated is all
    that is needed to pick the export back up.
    """
    src = ARTIFACTS / "hard_prompts.json"
    blob = json.loads(src.read_text(encoding="utf-8"))
    recs = blob["records"]
    n = blob["n_per_class"]
    seed = blob["seed"]

    harmful_rows = [r for r in recs if r["label"] == 1]
    benign_rows = [r for r in recs if r["label"] == 0]
    assert all(r["source"].startswith("JailBreakV-28K/") for r in harmful_rows)
    assert all(r["source"] == "XSTest/safe" for r in benign_rows)

    plan = [
        ("JailBreakV-28K/JailBreakV-28K", "hard_harmful_jailbreakv28k_n%d_s%d" % (n, seed),
         harmful_rows,
         "The obfuscated-harmful side of the adversarial OOD eval set "
         "(HARD_DATASETS.md): real JailBreakV-28K redteam_query rows wrapped "
         "in a text-native jailbreak format (Template/Persuade/Logic). Read "
         "from the already-committed artifacts/hard_prompts.json, not "
         "re-sampled. Meant to be scored zero-shot by a probe trained on "
         "sets 1/2 above -- recall here is the honest robustness number."),
        ("Paul/XSTest", "hard_benign_xstest_n%d_s%d" % (n, seed),
         benign_rows,
         "The superficially-alarming-but-benign side of the adversarial OOD "
         "eval set: XSTest 'safe' contrast prompts (e.g. 'how do I kill a "
         "Python process'). Read from artifacts/hard_prompts.json, not "
         "re-sampled. Stresses over-refusal / false-positive precision, the "
         "opposite failure mode from the harmful side."),
    ]

    written = []
    for source, name, rows, notes in plan:
        if source not in REDISTRIBUTABLE:
            print("[export] SKIP %s (source=%r): not yet in "
                  "dataset_export.REDISTRIBUTABLE (licence_of=%r). Verified "
                  "separately via HfApi.dataset_info as ungated/%s -- asked "
                  "ds-common (who owns dataset_export.py) to add it. Re-run "
                  "this script once that lands." % (name, source,
                                                     licence_of(source),
                                                     licence_of(source)))
            continue
        man = export_slice(HERE, name, source, rows, seed=seed, notes=notes,
                           extra={"hard_prompts_header":
                                  {k: v for k, v in blob.items() if k != "records"}})
        print("[export] %s: %d rows, %.2f KB gz, fp=%s"
              % (name, man["n_rows"], man["bytes_gz"] / 1e3, man["slice_fingerprint"]))
        written.append(man)
    return written


def main() -> None:
    export_jbb()
    export_large()
    export_hard()


if __name__ == "__main__":
    main()
