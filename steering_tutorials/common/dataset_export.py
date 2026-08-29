"""dataset_export.py -- ship the EXACT data each lesson used, or a way to refetch it.

WHY SLICES AND NOT CORPORA
--------------------------
Every lesson samples a deterministic slice (seed + N_PER_CLASS + a
pool_fingerprint) out of a much larger upstream dataset. The slice is what the
numbers were actually computed from, so the slice is what belongs beside them:
it is smaller, and unlike "we used lmsys/toxic-chat" it is checkable. Shipping
the upstream corpora instead would add gigabytes and still not pin which rows a
result came from.

THE LICENCE GATE IS STRUCTURAL, NOT A REMINDER
----------------------------------------------
These repos are PUBLIC. Re-publishing a gated dataset bypasses an access
decision its uploader made, whatever its licence says -- `walledai/HarmBench` is
MIT AND gated, and the gate is the intent that governs. `export_slice` therefore
REFUSES any source not present in REDISTRIBUTABLE, and there is no override
argument. A gated source gets a manifest (ids + fingerprint + loader) so a
reader can refetch it under their own accepted terms.

Licences below were read from each dataset's own card via HfApi.dataset_info on
2026-08-29, not from memory. `gated` is the hub's own field.

CPU-only, no model, no network. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

__all__ = ["REDISTRIBUTABLE", "GATED", "NO_LICENCE_STATED", "licence_of", "assert_redistributable",
           "export_slice", "write_refetch_manifest", "slice_fingerprint"]

# --- verified 2026-08-29 via HfApi.dataset_info(...).cardData['license'] ------
REDISTRIBUTABLE = {
    "JailbreakBench/JBB-Behaviors": "mit",
    "intrinsec-ai/cstm-bench": "mit",
    "lmsys/toxic-chat": "cc-by-nc-4.0",          # NON-COMMERCIAL; notice required
    "SafeMTData/SafeMTData": "mit",
    "HuggingFaceH4/ultrachat_200k": "mit",
    "tom-gibbs/multi-turn_jailbreak_attack_datasets": "mit",
    "AI45Research/ATBench": "apache-2.0",
    "nvidia/Aegis-AI-Content-Safety-Dataset-2.0": "cc-by-4.0",
    "jackhhao/jailbreak-classification": "apache-2.0",
    # verified 2026-08-29 by me, not taken on report. NOTE the hub's canonical
    # id capitalises differently from the form the lessons use
    # ("JailbreakV-28K/JailBreakV-28k"); HF resolves ids case-insensitively and
    # a dict does not, so lookups here are case-folded (see _norm) rather than
    # relying on either spelling being the one a caller happens to pass.
    "JailBreakV-28K/JailBreakV-28K": "mit",
    "Paul/XSTest": "cc-by-4.0",
    "PKU-Alignment/BeaverTails": "cc-by-nc-4.0",   # NON-COMMERCIAL
}

# Gated on the hub. NOT redistributable here even where the licence is
# permissive -- the gate is an access decision, and republishing routes around
# it. Value is (licence, why).
GATED = {
    "allenai/wildguardmix": ("odc-by", "gated: auto-approval still requires accepting terms"),
    "allenai/wildjailbreak": ("odc-by", "gated: auto-approval still requires accepting terms"),
    "ScaleAI/mhj": ("cc-by-nc-4.0", "gated AND non-commercial"),
    "walledai/HarmBench": ("mit", "MIT but GATED -- the gate governs, not the licence"),
    "lmsys/lmsys-chat-1m": ("other", "gated; per-record terms"),
}

# CHECKED, and the card states NO licence. This is a THIRD state, distinct from
# both "permissive" and "never looked at": absence of a licence is not
# permission, it is default copyright, so these may not be redistributed even
# though the hub does not gate them. Recorded explicitly so a later reader can
# tell a verified absence from an unexamined one -- absence of evidence is the
# defect, not a clean bill (CLAUDE.md 18.6).
NO_LICENCE_STATED = {
    "adityaasinha28/control_arena_agentdojo":
        "no license field in the dataset card (checked 2026-08-29); ungated, "
        "but silence is default copyright, not permission",
    "adityaasinha28/control_arena_shade":
        "no license field in the dataset card (checked 2026-08-29); ungated, "
        "but silence is default copyright, not permission",
}

_NC = ("This slice is NON-COMMERCIAL (cc-by-nc-4.0). It may be used for "
       "research and teaching only, and that restriction travels with it.")


def _norm(s: str) -> str:
    """Case-folded id. HuggingFace resolves ids case-insensitively; a dict does
    not. `JailbreakV-28K/JailBreakV-28k` (the hub's canonical spelling) and
    `JailBreakV-28K/JailBreakV-28K` (what the lesson passes) are the SAME
    dataset, and a case-sensitive lookup would refuse one of them with a message
    about licences that has nothing to do with licences.
    """
    return str(s).strip().lower()


_REDIST_CI = {_norm(k): v for k, v in REDISTRIBUTABLE.items()}
_GATED_CI = {_norm(k): v for k, v in GATED.items()}
_NOLIC_CI = {_norm(k): v for k, v in NO_LICENCE_STATED.items()}


def licence_of(source: str) -> str:
    k = _norm(source)
    if k in _REDIST_CI:
        return _REDIST_CI[k]
    if k in _GATED_CI:
        return _GATED_CI[k][0]
    if k in _NOLIC_CI:
        return "NO-LICENCE-STATED"
    return "UNKNOWN"


def assert_redistributable(source: str) -> str:
    """Return the licence, or raise. The only door to writing data to disk."""
    k = _norm(source)
    if k in _REDIST_CI:
        return _REDIST_CI[k]
    if k in _GATED_CI:
        lic, why = _GATED_CI[k]
        raise SystemExit(
            "REFUSING to export %r into a PUBLIC repo.\n"
            "  licence : %s\n"
            "  reason  : %s\n"
            "Ship a refetch manifest instead (write_refetch_manifest), which "
            "records the row ids and fingerprint without redistributing the "
            "rows themselves." % (source, lic, why))
    if k in _NOLIC_CI:
        raise SystemExit(
            "REFUSING to export %r into a PUBLIC repo.\n"
            "  licence : NONE STATED\n"
            "  reason  : %s\n"
            "Silence is not a permissive licence -- an unlicensed work is "
            "default-copyright, so the absence of terms grants nothing. Ship a "
            "refetch manifest instead."
            % (source, _NOLIC_CI[k]))
    raise SystemExit(
        "REFUSING to export %r: it is in neither REDISTRIBUTABLE nor GATED, so "
        "its licence has not been checked. Read the licence from the dataset "
        "card and add it to dataset_export.py before exporting. An unchecked "
        "licence is not a permissive one." % source)


def slice_fingerprint(rows) -> str:
    """Stable over row ORDER-INDEPENDENT content, so a reshuffle is detected."""
    h = hashlib.sha256()
    for blob in sorted(json.dumps(r, sort_keys=True, ensure_ascii=True)
                       for r in rows):
        h.update(blob.encode("ascii"))
    return h.hexdigest()[:16]


def _write_jsonl_gz(path: Path, rows) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path.stat().st_size


def export_slice(lesson_dir, name: str, source: str, rows, *, split: str = "",
                 seed=None, notes: str = "", extra: dict | None = None) -> dict:
    """Write `<lesson>/datasets/<name>.jsonl.gz` plus its manifest.

    Raises rather than writing if `source` is gated or unvetted.
    """
    lic = assert_redistributable(source)
    rows = list(rows)
    d = Path(lesson_dir) / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    data_path = d / ("%s.jsonl.gz" % name)
    size = _write_jsonl_gz(data_path, rows)
    fp = slice_fingerprint(rows)
    man = {
        "name": name, "source": source, "source_type": "huggingface",
        "licence": lic, "redistributable": True,
        "non_commercial": lic == "cc-by-nc-4.0",
        "split": split, "seed": seed, "n_rows": len(rows),
        "fields": sorted({k for r in rows for k in r}) if rows else [],
        "slice_fingerprint": fp, "bytes_gz": size,
        "file": data_path.name,
        "licence_verified": "2026-08-29 via HfApi.dataset_info cardData",
        "notes": notes,
    }
    if lic == "cc-by-nc-4.0":
        man["non_commercial_notice"] = _NC
    if extra:
        man["extra"] = extra
    mp = d / ("%s.manifest.json" % name)
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    os.replace(tmp, mp)
    return man


def write_refetch_manifest(lesson_dir, name: str, source: str, *,
                           row_ids=None, n_rows=None, seed=None,
                           loader_hint: str = "", notes: str = "") -> dict:
    """For a GATED source: record HOW to get the rows, never the rows."""
    lic, why = _GATED_CI.get(_norm(source),
                             (licence_of(source), "not gated; see notes"))
    d = Path(lesson_dir) / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    man = {
        "name": name, "source": source, "source_type": "huggingface",
        "licence": lic, "redistributable": False, "reason_withheld": why,
        "n_rows": n_rows if n_rows is not None else (len(row_ids) if row_ids else None),
        "row_ids": list(row_ids) if row_ids else None,
        "seed": seed, "loader_hint": loader_hint,
        "licence_verified": "2026-08-29 via HfApi.dataset_info cardData",
        "how_to_obtain": (
            "Accept the dataset's terms on its HuggingFace page, then "
            "`huggingface-cli login` and re-run the lesson's loader. The rows "
            "are NOT included here: republishing a gated dataset would bypass "
            "the access decision its uploader made."),
        "notes": notes,
    }
    mp = d / ("%s.refetch.json" % name)
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    os.replace(tmp, mp)
    return man


def _self_test() -> None:
    import tempfile

    d = Path(tempfile.mkdtemp())
    rows = [{"id": i, "text": "row %d" % i, "label": i % 2} for i in range(5)]
    m = export_slice(d, "smoke", "AI45Research/ATBench", rows, seed=0)
    assert m["licence"] == "apache-2.0" and m["n_rows"] == 5
    assert (d / "datasets" / "smoke.jsonl.gz").exists()
    print("OK  a permissive source exports data + manifest (%s, fp %s)"
          % (m["licence"], m["slice_fingerprint"]))

    back = []
    with gzip.open(d / "datasets" / "smoke.jsonl.gz", "rt", encoding="utf-8") as fh:
        for ln in fh:
            back.append(json.loads(ln))
    assert back == rows and slice_fingerprint(back) == m["slice_fingerprint"]
    print("OK  the written slice round-trips and re-fingerprints identically")

    assert slice_fingerprint(list(reversed(rows))) == m["slice_fingerprint"]
    assert slice_fingerprint(rows[:-1]) != m["slice_fingerprint"]
    print("OK  fingerprint is order-independent but content-sensitive")

    m2 = export_slice(d, "nc", "lmsys/toxic-chat", rows)
    assert m2["non_commercial"] and "non_commercial_notice" in m2
    print("OK  a cc-by-nc source carries its NON-COMMERCIAL notice")

    for bad in ("walledai/HarmBench", "ScaleAI/mhj", "allenai/wildjailbreak"):
        try:
            export_slice(d, "nope", bad, rows)
        except SystemExit as exc:
            assert "REFUSING" in str(exc)
        else:
            raise AssertionError("exported a GATED dataset: %s" % bad)
    print("OK  every GATED source is REFUSED -- including HarmBench, which is MIT")

    try:
        export_slice(d, "nope", "someone/never-vetted", rows)
    except SystemExit as exc:
        assert "licence has not been checked" in str(exc)
        print("OK  an UNVETTED source is refused too (unchecked != permissive)")
    else:
        raise AssertionError("exported an unvetted dataset")

    r = write_refetch_manifest(d, "mhj", "ScaleAI/mhj", row_ids=["a", "b"], seed=0)
    assert r["redistributable"] is False and r["row_ids"] == ["a", "b"]
    assert not (d / "datasets" / "mhj.jsonl.gz").exists()
    print("OK  a gated source yields a refetch manifest and NO data file")
    print("")
    print("OK -- dataset_export.py: slices ship, gated sources cannot.")


if __name__ == "__main__":
    _self_test()
