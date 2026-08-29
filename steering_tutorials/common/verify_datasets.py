"""verify_datasets.py -- re-check every shipped slice against its own manifest.

WHY
---
Three agents wrote the slices and each reported that they verified their own
output. That is worth something, but it is self-report: the manifest and the
data were produced by the same process in the same run, so a systematic error
would be recorded as consistent. This walks the repo afterwards and re-derives
every checkable claim from the FILES, which is the only form of verification
this program treats as evidence (CLAUDE.md 18.8: an artifact that cannot be
regenerated from the code beside it is not evidence).

WHAT IT CHECKS, per slice
-------------------------
  * the data file exists and parses as gzipped JSONL
  * row count matches the manifest
  * the fingerprint RECOMPUTES to the recorded value
  * the licence matches the central registry (a manifest may not carry its own
    private opinion about a licence)
  * nothing gated slipped through as data
  * cc-by-nc slices carry their NON-COMMERCIAL notice
  * USES_SHARED.json pointers resolve to a slice that actually exists

Exit code is non-zero if anything fails, so it can gate a commit.

CPU-only, no network, no model. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from steering_tutorials.common.dataset_export import (GATED, licence_of,
                                                      slice_fingerprint)

ROOT = Path(__file__).resolve().parent.parent


def _read_rows(path: Path) -> list:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("%s line %d is not JSON: %s"
                                 % (path.name, i, exc))
    return rows


def main() -> int:
    manifests = sorted(ROOT.glob("*/datasets/*.manifest.json"))
    shared = sorted(ROOT.glob("*/datasets/USES_SHARED.json"))
    refetch = sorted(ROOT.glob("*/datasets/*.refetch.json"))
    problems, total_rows, total_bytes = [], 0, 0

    print("%-26s %-34s %-14s %7s %9s  %s"
          % ("lesson", "slice", "licence", "rows", "gz KB", "fingerprint"))
    print("-" * 108)
    for mp in manifests:
        lesson = mp.parent.parent.name
        try:
            man = json.loads(mp.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append("%s: manifest unreadable (%s)" % (mp.name, exc))
            continue
        dp = mp.parent / man.get("file", "")
        if not dp.exists():
            problems.append("%s: data file %r missing" % (mp.name, man.get("file")))
            continue
        try:
            rows = _read_rows(dp)
        except ValueError as exc:
            problems.append(str(exc))
            continue

        name = man.get("name", mp.stem)
        src = man.get("source", "")
        lic_manifest = man.get("licence")
        lic_registry = licence_of(src)
        fp_now = slice_fingerprint(rows)
        size = dp.stat().st_size
        total_rows += len(rows)
        total_bytes += size

        if len(rows) != man.get("n_rows"):
            problems.append("%s/%s: manifest says %s rows, file has %d"
                            % (lesson, name, man.get("n_rows"), len(rows)))
        if fp_now != man.get("slice_fingerprint"):
            problems.append("%s/%s: FINGERPRINT MISMATCH -- manifest %s, "
                            "recomputed %s (the data changed under its manifest)"
                            % (lesson, name, man.get("slice_fingerprint"), fp_now))
        if lic_registry == "UNKNOWN":
            problems.append("%s/%s: source %r is not in the licence registry"
                            % (lesson, name, src))
        elif lic_manifest != lic_registry:
            problems.append("%s/%s: manifest licence %r disagrees with the "
                            "registry's %r -- a manifest does not get its own "
                            "opinion about a licence"
                            % (lesson, name, lic_manifest, lic_registry))
        if licence_of(src) != "UNKNOWN" and src.strip().lower() in {
                k.strip().lower() for k in GATED}:
            problems.append("%s/%s: GATED source %r shipped as DATA"
                            % (lesson, name, src))
        if lic_registry == "cc-by-nc-4.0" and not man.get("non_commercial_notice"):
            problems.append("%s/%s: cc-by-nc slice with no NON-COMMERCIAL notice"
                            % (lesson, name))

        print("%-26s %-34s %-14s %7d %9.1f  %s"
              % (lesson[:26], name[:34], lic_registry[:14], len(rows),
                 size / 1024.0, fp_now))

    print("")
    for sp in shared:
        lesson = sp.parent.parent.name
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append("%s: USES_SHARED.json unreadable (%s)" % (lesson, exc))
            continue
        # The pointer records a RELATIVE PATH to the shared file, so resolve it
        # rather than guessing a slice name. (My first version of this check
        # looked for a "shared_slice" key that does not exist and reported eight
        # false failures -- the audit was wrong, not the data.)
        rel = str(d.get("shared_data_file") or "")
        if not rel:
            problems.append("%s: USES_SHARED.json names no shared_data_file" % lesson)
            continue
        target = (sp.parent / rel).resolve()
        if not target.exists():
            problems.append("%s: USES_SHARED.json points at %r, which does not "
                            "exist -- a pointer to nothing is worse than a copy"
                            % (lesson, rel))
            continue
        man_rel = str(d.get("shared_manifest_file") or "")
        if man_rel and not (sp.parent / man_rel).resolve().exists():
            problems.append("%s: shared_manifest_file %r does not exist"
                            % (lesson, man_rel))
        print("shared-pointer  %-24s -> %s  (n_per_class=%s seed=%s)"
              % (lesson[:24], target.name, d.get("n_per_class"), d.get("seed")))

    for rp in refetch:
        d = json.loads(rp.read_text(encoding="utf-8"))
        if d.get("licence") in (None, "", "UNKNOWN"):
            problems.append("%s: source %r has an UNVETTED licence. Withholding "
                            "the rows was right, but the licence still has to be "
                            "read and registered before the lesson can claim a "
                            "provenance." % (rp.name, d.get("source")))
        if d.get("redistributable") is not False:
            problems.append("%s: refetch manifest not marked non-redistributable"
                            % rp.name)
        sib = rp.parent / (d.get("name", "") + ".jsonl.gz")
        if sib.exists():
            problems.append("%s: a GATED source has a data file beside its "
                            "refetch manifest (%s)" % (rp.name, sib.name))
        print("refetch-only    %-24s    %s" % (rp.parent.parent.name[:24],
                                               d.get("source")))

    print("")
    print("%d slices, %d rows, %.2f MB total"
          % (len(manifests), total_rows, total_bytes / 1024.0 / 1024.0))
    print("%d shared pointers, %d refetch-only manifests"
          % (len(shared), len(refetch)))
    if problems:
        print("")
        print("FAILURES (%d):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return 1
    print("")
    print("OK -- every slice re-parses, re-fingerprints, and agrees with the "
          "central licence registry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
