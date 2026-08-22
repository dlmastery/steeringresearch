"""artifact_paths.py -- keep a results file from being overwritten by a sibling arm.

WHY THIS EXISTS
---------------
Three lessons shipped the SAME defect, independently, and it cost a real artifact:

    EMB_CACHE   = {m: ARTIFACTS / f"cache_{m}.npz" for m in EMBEDDERS}   # keyed
    RESULTS_PATH = ARTIFACTS / "results.json"                            # NOT keyed

The caches were keyed by whatever varied; the results file was not. So running a
second arm silently destroyed the first arm's results while its cache survived,
which makes the loss easy to miss -- the expensive thing is still there.

  * `meerkat`  2026-08-21: running the `bge` arm DESTROYED the `minilm` results.
    Recovered from git only because they happened to be committed.
  * `cross_trajectory` hit it earlier and fixed it locally. The fix never
    propagated to its siblings, so the lesson stayed unlearned.
  * `rogue_scalpel` would have destroyed its `project_out` ladder when the
    `negative_add` mode was first run. Caught in pre-flight, not by luck.

The pattern is not a coincidence three times. It is what happens when caches are
written by an embedding helper that already thinks in variants, while the results
path is a module-level constant written once when the lesson had a single arm.

WHAT THIS DOES
--------------
`keyed_path` builds a per-variant filename and REFUSES an empty variant, so the
bare form cannot be reintroduced by passing an unset env var:

    RESULTS_PATH = keyed_path(ARTIFACTS, "results", ".json", EMBEDDER)
    # -> artifacts/results_embeddinggemma.json

`assert_no_bare_sibling` is the guard for the failure mode that actually bit us:
a bare `results.json` sitting NEXT TO keyed ones is the fossil of a run made
before the path was keyed, and it is indistinguishable from a current result.

CPU-only. Imports nothing heavy. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["keyed_path", "assert_no_bare_sibling", "audit_lesson_paths"]


def keyed_path(artifacts_dir, stem: str, ext: str, *variants) -> Path:
    """``<artifacts>/<stem>_<v1>_<v2><ext>``, refusing an empty variant.

    Every component that VARIES between runs of the same lesson belongs here --
    embedder, substrate, attack mode, base model. If two runs can differ in it
    and it is not in the filename, one run will overwrite the other.

    Raises
    ------
    ValueError
        If no variant is given, or any variant is empty/None. That is the bare
        form, and the bare form is the bug.
    """
    if not variants:
        raise ValueError(
            "keyed_path(%r) needs at least one variant. A results path with no "
            "variant is the defect this helper exists to prevent -- see the "
            "module docstring for the three lessons that shipped it." % stem)
    parts = []
    for v in variants:
        s = "" if v is None else str(v).strip()
        if not s:
            raise ValueError(
                "keyed_path(%r) got an EMPTY variant %r. An unset env var must "
                "not silently collapse the path back to the bare form." % (stem, v))
        parts.append(s.replace("/", "-").replace("\\", "-"))
    return Path(artifacts_dir) / ("%s_%s%s" % (stem, "_".join(parts), ext))


def assert_no_bare_sibling(artifacts_dir, stem: str, ext: str) -> None:
    """Raise if a bare ``<stem><ext>`` sits beside keyed ``<stem>_*<ext>`` files.

    That combination means an older, UNKEYED run left a file which is now
    indistinguishable from a current one -- exactly the state `meerkat` was in
    after its bge run overwrote minilm: one `results.json` on disk, no way to
    tell which arm produced it.
    """
    d = Path(artifacts_dir)
    bare = d / ("%s%s" % (stem, ext))
    if not bare.exists():
        return
    keyed = [p for p in d.glob("%s_*%s" % (stem, ext)) if p != bare]
    if keyed:
        raise SystemExit(
            "AMBIGUOUS ARTIFACT: %s exists alongside %d keyed sibling(s) (%s). "
            "The bare file predates the per-variant path and cannot be "
            "attributed to an arm. Rename it to its variant or delete it -- an "
            "artifact that cannot be attributed is not evidence."
            % (bare.name, len(keyed), ", ".join(p.name for p in keyed[:3])))


def audit_lesson_paths(root=None) -> list:
    """Find lessons whose results path is bare while their caches are keyed.

    Returns a list of ``(lesson, finding)``. Pure text inspection -- imports no
    lesson module, so a lesson with a heavy import cannot break the audit.
    """
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    out = []
    for cfg in sorted(root.glob("*/config.py")):
        try:
            src = cfg.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Match the ASSIGNMENT at line start, not the substring. A plain
        # `in src` also matches LEGACY_RESULTS_PATH = ARTIFACTS / "results.json",
        # which flagged cross_trajectory even though its ACTIVE path is keyed --
        # a false positive from exactly the substring-matching mistake this
        # module exists to catch elsewhere.
        bare = any(ln.strip().startswith("RESULTS_PATH")
                   and 'ARTIFACTS / "results.json"' in ln
                   for ln in src.splitlines())
        keyed_cache = ("_CACHE = {" in src) or ("%s.npz" in src)
        if bare and keyed_cache:
            out.append((cfg.parent.name,
                        "RESULTS_PATH is bare while caches are keyed -- a second "
                        "arm will overwrite the first arm's results"))
        elif bare:
            out.append((cfg.parent.name,
                        "RESULTS_PATH is bare; safe only while the lesson has "
                        "exactly one arm"))
    return out


def _self_test() -> None:
    import tempfile

    d = Path(tempfile.mkdtemp())
    p = keyed_path(d, "results", ".json", "embeddinggemma")
    assert p.name == "results_embeddinggemma.json", p.name
    assert keyed_path(d, "results", ".json", "shade", "seed0").name == \
        "results_shade_seed0.json"
    print("OK  keyed_path builds per-variant names")

    for bad in ((), (None,), ("",), ("  ",)):
        try:
            keyed_path(d, "results", ".json", *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("keyed_path accepted the bare form: %r" % (bad,))
    print("OK  the bare form is REFUSED, including via an unset env var")

    assert_no_bare_sibling(d, "results", ".json")          # nothing on disk yet
    (d / "results_minilm.json").write_text("{}", encoding="utf-8")
    assert_no_bare_sibling(d, "results", ".json")          # keyed only -> fine
    print("OK  keyed-only and empty directories pass")

    (d / "results.json").write_text("{}", encoding="utf-8")
    try:
        assert_no_bare_sibling(d, "results", ".json")
    except SystemExit:
        print("OK  a BARE file beside keyed siblings is REFUSED (the meerkat state)")
    else:
        raise AssertionError("bare sibling was not caught")

    findings = audit_lesson_paths()
    print("OK  audit_lesson_paths ran over the course: %d lesson(s) flagged" % len(findings))
    for name, why in findings:
        print("      %-28s %s" % (name, why))
    print("\nOK -- artifact_paths.py self-test passed CPU-only, no model, no GPU.")


if __name__ == "__main__":
    _self_test()
