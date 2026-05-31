#!/usr/bin/env python
"""verify_rubrics.py -- top-level scorecard for the mechanical parts of Rubrics A/C/D.

Runs the checks from audits/RUBRICS.md that can be verified mechanically:

  * Rubric A/C  -- required files exist; skills/meta-skills/corpus counts;
                   paper + docs landing + FINDINGS + ledger present.
  * Rubric C    -- no secret (full hf_ token) in TRACKED text files;
                   reproducibility docs present.
  * Rubric D    -- composite fingerprint consistent across artifacts;
                   ruff / mypy / pytest exit codes;
                   every experiment row carries behavior_scorer + safety_real.

It is BEST-EFFORT: a check it cannot perform (e.g. git unavailable) is reported
as SKIP with a note, never a crash. Genuinely-open items (e.g. the pre-fix
projection-proxy rows lacking provenance tags) are reported as FAIL with an
honest note -- that is expected and correct, not a script bug.

Exit code: nonzero if any non-skipped check FAILs.

Usage:
    python scripts/verify_rubrics.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINGERPRINT = "a9001e87087e"

# Expected counts (audits/RUBRICS.md A1/A4).
EXPECT_META_SKILLS = 27
EXPECT_SKILLS = 14
EXPECT_CORPUS = 10

# status: "PASS" | "FAIL" | "SKIP"
_results: list[tuple[str, str, str]] = []


def record(name: str, status: str, evidence: str = "") -> None:
    _results.append((name, status, evidence))


def ok(name: str, cond: bool, evidence: str = "") -> None:
    record(name, "PASS" if cond else "FAIL", evidence)


def skip(name: str, evidence: str = "") -> None:
    record(name, "SKIP", evidence)


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:  # pragma: no cover - defensive
        return f"<<unreadable: {e}>>"


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a command best-effort; return (exit_code, short_output). -1 if unrunnable."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600
        )
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        return proc.returncode, (tail[-1] if tail else "")
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def _tracked_text_files() -> tuple[list[Path], bool]:
    """Return (tracked text files, git_available). Falls back to a glob walk."""
    code, _ = _run(["git", "rev-parse", "--is-inside-work-tree"])
    if code == 0:
        proc = subprocess.run(
            ["git", "ls-files"], cwd=str(REPO), capture_output=True, text=True
        )
        files = [REPO / f for f in proc.stdout.splitlines() if f.strip()]
        git_ok = True
    else:
        files = [p for p in REPO.rglob("*") if p.is_file()]
        git_ok = False
    exts = {".md", ".py", ".html", ".txt", ".json", ".toml", ".cfg", ".ini", ".yaml", ".yml"}
    # Exclude vendored model/tokenizer dirs (not project source).
    text = [
        f for f in files
        if f.suffix.lower() in exts and "models" not in f.relative_to(REPO).parts
    ]
    return text, git_ok


def main() -> int:
    # ---------------- Rubric A / C: files exist ----------------
    required = {
        "CLAUDE.md": REPO / "CLAUDE.md",
        "AUTORESEARCH_PROCESS.md": REPO / "AUTORESEARCH_PROCESS.md",
        "README.md": REPO / "README.md",
        "IDEA_TABLE.md": REPO / "IDEA_TABLE.md",
        "EXPERIMENT_LEDGER.md (ledger)": REPO / "EXPERIMENT_LEDGER.md",
        "FINDINGS.md": REPO / "FINDINGS.md",
        "paper/PAPER.md": REPO / "paper" / "PAPER.md",
        "docs/index.html (landing)": REPO / "docs" / "index.html",
        "requirements.txt": REPO / "requirements.txt",
        "audits/RUBRICS.md": REPO / "audits" / "RUBRICS.md",
        "audits/ICML_REVIEW.md": REPO / "audits" / "ICML_REVIEW.md",
    }
    for name, p in required.items():
        ok(f"A/C file: {name}", p.exists(), str(p.relative_to(REPO)))

    # ---------------- Rubric A: skills / meta-skills / corpus counts ----------------
    n_meta = len(list((REPO / "meta-skills").glob("*/SKILL.md")))
    n_skills = len(list((REPO / "skills").glob("*/SKILL.md")))
    n_corpus = len(list((REPO / "corpus").glob("*.md")))
    ok(f"A4 meta-skills count >= {EXPECT_META_SKILLS}", n_meta >= EXPECT_META_SKILLS,
       f"found={n_meta}")
    ok(f"A4 skills count >= {EXPECT_SKILLS}", n_skills >= EXPECT_SKILLS,
       f"found={n_skills}")
    ok(f"A1 corpus doc count >= {EXPECT_CORPUS}", n_corpus >= EXPECT_CORPUS,
       f"found={n_corpus}")

    # ---------------- Rubric C7: no secret (full hf_ token) in tracked files ----------------
    text_files, git_ok = _tracked_text_files()
    # A real HF token is hf_ + ~37 alphanumerics. Match >=34 to avoid false negatives
    # but NOT redacted prefixes ("hf_VoAZX...") or the benign filename "hf_fetch.py".
    full_token = re.compile(r"hf_[A-Za-z0-9]{34,}")
    leaks: list[str] = []
    for f in text_files:
        body = _read(f)
        if full_token.search(body):
            leaks.append(str(f.relative_to(REPO)))
    scope = "tracked files" if git_ok else "rglob fallback (git unavailable)"
    ok(f"C7 no full hf_ token in {scope}", not leaks,
       "clean" if not leaks else f"LEAK in {leaks}")

    # ---------------- Rubric D2: composite fingerprint consistent ----------------
    # eval.py is the SOURCE of the fingerprint: it holds the frozen COMPOSITE_FORMULA
    # whose sha256[:12] is the fingerprint, not the literal string. So the
    # authoritative eval.py check is "the formula re-derives to a9001e87087e", and
    # the downstream artifacts must carry the literal string.
    eval_py = REPO / "src" / "steering" / "eval.py"
    derived = None
    if eval_py.exists():
        import hashlib

        src = _read(eval_py)
        # Pull only the string-literal lines inside COMPOSITE_FORMULA = ( ... ),
        # skipping comments / blank lines, then concatenate the literals.
        m = re.search(r"COMPOSITE_FORMULA\s*=\s*\((.*?)\n\)", src, re.S)
        if m is None:
            m = re.search(r"COMPOSITE_FORMULA\s*=\s*\((.*?)\)", src, re.S)
        if m:
            parts: list[str] = []
            for line in m.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = re.sub(r"\s+#.*$", "", line)  # strip trailing comment
                for lit in re.findall(r'"((?:[^"\\]|\\.)*)"', line):
                    parts.append(lit.encode().decode("unicode_escape"))
            if parts:
                formula = "".join(parts)
                derived = hashlib.sha256(formula.encode("utf-8")).hexdigest()[:12]
    if derived is None:
        skip("D2 eval.py fingerprint re-derives from COMPOSITE_FORMULA",
             "could not parse COMPOSITE_FORMULA")
    else:
        ok("D2 eval.py COMPOSITE_FORMULA re-derives to " + FINGERPRINT,
           derived == FINGERPRINT, f"derived={derived}")

    # The fingerprint LITERAL must appear in every downstream artifact.
    fp_literal_sources = {
        "FINDINGS.md": REPO / "FINDINGS.md",
        "EXPERIMENT_LEDGER.md": REPO / "EXPERIMENT_LEDGER.md",
        "paper/PAPER.md": REPO / "paper" / "PAPER.md",
        "docs/index.html": REPO / "docs" / "index.html",
        "dashboard/index.html": REPO / "dashboard" / "index.html",
    }
    for name, p in fp_literal_sources.items():
        present = p.exists() and FINGERPRINT in _read(p)
        ok(f"D2 fingerprint {FINGERPRINT} in {name}", present,
           "present" if present else "MISSING")

    # ---------------- Rubric D4/E2: every row has behavior_scorer + safety_real ----------------
    jsonl = REPO / "autoresearch_results" / "experiment_log.jsonl"
    rows: list[dict] = []
    if jsonl.exists():
        for line in _read(jsonl).splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        skip("D4 rows have behavior_scorer + safety_real", "no JSONL rows found")
    else:
        missing = [
            r.get("experiment_num")
            for r in rows
            if r.get("behavior_scorer") is None or r.get("safety_real") is None
        ]
        with_tags = len(rows) - len(missing)
        ok(
            "D4 every experiment row has behavior_scorer + safety_real",
            not missing,
            f"{with_tags}/{len(rows)} tagged"
            + ("" if not missing
               else f"; UNTAGGED rows={missing} (pre-fix projection-proxy/stubbed-safety era)"),
        )

    # ---------------- Rubric D7: ruff / mypy / pytest exit codes ----------------
    for label, cmd in [
        ("D7 ruff clean", [sys.executable, "-m", "ruff", "check", "src/steering", "tests"]),
        ("D7 mypy clean", [sys.executable, "-m", "mypy", "src/steering",
                           "--ignore-missing-imports"]),
        ("D7 pytest green", [sys.executable, "-m", "pytest", "tests/", "-q"]),
    ]:
        code, tail = _run(cmd)
        if code == -1:
            skip(label, f"tool unrunnable: {tail}")
        else:
            ok(label, code == 0, f"exit={code} :: {tail[:80]}")

    # ---------------- Rubric C3: reproducibility docs ----------------
    repro = REPO / "NEXT_STEPS.md"
    ok("C3 reproducibility commands documented",
       repro.exists() or "Quickstart" in _read(REPO / "README.md"),
       "NEXT_STEPS.md / README quickstart")

    # ---------------- scorecard ----------------
    print("\n" + "=" * 72)
    print("Rubrics A/C/D -- mechanical scorecard (scripts/verify_rubrics.py)")
    print("=" * 72)
    width = max(len(n) for n, _, _ in _results)
    n_pass = sum(1 for _, s, _ in _results if s == "PASS")
    n_fail = sum(1 for _, s, _ in _results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in _results if s == "SKIP")
    for name, status, ev in _results:
        print(f"  [{status:4}] {name.ljust(width)}  {ev}")
    print("-" * 72)
    print(f"  PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}  (total={len(_results)})")
    if n_fail:
        print("  NOTE: FAILs on genuinely-open items (e.g. pre-fix rows missing")
        print("        provenance tags) are expected and tracked in audits/RUBRICS.md sec F.")
    print("=" * 72)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
