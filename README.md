# steeringresearch

Autonomous, principled autoresearch on **conditional / activation steering of
small Gemma models** (Gemma-3-1B-it / Gemma-2-2B-it) on a single **RTX 4090
Laptop (16 GB)**. Publication-grade rigor, a CIFAR-style benchmark ladder, and a
transparent multi-page dashboard.

## Layout

| path | what |
|---|---|
| `CLAUDE.md` | the constitution — rules, winner def, dashboard mandate, rigor floor |
| `AUTORESEARCH_PROCESS.md` | the detailed inner loop |
| `meta-skills/` | **domain-agnostic** autoresearch process pack (reusable for any topic) |
| `skills/` | steering-specific instantiation skills |
| `corpus/` | the verbatim steering research corpus (literature, 12 axes, ladder, datasets, Rogue Scalpel, N1–N20) |
| `src/steering/` | the harness: intervention, extraction, eval bundle, runner, dashboard |
| `tests/` | Rung-0 unit tests |
| `ideas/` | per-hypothesis sub-projects (E1–E50, N1–N20) |
| `autoresearch_results/` | experiment log, champion, reasoning annotations |
| `dashboard/` + `docs/dashboard/` | master + per-hypothesis + per-experiment dashboards |
| `audits/` | impl-critic, sci-critic, leakage, meta-process audits |
| `IDEA_TABLE.md` / `EXPERIMENT_LEDGER.md` / `FINDINGS.md` | the ledgers |

## Quickstart

```bash
huggingface-cli login            # accept the Gemma license (gated)
pip install -r requirements.txt
pytest tests/                    # Rung 0 — plumbing
python -m steering.runner --help # the single-experiment executor
```

Start with `meta-skills/autoresearch-meta/SKILL.md` (the process spine), then
`IDEA_TABLE.md` (where the program is). All inherited corpus numbers are
`[NEEDS VERIFICATION]` until reproduced on the 4090 ladder.

## Docs, paper, and verification

| link | what |
|---|---|
| `docs/index.html` | GitHub-Pages landing page → dashboard, paper, findings, audits |
| `docs/dashboard/index.html` | the live multi-page dashboard (master + per-hypothesis + per-experiment) |
| `paper/PAPER.md` | honest ICML-format methods / harness / **n=1 screening** draft (no overclaiming) |
| `FINDINGS.md` | rigor-gated findings (currently zero external-ready; S-1..S-4 are screening-only) |
| `audits/RUBRICS.md` | the scoreable PASS/FAIL rubrics A–E |

Status: **methodology + harness + n=1 screening; external steering claims are gated**
on the required experiments enumerated in the paper and `FINDINGS.md`. Composite
formula fingerprint `a9001e87087e`.

Verify mechanically:

```bash
python scripts/verify_dashboard.py   # Rubric B (dashboard) — PASS/FAIL table
python scripts/verify_rubrics.py      # Rubrics A/C/D — scorecard (ruff+mypy+pytest+secrets+fingerprint)
```
