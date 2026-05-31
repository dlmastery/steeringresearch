# ICML Final Sign-Off Review — Steering Autoresearch Program

> Reviewer role: the same hostile-but-fair senior ICML Area Chair from
> `audits/ICML_REVIEW.md`, now performing the **final sign-off** against the
> explicit acceptance rubric (`audits/RUBRICS.md`, Rubric E). The project is
> scored **AS WHAT IT HONESTLY IS**: a methodology + reproducible-harness +
> n=1 screening-results contribution — **not** a steering-efficacy-claims paper.
> This is an **internal QA pass**, not an external seal of approval (see §7).
>
> Date of sign-off pass: 2026-05-31.

---

## 1. VERDICT

**CONDITIONAL ACCEPT — sign-off WITHHELD pending two non-scientific blockers.**

The scientific substance, scoping discipline, measurement-validity fixes, and
numeric reconciliation all **pass**. Both original cruxes (W1 circular behavior
proxy; W2 stubbed safety) are genuinely fixed in code and in the live log. No
SCREENING result has leaked into an external claim. Limitations are in the
abstract; the required-experiments list is present and honest; the
same-model-family circularity is disclosed. The Rubric-E target sign-off
statement is **earned on the science**.

However, I cannot sign the clean-gates attestation as written because **two
reproducible, mechanical contradictions** exist between the artifacts' own
claims and their current state. These are cheap to fix (minutes) and do not
touch any scientific conclusion, but the sign-off statement explicitly asserts
"no internal contradictions" (E8) and the paper asserts a clean `ruff` gate
(§8) — both currently false. **Fix the two blockers below and the sign-off is
immediate.**

### Blocking items (both non-scientific, both reproduced by me)

1. **BLOCKER-1 (lint regression — contradicts paper §8 and `CODE_QUALITY.md` §564).**
   `python -m ruff check src/steering tests` **FAILS** (exit 1, 3 errors) on
   `tests/test_geometry.py`: one `I001` (unsorted import block, L62) and two
   `E702` (multiple statements on one line via `;`, L68/L69). These rules are
   in the repo's own selected set (`pyproject.toml` ruff `select = E4,E7,E9,F,I`),
   so they are in-scope, not spurious. This file appears to post-date the
   `CODE_QUALITY.md` cleanup (it tests the new `angular_displacement` metric
   added in campaign C3). Paper §8 claims "`ruff check src/steering tests`
   (clean)" and `CODE_QUALITY.md` §564 claims "All gates now CLEAN … All checks
   passed!" — both are now false. `verify_rubrics.py` itself flags this as
   `[FAIL] D7 ruff clean`. **Fix:** `ruff check --fix src/steering tests` (one
   error is auto-fixable; split the two `;` lines), then the gate is green and
   the §8 claim becomes true.

2. **BLOCKER-2 (stale internal contradiction in `FINDINGS.md` — E8).**
   `FINDINGS.md` §Status (lines 25–29) still reads: *"No external-ready findings
   yet — program initialized 2026-05-30. The research program is in the
   pre-experiment backbone phase … The first experiments (E1-E3 infrastructure
   block) have not yet run."* This is **directly contradicted** by S-1…S-8
   immediately below it (which report exp#2–59 across three models) and by the
   entire paper §5. The "no external-ready findings" half is correct and must
   stay; the "experiments have not yet run / pre-experiment backbone phase" half
   is false and is an internal contradiction the rubric (E8) explicitly checks.
   Additionally, line 19 still carries the composite fingerprint as a literal
   placeholder stub: `` `[TO BE FILLED AFTER eval.py IS WRITTEN -- placeholder
   a9001e87087e]` `` — the fingerprint (`a9001e87087e`) is in fact final and
   verified (see §3), so the "TO BE FILLED / placeholder" wording is stale.
   **Fix:** update the §Status paragraph to "screening observations recorded
   (exp#2–59); no result has cleared the rung-3 contract, so no external-ready
   findings exist," and replace line 19 with the confirmed fingerprint.

*Neither blocker changes a single number, conclusion, or scope statement. Once
both are corrected, this review converts to an unconditional sign-off of the
Rubric-E target statement.*

---

## 2. Verify-script + lint results (reproduced read-only)

| Check | Command | Result |
|---|---|---|
| Rubric B (dashboard) | `python scripts/verify_dashboard.py` | **PASS 15/15** (exit 0) |
| Rubrics A/C/D | `python scripts/verify_rubrics.py` | **PASS 24 / FAIL 2** (exit 1) |
| ruff | `python -m ruff check src/steering tests` | **FAIL** — 3 errors in `tests/test_geometry.py` (1×I001, 2×E702) |
| mypy | `python -m mypy src/steering --ignore-missing-imports` | **PASS** — "no issues found in 10 source files" |
| pytest | (via `verify_rubrics.py` D7) | **PASS** (green) |

The two `verify_rubrics.py` FAILs:
- `D7 ruff clean` — exit 1 → **this is BLOCKER-1** (a real regression, not an
  expected-open item).
- `D4 every experiment row has behavior_scorer + safety_real` — 50/59 tagged;
  rows 1–9 untagged. **Expected and tracked**: these are the superseded
  pre-fix projection-proxy / stubbed-safety era rows (exp#2–9), retained for
  provenance and explicitly demoted in paper §5.1. Not blocking. (Recommend the
  D4 check exempt rows <10 or that those rows carry a `superseded=true` tag so
  the scorecard reads clean; optional polish, not a blocker.)

---

## 3. Cruxes fixed + numeric reconciliation (independent spot-checks)

**Cruxes from the prior review — both VERIFIED FIXED.**
- Recent `experiment_log.jsonl` rows carry `behavior_scorer = "generation"` and
  `safety_real = True` (checked last 5 rows; all generation/real). The original
  W1 (projection-of-edit-onto-itself) and W2 (hardcoded `_fake_responses`
  constant) instruments are replaced. The superseded projection-proxy rows are
  still present in the log (Qwen behavior saturating at 1.0 for all α≥1, CR=0.0)
  but are correctly labeled superseded and not used for any claim.
- `eval.py` `COMPOSITE_FORMULA` re-derives to fingerprint `a9001e87087e`
  (verified by `verify_rubrics.py` D2); present in PAPER, LEDGER, dashboard,
  docs.

**Numeric claims — all reconcile against the log / campaign files (no
discrepancies):**

| Claim (paper) | Source value found | Reconciles? |
|---|---|---|
| §5.5 C6 behavior@α1 = 0.646 (gemma-3-1b @L18) | log: 0.6463; base PPL 74→104 (+41%); peaks α1 then declines | ✓ |
| §5.5 C1: E2 Spearman(Fisher, behavior) = +0.14 (p=0.74) | `C1_C2_results.md`: +0.143 (p=0.74) | ✓ |
| §5.5 C2/§S-6: N5 law `log PPL = 5.40 + 2.87·Δ‖h‖`, R²=0.81 | `C1_C2_results.md`: R²=0.809; N17 Spearman +0.705 | ✓ |
| §5.1 Qwen-0.5B α-cliff (behavior 0.500/0.694/0.526/0.494/0.346; PPL 48.9/58.7/89.0/293.6/3787; CR 0.30/0.30/0.30/0.60/1.00) | log generation-set rows match exactly | ✓ |
| §5.2 cos(DiffMean,PCA) 0.996 Qwen / 0.994 Gemma-270m / 0.9945 1b | log: 0.9956 / 0.9945 (rounds correctly) | ✓ |

No number failed to reconcile. The paper tables are faithful to the logged data.

---

## 4. Rubric E scorecard (E1–E8)

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| **E1** | Reproducible harness, correct mechanics (hooks, extraction, composite) | **PASS** | mypy clean; pytest green; composite fingerprint re-derives to `a9001e87087e`; hook/extract mechanics validated in prior review §2.7 (no mechanical errors found); dashboard 15/15. |
| **E2** | Honest measurement instruments (non-circular behavior, real safety) | **PASS** | `behavior_scorer=generation` and `safety_real=True` in live rows; W1/W2 fixes documented in paper §4.1–4.3 with residual circularity disclosed; generation-set reconciles to paper §5.1. |
| **E3** | Rigor protocol present & followed (pre-reg, n-floor, fingerprint, ladder) | **PASS** | 7-step ritual, n≤3=SCREENING floor, frozen fingerprint, 5-rung ladder all present and exercised; campaign hypotheses falsified/supported/refined, never silently confirmed (§5.5). |
| **E4** | Findings correctly scoped to SCREENING; no overclaiming | **PASS** | FINDINGS §Status: zero external-ready findings; all S-1…S-8 tagged "SCREENING ONLY — not an external claim" with n=1; paper abstract + §5 banner + §7 gate every number. No n=1 external claim found. |
| **E5** | Limitations in the abstract; required-experiments enumerated | **PASS** | Abstract lines 28–40 enumerate n=1, synthetic mini-data, single behavior, no judge, sub-0.5B; §6 (10 limitations) + §7 (7 ordered required experiments) present and honest. |
| **E6** | Reproducibility: anyone can rerun from README + token | **PASS** | §8 gives exact commands, seeds, model+license note, FakeLM offline path; `verify_rubrics.py` C3 confirms reproducibility commands documented. (Caveat: the §8 ruff claim is currently false — see BLOCKER-1.) |
| **E7** | Novelty/value stated honestly (methods/infra + geometry findings) | **PASS** | §1 explicitly disclaims a new method/result; contribution framed as harness + Goodhart composite + measurement-validity analysis + screening directions; strongest screening result (N5/N17 R²=0.81 geometry law) honestly scoped. |
| **E8** | No internal contradictions; circularity disclosed | **PARTIAL** | Same-family circularity disclosed in abstract footer, §4.2/4.3, §6.9, and this review §7. **But** two stale internal contradictions remain: FINDINGS §Status "experiments have not yet run" vs S-1…S-8 (BLOCKER-2), and the FINDINGS line-19 fingerprint placeholder stub. Convert to PASS once BLOCKER-2 is fixed. |

**Rubric-E gate:** 7 PASS / 1 PARTIAL. The single PARTIAL (E8) is the stale-text
contradiction in BLOCKER-2; combined with the lint regression (BLOCKER-1, which
falsifies the E6 §8 reproducibility attestation as written), these are the only
two items standing between this and a clean sign-off.

### Checklist items (explicitly verified)
- **No SCREENING leaked into an external claim:** CONFIRMED. FINDINGS has zero
  F-entries; all observations are S-tagged screening; paper makes zero external
  claims and gates every number on §7.
- **Limitations in the abstract:** CONFIRMED (abstract lines 28–40).
- **Required-experiments list present and honest:** CONFIRMED (§7, 7 ordered
  items; mirrored in FINDINGS and inherited from the prior review).
- **No internal contradictions:** TWO found (BLOCKER-2) — must fix.
- **Same-model-family circularity disclosed:** CONFIRMED (abstract footer, §6.9,
  §4.2/4.3).

---

## 5. Sign-off statement (Rubric E target)

Conditional on BLOCKER-1 and BLOCKER-2 being corrected (both mechanical,
minutes of work, zero scientific impact), I endorse the Rubric-E target:

> *"ACCEPT as a methodology/infrastructure contribution with reproducible
> screening results; the steering EFFICACY claims remain correctly gated on the
> enumerated required-experiments (real AxBench + calibrated judge + n≥7 +
> prompting baseline + multi-behavior). No overclaiming detected."*

The scoping is honest, the two original cruxes are genuinely repaired, the
numbers reconcile, and the screening results are correctly fenced off from
external claims. The only thing blocking an unconditional signature is two stale
self-claims (clean-ruff and "experiments not yet run") that the artifacts have
outgrown.

---

## 6. Recommended (non-blocking) polish

- Exempt superseded rows <10 (or tag `superseded=true`) so `verify_rubrics.py`
  D4 reads clean rather than FAIL-with-explanation.
- Add `[tool.ruff]`/`[tool.mypy]` are already pinned in `pyproject`; add a CI
  step that runs the same gate over `tests/` to prevent a future test file from
  silently breaking the lint claim again (this is exactly how BLOCKER-1 arose).

---

## 7. Circularity disclosure (same-model-family reviewer)

This sign-off was produced by a Claude-family model, the same family that
authored, implemented, and audited the program under review (`CLAUDE.md` §14;
the program's own meta-skill mandates this disclosure). **Therefore this is an
internal-QA-grade pass, not an external seal of approval.** It has not been
calibrated against a known-good external reference codebase, so a PASS here
means "internally consistent and honestly scoped," not "externally validated."
An independent, different-family review — and the program's own dual-track audit
run against a real reference implementation to establish a false-positive
baseline — remains a prerequisite before any finding (or this sign-off itself)
is treated as settled. **External review still pending.**

---

## 8. Final sign-off (post-fix) — 2026-05-31

Both blocking items from §1 were fixed in commit `4b30cb6` and I have
**independently re-verified** them read-only:

- **BLOCKER-1 (lint) — RESOLVED.** `python -m ruff check src/steering tests`
  now returns **"All checks passed!"** (exit 0). The `tests/test_geometry.py`
  semicolons were split and the import block sorted. Paper §8's clean-ruff
  attestation is now true.
- **BLOCKER-2 (internal contradiction, E8) — RESOLVED.** `FINDINGS.md` §Status
  (L25) now reads *"No EXTERNAL-READY findings yet — 59 experiments run, all
  SCREENING (n=1)"* and explains exp#1–59 ran but none cleared the rung-3 gate
  ⇒ zero external-ready findings. The false "experiments have not yet run /
  pre-experiment backbone phase" wording is gone; the "no external-ready
  findings" scoping is preserved. Line 19's fingerprint stub is now the literal
  `` `a9001e87087e` ``. No remaining internal contradiction.

### Re-verified gate results (final)

| Check | Command | Result |
|---|---|---|
| Rubric B (dashboard) | `python scripts/verify_dashboard.py` | **PASS 15/15** (exit 0) |
| Rubrics A/C/D | `python scripts/verify_rubrics.py` | **PASS 25 / FAIL 1** (exit 1) |
| ruff | `python -m ruff check src/steering tests` | **PASS** — "All checks passed!" (exit 0) |
| mypy | `python -m mypy src/steering --ignore-missing-imports` | **PASS** — no issues, 10 files |
| pytest | (via `verify_rubrics.py` D7) | **PASS** (green) |

The single remaining `verify_rubrics.py` FAIL is `D4` — rows 1–9 untagged. This
is the **expected, tracked** pre-fix projection-proxy / stubbed-safety era
(exp#2–9, superseded and demoted in paper §5.1, tracked in `RUBRICS.md` §F). It
is **not** a blocker and does not affect any claim.

### Rubric E — final scorecard

| # | Criterion | Verdict |
|---|---|---|
| E1 | Reproducible harness, correct mechanics | **PASS** |
| E2 | Honest measurement instruments (non-circular behavior, real safety) | **PASS** |
| E3 | Rigor protocol present & followed | **PASS** |
| E4 | Findings correctly scoped to SCREENING; no overclaiming | **PASS** |
| E5 | Limitations in abstract; required-experiments enumerated | **PASS** |
| E6 | Reproducibility from README + token (now incl. true clean-gate claim) | **PASS** |
| E7 | Novelty/value stated honestly | **PASS** |
| E8 | No internal contradictions; circularity disclosed | **PASS** (contradiction removed; circularity disclosed) |

**8 / 8 PASS.**

### UNCONDITIONAL SIGN-OFF (Rubric E target statement)

> **ACCEPT as a methodology/infrastructure contribution with reproducible
> screening results; the steering EFFICACY claims remain correctly gated on the
> enumerated required-experiments (real AxBench + calibrated judge + n≥7 +
> prompting baseline + multi-behavior). No overclaiming detected.**

The two original cruxes (circular behavior proxy, stubbed safety) are genuinely
repaired; all spot-checked numbers reconcile against the log and campaign files;
no SCREENING result leaks into any external claim; limitations are in the
abstract; the required-experiments list is present and honest; and the
same-model-family circularity is disclosed. The two mechanical blockers are
cleared and independently re-verified. **Sign-off granted.**

**Circularity caveat (unchanged):** this is an internal-QA-grade pass by a
Claude-family reviewer (see §7), **not** an external seal of approval.
Independent, different-family external review remains pending.
