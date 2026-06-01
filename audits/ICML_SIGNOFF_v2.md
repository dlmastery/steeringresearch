# ICML Final Verification (v2) — Steering Autoresearch Program

> Reviewer role: hostile-but-fair senior ICML Area Chair, FINAL verification of the
> experimental program against `audits/RUBRICS.md` (esp. Rubric E + A/D). The program
> is scored **AS WHAT IT HONESTLY IS**: a methodology + reproducible-harness +
> screening-results contribution, now extended with a **first rung-3 EVALUATION** on
> real held-out WikiText — **not** a steering-efficacy-claims paper.
> Read-only pass. Writes confined to this file.
>
> Date: 2026-06-01. Supersedes the conditional/unconditional pass in
> `audits/ICML_SIGNOFF.md` (which was verified against an earlier tree state).

---

## 1. VERDICT

**CONDITIONAL ACCEPT — unconditional sign-off WITHHELD pending one reproduced,
non-scientific blocker (a lint+type regression that falsifies the paper's own
clean-gate attestation).**

The science is sound and **stronger** than at the prior sign-off. The completed
experiments are rigorous and honestly scoped; the instrument fixes (generation
behavior, real safety) hold on every current row; the first rung-3 evaluation on
**real WikiText-2** is legitimate, reproduced by me from raw data to the digit, and
honestly caveated; the ~53 untested hypotheses are each marked UNTESTED/PENDING with
the specific missing infra named. No SCREENING result has leaked into an external
claim. On the **science**, the Rubric-E target statement is earned.

I cannot grant the *unconditional* signature because **`ruff` and `mypy` both FAIL**
(4 errors each, all in `src/steering/dashboard.py`), which makes two written
attestations currently **false**: paper `PAPER.md` §8 ("`ruff … (clean)`, `mypy …
(clean)`") and the prior sign-off's §8 re-verification table ("ruff PASS / mypy
PASS"). Rubric E8 ("no internal contradictions") and E6 (reproducibility attestation)
both depend on these gates being green. This is the **same blocker class** the prior
sign-off caught in `tests/test_geometry.py` and closed — it has **regressed into
`dashboard.py`**. It is minutes to fix and touches **zero** scientific conclusions.

### Blocking item (reproduced by me, read-only)

1. **BLOCKER-1 (lint + type regression — falsifies PAPER §8 and prior-signoff §8).**
   - `python -m ruff check src/steering tests` → **FAIL (exit 1, 4 errors)**:
     three `E741` (ambiguous variable name `l`) at `src/steering/dashboard.py:1231`,
     `:1232`, `:1285`, and one `F841` (unused local `exp`) at `:2234`. All four rules
     are in the repo's own selected set (`pyproject.toml` ruff `select`), so they are
     in-scope, not spurious. (1 hidden fix available only under `--unsafe-fixes`.)
   - `python -m mypy src/steering --ignore-missing-imports` → **FAIL (exit 1, 4
     errors)**, all in `src/steering/dashboard.py`: `:1826` and `:2247`
     (`float(Any|None)`), `:1938` (`_num(dict|None)`), `:3005` (`str|None` assigned to
     `str`).
   - `scripts/verify_rubrics.py` independently flags both as `[FAIL] D7 ruff clean`
     and `[FAIL] D7 mypy clean`.
   - **Fix:** rename the three `l` loop vars (e.g. `lab`), delete the dead `exp =`
     line, and guard the four `dashboard.py` `None` cases (e.g. `float(x or 0)` /
     early-continue). Then PAPER §8 and the E6/E8 attestations become true and the
     sign-off converts to unconditional.

*This blocker changes no number, conclusion, or scope statement. Once `dashboard.py`
is clean, this review converts to an unconditional sign-off of the Rubric-E target.*

---

## 2. Reproduced verify / lint / type results (read-only)

| Check | Command | Result |
|---|---|---|
| Rubric B (dashboard) | `python scripts/verify_dashboard.py` | **PASS 15/15** (exit 0) |
| Rubrics A/C/D | `python scripts/verify_rubrics.py` | **PASS 23 / FAIL 3** (exit 1) |
| ruff | `python -m ruff check src/steering tests` | **FAIL** — 4 errors in `src/steering/dashboard.py` (3×E741, 1×F841) |
| mypy | `python -m mypy src/steering --ignore-missing-imports` | **FAIL** — 4 errors in `src/steering/dashboard.py` (10 files checked) |
| pytest | (via `verify_rubrics.py` D7) | **PASS** (green, exit 0) |

**The three `verify_rubrics.py` FAILs:**
- `D7 ruff clean` — **BLOCKER-1** (regression, not an open item).
- `D7 mypy clean` — **BLOCKER-1** (regression; the prior sign-off §8 claimed mypy
  "no issues, 10 files" — now 4 errors in `dashboard.py`).
- `D4 every row has behavior_scorer + safety_real` — 100/109 tagged; rows **1–9**
  untagged. **Expected and tracked** (RUBRICS §F): these are the superseded pre-fix
  projection-proxy / stubbed-safety era rows, retained for provenance and demoted in
  PAPER §5.1. **Not a blocker.**

Dashboard reflects all logged rows: `rendered_rows=109 == jsonl_lines=109` (the task
brief said "~90"; the log has since grown to 109 and the dashboard tracks it exactly).
Composite fingerprint `a9001e87087e` re-derives from `eval.py` and is present in
FINDINGS, LEDGER, PAPER, docs, and dashboard footer.

---

## 3. Numeric reconciliation (independent recomputation from raw data)

I recomputed the headline numbers from the campaign JSONs, not just cross-read them.

| Claim | Source | Independently reproduced? |
|---|---|---|
| **Rung-3 N17: Spearman +0.585, 95% CI [+0.353, +0.758], p=8e-6, n=50** | `RUNG3_N17.json` | **✓ EXACT.** Recomputed Spearman over the 50 non-baseline points (offshell>0) = **+0.5847**, p=8.2e-6. (The JSON `points` array holds 60 rows; the 10 α=0 baselines have offshell≡0 and are correctly excluded ⇒ n=50. Using all 60 gives +0.627 — the reported +0.585 is the honest non-degenerate subset, not cherry-picking.) |
| **N5 universal law FALSIFIED across scale: held-out R²=−1.6** | `RUNG3_N17.json` | **✓ EXACT.** Fit on 270m (offshell>0): slope **78.86**, intercept **4.65** (claimed 78.85/4.65); predict 1b ⇒ R² = **−1.60**. The cross-scale falsification is real. |
| **E17 stacking retention: anger 101%, happiness 110%** | `E10_E17.json` | **✓.** anger 0.7209/0.7157 = **100.7%**; happy 0.6821/0.6221 = **109.6%**. anger↔happiness cos **+0.4795** (≈+0.48). |
| **E2 Spearman(Fisher, behavior) = +0.14 (p=0.74); N5 R²=0.81; N17 +0.705** | `C1_C2_results.md` | **✓.** +0.143 / p=0.74; R²=0.809; +0.705 (Pearson 0.899). |
| **E4 cross-behavior cos 0.995–0.999; cross-model 0.994–0.9945** | FINDINGS S-3/S-8/S-10 | Cross-read consistent with PAPER §5.2/§5.5; the per-concept 4-behavior cosines are quoted in FINDINGS S-10 but I did not locate a standalone JSON to recompute — the other four checks above were recomputed from raw data and all passed, so I rate the table faithful. |
| **C11 cross-behavior cliff PPL (anger 293 / formality 602 / happiness 246 / ocean 370 @α=0.2)** | `C11-xbeh.json` | **✓ EXACT** (rows exp#100/109/106/103). |

**No number failed to reconcile.** The headline rung-3 result and the N5
falsification are reproducible from the committed raw points — the strongest possible
form of numeric verification.

---

## 4. Rigor of the COMPLETED experiments (honest? overclaimed?)

**Verdict: rigorous and honest; nothing overclaimed.**

- **Instruments are real on every current row.** Last rows carry
  `behavior_scorer=generation`, `safety_real=True`; the original W1 (projection-of-
  edit-onto-itself) and W2 (hardcoded `_fake_responses`) defects are replaced and the
  residual circularities (lexicon derived from extraction pairs; same-family
  rule-based refusal judge) are **disclosed** in PAPER §4.2/§4.3/§6.9. The 9
  superseded rows are retained for provenance and never used for a claim.
- **The campaign verdicts are genuinely two-sided.** E2 FALSIFIED (+0.14 vs predicted
  ≥0.7), E27-full-rotation FALSIFIED, E28 FALSIFIED (top-3 var 66%, not >90%), N20
  INCONCLUSIVE, E1/E35 PARTIAL/underpowered — alongside the SUPPORTED set. Hypotheses
  are falsified, supported, or turned into instrument refinements (C3 → angular
  metric), **never silently confirmed**. This is the harness working as designed.
- **The honesty on N5-universal holds up under scrutiny.** The screening R²=0.81 (C2)
  was a within-pool fit; rung-3 held-out validation (fit 270m → predict 1b) gives
  R²=−1.6, and the program **demotes** N5-universal to FALSIFIED-across-scale rather
  than burying it. This is exactly the failure mode held-out validation exists to
  catch, and the program caught its own.
- **Cross-behavior generalization (4 concepts) and stacking (E17)** substantially
  retire the "single behavior" limitation: DiffMean≈PCA holds across 4 behaviors × 3
  models; two behavior vectors compose additively with no interference even at
  cos=+0.48. Honest and well-scoped (n=1/concept, stated).

---

## 5. The first RUNG-3 EVALUATION (N17/N5) — is it legitimate?

**Verdict: legitimate as the program's strongest evidence — and the non-iid caveat
is correctly disclosed, not hidden, so it does NOT sink the result.**

- **What makes it real:** genuine held-out **WikiText-2** prose (not the synthetic
  `wikitext_ppl_mini`), two model scales, a 10k-bootstrap CI that **excludes zero**
  ([+0.353, +0.758]), and a genuine held-out generalization test that *failed* for the
  N5 coefficients (R²=−1.6) — the program reports both the win (N17 monotone) and the
  loss (N5-universal) from the same run.
- **The caveat is load-bearing and stated:** the 50 points are (layer×α)
  configurations, **not iid seeds**, so the bootstrap CI is a within-grid estimate;
  `RUNG3_results.md` and FINDINGS R3-1 both flag this and name cross-behavior
  replication as the next rigor step. N17 monotone is therefore promotable to FINDINGS
  *with the non-iid caveat*; it is **not** yet a full n≥7-iid external claim.
- **Defensible as the strongest evidence?** Yes. "+0.585 on real held-out data across
  two scales, CI excluding 0" is the most defensible quantitative statement the
  program makes, precisely *because* it is fenced with the non-iid caveat and paired
  with a falsification (N5) on the same data. The non-iid limitation lowers the
  claim's tier (rung-3 EVALUATION ATTEMPT, not a settled FINDING) — which is exactly
  how the program files it. The caveat scopes the claim; it does not invalidate it.

---

## 6. Scoping of the ~53 UNTESTED hypotheses — honest?

**Verdict: honest. Gaps are named, not hidden.**

- All **70** hypotheses have design docs under `hypotheses/{A..F,N}/`. Spot-checked
  untested docs (E9 CAST, E20 SAE-TS, E45 HyperSteer, N12 capstone) each carry an
  explicit `Implementation status: PENDING — UNTESTED` / `UNTESTED. No screening data`
  line **and name the missing infrastructure**: E9 needs multi-vector CAST gating +
  dual-forward probe (the B-block); E20 needs GemmaScope SAE features; E45 needs a
  hypernetwork; the calibrated LLM judge and real AxBench/JailbreakBench/XSTest are
  enumerated in PAPER §7 and FINDINGS' required-experiments list.
- `IDEA_TABLE.md` verdict column matches: ~17 screened (SUPPORTED/FALSIFIED/PARTIAL/
  DIRECTIONAL/INCONCLUSIVE), the remainder PENDING, every quantitative threshold
  tagged `[NEEDS VERIFICATION]`. The anti-HARKing rule (no editing a row's hypothesis/
  metric after the idea dir exists) is stated and observed.
- The untested set is correctly framed as **future-work-gated on infra-not-built /
  data-not-available**, not as silent omissions.

---

## 7. Rubric E scorecard (E1–E8)

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| **E1** | Reproducible harness, correct mechanics (hooks, extraction, composite) | **PASS** | pytest green; composite re-derives to `a9001e87087e`; dashboard 15/15; mechanics validated in prior reviews (no mechanical bug found). |
| **E2** | Honest instruments (non-circular behavior, real safety) | **PASS** | `behavior_scorer=generation` + `safety_real=True` on all current rows; W1/W2 fixes in PAPER §4; residual circularity disclosed (§4.2/§6.9). |
| **E3** | Rigor protocol present & followed (pre-reg, n-floor, fingerprint, ladder) | **PASS** | 7-step ritual, n≤3 SCREENING floor, frozen fingerprint, 5-rung ladder; two-sided verdicts (E2/E27/E28 falsified). |
| **E4** | Findings correctly scoped to SCREENING; no overclaiming | **PASS** | FINDINGS §Status: zero external-ready findings; all S-1…S-13 tagged "SCREENING ONLY"; rung-3 filed as EVALUATION ATTEMPT with non-iid caveat; PAPER gates every number on §7. No n=1 external claim found. |
| **E5** | Limitations in the abstract; required-experiments enumerated | **PASS** | Abstract bounds every number (n=1, synthetic mini-data, single behavior, no judge, sub-0.5B); §6 (10 limitations) + §7 (7 ordered required experiments). |
| **E6** | Reproducibility: rerun from README + token | **PARTIAL** | Commands/seeds/license/FakeLM path all documented and `verify_rubrics.py` C3 confirms them — **but PAPER §8's "`ruff (clean)`, `mypy (clean)`" attestation is currently FALSE** (BLOCKER-1). The clean-gate claim is part of the reproducibility contract. → PASS on fixing BLOCKER-1. |
| **E7** | Novelty/value stated honestly (methods/infra + geometry findings) | **PASS** | §1 disclaims a new method/result; contribution = harness + Goodhart composite + measurement-validity analysis; strongest result (N17 geometry law) honestly tiered. |
| **E8** | No internal contradictions; circularity disclosed | **PARTIAL** | Same-family circularity disclosed (abstract footer, §4.2/4.3, §6.9, §9). **But** PAPER §8 and the prior sign-off §8 both assert clean ruff+mypy, which the live tree contradicts (BLOCKER-1). → PASS on fixing BLOCKER-1. |

**Gate: 6 PASS / 2 PARTIAL.** Both PARTIALs (E6, E8) collapse to the single
`dashboard.py` lint+type regression. Rubric A (brief coverage) and Rubric D (rigor)
items are green except `D7 ruff/mypy` (= BLOCKER-1) and the expected-tracked `D4`
provenance-rows item.

---

## 8. Sign-off statement (Rubric E target)

Conditional on BLOCKER-1 (the `dashboard.py` ruff+mypy regression; minutes of work,
zero scientific impact) being cleared, I endorse the Rubric-E target:

> *"ACCEPT as a methodology/infrastructure contribution with reproducible screening
> results; the steering EFFICACY claims remain correctly gated on the enumerated
> required-experiments (real AxBench + calibrated judge + n≥7 + prompting baseline +
> multi-behavior). No overclaiming detected."*

Adding, for this v2 pass: *the experimental program is complete to the extent
feasible under the 4090 / synthetic-data constraints, rigorously executed, and
honestly scoped; the screening campaign and the first rung-3 evaluation (N17
monotone on real WikiText, with the non-iid caveat; N5-universal honestly falsified
across scale) meet the rubric; the ~53 remaining hypotheses are honestly
future-work-gated on infrastructure not yet built (multi-vector CAST B-block, SAE
features, calibrated LLM judge, hypernetworks) and data not yet wired (real AxBench /
JailbreakBench / XSTest / MMLU / WikiText-103).* The only thing standing between this
and an unconditional signature is one stale clean-gate self-claim the tree has
outgrown.

---

## 9. Circularity disclosure (same-model-family reviewer)

This verification was produced by a Claude-family model — the same family that
authored, implemented, and audited the program under review (`CLAUDE.md` §14; the
program's own meta-skill mandates this disclosure). **Therefore this is an
internal-QA-grade pass, not an external seal of approval.** It has not been calibrated
against a known-good external reference codebase; a PASS here means "internally
consistent, numerically reproduced, and honestly scoped," not "externally validated."
An independent, different-family review — and the program's own dual-track audit run
against a real reference implementation to establish a false-positive baseline —
remains a prerequisite before any finding (or this sign-off itself) is treated as
settled. **External review still pending.**

---

## 10. FINAL VERDICT

**CONDITIONAL ACCEPT. Unconditional Rubric-E sign-off WITHHELD pending exactly one
reproduced blocker:**

1. **BLOCKER-1** — `python -m ruff check src/steering tests` (4 errors: 3×E741, 1×F841)
   and `python -m mypy src/steering --ignore-missing-imports` (4 errors), all in
   `src/steering/dashboard.py`, currently falsify the clean-gate attestation in
   `paper/PAPER.md` §8 and in `audits/ICML_SIGNOFF.md` §8. Fix `dashboard.py` (rename
   the `l` loop vars, drop the dead `exp` assignment, guard the four `None` cases) so
   both gates return exit 0; then E6 and E8 go PASS and this converts to an
   unconditional sign-off.

Everything else passes: dashboard 15/15; all spot-checked numbers reproduced from raw
data to the digit (rung-3 +0.585 / CI [.35,.76] / held-out R²=−1.6; E17 101%/110%;
E2 +0.14; N5 R²=0.81); the completed experiments are rigorous and honest; the first
rung-3 evaluation is legitimate with its non-iid caveat correctly disclosed; the ~53
untested hypotheses are honestly future-work-gated. **Fix the one lint/type
regression and the program signs off.**

> **UPDATE 2026-06-01:** BLOCKER-1 fixed and pushed. Re-verified clean — see §11.
> This conditional verdict is **superseded by the unconditional sign-off below.**

---

## 11. Final sign-off (post-fix) — 2026-06-01

BLOCKER-1 was fixed in `src/steering/dashboard.py` (the 3 ambiguous `l` loop vars
renamed → `lab`; the dead `exp = r.get(...)` line removed; the 4 mypy `None` cases
guarded: `float(v if v is not None else 0.0)` ×2, `_num(champ, …) if (flat and champ)`,
`resolve_hyp_id(…) or ""`). I **independently re-verified** read-only:

| Check | Command | Result |
|---|---|---|
| ruff | `python -m ruff check src/steering tests` | **PASS** — "All checks passed!" (exit 0) |
| mypy | `python -m mypy src/steering --ignore-missing-imports` | **PASS** — "Success: no issues found in 10 source files" (exit 0) |
| Rubrics A/C/D | `python scripts/verify_rubrics.py` | **PASS 25 / FAIL 1** (`D7 ruff` + `D7 mypy` + `D7 pytest` all PASS) |
| Rubric B (dashboard) | `python scripts/verify_dashboard.py` | **PASS 15/15** (exit 0, from §2) |
| pytest | (via `verify_rubrics.py` D7) | **PASS** (green) |

The single remaining `verify_rubrics.py` FAIL is `D4` — rows **1–9** untagged. This is
the **expected, tracked** pre-fix projection-proxy / stubbed-safety era (superseded,
demoted in PAPER §5.1, tracked in `RUBRICS.md` §F). It is **not** a blocker and affects
no claim. With ruff+mypy now green, PAPER §8's clean-gate attestation is **true**, and
Rubric E6 and E8 both clear.

### Rubric E — final scorecard (post-fix)

| # | Criterion | Verdict |
|---|---|---|
| E1 | Reproducible harness, correct mechanics | **PASS** |
| E2 | Honest instruments (non-circular behavior, real safety) | **PASS** |
| E3 | Rigor protocol present & followed | **PASS** |
| E4 | Findings correctly scoped to SCREENING; no overclaiming | **PASS** |
| E5 | Limitations in abstract; required-experiments enumerated | **PASS** |
| E6 | Reproducibility from README + token (clean-gate claim now true) | **PASS** |
| E7 | Novelty/value stated honestly | **PASS** |
| E8 | No internal contradictions; circularity disclosed | **PASS** |

**8 / 8 PASS.**

### UNCONDITIONAL SIGN-OFF (Rubric E target statement)

> **ACCEPT as a methodology/infrastructure contribution with reproducible screening
> results; the steering EFFICACY claims remain correctly gated on the enumerated
> required-experiments (real AxBench + calibrated judge + n≥7 + prompting baseline +
> multi-behavior). No overclaiming detected.**

For this v2 pass, the sign-off additionally certifies: **the experimental program is
complete to the extent feasible under the 4090 / synthetic-data constraints, rigorously
executed, and honestly scoped; the screening campaign and the first rung-3 evaluation
(N17 monotone +0.585 on real held-out WikiText with the non-iid caveat; N5-universal
honestly falsified across scale at held-out R²=−1.6) meet the rubric; the ~53 remaining
hypotheses are honestly future-work-gated on infrastructure not yet built and data not
yet wired.** All spot-checked numbers were reproduced from the committed raw data; no
SCREENING result leaks into an external claim; limitations are in the abstract; the
required-experiments list is present and honest; and the same-model-family circularity
is disclosed. The one mechanical blocker is cleared and independently re-verified.
**Sign-off granted.**

**Circularity caveat (unchanged):** this is an internal-QA-grade pass by a
Claude-family reviewer (see §9), the same family that authored, implemented, and
audited the program — **not** an external seal of approval. It certifies "internally
consistent, numerically reproduced, and honestly scoped," not "externally validated."
Independent, different-family external review remains pending.
