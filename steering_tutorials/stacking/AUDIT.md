# AUDIT — `stacking` (Lesson 12): which steering priors STACK vs COMPETE

Independent paper-auditor pass, **re-run against the artifacts currently on disk**.
No external paper is claimed for the core stack-vs-compete rule — it is the
project's OWN hypothesis (CLAUDE.md §9 + `corpus/steering-stackable-vs-competing-analysis.md`).
This audit checks that framing is honest **and** that every number the docs quote
exists in an artifact beside them.

> **Why this file was rewritten.** The previous revision audited a run that is no
> longer on disk: it validated a refusal column of `0.667 / 0.333 / 0.667 / 0.50`
> and a gibberish column of `0 / 0 / 0 / 0.167` at "n≈12 held-out", none of which
> appear in the current `artifacts/results.json` (n=150/rung). Those verdicts are
> **withdrawn below, not deleted** — §1 lists each one against what the artifact
> actually says. `FINDING_hillclimb.md` flagged this staleness ("`AUDIT.md` is also
> **stale** — it validates numbers absent from the current `results.json`"); this
> file now agrees with that finding instead of contradicting it.

## 0. What is actually on disk (the audited surface)

| artifact | what it is | stamps it carries |
|---|---|---|
| `artifacts/results.json` | the **4-rung ladder** `[A] / [A,B] / [A,B'] / [A,B,B']`, **n = 150** generations per rung | `model_id`, layers, `stack_alpha` 0.08, `compete_add_alpha` 401.1256640625, `refusal_vector{norm 342.341, layer 12, n_extract 300}` — **no judge id, no seed, no dataset fingerprint** |
| `artifacts/hillclimb_results.json` | the **6-rung hill-climb** R0–R5 + 4 control cells, **n = 40 harmful / 20 benign** | `judge_id: Qwen/Qwen2.5-3B-Instruct`, `seed`, `tier: SCREENING…`, `preregistration`, and a `vector_check` block that **re-derives the refusal vector and confirms `cosine_saved_vs_recomputed = 1.0`** |
| `PREREGISTRATION_hillclimb.md` | P1–P6 + the STACK/COMPETE matrix, written before the hill-climb | states plainly which rungs are replications of the prior run |
| `FINDING_hillclimb.md` | the hill-climb write-up, incl. a self-reported **methodology violation** | tier, n, judge, artifact path |
| `artifacts/hillclimb_partial.json`, `ladder.png`, `hillclimb_ladder.png`, `refusal_vector.pt` | supporting | — |

**The two JSONs are different runs and must never be read as one series.** On the
identical config `[A]` — same model, same layer, same `stack_alpha` 0.08, same
`refusal_vector.norm` 342.3414306640625, same `MAX_NEW_TOKENS` 40 — they disagree
by 2×: `results.json` rung1 refusal **0.200**, gibberish **0.433**;
`hillclimb_results.json` R1 refusal **0.100**, gibberish **0.800**.
`single.B_refusal_rate` 0.2533 vs `S_B` 0.175 shows the same split. **No single
cause explains both halves:** the refusal gap is consistent with a judge swap
(check 5), but the gibberish gap is **not** — `judge.is_gibberish` is a
deterministic, model-free rule, so a different judge cannot move it. The eval
slices are nested (`run_hillclimb.py` takes `harmful[300:340]`, the first 40 of
the same 150 `run_stacking.py` uses), which makes subset composition a candidate,
but neither artifact stamps enough to decide. **Recorded as unexplained.** Quote
every rate **with the file it came from**, and do not difference across files.

## 1. WITHDRAWN — claims the previous audit passed that no artifact now supports

Kept as a record. "Absent" means the value is not in `artifacts/results.json`;
"superseded" means a current value occupies the same slot.

| previous audit claim | status | what the current artifact says |
|---|---|---|
| refusal `0.667 → 0.333 → 0.667 → 0.50` across rungs 1 / 2a / 2b / 3 | **WITHDRAWN — absent** | `0.200 → 0.060 → 0.0733 → 0.040` |
| gibberish `0 / 0 / 0 / 0.167` | **WITHDRAWN — absent** | `0.433 / 0.813 / 0.767 / 0.880` — the lesson has **no** zero-gibberish rung |
| norm_budget `0.077 → 0.225 → 0.136 → 0.277` | **WITHDRAWN — superseded** (close, but not the same run; not reconcilable by rounding) | `0.0786 → 0.2180 → 0.1411 → 0.2743` |
| marginals: stack **−0.333**, compete **0.0**, overstack gibberish **+0.167** | **WITHDRAWN — absent** | `stack_marginal −0.14`, `compete_marginal −0.1267`, `overstack_gibberish_delta +0.4467`. The compete marginal is **not** 0.0; it is clearly negative |
| "the disjoint-site rung actually lost refusal (marginal **−0.333**)" | **WITHDRAWN in magnitude, upheld in direction** | marginal is **−0.14** — still a loss, so the qualitative point (the disjoint-site rung lost refusal against prediction) survives; the number does not |
| `compete_add_alpha = 400.8` | **WITHDRAWN — superseded** | `401.1256640625` (it is measured from data each run: the hill-climb's is 399.469, so drift here is expected and benign) |
| "n≈12 held-out" | **WITHDRAWN — superseded** | `n = 150` per rung in `results.json`; `n = 40` harmful / 20 benign in the hill-climb |
| "abliterated 1B **self-graded**" | **WITHDRAWN as unverifiable** | `results.json` records no judge at all (see check 5). The hill-climb *does*: `Qwen/Qwen2.5-3B-Instruct` |
| `decision.verdict = "INCONCLUSIVE at this scale"` | **UPHELD** | present verbatim (`"INCONCLUSIVE at this scale (1B toy; effects noisy — see caveats)"`) |

## 2. Checks, re-run against the current artifacts

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | **Attribution** — internal claim honestly framed, not attributed to a paper that does not say it | **PASS (minor cite item still open)** | README opens by naming the lesson the tutorial instantiation of **CLAUDE.md §9** and the project's own corpus analysis — explicitly the project's hypothesis. `stacking.py:14-19` cites Rimsky et al. 2023 CAA (arXiv:2312.06681 — real, correctly used for the additive edit) plus Han et al. 2024 and a Wehner et al. 2025 survey **without arXiv ids**. The previous audit asked for those two to be tagged `[UNVERIFIED]`; they are **still untagged**. Motivational, not load-bearing — but the item is carried forward, not closed. |
| 2 | **Method fidelity** (code vs claim) | **PASS** | `build_priors` constructs A (refusal@L12, `relative_add`), B (same vector @L8 — disjoint site), B′ (same vector @L12, raw `add` — same site, incompatible op) from ONE shared direction, so only the *site*/*operation* varies. `stack_contexts` composes N `SteeringContext`s via `ExitStack` (every hook removed on exit); the CPU self-test checks the composed-delta math for the disjoint-site stack and the same-site sequential double-count, and asserts exact state restoration. B′ is rescaled at run time against measured `‖h‖` so B′-alone ≈ A-alone, making rung 2b a controlled comparison. The hill-climb's `vector_check` block independently re-derives the saved vector (`cosine = 1.0`, `reproduces: true`) — this lesson is one of the few here whose direction is provably regenerable from the code beside it. |
| 3 | **Claim accuracy in the README Results table** | **PASS** | The README §"The measured ladder" now transcribes the **current** `results.json`: refusal 0.20 / 0.06 / 0.073 / 0.04, gibberish 0.43 / 0.81 / 0.77 / 0.88, budget 0.079 / 0.218 / 0.141 / 0.274, `single.B_refusal_rate` 0.2533 with `B_gibberish_rate` 0.4133, `stack_marginal −0.14` (and −0.193 against the *best* single prior). The hill-climb table (R0–R5, n=40) likewise matches `hillclimb_results.json` including `norm_budget` R3 0.236 / R4–R5 0.223 and `harmful_gate_fire_rate` 0.525. **The README is current; this AUDIT file was the stale one.** |
| 4 | **Results honesty vs artifacts** | **PASS — the honesty is unusually strong** | The lesson does not claim its prediction held. `decision.verdict` is logged INCONCLUSIVE and the README says the disjoint-site rung *lost* refusal; only the over-stack prediction (all-on = most gibberish + most budget) survives. `FINDING_hillclimb.md` goes further and reports, against the lesson's own design: **no direction specificity** (A, the refusal diff-of-means, and C, a direction measured `cos(C, refusal) = 3.0e-08` at the same site and α, both score **0.100**), **no prior beats unsteered** (A 0.100, B 0.175, C 0.100, B′ 0.175 vs R0 0.275), and that the whole ladder is therefore measuring **coherence destruction** (gibberish 0.500 → 0.975), not steering. Self-reporting that the headline object is uninterpretable is the correct call. |
| 5 | **Stamping / regenerability of `results.json`** | **FAIL (new)** | `results.json` records **no judge id, no seed, no eval-set fingerprint, and no `n_eval` provenance**. `run_stacking.py` builds its judge as `Judge(model, tok)`, and `hello_world_steering/judge.py` resolves that to the **off-family model only if `STEER_JUDGE_MODEL` is set in the environment**, otherwise falling back to the **target model self-judging**. So the README's "off-family Qwen-3B judge, n=150/rung" is a claim about an environment variable that the artifact did not capture, and the *refusal* half of the 2× disagreement with the Qwen-judged hill-climb on identical configs (§0) is what a judge swap would look like (the *gibberish* half is not — see §0). **This is the §18.8 "stamp your inputs" defect**: the artifact cannot be attributed to an instrument from the code beside it. The fix is one line — `Judge` already exposes `self.judge_id` (`"self"` when no env var is set); `run_stacking.py` simply never writes it into the results dict. The hill-climb artifact does it right (`meta.judge_id`) and is the model to copy. Not a numbers error — an attribution gap. |
| 6 | **Methodology — does the lesson obey CLAUDE.md §1/§9?** | **VIOLATION, self-reported and correctly recorded** | `FINDING_hillclimb.md` §"METHODOLOGY VIOLATION" states it: **no prior clears rung 1** (all four score below the unsteered 0.275), so the revert rule terminates the climb at R0 and no ladder legitimately exists; the run instead stacked on every failing rung until **R5 had all five priors live — the "everything on" hybrid §9 forbids**. The two violations are the same mistake seen twice (nothing reverted ⇒ hybrid reached). This audit **confirms** the violation from the artifact and endorses the disposition: the R0–R5 table is a **record of what was run, not a ladder**, and its marginals are not stacking evidence. Note the lesson's *original* `rung3 = [A,B,B′]` all-on hybrid is a different case — the pre-registration declares it a deliberate demonstration of the forbidden configuration, which is legitimate as long as it is never read as a result. |
| 7 | **Pre-registration integrity** (P1–P6 vs measured) | **PASS with one unrecorded failure (new)** | Registered before the run and left unrevised. **P1 HELD** (R2 marginal refusal −0.075 ≤ 0, gibberish 0.800 → 0.950). **P2 FAILED — and this failure is not in `FINDING_hillclimb.md`'s contradiction list**: P2 predicted an orthogonal direction is "behaviourally near-inert", falsifier *C standalone refusal shift > 0.15*; measured `S_C` refusal **0.100** vs R0 **0.275** = a **−0.175** shift, and gibberish +0.200 against a predicted < 0.15. **The falsifier fired.** It points the same way as the recorded direction-specificity finding, so nothing downstream changes — but a fired falsifier belongs in the record. **P3 HELD but is uninformative** (gibberish R3−R2 = 0.025 < R2−R1 = 0.150 only because R2 is already at 0.950, i.e. against a ceiling). **P4 FAILED** (clamp moved gibberish −0.050 against a required ≥ −0.10). **P5: numeric falsifier did not fire but the premise did** — harmful refusal moved exactly 0.050 (< the 0.10 falsifier) and benign gibberish fell 0.900 → 0.000, yet the prediction assumed "the probe fires on nearly all harmful prompts" and it fires on **52.5%** (0% benign), so R5 is a 53/47 mixture of steered and unsteered and the clean-meta-layer reading is unavailable. `FINDING_hillclimb.md` reports this correctly. **P6 HELD** — the control `[A,B′]` scores **0.000**, below both constituents (each 0.175): the one unconditional COMPETE prediction is the one that held. |
| 8 | **Instrument validity** | **PASS (disclosed), and it is the binding limit** | Both the README and the pre-registration state up front that the judge family measured **ROC-AUC 0.665–0.751**, below this course's 0.85 bar (`../JUDGE_VALIDITY.md`), so differences at or below the judge's noise floor are not effects. Combined with check 5, the honest position is: **no rate on this page is instrument-grade**, and the ladder artifact cannot even name its instrument. |
| 9 | **§8c near-orthogonal arm — data floor and stamping** *(added 2026-08-22)* | **PASS on both, and it closes the lesson's hard violation** | `artifacts/near_ortho_results.json` **meets the ≥500/class rubric floor**: `data_floor.meets_floor: true`, 500 harmful / 500 benign achieved, `pool_capped: false`, `env_capped: false`, extract trimmed 300 → 292 from a 792/class pool with the reason recorded in `split_plan.note`. It is also **fully stamped** — `meta.judge_id: Qwen/Qwen2.5-3B-Instruct`, `is_self_judge: false`, `off_family: true`, `seed: 0`, plus a `directions_stamp` string that pins model, layer, n, seed, K, cosine and grid. This is the check-5 defect fixed, in the arm that landed last. **The 20-prompt benign arm the README §8d calls a hard violation no longer exists in this arm.** |
| 10 | **§8c result honesty** *(added 2026-08-22)* | **PASS — the null is reported as a null** | 0 of 4 ladder candidates kept (L1, L2 both DROP on COHERENCE+COMPETE; stop on `CONSECUTIVE_DROPS`), 0 of 5 cosine cells beat their best constituent. The README reports this without softening and, crucially, **prints the confound that dominates it**: `notes` records `SUBSTRATE CAVEAT: the UNSTEERED gibberish rate is 0.462`, so every coherence-driven DROP is partly a measurement of the abliterated base. The artifact also declines to claim the §9 clause is false — `NO CELL STACKED AT ANY COSINE … so the clause's boundary cannot be located here`. One self-correction is recorded rather than buried: the README predicted a measured/predicted N5 ratio **above** 1 from O(α²) compounding; the measured ratio is **0.96–0.97**, i.e. below, and the note says so. |

## Overall verdict: **PASS on framing and fidelity; FAIL on artifact stamping**

The central stack-vs-compete claim is honestly framed as the project's own
hypothesis (CLAUDE.md §9 / corpus), not attributed to any paper; the component
citation (Rimsky CAA) is real and correctly used; the prior construction and
composed-hook math are correct, unit-tested, and — in the hill-climb — provably
regenerable (`cosine 1.0`). The README is faithful to the artifacts, and the
lesson reports against itself: an INCONCLUSIVE ladder, a self-declared §1/§9
methodology violation, and a direction-specificity control that voids the rest.

Two items stand open, one of them new:

1. **`artifacts/results.json` is unstamped** (no judge id, seed, or eval
   fingerprint) and disagrees 2× with the Qwen-judged hill-climb on identical
   configs. Until a re-run stamps its judge, every rate in it should be quoted as
   *judge-unattributed*. Copy `hillclimb_results.json`'s `meta` block.
   *(Scope note added 2026-08-22: this applies to `results.json` **only**. Both other
   artifacts on this lesson — `hillclimb_results.json` and the new
   `near_ortho_results.json` — carry full `meta` blocks with `judge_id`,
   `is_self_judge: false` and `off_family: true`.)*
2. **Tag the Han/Wehner secondary cites `[UNVERIFIED]`** in `stacking.py:16-19`
   (carried forward from the previous audit; still open).

And one standing scientific caveat, which is larger than either: per
`FINDING_hillclimb.md`, at this α on this abliterated 1B model the measured
effect is **not direction-specific** (an exactly-orthogonal control scores what
the refusal direction scores), so **the stack-vs-compete question cannot be
answered by this lesson's artifacts at all** until a prior is found that beats its
own orthogonal control. That is the pre-registered next step, and no number in
any of the three JSONs should be cited as stacking evidence before it lands.

**The 2026-08-22 near-orthogonal arm strengthens that caveat rather than resolving it.**
It was the one arm built to test §9's third clause with a real stopping condition, it ran
at the full rubric floor with a stamped off-family judge, and it returned a flat negative
across the entire cosine dial (−0.094 to −0.114 vs best constituent at cos
−0.00/0.25/0.50/0.75/0.95). With an unsteered gibberish rate of **0.462**, that is a null
**about this substrate**, not a falsification of the clause. The α sweep down to where a
single prior is coherent remains the unblocking experiment for the whole lesson.

Internal QA pass — independent external review pending (auditor shares a model family with the author).
