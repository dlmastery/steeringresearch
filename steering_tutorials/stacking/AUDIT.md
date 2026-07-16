# AUDIT — `stacking` (Lesson 12): which steering priors STACK vs COMPETE

Independent paper-auditor pass. Scope: verify the lesson against `stacking.py`,
`run_stacking.py`, README, and `artifacts/results.json`. No external paper is
claimed for the core stack-vs-compete rule — it is the project's OWN hypothesis
(CLAUDE.md §9 + `corpus/steering-stackable-vs-competing-analysis.md`). This audit
checks that framing is honest.

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Attribution: internal claim honestly framed (not attributed to a paper that doesn't say it) | **PASS (minor secondary-cite note)** | The README opens by naming the lesson "the tutorial instantiation of **CLAUDE.md section 9 (stacking discipline)**" and the mechanism analysis in the project's own corpus — i.e. explicitly the project's hypothesis, not a paper's result. External citations in `stacking.py:14-19` are used only for *components*: Rimsky CAA (arXiv:2312.06681, the additive edit — real, correct) for "the additive CAA edit each prior applies"; plus Han et al. 2024 and a Wehner et al. 2025 survey for additive-composition / norm-budget context. **Minor:** those two secondary cites carry no arXiv IDs and were not verified in this audit — they are motivational, not load-bearing, but should be tagged `[UNVERIFIED]` for corpus discipline. No nonexistent citation is implied for the central claim. |
| 2 | Method fidelity (code vs claim) | **PASS** | `build_priors` constructs A (refusal@L12, relative_add), B (refusal@L8, relative_add — disjoint site), B′ (refusal@L12, `add` — same site, incompatible op) from ONE shared refusal direction, isolating the *site* variable exactly as the README claims. `stack_contexts` composes N `SteeringContext`s via `ExitStack` (guarantees all hooks removed). The self-test verifies the exact composed-delta math for both a disjoint-site stack (layer-1 delta = 0.5·‖h‖·û) and the same-site *sequential* double-count (B′ `add` fires first, then A's relative_add computes its norm on the contaminated state), and asserts zero residual hooks + exact restoration. B′ is rescaled at run time so B′-alone ≈ A-alone (`compete_add_alpha=400.8` in `results.json`, calibrated against `single.B_refusal_rate`), making rung 2b a controlled comparison. Faithful. |
| 3 | Claim accuracy in Results table | **PASS** | README transcribes `results.json` correctly: refusal 0.667(rung1)→0.333(2a)→0.667(2b)→0.50(3); gibberish 0/0/0/0.167; norm_budget 0.077→0.225→0.136→0.277; marginals stack −0.333, compete 0.0, overstack gibberish +0.167; `decision.verdict = "INCONCLUSIVE at this scale"`. All verdicts hedged to match. |
| 4 | Results honesty vs artifacts | **PASS — GOOD honesty** | The lesson does NOT claim its clean prediction held. `decision.verdict` in the artifact is logged as "INCONCLUSIVE at this scale" and the README states it plainly: the disjoint-site rung "actually lost refusal (marginal −0.333) rather than gaining," so the stack-vs-compete *separation* did not appear; only the over-stack prediction (all-on is the sole rung to break into gibberish, highest norm budget) survived. Naming the one surviving prediction and flagging the two that did not is exactly-right honesty. **Flag:** abliterated 1B self-graded (README §9 "Self-grading circularity" caveat), n≈12 held-out — screening tier; per the FLAS audit the 1B self-judge inflates refusal. Re-run with the off-family Qwen judge and n≥7 seeds + rigor contract before any non-screening claim. |

## Overall verdict: **PASS**

The central stack-vs-compete claim is honestly framed as the project's own
hypothesis (CLAUDE.md §9 / corpus), not attributed to any paper; the component
citation (Rimsky CAA) is real and correctly used; the prior construction and
composed-hook math are correct and unit-tested; the Results are transcribed
faithfully and the honest INCONCLUSIVE verdict — including the disjoint-site rung
losing refusal against prediction — is stated prominently. Minor: tag the
unverified Han/Wehner secondary cites `[UNVERIFIED]`. Standing caveat: 1B
self-judge screening tier — re-run with the off-family Qwen judge.

Internal QA pass — independent external review pending (auditor shares a model family with the author).
