# AUDIT — gavel

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id (primary) | **2601.19768 — VERIFIED (resolves).** |
| actual title | *GAVEL: Towards Rule-Based Safety Through Activation Monitoring* — matches the README's title verbatim (l.62). |
| actual authors | **Shir Rozenfeld, Rahul Pankajakshan, Itay Zloczower, Eyal Lenga, Gilad Gressel, Yisroel Mirsky** — matches README l.61-62 verbatim, correct order. |
| venue / dates | arXiv; submitted 2026-01-27 (v1), last revised 2026-04-30 (v3). README's "2026" is correct. |
| method in abstract | Yes — activations represented as fine-grained interpretable **cognitive elements** (examples: "making a threat", "payment processing"); composable **predicate rules**; safeguards **reconfigurable without retraining**; transparency/auditability. Matches README's method description (l.67-74) exactly. |
| arXiv id (secondary) | **2310.17389 — VERIFIED (resolves).** *ToxicChat…* (Lin, Wang, Tong, Wang, Guo, Wang, Shang). README cites "Lin et al. 2023" (l.90, l.277) — correct first-author + year + id. |
| verification | WebFetch of both `arxiv.org/abs/` pages confirmed titles, authors, method. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Primary paper exists & id resolves | **PASS** | 2601.19768 resolves; title + method match verbatim. |
| Citation attribution correct (primary) | **PASS** | All six authors named correctly and in order (README l.61-62; `monitor.py` l.27). No fabricated/wrong co-authors — the common bug is absent here. |
| Secondary (ToxicChat) attribution | **PASS** | 2310.17389 resolves; "Lin et al. 2023" is the correct first author/year; used correctly as the dataset source, not overclaimed. |
| Method fidelity — CE library + predicate rule | **PASS** | `monitor.py` implements `CEDetector` (diff-of-means direction + benign-calibrated `tau`), `Rule.any_of/all_of/at_least`, and `GavelMonitor.decide` returning an auditable `{block, scores, fired, triggered_by, reason}`. Directly operationalizes the paper's compositional-CE + reconfigurable-rule thesis. |
| Fidelity nuance (simplification honestly flagged) | **PASS (with disclosed gap)** | Paper's CEs are human-authored fine-grained factors; lesson's CEs are one diff-of-means direction per toxic-chat harm *category*. README l.76-81 and l.248-252 openly call this a "faithful miniature"/"simplification" and note the missing composite ("violence AND payment") example (l.132-134). Honest, not a faithful full reproduction — correctly stated as such. |
| Baseline design | **PASS** | Single broad all-harm direction (`run_gavel.py` l.247-248) is built as the "broad misuse detector" the paper argues past — a fair head-to-head. |
| Results honesty | **PASS** | Results table marked "[PENDING THE GPU RUN]" with no pre-written numbers (README l.220-237); screening-tier disclosed (n≤60/class, one seed, per §7); off-family `Qwen2.5-3B-Instruct` judge documented (l.198, l.244-247); self-judge/coarse-CE/monitor-not-a-fix caveats all stated. `results.json` written before summary print (`run_gavel.py` l.352); no unicode in prints. |

## Overall verdict

**PASS.** Both cited arXiv ids resolve; the primary GAVEL paper is real and its
six authors, title, and method are attributed **correctly and verbatim** (no
fabricated co-authors — the usual failure mode is absent). The code faithfully
operationalizes the compositional-CE + predicate-rule mechanism, and the lesson
is transparent that its per-category CEs are a deliberate simplification of the
paper's fine-grained human-authored elements. Results are honestly marked pending,
screening-tier, off-family-judged. No required fixes. *(Optional nit: README l.63-65
asserts its own WebFetch verification — harmless, and it checks out.)*

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*

---

## Addendum (2026-08-02) — the "Results honesty" PASS above is SUPERSEDED

The audit above was performed while the results table still read
**"[PENDING THE GPU RUN]"**. It therefore graded the lesson's *documented intent*
— which did specify the off-family `Qwen2.5-3B-Instruct` judge — and could not
have graded the artifact, because no artifact existed yet. The GPU run happened
afterwards, **without `STEER_JUDGE_MODEL` set**, and `results.json` recorded
`"judge_id": "self"`: the abliterated 1B graded its own generations.

| item | revised finding |
|---|---|
| Results honesty (pass-through) | **FAIL at the time of the run.** Self-judged numbers were published under a README that claimed an off-family judge — a CLAUDE.md §17.3 violation. |
| Results honesty (monitor half) | **Still PASS.** Block rates, broad baseline and per-CE firing are judge-free (thresholded dot products vs ground-truth labels); no grader is involved. |
| Current state | README now carries a judge-provenance correction, marks the pass-through figures **inadmissible pending re-judge**, and retains the self-judged column labelled as superseded. `rejudge.py` implements the single-variable judge swap; the measurement is pending a host memory window. |

**Process lesson (worth more than the fix).** An audit that clears a *pending*
results section grants a PASS that silently transfers to whatever numbers land
later. The audit's own evidence line — "off-family `Qwen2.5-3B-Instruct` judge
documented (l.198, l.244-247)" — cites the README's *instructions*, not the run's
*provenance*. **Verify the artifact, never the intent** (§18.8: "verify contents,
never listings"). A results-honesty check should be re-run against `results.json`
once it exists, and should assert `judge_id` explicitly.

**This is a class of bug, not one lesson's slip.** `Judge.__init__` falls back to
the self-judge branch whenever `STEER_JUDGE_MODEL` is unset, so the correct
instrument depends on an env var that no runner enforces. A sweep of the course
found the provenance stamped correctly in `contextual_steering`, `curveball`,
`fine_grained`, `flas`, `realignment`, `talan` and `stacking/hillclimb_results.json`
(all `Qwen/Qwen2.5-3B-Instruct`), `gavel` as the only artifact stamped `self` — and
several generation-judged lessons stamping **no judge id at all**, including
`stacking/artifacts/results.json` (which also records no seed), `hello_world_steering`,
`reft_r1`, `rogue_scalpel`, `multi_intent`, `non_identifiability`,
`decomposing_prompting` and `prompt_activation_duality`. An unstamped artifact is
*worse* than gavel's: gavel recorded its wrong judge honestly and was therefore
catchable, whereas an unstamped one cannot be shown to be either compliant or
violating. The durable fix is to stamp `judge_id` unconditionally and fail the
build when a generation-judged lesson's artifact lacks an off-family id.
