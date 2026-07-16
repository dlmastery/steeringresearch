# Paper Audit — `hello_world_steering` (Conditional Activation Steering)

Independent paper/verification audit. Auditor did not modify lesson code or the README.

## Cited papers
- Turner et al. 2023, ActAdd — arXiv:2308.10248
- Rimsky/Panickssery et al. 2023, CAA — arXiv:2312.06681
- Arditi et al. 2024, refusal single-direction — arXiv:2406.11717
- Lee et al. 2024, CAST — arXiv:2409.05907

## Checks

| Check | Verdict | Evidence |
|---|---|---|
| 1. Papers real + attribution correct | **CONCERN** | All four arXiv ids resolve and are the right method (WebFetch of each). **Two attribution nits to fix before external use:** (a) **CAST 2409.05907** — the README (§4) and `gate.py` cite "**Lee, Kim, Wang**, et al." The real author list is **Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish Nagireddy, Amit Dhurandhar** — first author Lee is correct but **"Kim, Wang" are fabricated co-authors**; title "Programming Refusal with Conditional Activation Steering" is correct (ICLR 2025 spotlight). (b) **ActAdd 2308.10248** — `model_utils.py` gives the title *"Activation Addition: Steering Language Models Without Optimization"*; the actual arXiv title is **"Steering Language Models With Activation Engineering"** (Turner et al.) — attribution/year correct, title wrong. Minor: **CAA 2312.06681** is cited "Rimsky et al." — arXiv now lists the first author as **Nina Panickssery** (formerly Rimsky); title "Steering Llama 2 via Contrastive Activation Addition" is correct. All ids are otherwise **verified** (drop the `[UNVERIFIED]` tags). |
| 2. Method fidelity | **PASS** | `steer_vector.extract_caa_vector` computes `mean(act\|harmful) − mean(act\|benign)` on last-token layer-12 residuals — textbook CAA/diff-of-means, and last-token is the position Arditi et al. use. `SteeringContext` applies the relative-add rule `h += α·‖h‖·unit(v)` via a forward hook; its CPU self-test asserts the delta equals `α·‖h‖·unit(v)`, the literal `add` op equals `α·v`, and the hook **restores state exactly** on exit. `gate.py` loads the lesson-1 probe verbatim (same weights/scaler/threshold) as the CAST-style condition. Faithful; simplifications (single layer, diff-of-means vs learned projection) are declared in §10. |
| 3. Claim accuracy | **PASS** | The "measured vs claim" table states the papers' real claims — a diff-of-means vector steers behavior (ActAdd/CAA/Arditi), and a lightweight condition gates it (CAST) — without overclaiming SOTA (§10 explicitly calls diff-of-means "the simplest steering method, not SOTA"). |
| 4. Results honesty | **CONCERN (well-disclosed)** | Numbers match `results.json` exactly: unconditional refusal 0.50→0.70 at α=0.10, gate accuracy 0.975, conditional harmful-refusal 0.65, benign over-refusal 0.50, all n=20/arm. Screening-tier (single seed) is stated. **The flag:** these refusal rates rest on the **1B self-judge**, and the committed `examples` show it mislabeling — e.g. the "make a bomb" steered reply (gives a TNT "outline") and the "identity theft" steered reply are both graded **REFUSAL** despite being hedged compliance. So the reported refusal numbers likely **overcount**. This is exactly the FLAS-audit effect, and the lesson **does carry the caveat prominently**: §5/§10 state "a 1B judge is weak… read verdicts as a demonstration of the loop, not a measured refusal rate," the benign-0.50 cell is explained as instrument-dominated, and `judge.py` adds an off-family `STEER_JUDGE_MODEL` hook citing the FLAS audit. Honesty preserved via disclosure; the numbers themselves remain instrument-limited. |

## Overall verdict
All four cited papers are real and the methods (CAA diff-of-means, relative-add steering, CAST-style probe gating) are implemented faithfully and honestly caveated. Two things to fix before any external claim: the **fabricated "Kim, Wang" CAST co-authors** and the **wrong ActAdd title** (both citation-text errors, not id errors). Results are consistent with artifacts and correctly framed as screening-tier, but the refusal rates depend on the flagged 1B self-judge — which the lesson discloses clearly and even provides an off-family escape hatch for. Net: **CONCERN** on citation text + self-judge dependence; no FAIL, no unverifiable id.

Internal QA pass — independent external review pending (auditor shares a model family with the author).
