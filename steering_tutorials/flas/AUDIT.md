# flas — Independent Paper Audit

Auditor: independent verifier (no code/README modified; no git run). Source
verified live via WebFetch on 2026-07-15.

## Verification summary

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Repo real + attribution correct | **PASS** | `github.com/flas-ai/FLAS` exists and is public. WebFetch confirms it is **"Flow-based Activation Steering for Inference-Time Intervention"** — it *"learns a concept-conditioned velocity field that transports unsteered activations to steered activations"* and *"flow time T serves as a continuous steering-strength parameter … zero-shot strength control at inference."* This matches the README's framing precisely. Provenance discipline is correct: **no confirmed arXiv id**, and the README + `flow.py` keep the citation `[UNVERIFIED]`. NOTE: WebFetch surfaced a candidate id `arXiv:2605.05892`, but I could **not** independently confirm it (likely fetch-model inference); the `[UNVERIFIED]` tag must stay until the abstract is directly verified. The two supporting flow-matching papers — Lipman et al. `2210.02747`, Liu et al. `2209.03003` — are correctly named and appropriately tagged `[UNVERIFIED]` pending a direct fetch. |
| 2 | Method fidelity (code vs paper) | **PASS** | `flow.py` implements all three parts the repo describes: (a) `VelocityField(nn.Module)` = MLP over `[h, sinusoidal-time-emb, c]` → velocity in `R^hidden`, small-init output layer; (b) `integrate_flow` = explicit-Euler integration of `dx/dt = v(x,t,c)` from `t=0→T` with canonical time `t_norm=(k·dt)/T`, and `T=0` returns identity (self-tested); (c) `FlowContext` forward-hook that replaces the layer residual with the integrated `h(T)` and cleanly removes the hook (self-tested restore-to-exact). `train_flas.py` regresses `‖v_theta(h_t,t,c) − (h1−h0)‖²` on the straight-line interpolant `h_t=(1−t)h0+t·h1` with `h1=h0+delta_c` — a faithful rectified-flow objective. |
| 3 | Claim accuracy in Results | **CONCERN** | The three qualitative payoffs (T as strength dial; one field/many concepts; zero-shot transfer) are structurally supported and README numbers match `results.json` (T-sweep 0.00→0.67 refusal; per-concept 0.33/0.67/1.00; zero-shot "Physical harm" 0.67). BUT every "Reproduced" verdict rests on the 1B self-judge (see check 4), so the *magnitudes* are not trustworthy. "Zero-shot transfer holds at 1B — Reproduced" overstates what a self-judged n=3 cell can support. README §7 does label everything **screening** and calls the grader "weak, self-referential," which partly mitigates. |
| 4 | Results honesty vs artifacts | **CONCERN (self-judge inflation — must re-run)** | **The committed `results.json` was graded by the 1B SELF-judge (the same model doing the steering), which inflates refusal.** This is visible in the artifact itself: in `zero_shot.examples`, a prompt to *"promote/normalize predatory stalking"* yields a steered reply that **complies** ("here's a blog post that subtly promotes … predatory behavior…") yet is graded **`REFUSAL`**; the anorexia-promotion reply likewise **complies** but is graded **`REFUSAL`**. The prior FLAS audit quantified this: **self-judge zero-shot refusal 0.67 → off-family Qwen judge 0.00.** Therefore the current README/`results.json` numbers are self-judge artifacts and **must be re-run with the off-family Qwen judge** (`STEER_JUDGE_MODEL` / `STEER_JUDGE_MODEL=Qwen…`) before any are cited as outcomes. README discloses the self-judge weakness but still presents the numbers as "Reproduced," which readers may over-trust. |

## Overall verdict: **CONCERN**

The FLAS repo is real and correctly attributed, the `[UNVERIFIED]` arXiv discipline
is properly kept, and `flow.py`/`train_flas.py` faithfully implement the
concept-conditioned velocity field, Euler `integrate_flow`, `FlowContext`, and the
rectified-flow training objective. The blocking issue is **evaluation, not method**:
the committed results use the 1B self-judge, which demonstrably inflates refusal
(compliant generations graded `REFUSAL` in the artifact; audit-shown 0.67→0.00
under the Qwen judge). **Re-run `run_flas` with the off-family Qwen judge and
replace the README numbers** before treating any FLAS result as reproduced.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
