# AutoResearch Process — Claude as Expert Steering Researcher

> The detailed inner loop. `CLAUDE.md` is the constitution; this is the
> operating procedure. Adapted from `dlmastery/autoresearch` (FX) and the
> `sacgeometry` skill pack for the steering domain.

## Karpathy's principle, our adaptation

Karpathy's autoresearch: modify → train (5 min) → check if improved →
keep/discard → repeat, agent runs autonomously until interrupted, one file
changed, one metric to beat. We keep the keep/discard loop but: (1) **never
deviate far from the winner** — single-axis perturbations of the sacred
champion; (2) **Claude is the expert** — diagnose per-axis, cite literature, form
falsifiable hypotheses, no blind search; (3) **ladder-bound** — clear rung *k*'s
gate before spending rung *k+1* compute (CLAUDE.md §4).

## The loop (every iteration)

### Step 1 — Read state
- `experiment_log.jsonl` (full history), `best_config.json` (champion),
  `IDEA_TABLE.md` (what's open), `EXPERIMENT_LEDGER.md` (rung status).
- Note which of the **five axes** moved vs the prior experiment, and which rung
  the active method currently sits on.

### Step 2 — Diagnose (where the work happens)
- **Per-axis forensics.** Which axis is the bottleneck of the composite?
  - Behavior low → wrong layer/direction/α, or wrong source (diffmean vs PCA)?
  - Capability drop → α over the coherence cliff (E3), or fragile-layer write (F3)?
  - Coherence (PPL) up → off-shell displacement Δ‖h‖ too large (N17)? norm budget
    spent (N5)? curvature too high at this layer (N20)?
  - Safety leak → mid-layer perturbation knocking the refusal ridge (Rogue Scalpel)?
  - Selectivity low → condition/behavior entangled (N6)? threshold θ miscalibrated?
- **Geometry leading indicators** (cheap, behavior-free): effective-rank drop,
  participation ratio at the injection layer, Δ‖h‖. These predict rogue-fragility
  and the coherence cliff BEFORE you spend a behavior eval.
- **Consistency.** Do SMOKE (1B) and DEV (2B) agree? If 1B says go and 2B
  regresses, suspect a scale-specific manifold effect (D-block hypotheses).
- **Trajectory.** Plot composite over experiments — progress or cycling? Have we
  exhausted this axis (3 failures ⇒ rethink the diagnosis, not the value)?

### Step 3 — Research (go deep when stuck)
Read the relevant `corpus/*.md` and the cited primary arXiv for the **diagnosed**
problem, not in general:
- Foundations: CAA (2312.06681), ITI (2306.03341), RepE (2310.01405),
  refusal-direction (2406.11717), CAST (2409.05907).
- Geometry: Manifold Steering (2605.05115), Curveball (2603.09313), Spherical/
  Angular (2510.26243), CRH (2605.01844), Non-Identifiability (2602.06801).
- Safety: Rogue Scalpel (2509.22067, incl. Appendix E).
- Eval: AxBench (2501.17148), the benchmark suite doc.
Cite in full format with real arXiv IDs; `[UNVERIFIED]` if unsure.

### Step 4 — Hypothesize (specific, falsifiable)
> "Behavior on AxBench-mini is low at layer 9 because the Fisher ratio peaks at
> layer 6 (E2); per CAST (2409.05907) the optimal injection layer is the layer of
> max linear separability. I predict moving injection to layer 6 raises behavior
> from 0.31 to ≥0.45 while MMLU-tiny drop stays <2 pp and PPL stays bounded."

NOT "let me try a different layer and see."

### Step 5 — Design ONE experiment
- Change exactly ONE of the 12 axes from the champion (or documented baseline).
- Justify every value with corpus/paper/prior result. Predict the outcome.

### Step 6 — Run & analyse
- Run via the runner at the appropriate rung (start at the lowest unproven rung).
- Compare to prediction across all five axes + geometry probes. KEEP iff the
  composite improves at matched coherence and no axis regresses past its gate;
  else REVERT and record why the hypothesis was wrong.

### Step 7 — Decide next direction
- KEEP → small single-axis tweaks around the new champion (local search), or
  promote it up a rung.
- REVERT after 1–2 tries → different axis for the same diagnosed problem.
- REVERT after 3+ on the same axis → the diagnosis is wrong; step back.
- Occasionally take a RADICAL step (new site, new metric per axis 9) to escape a
  local optimum — but only with a falsifiable hypothesis.

## Anti-patterns

| Anti-pattern | Do instead |
|---|---|
| "Try X and see" | "X because [diagnosis] + [paper] predicts [mechanism]" |
| Grid search over α | Diagnose → hypothesize → test ONE α with justification |
| Change 2+ axes at once | ONE axis. Sequence them if both matter. |
| Report aggregate only | Always read the per-axis + per-rung breakdown |
| Repeat a failed axis | 3 failures ⇒ rethink the diagnosis |
| Arbitrary α/layer | Every value justified: corpus, paper, or prior result |
| Run before diagnosing | Never. Diagnosis is authored before launch. |
| Assume determinism | Measure the noise floor: same config, different seed. |
| Win safety via gibberish | The composite's coherence term forbids it |
| "Everything on" hybrid | Orthogonal-axis additive ladder only (§9) |

## Why this beats blind hyperopt

Optuna explores a space; this process eliminates most of it with the 12-axis
geometry, forms testable mechanistic hypotheses, reads the primary literature for
diagnosed failures, predicts outcomes, and learns from failures by updating the
mental model. Optuna cannot read Rogue Scalpel Appendix E and realize the safety
leak is an off-manifold displacement, not an attack direction. Claude can.

## State files
See `CLAUDE.md §12`. The runner writes results; Claude authors the pre-run
reasoning. The runner refuses to fabricate pre-run fields — a missing diagnosis/
citation/hypothesis/prediction is a protocol violation, not a placeholder.
