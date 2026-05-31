# E38 — Function/Task Vectors and DiffMean Share Subspace; Task + Style in One Edit

> **One-line claim:** ICL-derived function/task vectors and DiffMean
> behavior vectors share a low-dimensional subspace; composing them
> transfers both task competence and stylistic behavior in a single
> residual-stream edit, outperforming either alone.
>
> **Block:** E — Mechanistic and interpretability-guided (E34-E40).
> **Primary axis:** A2 (WHAT — direction/subspace).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

Two streams of research have converged on residual-stream directions as the
primary steering medium: the activation-steering literature (CAA, DiffMean,
arXiv:2312.06681) extracts behavior directions from contrastive activation
differences, while the in-context learning (ICL) interpretation literature
(Todd et al. 2023, arXiv:2310.15916; Hendel et al. 2023, arXiv:2305.00586)
extracts "task vectors" or "function vectors" from the ICL-induced
activation differences. Both literatures independently discover that
task-level information is encoded linearly in mid-to-late residual-stream
directions. The hypothesis N4 in our corpus makes the strong prediction
that DiffMean(behavior) aligns with the ICL-induced activation delta
(cos > 0.6), treating steering and ICL as dual representations of the same
operator. E38 is a more practical, compositional test of this alignment:
if task vectors and behavior vectors share subspace, then adding them
together should transfer both the task structure (what kind of output to
produce) and the stylistic/behavioral modifier (how to produce it) in a
single edit, without double-counting or interference. The practical
motivation is significant: current steering workflows apply behavior vectors
independently of task vectors, requiring the model to already have the
desired task behavior in its default distribution. If the two vector types
compose, a single combined edit could both specify the task (e.g.,
"translate to French") and the style modifier (e.g., "in a formal, concise
register"), allowing fine-grained multi-property control without separate
prompting for each property. The mechanism requires that the two vector
types be at least partially orthogonal (so they compose additively without
destructive interference) AND that they share enough subspace structure
that the combined edit does not push the model off the data manifold
(the norm-budget constraint from E22/N5).

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, (i) ICL-derived function/task vectors (e.g.,
the "French translation" task vector from Hendel et al.) and DiffMean
behavior/style vectors (e.g., the "formal register" style vector from CAA)
will have cosine similarity > 0.3 when projected into the top-10 PCA
dimensions of the combined vector set, indicating shared subspace structure;
and (ii) the additive composition of one task vector + one style vector will
achieve >= 90% of the solo task-success rate AND >= 90% of the solo style-
success rate simultaneously, demonstrating that the two vector types compose
rather than compete.

---

## 3. Falsifier (>= 30 words)

If the additive composition achieves < 75% of EITHER the solo task-success
rate OR the solo style-success rate on two or more tested (task, style) pairs,
the non-interference claim is DISCARDED. If cosine similarity in the shared
PCA subspace is < 0.15, the shared-subspace claim is separately falsified.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Todd, Eric, et al. 2023 'Function Vectors in Large Language Models'
arXiv:2310.15916 — the primary function-vector paper; extracts "task
vectors" from ICL activations and shows they causally induce task behavior
when inserted into the residual stream; provides the vector extraction
methodology for the task-vector side of E38.

Hendel, Roee, et al. 2023 'In-Context Learning Creates Task Vectors'
arXiv:2305.00586 — companion work showing that ICL implicitly builds a
task vector in the residual stream; directly motivates the subspace
hypothesis by showing the task is represented linearly.

Zou, Andy, et al. 2023 'Representation Engineering: A Top-Down Approach
to AI Transparency' arXiv:2310.01405 — CAA / DiffMean baseline; the
behavior-vector side of E38; demonstrates that concept/style directions
are linear in the residual stream, creating the parallel to task vectors.

Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA paper with behavior-vector extraction;
provides the DiffMean-style vectors for style and behavior that are
composed with task vectors in E38.
```

---

## 5. Mechanism

Task vectors (Todd et al.) are computed as:

    v_task = mean(h_ICL) - mean(h_zero-shot)

across a set of input-output demonstrations. Style/behavior vectors
(DiffMean/CAA) are computed as:

    v_style = mean(h_style_A) - mean(h_style_B)

Both extractions take the difference of means over a contrastive
set. The hypothesis predicts that these two difference-of-means directions
land in a shared low-dimensional subspace of the residual stream —
specifically, the subspace of directions that the model uses to encode
"what kind of text to produce." If this shared subspace exists, the
sum v_task + v_style has two interpretable components: one that specifies
the task structure and one that specifies the style modifier. The combined
injection adds both:

    h_steered = h + alpha_task * v_task + alpha_style * v_style

Under the additive composition law (from Part 1 of the first-principles
corpus), this works cleanly if the two vectors are near-orthogonal
(|cos(v_task, v_style)| < 0.2). The shared subspace (slight positive
cosine in the top-PCA projection) is compatible with near-orthogonality
in ambient space if the shared PCA dimensions carry only a small fraction
of each vector's norm.

The N4 prediction (cos(DiffMean, ICL_delta) > 0.6) is a stronger version
of this claim; E38 tests the weaker, practical version: compositional
transfer with >= 90% solo efficacy retention, not requiring high cosine.

---

## 6. Predicted Delta

| Condition | Task success | Style success |
|---|---|---|
| Task vector only | 1.00x (baseline) | ~0.1x (no style) |
| Style vector only | ~0.1x (no task) | 1.00x (baseline) |
| Task + Style combined | >= 0.90x (prediction) | >= 0.90x (prediction) |
| Solo sum without norm control | 0.70-0.85x (norm-budget risk) |

Key: the combined condition must BOTH achieve >= 90% of each solo
metric — this is the joint success condition. Norm-budget check: if the
sum ||alpha_task * v_task + alpha_style * v_style|| exceeds the E22/N5
safe budget (~15% of ||h||), renormalise the sum to the budget boundary
before injection.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Task vectors: French translation, sentiment reversal (positive -> negative),
  arithmetic completion — extracted following Todd et al. (2310.15916)
  from ICL demonstrations on Gemma-2-2B.
- Style/behavior vectors: formal register, conciseness, refusal (DiffMean
  from CAA pairs / Anthropic model-written evals).
- Test combinations: (task=French, style=formal), (task=sentiment,
  style=concise), one additional pair.
- Injection: relative_add at alpha_task = 0.10, alpha_style = 0.10;
  apply at same layer (identified from prior sweep) or at task-optimal
  vs style-optimal layers separately.
- Eval: task-success rate (LLM-judge: is the output in French / is
  sentiment reversed?), style-success rate (LLM-judge: is the register
  formal / is the output concise?), WikiText PPL, MMLU-500.
- Seeds: 3 (screening), 7 for rung-3.
- Subspace check: compute top-10 PCA of {all task vectors, all style
  vectors}; report cosine of each vector to the shared subspace.

### 7.2 Where it shines

This is the natural landing point for the function-vector + DiffMean
literature synthesis. If it works, it demonstrates that the two research
streams have been extracting from the same representational medium and
that multi-property control (task + style) is achievable in a single
algebraic edit.

---

## 8. Cross-references

- IDEA_TABLE.md Block E row E38.
- N4 (steering as inverse ICL): E38 is the compositional practical test
  of N4's prediction; N4's cos > 0.6 prediction is the stronger form.
- E17-E19 (stacking and composition): task + style composition is a
  specific two-vector stacking case; Gram-Schmidt orthogonalisation (E19)
  is a potential improvement if the two vectors have significant overlap.
- E22 / N5 (norm budget): combined injection must be checked against
  the norm budget; renormalise if needed.
- arXiv:2310.15916 (function vectors): task vector extraction methodology.
- arXiv:2312.06681 (CAA): DiffMean/style vector extraction methodology.

---

## 9. Committee Q&A

**Q: How do you define "task success" when the task is "French translation"?**

> Task success is judged by an LLM-as-judge (or rule-based: does the
> generated text contain > 80% French words?). The same judge is calibrated
> on a small human-annotated set before the sweep.

**Q: If the two vectors have |cos| > 0.2, the E17/E18 stacking analysis
predicts interference. How does the experiment handle this?**

> First, measure |cos(v_task, v_style)| for each pair. If > 0.2, apply
> Gram-Schmidt orthogonalisation (E19 protocol) before summation. Report
> both the raw and orthogonalised conditions as sub-conditions.

**Q: Is this just a replication of Conceptor AND-composition (E21)?**

> No. Conceptors (Jaeger 2014) operate on activation covariance ellipsoids;
> E38 tests simple additive composition of linear directions. The
> compositional mechanism here is different (additive, not conceptor-based),
> and the question is whether this simpler approach is sufficient for task
> + style transfer.

---

## 10. Verification checklist

- [ ] Task-vector extraction code replicated from Todd et al. (2310.15916)
      methodology on Gemma-2-2B.
- [ ] DiffMean style vectors extracted from >= 50 contrast pairs.
- [ ] PCA subspace computed on the combined set of task + style vectors.
- [ ] Norm budget check: ||alpha_task*v_task + alpha_style*v_style||/||h||
      reported and flagged if > 0.20.
- [ ] LLM-judge calibrated on human-annotated slice (>= 50 examples per
      task/style combination).
- [ ] JailbreakBench CR baseline = 0% before any steering.
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block E, hypothesis E38.
  Status: `o UNTESTED`. Theoretically motivated by convergence of function-
  vector (arXiv:2310.15916) and DiffMean (arXiv:2312.06681) literatures.
  N4 (cos > 0.6 alignment prediction) and this experiment are companion
  tests; E38 is the compositional / practical version. No prior screening
  run.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-E (ICL + steering intersection specialist).*

### Prior plausibility
**MEDIUM.** Function vectors (Todd et al.) have been demonstrated primarily
on GPT-2 and Llama architectures; transfer to Gemma-2-2B requires
replication. The subspace-sharing claim is plausible from the linear
representation hypothesis but the 90% joint-success threshold is
ambitious — interference at moderate cosine overlap (0.1-0.2) could
reduce joint success by 15-25%.

### Mechanism scrutiny
The additive composition mechanism is the standard one from Part 1 of the
first-principles corpus. The key risk is norm budget: at alpha = 0.10
per vector, the combined displacement is ||0.10*v_task + 0.10*v_style||
which, if the vectors point in similar directions, could exceed the safe
budget and cause PPL inflation that degrades both task and style success.

### Confounds
1. "Task success" and "style success" as measured by an LLM judge are
   correlated — a French translation is also more formal by default in
   some registers. The joint-success metric may overcount if the two
   behaviors are semantically correlated.
2. Task vectors may be layer-specific in a different way than style vectors;
   injecting both at the same layer may not be optimal. Consider multi-layer
   injection for each vector at its optimal layer.

### Expected effect size
My prior: joint success at 75-90% (not always >= 90%). The 90% threshold
is achievable for orthogonal (task, style) pairs but may fail for pairs
with semantic overlap (e.g., "formal French translation").

### Verdict
**TESTABLE + PRACTICALLY VALUABLE** — Successful composition would unify
two research literatures and enable a powerful multi-property steering
interface. The 90% threshold is tight; a graduated verdict table is
recommended.
