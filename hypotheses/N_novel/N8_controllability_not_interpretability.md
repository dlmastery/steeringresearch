# N8 — The Controllability != Interpretability Decomposition

> **One-line claim:** Activation space splits into an interpretable-but-inert
> subspace and a controllable subspace; SAEs over-index on the former; a
> causal-objective dictionary yields fewer, more causal atoms than reconstruction
> SAEs and beats AxBench SAE results on steering.
>
> **Primary axes:** A7 (how-derived), A10 (identifiability/gauge), A12 (basis/superposition)
> **Status:** UNTESTED (tangentially supported by E2 FALSIFIED: Fisher ratio != steering
> efficacy, consistent with the interpretable != controllable split)

---

## 1. Motivation (>= 100 words)

Sparse Autoencoder (SAE) features dominate current LLM interpretability research.
The GemmaScope SAE decomposition produces thousands of monosemantic features that
are human-interpretable — concepts like "toxicity," "politeness," "refusal signal."
But AxBench (a benchmark of steering methods) shows that SAE-feature steering
systematically underperforms raw DiffMean vectors on behavioral tasks. How can
features that look interpretable fail to steer behavior? The controllability-
interpretability decomposition answers this: the reconstruction objective optimizes
for how well a basis explains natural variance in activations — but natural variance
is dominated by inert stylistic and positional variation, not the specific directions
that causally govern behavior. Think of a physical analogy: spectroscopy identifies
all the molecules in a sample (interpretability) but does not identify which molecules
are catalysts for a particular reaction (controllability). Most features identified
by reconstruction SAEs are spectroscopic — they label the activation landscape but
do not move behavior when intervened upon. A causal-objective dictionary, trained
to minimize the behavioral change from interventions rather than the reconstruction
error, would identify a smaller set of features that are causally potent. This
would explain the AxBench gap and provide a principled replacement for reconstruction
SAEs in steering applications. The E2 falsification (S-5: Fisher ratio does not
predict steering efficacy, rho=0.14) is consistent with this picture: the Fisher
ratio measures linear separability — an interpretability-adjacent criterion — and it
fails to predict controllability.

## 2. Formal Hypothesis (>= 50 words)

Let F_rec be the GemmaScope reconstruction SAE feature set and F_causal be a
causal-objective dictionary (trained to minimize intervention-induced PPL increase
while maximizing behavior-cosine shift, using a sparse intervention objective).
The claim is:

  |F_causal| < 0.3 * |F_rec|  [fewer features]

and for each behavior B in the AxBench suite:
  steering_efficacy(F_causal, B) > steering_efficacy(F_rec, B)  [more causal]

with F_causal requiring fewer non-zero atoms per behavior (< 5 vs > 20 typical
for F_rec). Additionally, the subspace spanned by F_causal should have < 0.3
cosine with the null space of downstream readouts (identified by N15 / Non-ID).

## 3. Falsifier (>= 30 words)

If F_causal yields steering efficacy <= F_rec on more than 2 of the 5 AxBench
behaviors, the causal-objective advantage is FALSIFIED. If |F_causal| > 0.5 * |F_rec|
(causal atoms are not significantly sparser), the atomicity claim fails. The E37
design (overlap of interpretable vs causal features) must show < 30% overlap.

## 4. Citations (Citation Rigor >= 80 words)

```
Venkatesh & Kurapath 2026. 'On the Non-Identifiability of Steering Vectors'
arXiv:2602.06801 (ICLR 2026 workshop). Proves that the effective steering subspace
has large null space; features outside the effective subspace are identifiable
but inert — precisely the interpretable-but-inert subspace of N8's claim.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. The M_h <-> M_y
bidirectional link means behaviorally active features are those that move M_y;
reconstruction SAE features that don't move M_y are interpretable but inert.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681 (CAA). Empirical anchor:
DiffMean (a simple non-SAE method) outperforms SAE feature steering on behavior
tasks despite being less "interpretable" — the direct empirical motivation for N8.

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Curveball finds the
optimal curved path; N8 claims the SAE basis misses the directions along which
curved paths are most efficient. A causal-objective basis aligned with the
behavioral manifold would make Curveball's path-finding more efficient.

Park et al. 2023. 'The Linear Representation Hypothesis and the Geometry of Large
Language Models' arXiv:2311.03658. Establishes that behavioral concepts ARE linearly
represented; if true, the causal dictionary should be a small linear basis, not the
thousands-of-features SAE. N8 reconciles these findings: behavioral concepts are
linear AND rare in the reconstruction SAE basis.
```

## 5. Mechanism

The reconstruction SAE minimizes E[||h - sum_i f_i * a_i||^2] + lambda * ||a||_1
where {f_i} are features and {a_i} are sparse activations. This objective rewards
features that explain activation variance. Most activation variance is in the
"inert" subspace — positional encoding effects, syntactic patterns, domain-level
statistics — that do not move behavior when intervened upon.

The causal-objective SAE replaces the reconstruction loss with a joint loss:
  L_causal = E[||h - sum_i f_i * a_i||^2]   [reconstruction, secondary]
           + mu * E[-behavior_cosine(h + delta_causal_i) + PPL(h + delta_causal_i)]
where delta_causal_i = eps * f_i is a small intervention in feature i's direction
and behavior_cosine is the projection onto the behavior target.

This causal regularizer pushes features toward directions where intervention actually
changes behavior. Features that are interpretable but inert (high reconstruction
weight, zero causal effect) are penalized; features that are causally potent but
low-variance are boosted.

The inert subspace size: from the Non-ID paper (arXiv:2602.06801), the null space
of downstream readouts is estimated to have dimension 0.80-0.95 * d_model. This
means 80-95% of directions are behaviorally inert. Reconstruction SAEs span all
these directions; causal SAEs should be concentrated in the 5-20% effective subspace.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| Causal feature count vs reconstruction | 20-30% of F_rec count | Effective subspace is ~10-20% of d_model |
| Atoms needed per behavior (causal) | 2-5 | Concentrated in effective subspace |
| Atoms needed per behavior (reconstruction) | 15-50 | Diluted by inert features |
| Steering efficacy causal vs DiffMean | Within 10% | Both in the effective subspace |
| Steering efficacy causal vs reconstruction SAE | +20% to +50% relative | Causal features avoid inert space |
| Overlap F_causal with null-space | < 30% | Key diagnostic |

## 7. Protocol

### 7.1 Primary experiment

This is a research-infrastructure experiment requiring training a causal SAE,
which is significantly more expensive than other N-hypothesis tests. Staged approach:

Stage 1 (diagnostic, ~2 hours): E37 protocol — measure overlap between
  GemmaScope SAE features sorted by interpretability score and features sorted
  by causal effect size (behavior-cosine shift per unit intervention magnitude).
  If overlap < 30%, the decomposition claim is supported and motivates Stage 2.

Stage 2 (full causal SAE, ~12 hours): Train a 2-layer causal-objective SAE on
  Gemma-3-1B-it @L16, using the mixed loss with mu=0.5. Compare feature count,
  atoms-per-behavior, and AxBench steering efficacy vs GemmaScope SAE.

- Model: Gemma-3-1B-it (primary) / Gemma-3-270m (diagnostic cross-check)
- Behavior suite: AxBench behaviors + 3 synthetic behaviors from the program
- Evaluation: behavior-cosine shift at 5% off-shell displacement; PPL
- Seeds: 3 SAE training seeds (Stage 2 only)
- Wall-clock: Stage 1: ~2 hours; Stage 2: ~12 hours

### 7.2 Where it shines

Any application where interpretability is NOT the goal and steering efficiency is
paramount: fewer features = lower memory, faster lookup, less interference in stacking.
The causal SAE is a drop-in replacement for reconstruction SAE in the N12 unified operator.

## 8. Cross-References

- N3 (orthogonal capacity): the effective-subspace dimension is the "capacity" that
  N3 predicts governs stacking; the causal SAE's feature count should approximate this
- N15 (coset min-collateral): selecting the coset representative with min projection
  onto inert subspaces is equivalent to selecting the causal-aligned coset rep
- E37 (interpretable != controllable): E37 is the diagnostic Stage 1 of N8's protocol
- E36 (SAE selection problem): E36 argues selection (not representation) is the issue;
  N8 argues BOTH: selection AND the representation (reconstruction objective)
- N12 (capstone): the "how derived" axis in the unified operator should use causal SAE
  features for minimum interference
- IDEA_TABLE.md: N8 row, axes A7+A10+A12

## 9. Committee Q&A

**Q: Training a causal SAE requires behavioral labels for every feature intervention.
This is extremely expensive for a large feature dictionary. How is this feasible?**

> Stage 1 uses GemmaScope features (already trained) and only computes the causal
> effect of each existing feature by running a small intervention. Cost: O(N_features)
> forward passes, feasible for 16k features on a 1B model in ~2 hours. Stage 2
> trains a NEW causal SAE; this is expensive (~12 hours) and is run only if Stage 1
> confirms the interpretable-inert split.

**Q: The causal objective requires knowing which behaviors to optimize for during
training. Isn't this circular — you're building a SAE for specific behaviors,
then testing it on those same behaviors?**

> Yes, this is the key limitation. The evaluation must use held-out behaviors not
> seen during causal SAE training. The protocol uses 3 behaviors for training and
> 3 different behaviors for evaluation (AxBench suite has sufficient diversity).
> If causal features generalize to held-out behaviors, the subspace is genuinely
> "causal" rather than behavior-specific.

**Q: DiffMean already outperforms SAE on steering. Why train a causal SAE at all?**

> Causal SAE provides SPARSE, INTERPRETABLE features that are ALSO causal — the
> best of both worlds. DiffMean gives one dense vector; causal SAE gives multiple
> atoms with independent interpretability. This enables algebraic composition (N10)
> and the N12 unified operator's basis term. DiffMean cannot be the basis for
> a general steering algebra.

## 10. Verification Checklist

- [ ] Stage 1: GemmaScope feature causal-effect ranking computed and cross-correlated
      with interpretability ranking
- [ ] Overlap percentage (target < 30%) reported with definition of "overlap"
- [ ] Stage 1 result triggers Stage 2 decision recorded in ledger
- [ ] Stage 2 (if run): causal SAE training logs saved
- [ ] Held-out behavior evaluation (3 training + 3 held-out behaviors)
- [ ] Comparison table: causal SAE vs reconstruction SAE vs DiffMean
- [ ] Null-space overlap metric computed (< 30% target)
- [ ] IDEA_TABLE.md N8 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. Tangential support from S-5
  (E2 FALSIFIED: Fisher ratio, an interpretability-adjacent metric, does not predict
  steering efficacy rho=0.14), consistent with the interpretable-inert split.
  E37 (the diagnostic Stage 1) is a prerequisite and also UNTESTED. Stage 2 (causal
  SAE training) is the most expensive experiment in the N-block (~12 hours) and
  requires Stage 1 confirmation before investment.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM-HIGH. The AxBench gap between SAE and DiffMean is empirically documented.
The Non-ID paper's large null-space estimate (80-95% inert) provides theoretical
support. The causal-objective SAE idea is not new in principle (causal representation
learning exists), but its application to steering vectors is novel.

### Mechanism scrutiny

The causal-objective loss requires behavioral labels at training time, creating a
chicken-and-egg problem: you need to know what behaviors matter before training the
SAE that will identify the causal features for those behaviors. This is not circular
IF the causal subspace generalizes across behaviors — which is the key empirical
claim. If the causal subspace is behavior-specific (each behavior has its own causal
subspace), Stage 2 produces per-behavior SAEs, not a general causal basis.

### Confounds

1. GemmaScope features were trained on different data than the steering evaluation
   prompts; the "inert" classification may be dataset-dependent.
2. The causal effect measurement at Stage 1 uses the same forward-pass setup as
   DiffMean; if DiffMean already finds the causal direction, Stage 1 simply
   confirms that DiffMean ≈ top causal feature, not a novel finding.
3. Small sample: causal effects estimated from ~50 interventions per feature may
   be noisy for features with small baseline activation.

### Does the specific causal-objective claim matter?

HIGHLY. If the claim holds, it resolves the AxBench paradox (interpretable but
ineffective SAE features) with a principled fix, and provides a sparse basis for
N10 (concept algebra) and N12 (unified operator). The practical value is large.
But the 12-hour Stage 2 investment should only follow confirmed Stage 1 (< 30%
overlap).

### Literature precedent

Causal representation learning: Schölkopf et al. 2021 (NeurIPS 2021) established
the principle of learning causal structure; applying it to SAE features for LLMs
is novel. Intervention-based feature evaluation: Tucker et al. 2021 (NeurIPS 2021,
Invariant Risk Minimization variants) use intervention objectives for feature learning.

### Skeptical effect-size estimate

Overlap (interpretable ∩ causal): 30-50% (vs claimed < 30%). Many GemmaScope
features are causal for SOME behaviors; the issue is that they are not causal for
the specific targeted behaviors. A 30-50% overlap means the decomposition exists
but is not as clean as claimed. Stage 2 efficacy improvement: +10-25% vs reconstruction
SAE (vs claimed +20-50%) — still practically significant.

### Minimum distinguishing experiment

Stage 1 only (2 hours): rank GemmaScope features by causal effect; report top-20
causal features and top-20 interpretability features; compute overlap. If overlap
< 40%, the decomposition claim is supported; commit to Stage 2. If overlap > 60%,
N8's strong form is FALSIFIED but the weak form (causal ordering improves feature
selection) remains valid and E36 already covers it.

### Verdict

TESTABLE-MEDIUM (Stage 1) / TESTABLE-HIGH-COST (Stage 2). Stage 1 is the
essential pre-check with clear go/no-go criteria. Stage 2 is a major commitment
(~12 hours) that should follow a positive Stage 1. The hypothesis has high
practical value if confirmed — resolving the AxBench gap is a significant result.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N8 claims reconstruction SAEs over-index on an interpretable-but-inert subspace; a causal-objective dictionary gives fewer, more causal atoms. **UNTESTED** — Stage 2 needs a causal SAE trainer.

### 1. Steering-vector recipe (causal vs reconstruction atoms)

```python
# Stage-1 diagnostic uses EXISTING GemmaScope features f_i; rank by CAUSAL effect, not reconstruction:
causal_score(f_i) = behavior_cosine(h + eps*f_i) - lambda * PPL(h + eps*f_i)   # small intervention
interp_rank  = sort(features, key=interpretability_score)
causal_rank  = sort(features, key=causal_score)
overlap = jaccard(top20(interp_rank), top20(causal_rank))

# steering edit uses a few causal atoms additively (METHODOLOGY §2):  h' = h + Σ alpha_i f_causal_i
# null-space check: cos(span(F_causal), null_space_of_readouts) < 0.30
```

### 2. Experiment procedure

```text
Stage 1 (~2h): rank GemmaScope features by causal_score; compute overlap with interpretability rank.
   If overlap < 30% -> the interpretable≠causal split is real -> trigger Stage 2.
Stage 2 (~12h, NEW trainer): train a causal-objective SAE (mixed reconstruction + causal loss, mu=0.5);
   compare |F_causal| vs |F_rec|, atoms-per-behavior, and AxBench steering efficacy.
   Evaluate on HELD-OUT behaviors (3 train / 3 eval) to rule out behavior-specific overfit.
```

### 3. Measurement & decision rule

- **Primary metrics:** interpretable∩causal overlap; AxBench steering efficacy F_causal vs F_rec vs DiffMean.
- **Pre-registered falsifier (§3):** F_causal ≤ F_rec efficacy on >2 of 5 behaviors, OR |F_causal| > 0.5·|F_rec|, OR Stage-1 overlap ≥ 30% ⇒ strong form FALSIFIED.
- **Verdict logic:** Stage-1 overlap <30% is the go/no-go gate for the expensive Stage 2.

### 4. Where the code is / status

UNTESTED. Tangentially consistent with S-5 (Fisher ratio fails to predict efficacy). The five-axis bundle exists, but the **causal-objective SAE trainer** (and the GemmaScope causal-ranking pass) are not implemented — that missing infrastructure is why N8 is UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N8.md`.
