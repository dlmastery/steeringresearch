# E15 — Learned Gate vs Fixed Cosine Threshold Under Distribution Shift

> **One-line claim:** A learned logistic gate on multi-layer activations beats
> a fixed cosine threshold under distribution shift.
>
> **Source design space:** Block B — Conditional / Gated Steering (E9–E16).
>
> **Implementation status:** `SCREENED (n=1) — FALSIFIED_OOD`; infrastructure
> implemented + offline unit-tested (`src/steering/gate.py` + `scripts/run_e15.py`).
> Learned multi-layer logistic gate overfits the tiny in-dist set and generalizes
> WORSE than the fixed cosine threshold under distribution shift (OOD AUC gap
> -0.17, far below the +0.06 falsifier).

---

## In Plain English

**What we're testing, simply:** The gate that decides "is this harmful?" can be a
simple fixed rule, or a small trained model that learns from many examples. This asks
whether the trained gate does better when the incoming requests look *different* from
what it was trained on. (Early result: it actually did worse.)

**Key terms (defined here so you don't have to look anything up):**
- **Language model (LLM):** an AI that predicts the next word; here, small Gemma
  models.
- **Steering:** nudging the model's behavior by adding a direction to its internal
  state while it writes.
- **Residual stream / hidden state:** the model's running internal state, which the
  gate reads.
- **Layer:** one of the model's stacked processing steps.
- **Conditional steering / the gate:** applying the nudge only when relevant; the
  gate is the check that decides.
- **CAST (fixed cosine threshold):** the simple gate — a single fixed cut-off line on
  how harmful-looking a request is. Set once, used unchanged forever.
- **Learned (logistic) gate:** a small trained detector that combines clues from
  several layers and learns its own decision boundary from example data.
- **Distribution shift / OOD ("out-of-distribution"):** when real requests look
  different from the training examples — a new topic or style. The realistic hard
  case for any gate.
- **Gate AUC:** a 0.5-to-1.0 score for how well the gate tells harmful from harmless;
  higher = better.
- **Overfitting:** when a trained model learns the training examples too literally and
  then fails on anything that looks a bit different.

**Why we're doing this (the point):** Deployment requests never look exactly like the
training set. We want to know if spending extra effort on a smart, trained gate
actually pays off when inputs shift — or if the simple fixed rule is more robust.

**What the result would mean:** A positive result would favor the trained gate for
real-world robustness. The actual (negative) result shows the trained gate overfit
its small training set and generalized *worse*, so the simple fixed rule is the safer
choice here.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The CAST cosine gate and the SCS energy-ratio gate (E12) are both fixed
parametric functions: once theta_c is chosen on a development set, it is applied
unchanged to all future inputs, including those from different distributions
than the training prompts. Distribution shift is a fundamental challenge for any
fixed-threshold gate: harmful inputs from a new domain (e.g., chemistry vs
the social-harm domain used for training) may have activations that are
semantically harm-relevant but geometrically distant from the training contrast.
A learned logistic gate trained on multi-layer activation features from a diverse
training set can learn richer, nonlinear decision boundaries — using multiple
layers simultaneously to increase robustness to distribution shift. The cost
is additional training data and a small inference overhead (the logistic gate
evaluates a feature vector from multiple layers). This experiment tests whether
this added complexity buys measurable gate-quality improvement under distribution
shift — the operationally realistic scenario in which training prompts and
deployment prompts are drawn from different distributions.

---

## 2. Formal Hypothesis (>= 50 words)

Because a logistic gate trained on multi-layer activation features can exploit
the richer, complementary information carried by different layers simultaneously
(e.g., shallow layers may encode syntax-level harm signals while deep layers
encode semantic intent), it should generalize better to OOD harmful prompts
than a single fixed cosine threshold at a single layer. The mechanism is that
the logistic gate learns a multi-layer feature combination that is invariant to
the surface-form variation in OOD prompts, while the fixed cosine gate relies
on a single-layer dot product that may vary substantially with prompt style.
Formal claim: on OOD harmful prompts (drawn from a different distribution than
the gate training set), the learned logistic gate achieves gate AUC at least
0.06 higher than the best fixed cosine threshold (E9 optimal) at 3-seed median
on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If the learned logistic gate does not exceed the best fixed cosine threshold
(E9 optimal theta_c) by at least 0.06 gate AUC on the OOD evaluation set,
the hypothesis is FALSIFIED — the added training complexity is not worth the
OOD gain, and fixed-threshold CAST is sufficient. If the learned gate overfits
to the training distribution (in-distribution AUC high but OOD AUC low), status
is FALSIFIED on OOD, SUPPORTED on in-distribution (partial).

---

## 4. Citations (Citation Rigor >= 80 words)

```
Tang, Haoran, et al. 2025 arXiv 'FineSteer: Fine-Grained Steering of Large
Language Models via Subspace-Guided Conditional Activation Steering'
(arXiv:2604.15488) — SCS energy-ratio gate is the strongest fixed-parameter
alternative to the cosine gate; the learned logistic gate should be compared
to SCS as well as CAST cosine.

Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering: Concept-Level
Control via Conditional Vectors' (arXiv:2409.05907) — CAST fixed-threshold
baseline; the paper does not evaluate under distribution shift; this experiment
fills that gap.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — finding F4: "poor cross-prompt
generalization — you CANNOT pre-screen dangerous features; monitoring is
infeasible by enumeration." This finding motivates the OOD evaluation: if
features are direction-invariant under the rogue-scalpel mechanism, a learned
gate that captures this invariance is needed.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — the refusal direction is learned from
in-distribution contrast; OOD harmful prompts may activate the refusal direction
differently; a multi-layer learned gate can capture these variations.
```

---

## 5. Mechanism

### 5.1 Logistic gate architecture

The learned gate g_theta is a logistic regression on a feature vector drawn
from multiple layers:

    features(h) = [h_L1 @ v_L1, h_L2 @ v_L2, ..., h_Lk @ v_Lk]
                  (dot products with condition vectors at layers L1,...,Lk)
    g_theta(h) = sigmoid(W @ features(h) + b) > 0.5

W is a k-dimensional weight vector, b is a scalar bias. Trained on labeled
(harmful/benign) activation pairs using binary cross-entropy. Inference cost:
k dot products (fast) + one linear combination (trivial).

### 5.2 Training protocol

- Training set: 200 harmful + 200 benign prompts (disjoint from eval set)
- Feature layers: L in {6, 10, 14, 18} (4 layers; condition vectors from E9)
- Regularization: L2 with lambda in {0.01, 0.1, 1.0} (cross-validated)
- Evaluation set (OOD): 100 harmful + 100 benign from a different distribution

### 5.3 OOD evaluation

OOD set is constructed from harm categories NOT in the training set, or from
stylistically different prompts (e.g., training on direct requests, OOD on
indirect/jailbreak-style phrasing).

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Gate AUC (fixed cosine, in-dist) | [0.80, 0.90] | E9 result |
| Gate AUC (learned logistic, in-dist) | [0.85, 0.93] | Small in-dist gain |
| Gate AUC (fixed cosine, OOD) | [0.65, 0.80] | Distribution shift degrades cosine gate |
| Gate AUC (learned logistic, OOD) | [0.75, 0.88] | Core claim: >= 0.06 above cosine OOD |
| AUC gap (learned vs cosine, OOD) | >= 0.06 | Falsifier threshold |
| Training cost (logistic gate) | < 5 min | 400 examples, k=4 features |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit
- **Gate training:** logistic regression on {L6, L10, L14, L18} activation
  dot products; 200 harmful + 200 benign (in-distribution)
- **Evaluation:** fixed cosine gate (E9 optimal) and learned gate on (a) in-
  distribution 200-prompt set and (b) OOD 200-prompt set
- **Metric:** gate PR-AUC on both sets
- **Seeds:** 3 (training seed; evaluation seed)
- **Wall-clock:** < 15 min (activation extraction + logistic training)

### 7.2 Where it shines

Distribution shift is the key condition. If in-distribution AUC is similar for
both gates (within 0.03), the OOD comparison is the primary result. OOD must
genuinely be OOD — not just a restyled version of the training prompts.

---

## 8. Cross-References

- **E9** (CAST baseline): provides fixed cosine gate comparison point
- **E12** (SCS gate): second fixed-parameter comparison
- **E13** (early condition layer): multi-layer features in E15 subsume the
  single early-layer gate tested in E13
- **N5** (norm budget): gate training does not affect the activation budget
- **IDEA_TABLE.md** Block B row E15

---

## 9. Committee Q&A

**Q: Isn't a logistic gate equivalent to a prompt classifier?**

> Functionally yes — both classify whether the input is harm-relevant. The
> difference: the logistic gate uses internal activations (activation-space
> features) while a prompt classifier uses token-space features. Activation-
> space features have the advantage of capturing semantic content that is
> invisible at the token level (e.g., indirect harm phrasing that activates
> harm-relevant features despite benign surface form).

**Q: What if the logistic gate overfits to 400 training examples?**

> Logistic regression is a convex model with L2 regularization — overfitting
> is controlled. The 400-example regime is the realistic constraint at 4090-scale;
> if the OOD gap is not achieved with 400 examples, a larger training set is
> the next step.

---

## 10. Verification Checklist

- [ ] Training set: 200 harmful + 200 benign, disjoint from eval set
- [ ] OOD eval set: genuinely different distribution, 200 prompts
- [ ] Logistic gate trained with L2 cross-validation; training AUC logged
- [ ] Gate PR-AUC: both conditions (in-dist and OOD) for cosine and logistic
- [ ] AUC gap (learned vs cosine, OOD) compared to 0.06 falsifier threshold
- [ ] IDEA_TABLE.md row E15 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Requires:
  (a) E9 CAST pipeline (blocked); (b) OOD evaluation set construction;
  (c) logistic gate training script (trivial once activations are extracted).

- 2026-06-01 — SCREENED (n=1) on real Gemma-3-270m-it. **Verdict: FALSIFIED_OOD.**
  Built `src/steering/gate.py` (the multi-layer logistic gate + the fixed cosine-
  threshold baseline) driven by `scripts/run_e15.py` (exp#110, tag
  `E15-learned-gate`). Result on the tiny 10-harmful / 8-benign contrast set:
  **in-distribution PR-AUC** is 1.000 for BOTH gates (cosine 1.000, logistic
  1.000 — perfect resubstitution, a textbook overfitting signature on so few
  points). **OOD PR-AUC**: cosine 0.7177 vs logistic 0.5498, so the OOD AUC gap
  (logistic − best cosine) = **-0.1679** — NEGATIVE and far below the
  pre-registered +0.06 falsifier threshold (§3). Interpretation: the multi-layer
  logistic gate OVERFITS the tiny in-dist set and generalizes WORSE than the
  simple fixed cosine threshold under distribution shift — exactly the §3
  overfitting failure mode ("in-distribution AUC high but OOD AUC low"). The
  mechanism itself is sound on clean data: the offline unit test shows the
  multi-layer logistic gate at AUC 0.998 vs single-layer cosine 0.70 when no
  single feature separates the classes. The falsification here is driven by the
  tiny 270m-scale training data, NOT a code bug — the implementation is
  offline-unit-tested and the in-dist numbers confirm it learns the boundary.
  Next step (does not change the pre-registered hypothesis): re-screen with a
  larger, genuinely-OOD training set before any verdict beyond screening.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-B.*

### Prior plausibility

MEDIUM. The OOD improvement of a learned gate is plausible but depends on
the training set covering the distribution shift axes. A logistic gate trained
on direct harmful requests may not generalize to jailbreak-style phrasing —
precisely the OOD scenario that matters most for safety.

### Confounds

1. **OOD set construction:** if the OOD set is not genuinely OOD (superficially
   different but same underlying features), the OOD improvement will be inflated.
2. **Feature layer selection:** the four chosen layers (6, 10, 14, 18) may not
   be optimal; a fuller sweep would need feature selection.

### Verdict

**NOVEL+TESTABLE.** The OOD evaluation is the critical condition. The cost is
low (logistic training on 400 examples). The falsifier (0.06 AUC gap) is
specific and defensible.

---

## Pseudocode & Methodology

This is the **one IMPLEMENTED Block-B hypothesis**. It compares a **learned multi-layer
logistic gate** against the **fixed cosine threshold** (E9) under distribution shift. The
knob varied is the **gate source** (learned logistic vs fixed cosine); the decisive axis
is in-distribution vs OOD.

### 1. Steering-vector recipe

```python
# Per-layer DiffMean CONDITION vectors at a set of feature layers (METHODOLOGY §1.3).
feature_layers = [6, 10, 14, 18]
v_c = { L: diffmean_vector(*collect_activations(model, tok, harmful_vs_harmless, L))
        for L in feature_layers }
# gate.condition_features builds the multi-layer feature vector:
features(h) = [ h@L · v_c[L]  for L in feature_layers ]      # k dot products
```

### 2. Experiment procedure

```text
1. Build the k-layer feature vector per prompt (gate.condition_features).
2. FIXED gate (CosineGate): score = cos(h@L_best, v_c[L_best]); threshold = E9 optimal theta_c.
3. LEARNED gate (LogisticGate): g(h) = sigmoid(W·features(h) + b) > 0.5,
       trained by BCE with L2 (lambda CV'd) on 200 harmful + 200 benign (in-dist).
4. Evaluate BOTH gates' PR-AUC (gate.pr_auc) on:
5.     (a) in-distribution eval set, and
6.     (b) OOD eval set (harm categories / phrasings NOT in training).
7. AUC gap = PR-AUC(learned, OOD) - PR-AUC(best fixed cosine, OOD).
```

### 3. Measurement & decision rule

PRIMARY metric: **OOD PR-AUC gap (learned − fixed cosine)**. FALSIFIER (§3): FALSIFIED if
the learned gate does not exceed the best fixed cosine threshold by `>= 0.06` OOD AUC; if
in-dist AUC is high but OOD AUC low ⇒ FALSIFIED_OOD (overfitting). Pre-registered (§6):
fixed-cosine OOD `[0.65,0.80]`, learned OOD `[0.75,0.88]`, gap `>= 0.06`. **Observed
(exp#110, n=1, Gemma-3-270m)**: in-dist PR-AUC = 1.000 for BOTH gates (perfect
resubstitution on 10 harmful / 8 benign — a textbook overfit signature); OOD PR-AUC
cosine 0.7177 vs logistic 0.5498, so the gap = **−0.1679**, NEGATIVE and far below the
+0.06 threshold. Verdict: **FALSIFIED_OOD (screening, n=1)** — the multi-layer logistic
overfits the tiny set and generalizes worse than the simple cosine threshold (exactly the
§3 overfit failure mode). The mechanism is sound on clean data (offline unit test: logistic
AUC 0.998 vs single-layer cosine 0.70 when no single feature separates the classes); the
falsification is driven by tiny 270m-scale training data, not a code bug.

### 4. Where the code is / status

**Implemented and offline unit-tested.** `src/steering/gate.py` provides both
`CosineGate` and `LogisticGate` (plus `condition_features`, `pr_auc`, `roc_auc`,
`evaluate_gates`); the driver is **`scripts/run_e15.py`** (exp#110, tag
`E15-learned-gate`). This is the existing learned-gate machinery the rest of Block B
lacks. Status **SCREENED (n=1) — FALSIFIED_OOD**; next step (not a hypothesis change) is
to re-screen with a larger, genuinely-OOD training set.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E15.md`](../PROVENANCE/E15.md).

- **Experiments:** exp# 110 (`autoresearch_results/experiment_log.jsonl`).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/run_e15.py --model models/google/gemma-3-270m-it --quant none # trains the multi-layer logistic gate + fixed cosine baseline, in-dist + OOD PR-AUC
```
