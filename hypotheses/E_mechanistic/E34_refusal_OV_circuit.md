# E34 — Refusal Steering Acts Mainly Through the OV Circuit

> **One-line claim:** Refusal steering modifies model outputs primarily via
> the attention output-value (OV) circuit; freezing the attention score
> (QK) computation costs less than 10% of steering efficacy.
>
> **Block:** E — Mechanistic and interpretability-guided (E34-E40).
> **Primary axis:** A1 (WHERE — intervention site).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

Transformer models decompose the attention block into two semi-independent
circuits: the QK circuit, which decides WHERE to attend (softmax over
query-key products), and the OV circuit, which decides WHAT to write given
that attention pattern (value * output projection). Mechanistic
interpretability work on GPT-2 (Meng et al. 2022, ROME) established that
factual associations live in MLP weights, while Conmy et al. (2023, ACDC)
showed that specific attention heads' OV circuits execute high-level
reasoning moves such as copying, indirect object identification, and
inhibition. Subsequent refusal-circuit analyses (Arditi et al. 2024,
arXiv:2406.11717; Wei et al. 2024) identified that refusal behaviour in
instruction-tuned models traces back to specific attention-head OV outputs
that up-weight the refusal token and related continuations. If this
mechanistic picture is correct, then activation steering that targets the
residual stream is effectively injecting into the summed OV outputs, and
the QK routing is a bystander. The practical consequence is that one could
freeze attention-score computation (preventing any change in which tokens
are attended to) and still obtain the same steering effect as unrestricted
steering, because the residual-stream write from the OV circuit is what
carries the behaviour direction. This decomposition would be theoretically
significant: it would confirm that steering hijacks the WRITING half of
attention, not the routing half, and would permit targeted interventions
that touch only OV projections. It also has safety implications under the
Rogue Scalpel framework (arXiv:2509.22067): if refusal assembly is an OV-
circuit phenomenon and the fragile mid-layer ridge (Rogue Scalpel F3) is
located precisely there, then Guard Layer C (avoid fragile layers) should
be calibrated at the layers where OV circuits are refusal-active, not at
arbitrary mid-layers.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, steering the residual stream with a DiffMean
refusal vector while freezing the QK attention scores (replacing them with
the unsteered baseline scores at each head) will yield behavior-success
rate within 10 percentage points of unrestricted steering at matched alpha,
with perplexity change also within 10% of the unrestricted condition. This
confirms that the OV circuit, not QK routing, is the load-bearing
sub-circuit for residual-stream steering of refusal behaviour; the QK
scores are a downstream spectator in the steering mechanism.

---

## 3. Falsifier (>= 30 words)

If freezing QK attention scores reduces steering efficacy by MORE than 10
percentage points on the primary behavior-success metric (at alpha values
within the clean steering window), the hypothesis is DISCARDED and Status
moves to `x disproved`. A frozen-QK efficacy drop >= 10pp on the refusal
behavior direction signals that QK routing is causally load-bearing and not
separable from the OV write in the steering mechanism.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Arditi, Andy, et al. 2024 'Refusal in Language Models Is Mediated by a
Single Direction' arXiv:2406.11717 — establishes the refusal-direction
finding; shows a single linear direction mediates refusal across Llama,
Gemma, and other open models; the paper's ablation methodology (ablate a
direction in the residual stream) is the direct precursor to this
hypothesis's OV-vs-QK decomposition.

Conmy, Arthur, et al. 2023 'Towards Automated Circuit Discovery for
Mechanistic Interpretability' arXiv:2304.14997 — ACDC; the canonical
framework for isolating which attention-head sub-circuits (QK vs OV) cause
a behavior; motivates the frozen-QK protocol.

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — shows the refusal ridge is a fragile mid-
layer phenomenon (Finding F3); if OV circuits host this ridge, Guard C
calibration maps directly onto OV-active layers.

Wei, Jason, et al. 2024 'Refusal Circuit' (related mechanistic work on
refusal implementation in instruction-tuned models, arXiv:2604.08524) —
identifies the specific transformer sub-circuit components executing
refusal; this E34 hypothesis directly tests whether steering piggybacks on
those same OV pathways.
```

---

## 5. Mechanism

The residual stream at layer l is the sum of all prior OV outputs and MLP
writes:

    h_l = embed + sum_{k < l} (OV_k * a_k + MLP_k)

where a_k is the attention score vector and OV_k is the head's value-output
product. Steering injects alpha*v directly into h_l, which is then ADDED
to subsequent OV reads. If the refusal direction v lies primarily in the
subspace spanned by OV write vectors (i.e., refusal is implemented as a
specific OV output), then re-routing QK scores changes which tokens
contribute to future OV writes but does not alter the already-injected
direction. The frozen-QK condition holds a_k constant at the unsteered
values: any downstream behavior change is attributable purely to the
injected alpha*v being processed by subsequent OV matrices. If efficacy
holds, it confirms v operates as an OV-style write that subsequent layers
read and amplify without needing updated attention routing.

The protocol patches a specific set of heads identified as refusal-relevant
by prior circuit analysis (or by an activation-patching sweep), freezes
their attention score softmax outputs, and then applies the normal
residual-stream add. The delta in efficacy directly measures the causal
role of QK routing in the steering mechanism.

---

## 6. Predicted Delta

| Metric | Predicted Delta (frozen-QK vs unrestricted) |
|---|---|
| Behavior success rate | < -10 pp (hypothesis predicts < 10pp loss) |
| PPL (WikiText-103) | < +10% relative |
| MMLU delta | within ±1 pp |
| Composite score | within ±0.05 |

Directional prediction: less than 10 percentage-point drop in behavior
success under frozen-QK; efficacy should be largely preserved because OV
output writes are the operative channel. If the drop exceeds 10 pp the
falsifier fires.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit quantised), single RTX 4090.
- Extract DiffMean refusal vector from >=50 contrast pairs (harmful vs
  harmless instructions from Sorry-Bench / CAA suite).
- Two conditions: (A) standard residual-stream add at the identified
  injection layer; (B) same add but with QK attention score tensors
  replaced by their unsteered-forward-pass values (freeze hook on
  attn_weights before softmax).
- Injection layer: identified from prior E2/layer-sweep results.
- Eval: behavior success rate on CAA behavior suite (refusal subset),
  WikiText-103 PPL, MMLU-500 accuracy, JailbreakBench CR (baseline must
  be 0%).
- Alpha grid: use relative_add parameterisation (alpha = fraction of ||h||)
  following C9b findings; sweep alpha in {0.05, 0.10, 0.20}.
- Seeds: 3 (screening level); promote to 7 for rung-3 eval.

### 7.2 Where it shines

This experiment distinguishes E34 (OV-primary) from a competing account
where QK routing changes (e.g. attending to the "I cannot help" token more
strongly) are the mechanism. The protocol cleanly isolates the two sub-
circuits and is the minimum-cost test of mechanistic specificity.

---

## 8. Cross-references

- IDEA_TABLE.md Block E row E34.
- E37 (interpretability != controllability): if OV circuits are controllable
  but QK circuits are not, the controllable subspace is specifically OV.
- E40 (parallel transport across layers): if refusal direction is OV-carried,
  multi-layer transport should preserve OV-type writes.
- Rogue Scalpel Guard Layer C (fragile-layer avoidance): calibrate guard at
  OV-active refusal layers.
- arXiv:2604.08524 (refusal circuit mechanistic): direct prior mechanistic work.
- arXiv:2509.22067 (Rogue Scalpel): fragility at mid-layer OV writes.

---

## 9. Committee Q&A

**Q: Could freezing QK trivially break coherence (causing PPL spike) that
is then confounded with efficacy loss?**

> The falsifier is on behavior-success rate, which is measured on the
> refusal direction independently of coherence. PPL is tracked separately.
> If PPL spikes but behavior holds, the hypothesis passes on its primary
> claim. The secondary coherence check is reported as a separate metric.

**Q: Isn't "freeze QK" implementation-dependent and model-specific?**

> Yes; the hook must be inserted at the correct point in Gemma's attention
> implementation (after QK dot product, before softmax, to freeze the
> routing distribution). The protocol specifies this precisely and the
> implementation is audited before the main sweep.

**Q: What if only a subset of heads matters?**

> The protocol runs two sub-conditions: (B1) freeze QK at ALL heads at the
> injection layer; (B2) freeze QK only at the heads identified by
> activation-patching as refusal-relevant. If B2 shows more efficacy loss
> than B1, the relevant heads are identified; if B2 shows minimal loss, the
> refusal-relevant heads are OV-primary as predicted.

---

## 10. Verification checklist

- [ ] Frozen-QK hook verified to leave attention scores unchanged (test on
      a zero-steering baseline: both conditions must give identical outputs).
- [ ] Contrast-pair extraction logged with pair count >= 50.
- [ ] Injection layer pre-specified from prior sweep; not chosen post-hoc.
- [ ] JailbreakBench CR baseline = 0% before steering confirmed.
- [ ] Behavior-success metric uses independent judge (LLM-as-judge or AxBench
      scorer on real generated text), not projection proxy.
- [ ] Results reported at matched fractional alpha (relative_add).
- [ ] Rung-3 gate: n >= 7 seeds, Wilcoxon signed-rank p < 0.05 (Holm-
      Bonferroni corrected), bootstrap CI excludes zero.
- [ ] Row updated in IDEA_TABLE.md and EXPERIMENT_LEDGER.md.

---

## 11. Status Journal

- 2026-05-31 — Created from autoresearch corpus block E, hypothesis E34.
  Status: `o UNTESTED`. No prior screening run. OV-circuit mechanistic
  interpretation is theoretically motivated by arXiv:2406.11717 and
  arXiv:2604.08524 but has not been tested on Gemma-2-2B in our harness.
  Dependency: requires working generation-based behavior eval (in progress
  per FINDINGS.md required-experiments list).

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-E (elite-research-scientist critic, mechanistic
interpretability specialisation).*

### Prior plausibility (independent of proposed mechanism)
**MEDIUM.** Arditi et al. (arXiv:2406.11717) already showed refusal is
mediated by a single direction; whether that direction is "OV-carried" vs
merely present in the residual stream (which is the sum of OV and MLP
outputs) is a finer decomposition. The OV-primary story is consistent with
the mechanistic-interpretability literature but competes with the MLP-
midlayer account (Geva et al. 2022, FFN as key-value stores).

### Mechanism scrutiny
The mechanism claim is specific and testable, but the protocol does not
control for the MLP pathway. If the refusal direction is partially written
by MLPs, freezing QK alone is not a clean isolation — the MLP contribution
persists even in the frozen-QK condition. A complete mechanistic test would
also freeze MLP activations selectively, which is more complex.

### Confounds
1. QK freezing may inadvertently modify residual-stream norms at subsequent
   layers (because attention outputs normally depend on the current h, which
   now has alpha*v injected). This is a second-order effect but could inflate
   the apparent OV contribution.
2. The "refusal-relevant heads" selection requires an activation-patching
   sweep that itself needs careful methodology (patching noise source,
   threshold choice).

### Numerology check
The 10 percentage-point threshold for the falsifier is somewhat arbitrary;
the natural null is whether QK and OV contribute equally (50/50 split). A
more principled threshold would be derived from an information-theoretic
decomposition of the attention block's contribution to the refusal
direction.

### Literature: precedent
arXiv:2604.08524 (refusal circuit) is the direct predecessor; this
experiment should be compared to its head-level attribution results to
avoid redundancy. Key novelty here is applying the decomposition to the
STEERING context rather than the natural-behavior context.

### Expected effect size — skeptical a-priori
My prior: efficacy loss under frozen-QK is 10-30% (not < 10%), because QK
routing changes which tokens attend to the injected vector and this
secondary routing effect is non-negligible. The 10pp threshold may be too
tight for a clean "pass."

### Minimum-distinguishing experiment
Run both conditions at three alpha values and report the frozen-QK efficacy
drop. If it clusters below 10pp, the OV-primary story holds; if it
clusters at 20-30pp, QK and OV are both load-bearing and the story needs
revision.

### Verdict
**TESTABLE + MECHANISTICALLY INTERESTING** — A clean decomposition
experiment that could confirm or sharpen the refusal-circuit story.
The 10pp falsifier threshold is tight; consider pre-registering a
graduated verdict (< 5pp = OV-primary confirmed; 5-20pp = partial; > 20pp
= falsified).

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to E34, a
**mechanistic** test: does residual-stream refusal steering act through the OV
circuit (WHAT attention writes) rather than QK routing (WHERE it attends)? The
vector is standard DiffMean; the novelty is a frozen-QK ablation hook.

### 1. Steering-vector recipe (DiffMean refusal direction)

```python
# §1.3 METHODOLOGY: closed-form DiffMean, no training.
bank = extract.build_vector_bank(model, tok, load_concept("refusal"), layer_inj)
                                                  # >=50 harmful-vs-harmless pairs (Sorry-Bench/CAA)
v_refusal = bank[layer_inj]["diffmean"]           # mean(h_harmful) - mean(h_harmless)
# layer_inj is the prior E2/layer-sweep injection layer; relative_add normalises magnitude.
```

### 2. Experiment procedure (frozen-QK vs unrestricted)

```text
1. Capture the UNSTEERED forward pass; cache its attention-score tensors a_k
   (post QK dot-product, pre-softmax) at the injection layer's heads.
2. Condition A (unrestricted): hooks.apply_operation(h, v_refusal,
      operation="relative_add", alpha)  -> h' = h + alpha*||h||*unit(v)  (§2).
3. Condition B (frozen-QK): same relative_add, PLUS a freeze hook that overwrites
   the steered pass's attention scores with the cached unsteered a_k, so only the
   OV write of the injected direction can propagate. Sub-conditions:
      B1 = freeze QK at ALL heads of the injection layer;
      B2 = freeze QK only at activation-patching-identified refusal heads.
4. Alpha grid: relative_add alpha in {0.05, 0.10, 0.20}.
5. MEASURE (§3 METHODOLOGY): behavior efficacy (off-family judge.GeminiJudge on
   real generated text); WikiText-103 PPL; MMLU-500; JailbreakBench CR (baseline
   0%); geometry probes. Sanity: at alpha=0 both conditions must be byte-identical.
```

### 3. Measurement & decision rule

- **PRIMARY metric:** behavior-success-rate delta of frozen-QK (B) vs
  unrestricted (A) at matched alpha. (PPL is reported separately; the falsifier is
  on behavior, not coherence — §9.)
- **Hypothesis (§2/§6):** frozen-QK efficacy within 10 pp of unrestricted, PPL
  within 10% — the OV circuit is load-bearing, QK is a spectator.
- **Pre-registered FALSIFIER (§3):** if freezing QK reduces efficacy by MORE than
  10 pp on behavior-success, DISCARD (`x disproved`) — QK routing is causally
  load-bearing. (SciCritic suggests a graduated read: <5pp OV-primary; 5-20pp
  partial; >20pp falsified — but the registered threshold is 10 pp.)

### 4. Where the code is / status — UNTESTED

- **No driver yet** (campaign + `scripts/build_provenance.py` -> `PROVENANCE/E34.md`).
- **Missing machinery (why UNTESTED):** the **frozen-QK hook** — a hook that
  intercepts Gemma's attention after the QK dot product and before softmax and
  substitutes cached unsteered scores — does not exist in `hooks.py`; and an
  **activation-patching sweep** to identify refusal-relevant heads (for sub-
  condition B2). Generation-based behavior eval is also a prerequisite.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E34.md`.
