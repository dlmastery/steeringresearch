# E13 — Early Condition Layer Latency

> **One-line claim:** The CAST condition check can move to an earlier layer than
> the behavior injection with no gating-accuracy loss, cutting per-token latency.
>
> **Source design space:** Block B — Conditional / Gated Steering (E9–E16).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires the CAST pipeline
> from E9 and latency profiling infrastructure.

---

## In Plain English

**What we're testing, simply:** The gate that decides "is this harmful?" and the
nudge that acts on it happen at different processing steps inside the model. This asks
whether we can run the gate at a much *earlier* step without losing accuracy — which
would let the system decide sooner and run faster.

**Key terms (defined here so you don't have to look anything up):**
- **Language model (LLM):** an AI that predicts the next word; here, small Gemma
  models.
- **Steering:** nudging the model's behavior by adding a direction to its internal
  state while it writes.
- **Residual stream:** the model's running internal state, where the nudge is added
  and the gate reads.
- **Layer:** one of the model's stacked processing steps (Gemma-2-2B has 26). The
  model runs them in order, early to late.
- **DiffMean:** the simple recipe for building a direction from "yes/no" examples.
- **Conditional steering / the gate:** applying the nudge only when relevant; the
  gate is the check that decides.
- **CAST:** the gating method these experiments build on.
- **Condition layer vs injection layer:** the step where the gate checks (condition)
  vs. the step where the nudge is applied (injection). They don't have to be the same.
- **Gate AUC:** a 0.5-to-1.0 score for how well the gate tells harmful from harmless.
  Higher = better; we want the early-layer gate to score within 0.03 of the late one.
- **Latency:** how long the model takes to respond. Checking earlier means the system
  can skip work and answer faster.
- **Early-exit:** deciding early so later, expensive steps can be short-circuited.

**Why we're doing this (the point):** Speed matters in real deployment. If the gate
works just as well at an early step, the system can decide whether to steer before
doing most of its work — saving time on every request.

**What the result would mean:** A positive result means we get the same safety
accuracy with lower latency. A negative result means the gate must wait until a late
step, so there's no speed-up to be had.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

In production deployment the latency of a conditional steering system is
dominated by two costs: (a) the behavior injection, which modifies the residual
stream during the forward pass, and (b) the condition check, which must be
executed before the behavior write can be dispatched. If the condition check
is performed at the same layer as the injection, it adds only a dot-product
cost at that layer. But if the condition check must be performed at a late
layer (e.g., layer 18 of 26), all 18 layers must complete before the gate
fires — meaning there is no opportunity to short-circuit early. The practical
question is whether a condition check at an early layer (e.g., layer 6)
achieves the same gate AUC as a condition check at the injection layer (e.g.,
layer 18). If yes, the gate can fire early and the steering infrastructure
knows whether to modify the residual stream before reaching the expensive
later blocks, enabling early-exit optimization. The 7-axis framework (AXIS 1:
site; AXIS 5: condition) makes this question precise: can the condition axis
operate at a different layer from the write axis without loss of precision?

---

## 2. Formal Hypothesis (>= 50 words)

Because the DiffMean condition vector captures a semantic direction that encodes
harm-relevant content, and because the Linear Representation Hypothesis predicts
that semantic concepts are present across multiple layers of the residual stream
(not only at late layers), the gate AUC of a condition check at an early layer
(L_c in {4, 6, 8, 10}) should be within 0.03 of the gate AUC at the behavior
injection layer (L_b in {14, 18}). If this holds, moving the condition check
to L_c = 6 allows the steering infrastructure to branch before reaching L_b,
reducing effective per-prompt overhead to the cost of early-layer condition
extraction (a single dot product at layer 6) rather than a full forward pass
to layer 18. Formal claim: gate AUC at optimal L_c < L_b is within 0.03 of
gate AUC at L_b, measured on the E9 evaluation set (3-seed median) on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If the gate AUC at the best early layer (L_c in {4, 6, 8, 10}) is more than
0.03 below the gate AUC at the injection layer L_b (at 3-seed median), the
condition check cannot be moved early without precision loss, and the latency
benefit is unavailable. Status FALSIFIED. Additionally, if latency is not
reduced (overhead of early-layer probe equals savings from early-exit), the
engineering motivation is removed and status is NEAR-MISS.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering: Concept-Level
Control via Conditional Vectors' (arXiv:2409.05907) — CAST does not specify
L_c = L_b; the paper's condition-extraction methodology allows early-layer
condition checks; the specific layer choice is left open, which is the gap
this experiment fills.

Goh, Gabriel, et al. (anon) 2025 arXiv 'Selective Steering: Discriminative
Layer Identification for Targeted Activation Steering' (arXiv:2601.19375) —
Selective Steering demonstrates that discriminative information for a behavior
is NOT uniformly distributed across layers; some layers carry the diagnostic
signal more cleanly; this finding motivates searching for an early layer where
the harm-relevant signal is already linearly separable.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — the refusal direction emerges in middle
layers but is readable (via probes) from as early as layer 5 in tested models;
if harm-detection exhibits the same early emergence, the condition check can
be moved to layer 5-8 without loss.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — finds that the mid-layer (e.g.,
Llama layer 16) is the most fragile for rogue compliance; an early condition
check (layer 6) avoids this fragile region entirely, providing a safety
co-benefit beyond latency.
```

---

## 5. Mechanism

### 5.1 Layer-resolved gate AUC

Condition vector v_C extracted at each candidate layer L_c:

    v_C(L_c) = DiffMean(harmful_activations@L_c, harmless_activations@L_c)
    gate_AUC(L_c) = PRAUC( cos(h@L_c, v_C(L_c)) > theta_c | label )

If gate_AUC is approximately constant across L_c in {4, 6, 8, 10, 14, 18},
the semantic direction is present at all layers. If it peaks at a specific
early layer, that layer is the optimal early-condition site.

### 5.2 Latency model

Without early exit (condition at L_b): full forward pass to L_b before gate.
With early exit (condition at L_c < L_b): gate fires at L_c; if gate = False,
no behavior write at L_b; if gate = True, continue to L_b and write.
Latency saved when gate = False = cost of layers L_c+1 to L_b (relative
to no-early-exit baseline). Expected savings: ~(L_b - L_c)/L_total of compute
per benign prompt (benign prompts dominate in deployment).

### 5.3 Expected result

From the corpus (steering-first-principles-v2): the refusal circuit assembles
in early-middle layers (Rogue Scalpel finding: peak damage at layer 16 for
Llama). The condition signal (harm-relevance) should be readable even earlier,
at layer 6-10, because the model encodes semantic content before it resolves
the refusal decision. This predicts gate_AUC(L_c=6) ~ gate_AUC(L_b=18) - 0.02.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Gate AUC at L_c = 6 | [0.78, 0.88] | Early layer; slightly below late layer |
| Gate AUC at L_c = 10 | [0.80, 0.90] | Near-optimal early layer |
| Gate AUC at L_b = 18 | [0.82, 0.92] | Full-layer baseline from E9 |
| AUC gap (L_c=6 vs L_b=18) | [0, 0.04] | Falsifier: must be < 0.03 |
| Latency saving (benign prompts, L_c=6 vs L_b=18) | [30%, 50%] | ~12/26 layers saved for benign |
| Latency saving (harmful prompts) | 0% | Harmful prompts go all the way to L_b |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Layer grid (L_c):** {4, 6, 8, 10, 12, 14, 18 (= L_b)} — covers full range
- **Evaluation set:** same 200-prompt set as E9 (100 harmful, 100 benign)
- **Metrics:** gate PR-AUC per L_c; latency per prompt (torch.cuda.Event timing)
- **Control:** same-layer (L_c = L_b) baseline
- **Seeds:** 3 (screening)

### 7.2 Where it shines

Early-layer gating works best when: (a) harm-relevant semantic content is
encoded early (e.g., explicit harm keywords trigger early embedding differences);
(b) the benign prompt distribution is not "borderline" (ambiguous prompts
require later-layer disambiguation). The experiment should report separately
for "obvious" vs "subtle" harmful prompts.

---

## 8. Cross-References

- **E9** (CAST gate): baseline layer choice L_b; evaluation protocol
- **E12** (SCS vs cosine): SCS should also be tested across the L_c grid
- **E26** (gate-before-behavior injection order): tests the reverse (L_c > L_b)
- **N20** (curvature as fragility sensor): early layers may have lower curvature
  and be safer for condition reads (avoiding the fragile mid-layer ridge)
- **IDEA_TABLE.md** Block B row E13

---

## 9. Committee Q&A

**Q: Isn't this just a layer-sweep of a known technique?**

> Partly — but the engineering consequence (early-exit latency saving) is novel
> as a design target and the layer-resolved AUC curve has not been reported for
> Gemma-2-2B specifically. The falsifier is numeric and the latency measurement
> is a concrete deliverable.

**Q: Does early-layer condition extraction change the condition vector itself?**

> Yes — the condition vector at L_c = 6 is different from the one at L_c = 18.
> It encodes the same semantic contrast but at an earlier representational stage.
> The AUC measurement captures whether this earlier representation is
> discriminative enough for the gate purpose.

---

## 10. Verification Checklist

- [ ] Gate PR-AUC computed at each L_c in {4, 6, 8, 10, 12, 14, 18}
- [ ] Latency (ms/prompt) measured at each L_c on Gemma-2-2B (3 seeds)
- [ ] AUC gap (best early L_c vs L_b) compared to 0.03 threshold
- [ ] Latency saving for benign prompts reported
- [ ] IDEA_TABLE.md row E13 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Code needed:
  (a) E9 CAST pipeline with configurable L_c; (b) latency profiling
  (torch.cuda.Event per prompt). Blocked on E9.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-B.*

### Prior plausibility

HIGH. The LRH and early-emergence of semantic features in LLMs are
well-established. The main uncertainty is the AUC gap (will early layers be
within 0.03?). The latency benefit is real if the gap holds.

### Confounds

1. **KV-cache:** in autoregressive decoding, all layers are computed per token
   anyway; the early-exit benefit applies mainly to prefill or if gating is
   done once at the first token of the prompt.
2. **Condition-vector mismatch:** extracting the condition vector at L_c = 6
   and then reading h at L_c = 6 for each new prompt is fine, but the vector
   must be re-extracted at that layer, not transferred from L_b.

### Verdict

**NOVEL+TESTABLE** as an engineering result. Low compute cost, clear falsifier,
practical consequence for deployment.

---

## Pseudocode & Methodology

This hypothesis sweeps the **condition-read layer `L_c`** (independent of the behavior
write layer `L_b`) and checks whether the gate stays accurate at an *early* `L_c`, buying
an early-exit latency win. The knob varied is **`L_c`**.

### 1. Steering-vector recipe

```python
# A separate DiffMean CONDITION vector is RE-EXTRACTED at each candidate L_c
# (the early-layer vector differs from the late-layer one). METHODOLOGY §1.3.
def condition_vector(L_c):
    H = collect_activations(model, tok, harmful_vs_harmless, layer=L_c)
    return diffmean_vector(H.pos, H.neg)        # harm-relevance direction @ L_c
```

### 2. Experiment procedure

```text
1. for L_c in {4, 6, 8, 10, 12, 14, 18(=L_b)}:                 # the ONE knob
2.     v_c = condition_vector(L_c)
3.     gate signal = cos(h@L_c, v_c)                           # CosineGate
4.     gate_AUC(L_c) = pr_auc( gate signal vs harmful/benign label )   # gate.pr_auc
5.     latency(L_c)  = torch.cuda.Event timing per prompt (early-exit model)
6. CONTROL: same-layer (L_c == L_b) baseline.
7. Eval set: the same 200-prompt set as E9 (100 harmful, 100 benign).
8. Report gate_AUC vs L_c and the benign-prompt latency saving of early L_c.
```

Latency model: with the condition at early `L_c`, a benign prompt whose gate is False can
skip the `L_c+1 .. L_b` blocks of behavior dispatch — saving ~`(L_b - L_c)/L_total`.

### 3. Measurement & decision rule

PRIMARY metric: `gate_AUC(best early L_c) − gate_AUC(L_b)`. FALSIFIER (§3): FALSIFIED if
the best early-layer gate AUC is more than `0.03` below the L_b gate AUC (3-seed median);
NEAR-MISS if latency is not actually reduced (early-probe overhead == early-exit savings).
Pre-registered (§6): gate AUC at `L_c=6` `[0.78,0.88]`, at `L_c=10` `[0.80,0.90]`, at
`L_b=18` `[0.82,0.92]`; AUC gap `[0, 0.04]` (must be `< 0.03`); benign latency saving
`[30%, 50%]`. Verdict: SUPPORTED iff the early-layer gate is within 0.03 AUC AND the
latency saving is realized.

### 4. Where the code is / status

The per-layer gate-AUC machinery exists (`gate.CosineGate` + `gate.pr_auc`; condition
vectors from `extract.diffmean_vector`). **New machinery**: a configurable-`L_c` CAST
dispatch with an early-exit path and `torch.cuda.Event` latency profiling. Status
**UNTESTED**, blocked on the E9 CAST pipeline. (Caveat noted in §9: in pure autoregressive
decoding all layers run anyway, so the early-exit win is mainly a prefill / first-token
effect.)

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E13.md`.
