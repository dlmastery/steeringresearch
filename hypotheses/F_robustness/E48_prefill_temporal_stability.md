# E48 — Steering Directions Are Temporally Stable; Prefill-Only Suffices

> **One-line claim:** Behavior steering directions are temporally stable
> within a generation: a once-at-prefill injection achieves the same
> behavior-success rate as per-token recomputation of the steering vector,
> because the residual stream's direction-encoding of the behavior does not
> drift meaningfully token-by-token within a single generation.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axis:** A6 (WHICH TOKENS — span/temporal).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

Activation steering is typically implemented as a once-at-prefill (or once-
at-a-fixed-layer) injection: the steering vector is added to the hidden
state at the designated layer during the prefill (prompt processing) phase.
During autoregressive generation, the steering vector is typically NOT
re-injected at each generated token — the assumption being that the initial
injection propagates through the residual stream and maintains the behavioral
modification throughout the generation. An alternative implementation is
per-token recomputation: at each token generation step, the steering
vector is re-injected into the hidden state at the designated layer. This
is more computationally expensive (one extra matrix-vector product per
token per injection layer) but could in principle be more robust if the
behavior direction "decays" over generated tokens. The hypothesis is that
the prefill-only approach is sufficient because the residual stream encodes
the behavioral direction in a persistent way: the initial injection rotates
or translates the residual stream into the "behavior-active" region, and
subsequent token generation stays in that region without requiring
re-injection. This connects to N9 (steering as closed-loop dynamical
control) — N9 proposes a feedback controller that adjusts alpha based on
the current behavior projection error; E48 tests the simpler claim that
no such feedback is needed within a single generation because the projection
does not drift. If E48 is confirmed (prefill-only is sufficient), it reduces
the per-token overhead of steering to zero after the prefill, which is
critical for efficient deployment on long generations.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, the behavior-success rate of once-at-prefill
steering (inject alpha*v only during prompt processing) will be within 5
percentage points of per-token steering (inject alpha*v at each generated
token) on generations of up to 256 tokens, and the perplexity of prefill-
only steering will be within 5% of per-token steering. This confirms that
the behavior direction is temporally stable within typical generation
lengths and per-token recomputation is unnecessary.

---

## 3. Falsifier (>= 30 words)

If prefill-only steering achieves > 10 percentage-point LOWER behavior-
success rate than per-token steering on generations of 128-256 tokens,
temporal drift is real and the prefill-only approach is insufficient.
Status moves to `x disproved` and N9 (closed-loop control) becomes
higher priority.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA; the standard once-at-prefill injection
methodology; does not explicitly address temporal stability but implicitly
assumes it by reporting efficacy on fixed-length completions.

Zou, Andy, et al. 2023 'Representation Engineering: A Top-Down Approach
to AI Transparency' arXiv:2310.01405 — RepEng; measures behavior on
generated completions of 200+ tokens with prefill-only injection; implicit
assumption of stability motivates E48's explicit test.

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; the dual-forward verdict
check (Guard D) runs once at the prefill level; its design implicitly
assumes the verdict is stable for the full generation; E48 validates this
assumption at the generation-dynamics level.

Turner, et al. 2023 'Activation Addition: Steering Language Models Without
Optimization' — ActAdd / the original once-at-prefill implementation;
E48 provides the systematic temporal-stability test that this foundational
methodology implicitly relies on.
```

---

## 5. Mechanism

During autoregressive generation, the model processes each new token t by:

1. Reading the current KV cache (past token representations).
2. Computing h_t (the hidden state at the new token) from the embedding
   of token t and the attended-to past context.
3. Producing the next-token logits from h_t.

When steering is applied prefill-only, the initial injection modifies h_0
(the last prefill token's hidden state). This modified h_0 enters the KV
cache. At subsequent tokens, the model attends to the modified KV cache,
which may propagate the behavior-direction signal forward. The stability
question is: does this KV-cache propagation maintain the behavior direction
sufficiently, or does the direction "dilute" as new, unsteered tokens are
generated?

The dilution mechanism would be: as new tokens t > 0 are generated, their
h_t is computed from the embedding of the current token (which is sampled
from the unsteered distribution) + the attention over past modified hidden
states. If the attention weight on h_0 decreases (as the sequence grows
and later tokens dominate the attention), the behavior direction fades.

The stability argument is that (a) the model's output circuit has
"committed" to the behavior direction by the first generated tokens, and
the generation dynamics are an attractor that maintains the direction; (b)
the KV-cache propagation of the modified h_0 provides sufficient ongoing
signal.

Testing: compare short (32 tokens), medium (128 tokens), and long (256
tokens) generations under prefill-only vs per-token steering; measure
behavior-success rate at each length. Drift would appear as a systematic
efficacy gap that grows with generation length.

---

## 6. Predicted Delta

| Generation length | Prefill-only efficacy (vs per-token) |
|---|---|
| 32 tokens | within 2% (both methods fully effective) |
| 128 tokens | within 5% (hypothesis threshold) |
| 256 tokens | within 5% to 10% |

If the 256-token gap exceeds 10%, the drift is real and a midsequence
re-injection (e.g., at token 128) may be needed. The PPL difference
should also remain within 5% at all lengths.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Behavior: refusal (primary); formality (secondary, for a non-safety
  behavior to test generality).
- Injection conditions: (A) prefill-only (inject at last prefill token,
  layer from prior sweep); (B) per-token (inject at each generated token);
  (C) re-inject at every 32nd token (hybrid, tests whether periodic re-
  injection helps on long generations).
- Generation lengths: 32, 64, 128, 256 tokens.
- Alpha: relative_add at alpha = 0.10.
- Eval: behavior-success rate (LLM-judge on full generation); PPL (on
  the generated text, compared to unsteered baseline); token-level behavior
  projection <h_t, v> at each generated token (internal diagnostic: does
  the projection decay over generation length?).
- Computational cost: report per-token overhead of condition (B) vs (A).
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

If prefill-only is confirmed sufficient, it eliminates the per-token
steering overhead entirely, making steering a zero-cost operation during
autoregressive generation. This is the key efficiency result for deployment.

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E48.
- N9 (closed-loop dynamical control): if drift exists (E48 falsified),
  N9's feedback controller is the solution; E48 determines whether N9 is
  needed.
- E24 (residual + KV-cache composition on long generations): E24 tests
  KV-cache contamination in multi-site steering; E48 tests within-site
  temporal stability of residual-only steering.
- E39 (persona drift monitoring): if per-token behavior drift is real,
  the persona monitor (E39) becomes the detection mechanism for that drift.
- arXiv:2312.06681 (CAA): the primary implementation with implicit
  prefill-only assumption.

---

## 9. Committee Q&A

**Q: Wouldn't per-token steering trivially win because it provides more
"signal" per token? Is the comparison fair?**

> Per-token steering is also more expensive and potentially more disruptive
> (it applies alpha*v to every generated token, increasing cumulative off-
> shell displacement proportional to generation length). The comparison is
> fair in the sense that at matched per-token alpha, per-token steering
> may produce MORE off-manifold displacement and higher PPL for long
> generations. The hypothesis is that prefill-only is SUFFICIENT (within
> 5 pp), not that per-token is worse.

**Q: Does the token-level behavior projection diagnostic require a forward
pass at each token?**

> No; the hook captures the hidden state at the injection layer at each
> token during the single autoregressive forward pass. The projection is
> computed as a post-hoc scalar from the captured h_t and the behavior
> direction v. This adds minimal overhead (one dot product per token).

---

## 10. Verification checklist

- [ ] Three generation lengths (64, 128, 256) tested in the same sweep.
- [ ] Token-level projection <h_t, v> recorded at all generated tokens.
- [ ] Prefill-only vs per-token conditions run at identical random seeds
      and generation settings.
- [ ] PPL computed on the generated text (not on WikiText-103 for this
      experiment — the coherence of the generated text itself is what matters).
- [ ] Computational overhead of per-token injection measured and reported.
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E48.
  Status: `o UNTESTED`. Theoretically motivated by the autoregressive
  generation dynamics and the implicit assumption of temporal stability
  in all prefill-only steering papers. No prior screening run. This is
  a relatively low-cost experiment (no new infrastructure beyond tracking
  per-token projections) and should be run early in the program.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (autoregressive generation dynamics specialist).*

### Prior plausibility
**MEDIUM-HIGH.** The prefill-only assumption is implicitly made by virtually
every steering paper in the literature (they all report behavior on full
completions without per-token reinjection). This implicit assumption has
not been systematically tested, making E48 a gap-filling experiment. The
mechanism (KV-cache propagation of the initial steering) is plausible.

### Mechanism scrutiny
The KV-cache propagation argument is sound but depends on the specific
attention pattern of Gemma-2-2B. If Gemma's attention is highly local
(attending mainly to recent tokens), the h_0 injection may decay quickly
as the generation proceeds and recent tokens dominate. Gemma-2-2B uses
grouped query attention and sliding-window attention in some layers, which
could affect the stability prediction.

### Confounds
1. At 256 tokens, the generation may naturally drift toward a stable
   attractor (the language model's prior for long text), independent of
   the steering injection. This makes it hard to attribute any drift to
   "fading steering" vs "natural generation dynamics." The control is the
   per-token condition, which re-anchors at each step.
2. Greedy vs sampled decoding may affect the stability differently:
   sampled decoding introduces token-to-token randomness that could
   amplify drift; greedy decoding provides a deterministic trajectory.
   Run both.

### Expected effect size
My prior: prefill-only is within 5 pp of per-token for generations up to
128 tokens; at 256 tokens, the gap may grow to 5-15 pp, potentially firing
the falsifier. The hypothesis may be partially confirmed (stable for short
generations, drifts for long ones).

### Verdict
**TESTABLE + DEPLOYMENT-CRITICAL** — A confirmed result eliminates per-
token overhead; a falsified result motivates N9's closed-loop controller.
Either way, E48 is a necessary characterisation of the temporal properties
of activation steering. Recommend running at generation lengths up to 512
tokens to fully characterise the drift curve.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E48.md`.
