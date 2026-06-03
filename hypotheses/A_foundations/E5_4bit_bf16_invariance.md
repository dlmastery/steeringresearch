# E5 — 4-bit vs bf16 Steering Invariance

> **One-line claim:** Steering efficacy is invariant (within noise) to 4-bit NF4
> quantization vs bf16 full precision, so 4090-scale results on quantized models
> transfer to full-precision deployment.
>
> **Source design space:** Block A — Foundations and measurement tooling (E5).
> **Primary axis:** A7 (HOW DERIVED — extraction precision / quantization format).
> **Implementation status:** `o UNTESTED (bitsandbytes not installed at time of screening)`.

---

## 1. Motivation (>=100 words)

The practical case for quantization in this research program is overwhelming: Gemma-2-2B
in bf16 requires ~4.5 GB VRAM, while 4-bit NF4 (bitsandbytes) fits in ~2.5 GB —
a 45% VRAM reduction that enables larger batch sizes, faster iteration, and running
two models simultaneously. Virtually all steering papers that target consumer hardware
use quantized models (4-bit bitsandbytes or GPTQ). But quantization perturbs the
weight matrices in a non-uniform, input-dependent way: NF4 quantization introduces
quantization noise proportional to the range of each weight block, which in principle
could alter the geometry of the activation manifold, change the direction of behavioral
concept encodings, and therefore shift the optimal steering vector. If the direction
v_4bit differs significantly from v_bf16, all findings from the 4090 program must carry
a "quantized model only" qualifier and cannot be presented as claims about the base
model. Conversely, if v_4bit ~= v_bf16 in cosine and efficacy (within noise), the
quantization gap is absorbed by the existing noise floor and the 4090 results are
valid proxies for full-precision deployment. This matters for the AxBench leaderboard
(arXiv:2501.17148) where reported results are on bf16 or float32 models. E5 closes
this gap by direct measurement. The interaction with E4 is also important: the
DiffMean ~= PCA cosine (>0.99) found in screening is robust across two architectures —
is it equally robust across quantization formats?

---

## 2. Formal hypothesis (>=50 words, falsifiable)

On Gemma-2-2B-it, the cosine similarity between the DiffMean steering vector extracted
at bf16 precision and the DiffMean vector extracted at 4-bit NF4 precision exceeds 0.95
at the same layer (L16) for the same contrast set. Furthermore, steering at matched
alpha with the bf16 vector (applied to a bf16 model) vs. the NF4 vector (applied to
a NF4 model) produces behavior success rates that agree within 3% relative error and
PPL values that agree within 10% relative error. The 3% behavior threshold is
pre-registered as the "within noise" criterion for this research program.

---

## 3. Falsifier (>=30 words)

**Fired if:** cos(v_4bit, v_bf16) < 0.95 at L16 for any tested behavioral concept, OR
if the behavior success rate differs by more than 3% relative between matched-alpha
4-bit and bf16 steering runs, OR if the PPL gap exceeds 10% relative. Any of these
would require a quantization-format correction factor on all steering results from
the 4090 program.

---

## 4. Citations (>=80 words, Citation Rigor format)

```
Zou, Andy, et al. 2025 arXiv 'AxBench: Steering LLMs? Benchmarks Matter!'
(arXiv:2501.17148) — the primary evaluation benchmark uses full-precision
models; E5 determines whether our 4-bit results can be compared to their
reported numbers.

Dettmers, Tim, et al. 2022 NeurIPS 'LLM.int8(): 8-bit Matrix Multiplication
for Transformers at Scale' (no arXiv, NeurIPS proceedings) — introduces the
bitsandbytes quantization framework used in this project; documents that
quantization noise is bounded but non-zero; the question is whether it is
above or below the steering-efficacy noise floor.

Panickssery, Aryan, et al. 2023 arXiv 'Steering Llama 2 via Contrastive
Activation Addition' (arXiv:2312.06681) — uses quantized models without
ablating quantization; the implicit assumption E5 tests.

Venkatesh, T. G. and Kurapath, A. 2026 ICLR workshop 'On the Non-
Identifiability of Steering Vectors' (arXiv:2602.06801) — large equivalence
classes of steering directions produce equivalent behavior; if 4-bit and bf16
vectors lie in the same class (high cosine), E5 is confirmed by the
non-identifiability theory.
```

---

## 5. Mechanism (deep technical)

NF4 quantization applies a block-wise quantization of weight matrices with block size
64 (bitsandbytes default): each 64-element block of a weight row is scaled by its
absolute maximum, mapped to 16 representable NF4 values, and stored as 4 bits. The
quantization error per element is bounded by (W_max - W_min) / 32 (one-half step)
within the block.

The activation h at layer L is h_L = h_{L-1} + delta_L, where delta_L = Attn(h_{L-1})
+ MLP(h_{L-1}). Under quantization, delta_L^{4bit} = delta_L^{bf16} + epsilon_L,
where epsilon_L is the quantization noise introduced by the weight approximation in
all linear layers up to and including layer L. Because NF4 is a non-uniform
quantization optimized for normally distributed weights, epsilon_L is small for
Gaussian weights (which transformer weights approximately are post-training).

The direction of the concept in activation space (v = mu_pos - mu_neg) is an
average over many activations. Quantization noise epsilon_L is additive per-token
noise with mean approximately zero (NF4 is designed to be unbiased in expectation
for the trained weight distribution). Therefore:

    v_4bit = E[h_pos^{4bit}] - E[h_neg^{4bit}]
            = E[h_pos^{bf16} + epsilon_pos] - E[h_neg^{bf16} + epsilon_neg]
            = v_bf16 + (E[epsilon_pos] - E[epsilon_neg])

If the quantization noise is uncorrelated with the concept (i.e., epsilon does not
systematically differ between harmful and harmless prompts on the same layer), then
E[epsilon_pos] ~= E[epsilon_neg] and v_4bit ~= v_bf16. This is the "noise cancels
in the mean" argument for why E5 should hold — it is an instance of the same logic
as E4 (why PCA and DiffMean align: noise cancels in the mean).

The key risk: if quantization noise IS correlated with the concept — e.g., if harmful
prompts activate different attention heads with different weight magnitudes (hence
different quantization errors) — the cancellation fails and v_4bit and v_bf16 diverge.
This is the mechanism by which E5 could be falsified.

Behaviorally, the steering efficacy gap between 4-bit and bf16 is further attenuated
by the off-manifold budget (N5): even if v_4bit has a slightly different direction,
the manifold the 4-bit model has learned is itself slightly different from bf16, so
the "right" vector for the 4-bit model IS v_4bit, not v_bf16. The correct comparison
is: vector from 4-bit model steers 4-bit model vs. vector from bf16 model steers
bf16 model — both at matched relative alpha (E7). A transfer test (4-bit vector in
bf16 model) is a secondary ablation.

---

## 6. Predicted Delta (pre-registered numbers)

| Metric | Predicted value | Observed |
|--------|----------------|---------|
| cos(v_4bit, v_bf16) at L16 | >= 0.95 | **UNTESTED** |
| Behavior gap 4-bit vs bf16 (matched relative alpha) | < 3% relative | **UNTESTED** |
| PPL gap 4-bit vs bf16 | < 10% relative | **UNTESTED** |
| Cross-format transfer (4-bit vector in bf16 model) | behavior > 80% of matched-format | **UNTESTED** |

---

## 7. Experimental protocol

### 7.1 Primary experiment

- **Model pair:** Gemma-2-2B-it in bf16 (requires ~4.5 GB VRAM) vs. 4-bit NF4
  (bitsandbytes bnb_4bit_quant_type="nf4", ~2.5 GB).
- **Layer:** L16 (pre-registered best from C1/E2 on 270m; verify on 2B-it first).
- **Concepts:** same 3 concepts as E3/E4 primary experiments.
- **Metrics:** cos(v_4bit, v_bf16) per layer × concept; behavior success (LLM-judge)
  at matched relative alpha; PPL on WikiText; MMLU on >=100 items.
- **Seeds:** n=7 for each format (n=14 total).
- **Prerequisite:** bitsandbytes must be installed and verified (`import bitsandbytes`
  runs without error; a UNIT-level sanity check that the 4-bit model loads and produces
  outputs is required before this experiment launches).

### 7.2 Where this should SHINE

Safety concepts, where the direction is strongly linear and the quantization noise
is not systematically correlated with the concept. If E5 holds there, it extends
straightforwardly to other concept types.

### 7.3 Risk mitigation for bitsandbytes

The C5/C6/C9 screening used float16/bfloat16 models because bitsandbytes was not
installed. Installing bitsandbytes on Windows requires a pre-built wheel (the official
pip package supports Linux primarily). A Windows-compatible build via
`pip install bitsandbytes --prefer-binary` or a WSL2 fallback is the recommended
approach. Log the exact version and wheel provenance in the verification artifacts.

---

## 8. Cross-references

- **E4** (DiffMean vs PCA): the same "noise cancels in the mean" argument applies here.
- **E7** (norm-relative alpha): both 4-bit and bf16 experiments must use matched
  fractional displacement, not matched absolute alpha (||h|| differs between formats).
- **N5** (norm-budget): the norm-budget law log PPL = 5.40 + 2.87*Δ‖h‖ was measured
  on unquantized models; E5 tests whether it transfers to the 4-bit setting.
- **AxBench** (arXiv:2501.17148): all leaderboard comparisons assume bf16; E5 provides
  the calibration factor for our 4-bit results.
- **IDEA_TABLE.md** Block A row E5.

---

## 9. Committee Q&A

**Q: Why not just run everything in bf16 and skip quantization?**

> VRAM budget: Gemma-2-2B-it in bf16 requires ~4.5 GB; with activations cached for
> 200 contrast pairs at L16, peak VRAM exceeds 8 GB. Running the full sweep
> (layer × alpha × behavior) in bf16 would saturate the 16 GB laptop GPU with no
> headroom for the generation step. The 4-bit format is not a compromise — it is
> the hardware constraint.

**Q: What if E5 fails? How bad is that?**

> All screening results from S-1 to S-9 used unquantized models (no bitsandbytes).
> If E5 fails (v_4bit != v_bf16 by >5% cosine), the gap would need to be estimated
> per concept and reported as a correction factor. The findings would still be valid
> for quantized deployment specifically; the claims about "what the model knows"
> (the representational structure) would still hold — the difference is a format
> artifact, not a representational one.

**Q: Isn't this a solved problem? Papers report similar results on quantized and
full-precision models all the time.**

> "Similar results" is informal. E5 formalizes what "similar" means (3% behavior,
> 10% PPL, 0.95 cos) and tests it on steering specifically. Steering is more sensitive
> to the precise direction of the intervention than most downstream tasks (which
> average over many heads and layers). A 5% shift in activation geometry could matter
> more for steering than for, say, text classification.

---

## 10. Verification artifacts checklist

- [ ] bitsandbytes installed and UNIT-level test passes (model loads, outputs non-trivial text).
- [ ] cos(v_4bit, v_bf16) per layer and concept logged.
- [ ] Behavior + PPL comparison table (matched relative alpha, n=7 each format).
- [ ] Cross-format transfer ablation (4-bit vector in bf16 model).
- [ ] Result row in `EXPERIMENT_LEDGER.md` (Rung 1 minimum).
- [ ] N5 law (log PPL = 5.40 + 2.87*Δ‖h‖) re-fit for 4-bit model.

---

## 11. Status journal

- 2026-05-27 — hypothesis inherited from corpus E5; 3% behavior threshold pre-registered.
- 2026-05-29 — C5/C6/C9 screening ran without bitsandbytes (Windows install failure).
  **E5 UNTESTED.** All screening to date on float16 / bfloat16 unquantized models.
- 2026-05-31 — Design doc written. Status: **UNTESTED**. Priority: install bitsandbytes,
  then run alongside E3/E4 primary experiments as a calibration check.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-A. This is an important calibration experiment that is genuinely
untested — the critique is on the predicted outcome and experimental design.*

### Prior plausibility

**HIGH for the hypothesis holding; MEDIUM for the specific thresholds.** The "noise
cancels in the mean" argument is sound for DiffMean extraction. The 3% behavior
threshold is somewhat loose — even a 10% gap might be acceptable in practice. The
0.95 cosine threshold is reasonable but untested for quantized vs. full-precision
activations specifically.

### Mechanism scrutiny

The quantization-noise-cancellation argument is valid for NF4 quantization of
normally distributed weights. It is less robust if the concept direction is sensitive
to a specific attention head whose weights have high quantization error (unusual
weight magnitude). In practice, NF4 is calibrated to minimize such worst-case errors,
so the mechanism is plausible.

### Confounds

1. **Windows bitsandbytes**: the exact quantization behavior may differ on Windows vs
   Linux due to CUDA kernel differences in the bitsandbytes wheel. Document the wheel
   provenance.
2. **Gemma-3 vs Gemma-2**: screening was on Gemma-3-270m/1b (no bitsandbytes); the
   primary experiment is Gemma-2-2B. The architecture differs (gating in MLP, etc.),
   which could change the quantization noise distribution.
3. **NF4 vs GPTQ**: E5 pre-registers NF4 (bitsandbytes). GPTQ quantization has
   different error characteristics and should be a separate ablation if needed for
   AxBench comparison.

### Does the 3% threshold specifically matter?

**Yes, for AxBench calibration.** The AxBench benchmark reports efficacy on bf16
models. If our 4-bit results have a systematic 3% downward bias, we should report
that correction factor. The threshold is the minimum gap that would require a
correction in the paper.

### Literature precedent

No steering paper directly measures the 4-bit vs bf16 vector cosine or behavior gap.
The closest: CAA (arXiv:2312.06681) Appendix reports similar layer-sweep results on
both quantized and unquantized models qualitatively. This is insufficient for the
AxBench calibration claim. E5 fills a genuine gap.

### Skeptical effect-size re-prediction

For NF4 on Gemma-class weights: cos(v_4bit, v_bf16) in [0.97, 0.999], 80% CI.
Behavior gap: < 2% relative, 80% CI. The hypothesis is likely to hold — the main
risk is a pathological concept or layer where quantization noise happens to correlate
with the concept direction.

### Minimum-distinguishing experiment

One concept × L16 × {n=3 4-bit, n=3 bf16} × behavior + PPL. ~1 hour once
bitsandbytes is installed. If cos > 0.97 and behavior gap < 2%, done.

### Verdict

**NOVEL+TESTABLE (calibration experiment). Expected to pass.** The primary blocker
is the Windows bitsandbytes installation. Once unblocked, this is a fast experiment
that provides a necessary calibration for all results. The risk of falsification is
low but real. Priority: unblock bitsandbytes, then run as part of the E3/E4 primary
evaluation.

---

## Pseudocode & Methodology

This hypothesis tests **4-bit NF4 vs bf16 invariance** of the steering vector. The
knob varied is the **quantization precision** of the model the vector is extracted
from (and applied to). Source = DiffMean, layer L16, matched relative alpha.

### 1. Steering-vector recipe

```python
# Extract the SAME DiffMean direction from each precision (METHODOLOGY §1.3).
# The cache key includes model id + quantization (collect_activations_cached).
for quant in {"nf4", "bf16"}:
    model_q = load_model(model_id, quant=quant)          # model.py
    H_q     = collect_activations(model_q, tok, load_concept(behavior), layer=16)
    v_q     = diffmean_vector(H_q.pos, H_q.neg)           # mean(pos)-mean(neg)
cos_q = cosine(v_nf4, v_bf16)                             # PRIMARY direction metric
```

### 2. Experiment procedure

```text
1. Extract v_nf4 (from 4-bit model) and v_bf16 (from bf16 model) at L16, same pairs.
2. Record cos(v_nf4, v_bf16).
3. Matched-precision steering (each vector in its own model), matched relative alpha:
4.   h' = apply_operation(h, v_q/||v_q||, "relative_add", alpha_rel=0.10)
5.   record behavior (judge), PPL (WikiText), MMLU (>=100 items).
6. Compare behavior gap and PPL gap (nf4 vs bf16).
7. SECONDARY ablation: cross-format transfer (4-bit vector injected into bf16 model).
8. Re-fit the N5 law (log PPL = 5.40 + 2.87*offshell) on the 4-bit model.
```

### 3. Measurement & decision rule

PRIMARY metric: `cos(v_4bit, v_bf16)` at L16. FALSIFIER (§3): fires if `cos < 0.95`,
OR behavior gap `> 3%` relative, OR PPL gap `> 10%` relative. Pre-registered (§6):
`cos >= 0.95`; behavior gap `< 3%`; PPL gap `< 10%`; cross-format transfer behavior
`> 80%` of matched-format. Verdict: SUPPORTED iff all three thresholds hold (then 4090
4-bit results are valid bf16 proxies); otherwise a per-concept correction factor is
reported. Mechanism: NF4 noise is ~zero-mean over the trained weight distribution, so
`v_4bit ≈ v_bf16 + (E[ε_pos]-E[ε_neg]) ≈ v_bf16` (noise cancels in the mean — the
same logic as E4).

### 4. Where the code is / status

Driver: `scripts/campaign_sweep.py` with two model loads (4-bit + bf16 via `model.py`),
reusing `extract.collect_activations_cached`, `eval.py`, and `geometry.py`. Status
**UNTESTED** — and the specific blocker is **infrastructure**: bitsandbytes was not
installed at screening time (Windows wheel failure), so all S-1…S-9 runs used
unquantized fp16/bf16. The experiment needs (a) a working bitsandbytes NF4 install +
UNIT-level load check, then (b) the paired-precision sweep. No algorithmic machinery is
missing; only the quantized model backend.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E5.md`.
