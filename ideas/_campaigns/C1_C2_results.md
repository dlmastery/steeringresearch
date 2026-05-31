# Overnight campaigns C1 + C2 — results (SCREENING, n=1 per cell)

Model: Gemma-3-270m-it (real). Behavior = synthetic-lexicon proxy (weak);
PPL / off-shell Δ‖h‖ are the real signals. Composite fingerprint a9001e87087e.

## C1 / E2 — is max-Fisher the best injection layer? (FALSIFIED, screening)

Layer sweep @α=2, add, on Gemma-3-270m (exp#20–27):

| layer | Fisher | behavior | PPL | off-shell | composite |
|------:|------:|------:|------:|------:|------:|
| 2 | 0.20 | 0.395 | 133.6 | 0.012 | −1.456 |
| 4 | 10.73 | 0.363 | 195.9 | 0.013 | −1.633 |
| 6 | 2.05 | 0.351 | 245.4 | 0.000 | −1.767 |
| 8 | 0.29 | 0.310 | 222.2 | 0.006 | −1.680 |
| 10 | 1.27 | 0.485 | 252.2 | 0.003 | −2.471 |
| 12 | **30.57** | 0.319 | 322.3 | 0.057 | −2.889 |
| 14 | 17.23 | 0.478 | 253.2 | 0.023 | −2.288 |
| 16 | 18.25 | **0.534** | 205.3 | 0.038 | −2.271 |

- **best BEHAVIOR layer = L16** (0.534, and lower PPL than L12); **max-Fisher = L12**.
- **Spearman(Fisher, behavior) = +0.143 (p=0.74)** — E2 predicted ≥0.7. **E2 is
  FALSIFIED on this setup**: linear separability (Fisher) does NOT predict the
  best steering layer on a small model. (Consistent with N8/E37:
  controllability ≠ interpretability/separability.)
- **Actionable**: the earlier E3 cliff ran at max-Fisher L12 — a *suboptimal*
  layer. L16 gives more behavior at lower PPL. Future Gemma steering → L16.

*Caveat: behavior is the synthetic proxy, n=1, single concept. But the low
Spearman holds regardless of proxy magnitude, and the PPL/off-shell columns are
real.*

## C2 / N17 + N5 — does off-shell displacement govern incoherence? (SUPPORTED)

Pooled over **23 real steered rows** (Gemma-3-270m + Qwen-0.5B; 8 layers; α∈{1..24}):

- **N17**: Spearman(off-shell Δ‖h‖, log PPL) = **+0.705** (p=1.7e-4);
  Pearson = **+0.899**. The cheap, behavior-free geometry probe predicts
  incoherence.
- **N5 data-collapse**: a single linear law `log PPL = 5.40 + 2.87·Δ‖h‖`
  fits the pooled data with **R² = 0.809** — across two architectures, eight
  layers, and the full α range. Coherence collapse is governed by off-shell
  DISPLACEMENT, not raw α (supports the norm-budget conservation hypothesis N5).

This is the strongest screening result so far: an architecture- and
layer-independent geometric predictor of the coherence cliff. Promotion to an
external claim still requires n≥7 + real perplexity corpus (WikiText) + a
prompting baseline; but the relationship is robust within the screening data.
