# FINDING — Norm-preserving rotation is NOT more coherent than addition at matched budget

**Status:** SCREENING tier. Judge-free (unaffected by the H0 judge failure).
**Date:** 2026-07-25 · **Artifacts:** `autoresearch_results/matched_budget.json`,
`scripts/run_matched_budget.py` · **Pre-registered** in the script header before the run.

---

## The gap this fills

`2601.19375` (*Selective Steering: Norm-Preserving Control Through Discriminative Layer
Selection*, Dang & Ngo, Jan 2026) ships norm-preserving rotation and dismisses additive
steering as causing "catastrophic degradation on smaller models" — but it compares three
**rotation** methods to each other and **never holds displacement fixed**. The literature
scan (`corpus/LIT_2026-07_geometry.md`) confirms no matched-budget equivalence test
exists. The field's working assumption — *norm preservation buys coherence* — is
therefore untested.

## The controlled comparison

Both operations parameterised by the same chord displacement `f = ‖Δh‖/‖h‖`:

| operation | parameter | chord | norm |
|---|---|---|---|
| `relative_add` | α = f | f·‖h‖ | **changes** (goes off-sphere) |
| `rotate` | θ = 2·arcsin(f/2) | f·‖h‖ | **exactly preserved** |

Equal distance travelled, different path. Endpoint: **WikiText perplexity — no LLM judge**.

## Result

Model: `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated` (4-bit), layer 12,
40 WikiText passages.

| budget f | add PPL | rotate PPL | rotate − add |
|---|---|---|---|
| 0.00 (control) | 92.326 | 92.293 | −0.033 |
| 0.02 | 88.744 | 89.542 | +0.798 |
| 0.05 | 87.652 | 117.373 | +29.72 |
| 0.10 | 117.983 | 479.179 | +361.2 |
| 0.15 | 198.658 | 4,071.64 | +3,873.0 |
| 0.20 | 375.550 | 119,316.92 | +118,941.4 |

**Rotation is worse at 5/5 budgets**, and the penalty grows super-linearly.

### Pre-registered falsifier: TRIPPED

The field's assumption predicts `PPL(rotate) < PPL(add)` at matched f. Observed: rotation
is never better. **"Norm preservation buys coherence" is not supported at this scale.**

## Interpretation — a re-attribution, not just a negative

Norm preservation is not protective **because the conserved quantity is the wrong one.**
At matched chord, rotation redirects the *entire* residual vector toward `v` (as θ grows,
`h` is increasingly dominated by its `v`-orthogonal component `e₂`), whereas addition
perturbs `h` while leaving its original direction largely intact. Same distance
travelled, very different amount of *information destroyed*.

This says the coherence tax is priced by **angular** displacement, not radial — and that
holding the norm fixed while spending the whole budget on angle is the *worst* way to
spend it. That is a concrete re-attribution of the norm-vs-angle decomposition asserted
(but not measured as a controller) in `2606.06735`.

### Secondary observation

At small budgets (f = 0.02, 0.05) additive steering **lowers** perplexity below the
unsteered baseline (88.7 and 87.7 vs 92.3). Not investigated here; recorded so it is not
silently dropped.

## Honest limitations — this is SCREENING, not an evaluation claim

1. **n = 1 per cell.** No seeds, no CI, no significance test. Directional only.
2. **The direction is weakly estimated** — diff-of-means from **10 harmful / 8 harmless**
   prompts. This is far below the project's own data floor and is the single biggest
   threat to the result.
3. **One model, one layer (12), one corpus.** No cross-scale or cross-layer check.
4. Perplexity is a coherence proxy, not a capability measure.
5. The abliterated model is a specific, unusual substrate.

**None of these undermine the *direction* of the effect** — a 300× perplexity gap at
f = 0.20 is not a seed artifact — but the magnitude must not be quoted as an evaluation
result until re-run at n ≥ 8 with a properly estimated direction across ≥2 layers.

## Next (pre-registered)

- **M-a** Re-run at n ≥ 8 extraction resamples, ≥2 layers, with bootstrap CIs on the
  paired PPL delta; check the power contract (`power_note(..., family_size)`).
- **M-b** Decompose the budget into radial and angular components and measure the tax of
  each separately — the actual attribution claim.
- **M-c** Repeat on Gemma-3-4B abliterated for a cross-scale check.

## Reproduce

```bash
PYTHONPATH=src python scripts/run_matched_budget.py --layer 12 --n-ppl 40
```
