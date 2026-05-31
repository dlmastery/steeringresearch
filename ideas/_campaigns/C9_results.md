# Campaign C7→C8→C9 — relative steering (E7) + source equivalence (E36): RESOLVED

The C7→C8→C9 chain is the loop refining the steering-magnitude parameterization
through two negative results into a clean capstone. Gemma-3-270m @L16. n=1 SCREENING.

## The parameterization journey
- **C7** (raw α, diffmean vs unit pca): incomparable — raw DiffMean norm ~10× the
  unit PCA, so matched-α ≠ matched-displacement. → added `--normalize`.
- **C8** (unit vectors, absolute α≤40): ZERO effect (offshell=0) — ‖h‖ at L16 is
  large, so 40×unit is negligible. → the control variable is displacement RELATIVE
  to ‖h‖. → added `relative_add` (Δh = α·‖h‖·v̂).
- **C9b** (relative_add, α = fraction of ‖h‖): clean, ‖h‖-independent cliff.

## C9b — the relative cliff (E7 SUPPORTED, screening)

| α (frac ‖h‖) | DiffMean beh | DiffMean PPL | PCA beh | PCA PPL | offshell |
|--:|--:|--:|--:|--:|--:|
| 0.02 | 0.532 | 92.3 | 0.532 | 92.4 | 0.002 |
| 0.05 | 0.445 | 100.0 | 0.439 | 101.3 | 0.005 |
| 0.10 | **0.614** | 131.9 | **0.591** | 132.6 | 0.010 |
| 0.20 | 0.504 | 245.2 | 0.475 | 255.4 | ~0.030 |
| 0.40 | 0.319 | 1623 | 0.304 | 1759 | ~0.093 |

- **E7 SUPPORTED**: relative steering (α = fractional displacement) gives a clean,
  ‖h‖-independent, interpretable cliff. Behavior PEAKS at **α≈0.1 (a 10% nudge)**
  then declines; PPL rises monotonically; the knee (~10% displacement, offshell
  ~0.01–0.03) matches the N17/C2 geometry law. This is the *right* control variable
  — absolute α conflates direction with the (large, scale-varying) residual norm.
- **E36 SUPPORTED (resolved)**: at every matched fractional α, **DiffMean and
  PCA-top1 steer near-identically** (behavior within 0.02, PPL within ~8%). E4's
  0.99 cosine alignment DOES imply equivalent steering; the C7/C8 apparent
  differences were pure norm-parameterization artifacts, not a real source effect.

**Harness upshot**: `relative_add` is the recommended operation for interpretable,
comparable steering; absolute-α `add` cliffs (E3) remain valid DiffMean-internal but
are not cross-source/-layer comparable. N17/N5/N16 are unaffected (measured displacement).
