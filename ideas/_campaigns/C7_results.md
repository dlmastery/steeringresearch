# Campaign C7 — DiffMean vs PCA source (E36): INCONCLUSIVE + a norm-control finding

Gemma-3-270m @L16, add, α∈{0.5,1,2}, source∈{diffmean, pca}. exp#60–63 (+2). n=1.

| source | α | behavior | PPL | angular | composite |
|---|--:|--:|--:|--:|--:|
| diffmean | 0.5 | 0.485 | 99.1 | 0.0013 | −1.42 |
| diffmean | 1.0 | 0.468 | 120.0 | 0.0054 | −1.76 |
| diffmean | 2.0 | 0.534 | 205.3 | 0.0205 | −2.27 |
| pca | 0.5 | 0.500 | 89.6 | 0.0000 | −1.11 |
| pca | 1.0 | 0.470 | 89.6 | 0.0000 | −1.11 |
| pca | 2.0 | 0.500 | 90.0 | 0.0000 | −1.11 |

## Verdict: E36 INCONCLUSIVE at matched-α; a real norm-control finding

PCA registers **angular≈0 and flat PPL** — it is barely steering. Cause: DiffMean
is applied **raw** (its norm is large, ~order 10), while PCA-top1 is a **unit**
vector (SVD). So at the same raw α the PCA edit is ~10× smaller in magnitude — the
comparison is matched-α, NOT matched-displacement, so it cannot answer E36.

**Finding (real, affects the harness):** for α to be a meaningful, source-comparable
coefficient, the steering vector should be **unit-normalized** before applying α
(then α = displacement in units of ‖h‖-ish, the E7 "relative steering" idea). Raw
DiffMean conflates direction with magnitude. Queued fix: a `--normalize` option +
C7b re-run with both sources unit-normalized.

**Important scope note:** this does NOT affect the headline geometry results. N17/N5
(off-shell Δ‖h‖ → log PPL, R²=0.81) and N16 (angular → rotation PPL, R²=0.997) use
the **measured displacement**, not α, so they are unaffected by the raw-vs-unit α
parameterization. The E3 cliffs are all DiffMean-internal (self-consistent). Only
*cross-source* α comparisons (E36) need the normalization fix.
