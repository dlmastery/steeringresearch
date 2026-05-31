# Campaign C3 — operation comparison (E27): INCONCLUSIVE + a real method finding

Model: Gemma-3-270m @L16 (best layer from C1). add/rotate/project_out, α∈{1,2,4}.
exp#28–36. n=1/cell, SCREENING.

| α | op | behavior | PPL | off-shell Δ‖h‖ | composite |
|---|----|------:|------:|------:|------:|
| 1 | add | 0.468 | 120.0 | 0.013 | −1.76 |
| 1 | project_out | 0.271 | 96.2 | 0.003 | −1.37 |
| 1 | rotate | 0.197 | 3.9e10 | 0.001 | −2.1e8 |
| 2 | add | 0.534 | 205.3 | 0.038 | −2.27 |
| 2 | project_out | 0.197 | 110.9 | 0.001 | −1.78 |
| 2 | rotate | 0.197 | 1.1e17 | 0.000 | −6e14 |
| 4 | add | 0.296 | 1139 | 0.110 | −7.76 |
| 4 | rotate | 0.197 | 2.2e18 | 0.000 | −1e16 |

## Verdict: E27 NOT cleanly testable here (INCONCLUSIVE), but two findings

1. **α semantics are incommensurable across operations.** `rotate` interprets α
   as a rotation ANGLE in radians, so α=1–4 rad (57–229°) catastrophically
   scrambles every token's hidden state → PPL 1e10–1e18 and a degenerate constant
   output (behavior pinned at 0.197 for all rotate cells). A fair add-vs-rotate
   test must match by DISPLACEMENT or angle, not raw α, with small rotation angles.
   → Follow-up **C3b**: rotate at α∈{0.05,0.1,0.2,0.3,0.5} rad.

2. **Off-shell Δ‖h‖ is a RADIAL-only predictor (refines N17).** Rotation moves h a
   huge ANGULAR distance while preserving norm, so Δ‖h‖≈0 even as PPL explodes —
   the off-shell (norm-change) metric completely misses rotation's damage. This is
   exactly the Cylindrical Representation Hypothesis (corpus CRH / N16): activation
   change = radial × angular, and a coherence predictor needs BOTH components. N17's
   Δ‖h‖ law (C2, R²=0.81) holds for ADDITIVE steering (which changes norm) but is
   blind to norm-preserving operations. **Action: add an angular-displacement
   metric (cos angle between steered/unsteered h) to geometry.py** and re-test N17
   with the combined radial+angular displacement.

3. **project_out** is the gentlest op (lowest PPL: 96–179) but also lowest behavior
   — it removes the concept component rather than adding it; expected.

This is a productive negative result: the loop surfaced (a) an operation-parameter
incommensurability and (b) an incompleteness in the N17 geometry predictor that the
corpus's CRH anticipated. Both are queued as concrete fixes.
