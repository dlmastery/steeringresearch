# AUDIT — contextual_steering

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2604.24693 — VERIFIED (resolves).** |
| actual title | *Contextual Linear Activation Steering of Language Models* — README's short title matches (drops the "of Language Models" suffix; fine). |
| actual authors | Brandon Hsu, Daniel Beaglehole, Adityanarayanan Radhakrishnan, Mikhail Belkin. |
| venue / dates | arXiv cs.CL; submitted 2026-04-27. Method name: **CLAS**. |
| method in abstract | Yes — context-dependent (per-token) steering strengths replacing a fixed scalar; evaluated on 11 benchmarks / 4 model families; matches or beats ReFT and LoRA in low-label settings. |
| verification | WebFetch of the HTML + WebSearch confirmed title/authors/id and the CLAS mechanism. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Paper exists & id resolves | **PASS** | id 2604.24693 resolves; title/authors confirmed. |
| High-level idea fidelity — per-input adaptive strength | **PASS** | `contextual.py` reads a per-prompt signal `proj = cos(mean_pool(h@layer), v_unit)` and scales `alpha` by it, dropping lesson 2's separate probe gate. This matches the paper's core contribution: make the steering coefficient a function of the input activation rather than a fixed scalar. |
| Specific formula attribution | **FAIL** | README l.117 and `contextual.py` l.20–24 claim the schedule "is **exactly** the paper's `relu((proj − tau)/(1 − tau))`." It is not. Per the paper body, CLAS computes `α_{ℓ,t} = c_ℓ · [h_{ℓ,t} 1]` — a **learned sensing vector** `c` (trained by backprop, min next-token loss, zero-init), a linear/affine read with **no relu, no tau/ref thresholding, no cosine-onto-v**. The WebFetch of the paper explicitly notes CLAS "uses a simple linear projection without normalization or thresholding mechanisms." |
| Claim accuracy | **CONCERN** | The lesson's relu/tau/ref cosine ramp is a reasonable *alternative* realization (untrained, uses the steering direction itself as the sensing direction), but attributing that exact formula to the paper is incorrect. Recommend: keep the citation for the **idea** (per-input adaptive strength — genuinely CLAS's contribution), relabel the specific ramp as "our own construction inspired by CLAS," and delete the "exactly the paper's" sentence and the `[UNVERIFIED]` tag. |
| Implementation correctness | **PASS** | `cosine_projection`, `contextual_alpha` (relu ramp, cap, degenerate `ref≤tau` hard-step), and `calibrate_schedule` (tau = 90th-pct benign, ref = mean harmful, extract-split only) are pure, unit-tested, and internally consistent. Extract/eval kept disjoint. |
| Results honesty (code-only) | **PASS** | Results table marked "pending the GPU run"; screening-tier disclosed (n=25/class, one seed, 1B model + 3B judge); off-family `Qwen2.5-3B-Instruct` judge documented; adversarial-low-projection and mean-pool/position confounds flagged in caveats. |

## Overall verdict

**CONDITIONAL PASS — one required fix.** The paper is real and the lesson's
high-level idea (input-adaptive steering strength) faithfully reflects CLAS. But
the lesson **misattributes a specific formula** to the paper: CLAS uses a learned
sensing vector `c·[h,1]`, not the relu/tau/ref cosine ramp implemented here.
Correct the "exactly the paper's formula" claim to "inspired by CLAS; our own
untrained schedule," and drop `[UNVERIFIED]`. No code bug found.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
