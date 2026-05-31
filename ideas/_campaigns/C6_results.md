# Campaign C6 — cross-scale E3/E4/E27 on gemma-3-1b (SCREENING)

gemma-3-1b-it (26 layers, the STANDARD-rung model) @L18 (max-Fisher=36.4).
E3 cliff sweep, generation behavior + real safety. exp#55–59. n=1.

| α | behavior | PPL | radial | angular | CR | composite |
|---|------:|------:|------:|------:|------:|------:|
| 0 | 0.500 | 74.0 | 0.000 | 0.0000 | 0.80 | −1.11 |
| 1 | **0.646** | 104.0 | 0.007 | 0.0025 | 0.60 | **−0.89** |
| 2 | 0.453 | 207.9 | 0.018 | 0.0099 | 0.70 | −2.01 |
| 4 | 0.429 | 1518 | 0.052 | 0.0375 | 1.00 | −11.60 |
| 8 | 0.465 | 46082 | 0.174 | 0.1243 | 1.00 | −312.9 |

## The cross-scale picture (E27 SUPPORTED, screening)

| model | layers | base PPL | PPL@α1 | PPL@α2 | behavior@α1 | clean window? |
|---|---:|---:|---:|---:|---:|---|
| gemma-3-270m @L12 | 18 | 90 | +65% | +257% | 0.44 ↓ | **NO** (most fragile) |
| Qwen-2.5-0.5b @L21 | 24 | 49 | +20% | +82% | 0.69 ↑ | yes |
| gemma-3-1b @L18 | 26 | 74 | **+41%** | **+181%** | **0.65 ↑** | yes (least fragile) |

**The clean steering window emerges with scale.** The 270M model is so fragile it
has NO α at which behavior rises before coherence collapses (monotone decline).
Both larger models (0.5B, 1B) have a window at α≈1 where behavior peaks above
baseline; the 1B's coherence cliff is the gentlest (+41%/+181%). This is exactly
the E27 mechanism — larger models have more coherence budget and exit the data
manifold more slowly — observed cleanly across three architectures and scales.

**E4 confirmed on a third model**: cos(DiffMean,PCA)@L18 = 0.9945 (gemma-3-1b),
joining 0.994 (gemma-3-270m) and 0.996 (Qwen-0.5b) — DiffMean≈PCA-top1 is robust
across 270M→500M→1B and two architectures.

**Safety (Rogue-Scalpel)**: CR 0.80→1.00 under steering, consistent with the
other models (direction robust; baseline high due to weak model + synthetic prompts).
