---
name: steering-paper-rigor
description: >
  Use when evaluating whether a steering claim meets the rigor bar for
  external presentation, grading a hypothesis verdict tier, or auditing
  pre-registration completeness. Enforces the CLAUDE.md Section 7 four-part
  statistical contract. This is the steering instantiation of the meta
  autoresearch-paper-rigor process.
---

# Skill — steering-paper-rigor

This is the steering instantiation of [meta-skills/autoresearch-paper-rigor/SKILL.md](../../meta-skills/autoresearch-paper-rigor/SKILL.md).

Read that file first for the general rigor framework. This file adds the
steering-specific verdict tiers and the corpus discipline for inherited numbers.

---

## Steering-specific notes

### The four-part statistical contract (verbatim from CLAUDE.md Section 7)

A claim using "winner" / "beats baseline" / "outside seed noise" /
"statistically significant" requires ALL FOUR simultaneously:
1. Paired Wilcoxon signed-rank test on the delta (p < 0.05 after correction).
2. 95% bootstrap CI (>= 10,000 resamples) on the delta excludes zero.
3. Holm-Bonferroni correction across the sweep family.
4. Empirically-derived per-model noise band (2*sigma_seed, not a rule-of-thumb).

n <= 3 is SCREENING. n >= 7 is required for EVALUATION. There is no middle ground.

### Ordinal gate (the strongest requirement)

A result is EXTERNAL-READY only when the WORST evaluation seed beats the BEST
baseline seed. This is stronger than p < 0.05 and must be reported explicitly.

### Verdict tiers for steering hypotheses

| Tier | Meaning |
|------|---------|
| NOVEL+TESTABLE | A genuinely new prediction not derivable from prior corpus; has a specific falsifier |
| DERIVATIVE+TESTABLE | Extends a corpus result; has a specific falsifier |
| NUMEROLOGY | Any nearby axis value (layer +/- 2, alpha x 1.2) would give the same result; the specific threshold is not load-bearing |
| UNFALSIFIABLE | No single observation could discard it; too vague |
| FALSIFIED | Pre-registered falsifier fired; idea is DISCARDED |
| UNTESTED_ON_RIGHT_DATASET | Test ran but on wrong model / split / rung; result is inconclusive |

NUMEROLOGY is the steering analogue of phi-numerology: if the hypothesis is
"layer 18 is optimal" but layers 16-20 all produce similar efficacy, the claim
is NUMEROLOGY — the mechanism predicts a broad optimum, not a point optimum.

### Corpus discipline for inherited numbers

Any number copied from `corpus/*.md` is tagged [NEEDS VERIFICATION] in:
- IDEA_TABLE.md (at registration)
- IDEA.md (in the pre-registered prediction)
- FINDINGS.md (if it appears there — a verified reproduced number is untagged)

An individual paper claim is tagged [UNVERIFIED] until a reproduction run
on our 4090 ladder is logged in EXPERIMENT_LEDGER.md with n >= 1.
It becomes [VERIFIED] after n >= 3 seeds reproduce it within +/-20% (E49 protocol).

### L7 — Single-scale results are PROVISIONAL by construction

Any result characterized on a single model size (here: Gemma-3-1B-it or
Gemma-2-2B-it alone) is tagged **PROVISIONAL** — not SUPPORTED, not
EXTERNAL-READY. Phenomena in activation steering can change sign across
scale (held-out R² going negative has been observed in this program when
transferring a "universal law" from the smoke model to the standard model).

**Requirements before upgrading from PROVISIONAL to SUPPORTED:**

- At least ONE cross-scale replication: the effect must be replicated on a
  model that is ≥ 3× larger in parameter count than the primary model.
- For this project: a result on Gemma-3-1B-it alone is PROVISIONAL; to
  become SUPPORTED it must replicate on Gemma-2-2B-it; to become
  EXTERNAL-READY it must replicate on at least Gemma-2-2B-it (and ideally
  Gemma-2-9b-it at Rung-4, per CLAUDE.md §2 scale-check rules).
- If a replication FAILS, the original result is tagged SCALE-LIMITED:
  the effect is real on the tested scale but does not transfer.

Report the scale coverage in every FINDINGS.md entry: "tested on
[model_ids]; replicated on [model_ids]; PROVISIONAL/SUPPORTED/SCALE-LIMITED."

### L8 — Program-level multiple comparisons control

Per-hypothesis pre-registration is necessary but not sufficient. When this
program screens N hypotheses and promotes the ones that look good, it runs a
garden-of-forking-paths false-positive engine at the program level — even if
each individual hypothesis is correctly pre-registered.

**Requirements (binding at Rung 3+ and for any EXTERNAL-READY claim):**

1. **Confirmation holdout:** designate a set of concepts/prompts/seeds that
   is NEVER touched during screening or hill-climbing. Call it the
   CONFIRMATION SET. All EXTERNAL-READY claims must be confirmed on it.
   Pre-register the confirmation set composition in git BEFORE any screening
   runs.

2. **Promotion budget:** pre-commit to a maximum number K of hypotheses that
   may advance to the confirmation rung (e.g., K = 5 of the N registered
   hypotheses). This is a family-size declaration. Log it in IDEA_TABLE.md
   alongside the N total hypotheses. The Holm-Bonferroni correction in the
   four-part contract uses this K as the family size.

3. **No retroactive expansion of K.** After screening is complete, the
   promotion budget is fixed. Promoting hypothesis N+1 beyond K requires a
   documented protocol amendment committed to git with justification.

4. **Confirmation set integrity.** Any use of confirmation-set prompts for
   debugging, calibration, or pilot runs invalidates those prompts for
   confirmation and they must be replaced with fresh holdout examples. Log
   any such invalidation in EXPERIMENT_LEDGER.md.

**Why this matters:** selecting 5 of 20 hypotheses based on screening results
and then confirming them inflates the family-wise false positive rate. The
promotion budget + confirmation holdout together control this inflation at the
program level, not just the hypothesis level.

### L9 — Power for an effect size, not just a p-value

Pre-register the **minimum delta of interest** (the smallest real effect you
care about), and choose the seed count to provide adequate power for THAT
delta — not for any delta, no matter how small.

**Steering-specific guidance:**

- Typical seed variance in this program: σ_seed ≈ 0.03–0.05 composite units
  (empirically derived from E-series reproducibility runs). The 2σ seed noise
  band is therefore ≈ 0.06–0.10.
- A Δcomposite < 0.05 is likely inside seed noise even at n=7 Wilcoxon. The
  minimum meaningful delta for this program is pre-registered as
  **Δcomposite ≥ 0.05** (i.e., a 5 pp improvement over the champion).
- For Δcomposite = 0.05 and σ_seed = 0.04, the required n for 80% power at
  one-sided Wilcoxon α=0.05 is approximately n=11–13, not n=7.
- n=7 paired Wilcoxon (the Section-7 minimum) provides adequate power for
  Δcomposite ≥ 0.08 at σ_seed ≤ 0.04. For smaller effects, n must increase.

**Pre-registration requirement (binding before any evaluation sweep):**

```yaml
# In pre-registration.yaml for any EVALUATION-tier sweep:
minimum_delta_of_interest: 0.05       # composite units
estimated_sigma_seed: 0.04            # derived from [list of baseline runs]
target_power: 0.80
required_n_seeds: 11                  # computed from the power analysis
seeds_planned: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
power_justification: |
  Wilcoxon power analysis at delta=0.05, sigma=0.04, alpha'=0.0167 (K=3
  family, Holm-Bonferroni). Computed via src/steering/stats.py:power_analysis.
```

**Reporting rule:** every EXTERNAL-READY claim must include:
- The minimum delta of interest (pre-registered)
- The achieved effect size (Δcomposite ± 95% bootstrap CI)
- The power at that effect size and the actual n used
- An explicit statement: "This result is [adequately / inadequately] powered
  for the pre-registered minimum delta of interest."

An INCONCLUSIVE result (p > α but no evidence of absence) must be labeled as
such — not as DISCARD. INCONCLUSIVE means "we ran a properly powered
experiment and could not reject the null." It does NOT mean the hypothesis is
false.

**New harness support:** `src/steering/stats.py:rigor_report.external_ready`,
`src/steering/stats.py:bootstrap_ci`, `src/steering/stats.py:holm_bonferroni`.

### What "external-ready" means in this project

A finding graduates to FINDINGS.md only when:
- It clears the four-part contract above AND the ordinal gate
- It reports ALL five axes (not just behavior efficacy)
- It uses the fingerprinted composite formula
- The verdict is KEEP at rung >= 3 (STANDARD)
- It was confirmed on the pre-designated CONFIRMATION SET (L8)
- Cross-scale replication on at least Gemma-2-2B-it (SUPPORTED, not PROVISIONAL) (L7)
- Power for the pre-registered minimum delta of interest (L9)
- It carries the qualifier: "Internal QA pass — independent external review pending"

### FINDINGS.md self-contained mandate (steering instantiation)

When a finding is added to or updated in `FINDINGS.md`:

1. **Preamble present**: `FINDINGS.md` must open with: how to read the document,
   a glossary defining all hypothesis IDs (E1–E50, N1–N20), all result IDs (S-N),
   all metric abbreviations (behavior_efficacy, MMLU-delta, PPL, CR_jailbreak,
   over_refusal, composite, delta_norm, eff_rank_drop, norm_budget, part_ratio),
   and all verdict tiers.

2. **Zero-cross-reference rule**: every sentence in FINDINGS.md is independently
   interpretable. A finding saying "E4 at layer 6 achieves composite 0.5414" is
   invalid unless E4 is defined in the preamble as "E4 (Curvature-aware injection,
   ideas/04/)". No reader should need to open IDEA_TABLE.md, EXPERIMENT_LEDGER.md,
   or eval.py to understand a sentence in FINDINGS.md.

3. **Summary table**: a table at the top listing all current findings with their
   IDs, one-line summaries, verdicts, composite deltas, n, and tier.

4. **Strongest result stated plainly**: one sentence naming the strongest result,
   or explicitly stating "no EXTERNAL-READY result confirmed yet."

See `../../meta-skills/autoresearch-findings-ledger/SKILL.md` for the full
mandate and template.

### Citation format (mandatory)

Every cited paper: `Author1, Author2, ..., YEAR VENUE 'Title' (arXiv:XXXX.XXXXX)`
- Real arXiv IDs only; mark `[UNVERIFIED]` if unsure of the ID
- For ICLR/NeurIPS/ICML papers, include the venue
- For workshop papers, include `Workshop` in the venue field
- For preprints, use `Preprint` as venue

---

## Quick checklist (rigor audit)

- [ ] n >= 7 seeds for evaluation claims (n <= 3 = screening, clearly labeled)
- [ ] n chosen to provide ≥ 80% power for the pre-registered minimum delta (L9);
      if n=7 is insufficient for that delta, escalate to the required n
- [ ] Paired Wilcoxon p-value reported with Holm-Bonferroni correction
      (K = pre-registered promotion budget, per L8)
- [ ] Bootstrap CI (>= 10k resamples) reported; CI includes achieved effect size
- [ ] Ordinal gate: worst eval seed vs best baseline seed
- [ ] Evaluated on CONFIRMATION SET (never touched during screening/hill-climb) (L8)
- [ ] All five axes reported (behavior, capability, coherence, safety, selectivity)
- [ ] Composite formula fingerprint matches eval.py
- [ ] Geometry leading indicators reported (delta_norm, eff_rank_drop, norm_budget)
- [ ] Cross-scale replication documented; result tagged PROVISIONAL/SUPPORTED/
      SCALE-LIMITED (L7)
- [ ] Controls both beat: delta_vs_random_direction > 0 AND
      delta_vs_shuffled_label > 0 (L6)
- [ ] Metric calibration documented: ρ >= 0.70 against reference labels (L1)
- [ ] Off-family judge used at Rung 2+; judge id logged (L2)
- [ ] Extraction stability: bootstrap_cosine_p5 >= 0.85 for any vector used (L10)
- [ ] No inherited numbers presented without [NEEDS VERIFICATION] tag
- [ ] Citation format correct for every cited paper
- [ ] Verdict tier assigned (NOVEL+TESTABLE / DERIVATIVE+TESTABLE / NUMEROLOGY / etc.)
- [ ] Qualifier "Internal QA pass — independent external review pending" on any ACCEPT

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-paper-rigor/SKILL.md`
- Findings and ledger discipline: `../../meta-skills/autoresearch-findings-ledger/SKILL.md`
- Findings gate: `../../FINDINGS.md`
- Statistical contract: CLAUDE.md Section 7
- Verdict tiers: CLAUDE.md Section 7
- E49 (reproducibility audit): `../../IDEA_TABLE.md`
- Metric calibration + off-family judge: `../steering-eval-bundle/SKILL.md` §10–11
- Controls + extraction stability: `../steering-experiment/SKILL.md` §L6, §L10
- Harness modules: `src/steering/stats.py` (paired_wilcoxon, bootstrap_ci,
  holm_bonferroni, seed_noise_band, ordinal_gate, rigor_report.external_ready)
