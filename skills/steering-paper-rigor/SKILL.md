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

### What "external-ready" means in this project

A finding graduates to FINDINGS.md only when:
- It clears the four-part contract above AND the ordinal gate
- It reports ALL five axes (not just behavior efficacy)
- It uses the fingerprinted composite formula
- The verdict is KEEP at rung >= 3 (STANDARD)
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
- [ ] Paired Wilcoxon p-value reported with Holm-Bonferroni correction
- [ ] Bootstrap CI (>= 10k resamples) reported
- [ ] Ordinal gate: worst eval seed vs best baseline seed
- [ ] All five axes reported (behavior, capability, coherence, safety, selectivity)
- [ ] Composite formula fingerprint matches eval.py
- [ ] Geometry leading indicators reported (delta_norm, eff_rank_drop, norm_budget)
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
