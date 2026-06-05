# E49 — Meta-Reproducibility Audit: >= 70% of Source Claims Reproduce

> **One-line claim:** Quantitative steering claims from the primary corpus
> papers (CAA, AxBench, Rogue Scalpel, HyperSteer, Selective, Persona
> Vectors) reproduce within +-20% on our Gemma-2-2B harness for at least
> 70% of tested claims, establishing the reliability of the source corpus
> as a basis for the 50-experiment program.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axis:** A7 (HOW DERIVED — source / meta-level).
> **Implementation status:** `o planned / UNTESTED`.

---

## In Plain English

**What we're testing, simply:** This whole project builds on numbers reported in
other researchers' papers. We ask: when we redo their experiments ourselves on
our own model, do their headline numbers actually come out roughly the same?

**Key terms (defined here):**
- **Language model** — an AI that writes text one word at a time.
- **Steering** — changing the model's behavior by editing its internal state
  mid-sentence, without retraining.
- **Steering vector** — the nudge we add to push toward a behavior.
- **Residual stream** — the model's running internal scratchpad; the nudge goes
  here.
- **Layer** — one of the model's stacked processing steps.
- **alpha / strength** — how hard we push.
- **DiffMean** — the simplest nudge recipe: average internal state on "yes"
  examples minus "no" examples. No training.
- **Coherence** — whether the steered text stays fluent and sensible.
- **Reproduce** — redo someone else's experiment and check you get a similar
  result. The heart of this test.
- **Source corpus / source claims** — the published papers (and their reported
  numbers) that the rest of this project relies on.
- **Within 20%** — our "close enough" bar: a number counts as reproduced if it
  lands within a fifth of the original.
- **Robustness / monitoring** — making sure the foundation is solid before we
  build a 50-experiment program on top of it.

**Why we're doing this (the point):** If we're going to trust other people's
results as our starting point, we'd better check they hold up. This is a
foundation-inspection step — boring but essential for honesty.

**What the result would mean:** If most of the borrowed numbers reproduce, our
foundation is sound and we can build on it confidently. If many don't, parts of
the literature we relied on are shaky, and we have to be far more careful about
what we claim.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The entire autoresearch program rests on a set of quantitative claims
inherited from published papers: DiffMean achieves >= 90% of asymptotic
efficacy at 50 pairs (E1); CAST keeps harmless-input refusal < 3% (E9);
SAE-feature steering underperforms DiffMean by a specific margin (E36);
the Rogue Scalpel's universal attack raises compliance by ~ 4x (F5). These
claims were measured on different models (Llama, Qwen, Falcon), different
benchmarks (JailbreakBench, AxBench, custom evals), and different
experimental conditions than our Gemma-2-2B 4090 harness. Reproduction
failures are common in machine learning: claimed effects may be model-
specific, dataset-specific, or the result of undisclosed hyperparameter
tuning. A meta-reproducibility audit systematically attempts to reproduce
a representative sample of quantitative claims from the source corpus on
our specific setup, establishing which claims transfer and which are
artifacts of the original experimental conditions. The 70% reproduction
rate threshold is pre-registered as the minimum for treating the source
corpus as a reliable foundation. Below 70%, the program must revisit its
pre-registered thresholds (which are inherited from the corpus). This
experiment is also a calibration of our experimental harness: if we cannot
reproduce known results, our novel experiments are unlikely to be reliable.
The audit directly supports the program's internal validity.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** For a sample of 20 specific quantitative claims drawn from the six
primary corpus papers (CAA/DiffMean: arXiv:2312.06681; AxBench:
arXiv:2501.17148; Rogue Scalpel: arXiv:2509.22067; HyperSteer:
arXiv:2506.03292; Selective: arXiv:2601.19375; Persona Vectors:
arXiv:2507.21509), at least 14 claims (70%) will reproduce within +-20%
of the reported value on Gemma-2-2B-it (4-bit) in our harness. Claims are
tested using the closest available approximation of the original protocol
on our hardware and model.

---

## 3. Falsifier (>= 30 words)

If fewer than 12 of the 20 tested claims (60%) reproduce within +-20%,
the source corpus is unreliable as a basis for pre-registered thresholds
in the autoresearch program, and all inherited pre-registered thresholds
must be flagged `[UNVERIFIED, corpus non-replicable]` until re-derived
from Gemma-2-2B experiments. Status moves to `x disproved (meta)`.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA; source of DiffMean extraction claims
and behavioral efficacy numbers; claims to audit: pair-count knee at ~50
pairs (E1), behavior-success rate at optimal alpha for sycophancy and
refusal, cross-layer efficacy profile.

Zhong, Zeping, et al. 2025 'AxBench: Steering LLMs? Benchmarks Matter'
arXiv:2501.17148 — AxBench; source of the SAE-feature vs DiffMean gap;
claims to audit: DiffMean concept-incorporation score on their eval set,
SAE-feature underperformance margin, prompting vs steering gap.

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; source of compliance-rate
numbers; claims to audit: random-vector compliance rate (F1: 1-13%),
benign-SAE-feature compliance (F2: comparable to random), universal-attack
construction amplification (~4x).

Hernandez, Evan, et al. 2025 'HyperSteer: Concept-based Activation
Steering via Hypernetworks' arXiv:2506.03292 — HyperSteer; zero-shot
efficacy claims; the main E45 paper; claims to audit: hypernetwork vs
supervised efficacy gap on held-out behaviors.

Perez, Ryan, et al. 2025 'Persona Vectors: Representations of Character
and Disposition in LLMs' arXiv:2507.21509 — Persona Vectors; source of
persona-direction extraction methodology and causal control claims;
claims to audit: persona-direction extraction efficacy, causal control
rate for sycophancy and aligned-assistant persona.
```

---

## 5. Mechanism

The audit is methodological rather than mechanistic: it follows the
replication protocol (attempt to reproduce each claim as closely as
possible on our hardware/model) and categorises each claim as:

- REPRODUCED (within +-20% of reported value)
- PARTIAL (within +-50%, outside +-20%)
- FAILED (outside +-50% or qualitatively different direction)
- NOT TESTABLE (requires infrastructure not available on 4090 single-GPU)

The +-20% threshold accounts for model-to-model variation (Llama vs
Gemma), quantisation effects (fp16 -> 4-bit), dataset-size differences,
and minor protocol variations. The +-50% "partial" category captures
claims that show the right qualitative effect but at a different magnitude.

Claims are selected to span: (a) efficacy numbers (behavior-success rates);
(b) safety numbers (compliance rates); (c) capability costs (MMLU drops);
(d) geometric properties (cosine alignments, norm ratios). At least 3
claims from each paper are audited.

The audit also produces a "deviation catalogue" — for each non-reproduced
claim, a hypothesis about why it failed to replicate (model-specific,
quantisation artifact, dataset-specific, protocol underspecified). This
catalogue directly informs which pre-registered thresholds in other
experiments should be relaxed or tightened.

---

## 6. Predicted Delta

| Paper | Claims audited | Expected reproduction rate |
|---|---|---|
| CAA/DiffMean (arXiv:2312.06681) | 4 | 75-100% |
| AxBench (arXiv:2501.17148) | 4 | 50-75% (Gemma vs AxBench target models) |
| Rogue Scalpel (arXiv:2509.22067) | 4 | 75-100% (our S-4/S-8 already screens CR) |
| HyperSteer (arXiv:2506.03292) | 3 | 50-75% (requires training) |
| Selective (arXiv:2601.19375) | 2 | 60-80% |
| Persona Vectors (arXiv:2507.21509) | 3 | 60-80% |
| Overall (20 claims) | 20 | >= 70% (14/20) |

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Claim selection: 20 specific quantitative claims pre-specified before
  running (list locked in EXPERIMENT_LEDGER.md); selection balanced across
  papers and claim types. Claims specify the metric, the expected value,
  and the +-20% window.
- Protocol: for each claim, follow the closest available reproduction of
  the original paper's method. Where the original used a different model,
  adapt the protocol to Gemma-2-2B while preserving the key design choices.
- Reporting: for each claim: (a) original reported value, (b) our
  reproduced value, (c) relative deviation, (d) REPRODUCED/PARTIAL/FAILED,
  (e) hypothesis for deviation if not reproduced.
- Seeds: 1 (screening); 3 for important claims.
- Outcome: reproduction rate (REPRODUCED / total) vs the 70% threshold.

### 7.2 Where it shines

This is the calibration backbone of the entire program. It is the only
experiment that validates the program's pre-registered claims against their
source papers. Running it early (before other experiments) reduces the risk
of building on unreproducible foundations.

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E49.
- ALL other experiments: E49 validates the pre-registered thresholds
  inherited from the corpus; non-replication changes those thresholds.
- FINDINGS.md screening observations (S-4, S-8): partially confirm the
  Rogue Scalpel CR numbers on Gemma; E49 provides the systematic audit.
- FINDINGS.md S-1 through S-9: our own screening results that can be
  compared to corpus claims for the overlapping experiments.

---

## 9. Committee Q&A

**Q: How do you select the 20 claims without cherry-picking the easiest
ones to reproduce?**

> The 20 claims are pre-specified in EXPERIMENT_LEDGER.md before any
> reproduction attempts. The selection rule is: for each paper, include at
> least one "main result" claim (a headline number from the paper's abstract
> or key figures) and at most two "auxiliary" claims (supporting numbers).
> The selection cannot be revised after reproduction attempts begin.

**Q: A +-20% window is generous. Shouldn't a reproduction require closer
agreement?**

> For cross-model reproduction (Llama -> Gemma, fp16 -> 4-bit), +-20% is
> a reasonable threshold. Same-model, same-setup reproductions would use
> a tighter +-5% window, but that is not the context here. The +-20% window
> is pre-registered and will not be changed after running.

**Q: What counts as "not testable" to avoid artificially inflating the
reproduction rate by excluding hard claims?**

> "Not testable" requires explicit justification (e.g., the claim requires
> A100-scale compute, or a proprietary dataset not publicly available).
> Not-testable claims do not count toward the denominator OR the numerator.
> They are reported separately with the justification. The 70% threshold
> applies to testable claims only.

---

## 10. Verification checklist

- [ ] 20 specific claims pre-registered in EXPERIMENT_LEDGER.md before
      any reproduction attempt; list is locked.
- [ ] Each claim specifies: paper, metric, reported value, +-20% window,
      protocol adaptation for Gemma-2-2B.
- [ ] "Not testable" category requires written justification; max 4 claims
      (20%) can be not-testable before the threshold drops below 70% of
      the 16 testable claims.
- [ ] Results table with REPRODUCED/PARTIAL/FAILED and deviation for each
      claim.
- [ ] Deviation catalogue: hypothesis for each non-reproduced claim.
- [ ] Reproduction rate reported with 95% CI (binomial).
- [ ] IDEA_TABLE.md row updated; implications for other experiments' pre-
      registered thresholds noted.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E49.
  Status: `o UNTESTED`. Some pre-cursors exist: FINDINGS.md S-4 and S-8
  partially reproduce the Rogue Scalpel compliance-rate finding on Gemma
  (CR 0.80 -> 1.00 under steering, qualitatively matching the paper's
  finding that any steering increases compliance). S-1/S-3 reproduce E4's
  cos > 0.95 cosine alignment finding. These screening observations are
  not formal audit entries but increase confidence in the 70% reproduction
  target.

  Dependency: the 20-claim list must be pre-specified before running
  (EXPERIMENT_LEDGER.md entry required first). Most claims can be tested
  with the existing harness.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (reproducibility + meta-science specialist).*

### Prior plausibility
**MEDIUM-HIGH** for >= 70% reproduction rate overall. The steering
literature has a mixed reproducibility record: geometric claims (cosine
alignments, PPL curves) tend to reproduce well across models; quantitative
behavioral claims (behavior-success rates at specific alpha values) are
more model-specific and harder to reproduce. Our screening results (S-1
through S-9) suggest the geometric claims reproduce well.

### Mechanism scrutiny
The audit methodology is sound. The key risks are: (a) protocol
underspecification in the original papers (what exactly is "the same
method"?); (b) dataset unavailability (some AxBench evals use proprietary
sub-splits); (c) judge disagreement (different LLM-as-judge systems may
give different compliance rates).

### Confounds
1. The +-20% threshold is calibrated to cross-model variation but not to
   the specific Llama-to-Gemma gap. If Gemma-2-2B is systematically more
   or less steerable than Llama-3.1-8B, many claims from the Rogue Scalpel
   (which uses Llama) may be outside +-20%. Run a quick calibration: pick
   one claim from each paper and reproduce it before locking the 20-claim
   list, to verify the +-20% window is appropriate.
2. The 4-bit quantisation may systematically affect specific types of
   claims (e.g., compliance rates may be more sensitive to quantisation
   than cosine alignments).

### Expected effect size
My prior: 65-80% reproduction rate (straddling the 70% threshold). The
failure cases will cluster on the AxBench efficacy numbers (Gemma may
differ from the AxBench target models) and the Rogue Scalpel quantitative
compliance numbers (Llama vs Gemma architecture gap).

### Verdict
**NECESSARY + HIGH PROGRAMMATIC VALUE** — This is the calibration
experiment that validates or invalidates the program's foundation.
It should be run in parallel with, not after, the other experiments.
A 70% reproduction rate would be a strong endorsement of the inherited
thresholds; below 60% would require a program-wide threshold revision.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to E49, a
**meta-level robustness** audit: do the corpus papers' quantitative claims
reproduce on our Gemma-2-2B harness? There is no single steering vector — E49
re-runs OTHER hypotheses' recipes and compares to reported numbers.

### 1. Recipe (re-run each claim's native METHODOLOGY path)

```python
# Each claim is reproduced with the CLOSEST available harness path, e.g.:
#   CAA pair-count knee (E1)      -> extract.diffmean_vector over pair budgets
#   E4 cosine alignment           -> cos(diffmean_vector, pca_top1_vector)
#   Rogue Scalpel random-vec CR   -> controls.matched_norm_random + JailbreakBench CR
#   HyperSteer held-out efficacy  -> src/steering/hypersteer.py (see E45)
#   Persona causal control        -> DiffMean persona steer (see E39)
# All injection via hooks.apply_operation; all behavior via judge.GeminiJudge (§3,§5).
```

### 2. Experiment procedure (20-claim locked audit)

```text
1. PRE-REGISTER 20 specific claims in EXPERIMENT_LEDGER.md (>=3 per paper, >=1
   headline each); each row = {paper, metric, reported value, ±20% window,
   Gemma-2-2B protocol adaptation}. List LOCKED before any reproduction.
2. For each claim: run the closest harness reproduction; record our value.
3. Classify: REPRODUCED (within ±20%) / PARTIAL (±50%) / FAILED (outside ±50%
   or wrong direction) / NOT-TESTABLE (needs A100-scale or proprietary data;
   excluded from numerator AND denominator, with written justification).
4. Build a deviation catalogue (why each non-reproduced claim failed).
```

### 3. Measurement & decision rule

- **PRIMARY metric:** reproduction rate = REPRODUCED / testable claims, vs the
  pre-registered 70% (14/20) threshold; reported with binomial 95% CI.
- **Hypothesis (§2):** >= 14/20 reproduce within ±20% on Gemma-2-2B.
- **Pre-registered FALSIFIER (§3):** if fewer than 12/20 (60%) reproduce within
  ±20%, the corpus is unreliable as a threshold source; all inherited
  pre-registered thresholds are flagged `[UNVERIFIED, corpus non-replicable]` and
  Status -> `x disproved (meta)`.

### 4. Where the code is / status — UNTESTED (some screening pre-cursors)

- **No driver yet** (campaign + `scripts/build_provenance.py` -> `PROVENANCE/E49.md`).
- **Pre-cursors:** FINDINGS.md S-4/S-8 partially reproduce Rogue Scalpel CR on
  Gemma; S-1/S-3 reproduce E4's cos > 0.95. **Missing machinery (why UNTESTED):**
  the **locked 20-claim list** must be written into EXPERIMENT_LEDGER.md FIRST; and
  many claims depend on infra not yet built (GemmaScope SAE for AxBench claims,
  JailbreakBench for Rogue Scalpel CR, the E45 hypernetwork for HyperSteer,
  persona vectors). The audit is a meta-experiment gated on the other blocks'
  tooling and a calibrated judge.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E49.md`.
