# ICML Area-Chair Review — Steering Autoresearch Program

> Reviewer role: hostile-but-fair senior Area Chair. Scope: the **methodology
> and experimental design** of the activation-steering autoresearch program, held
> to ICML acceptance bar. Empirical numbers are still running; this reviews the
> **program**, not the results. The current state is treated as
> *"infrastructure + first screening run."*
>
> Files audited (read-only): `CLAUDE.md`, `AUTORESEARCH_PROCESS.md`,
> `IDEA_TABLE.md`, `meta-skills/autoresearch-paper-rigor/SKILL.md`,
> `meta-skills/autoresearch-meta/SKILL.md`, `src/steering/eval.py`,
> `src/steering/extract.py`, `src/steering/hooks.py`, `src/steering/runner.py`,
> `src/steering/datasets.py`, `src/steering/model.py`,
> `corpus/steering-benchmark-datasets-suite.md`.

---

## 1. Summary

The program is an autonomous, pre-registered research engine for activation /
conditional steering of small Gemma models (Gemma-3-1B-it smoke, Gemma-2-2B-it
standard) on a single 16 GB laptop GPU. It encodes a Karpathy-style
keep/discard loop with three disciplines layered on top: (i) every experiment is
a single-axis perturbation of a "sacred" champion config, authored through a
7-step Diagnose→Cite→Hypothesize→Predict→Execute→Analyse→Checkpoint ritual whose
pre-run fields are gated against placeholders (`CLAUDE.md` §1, §5); (ii) a
5-rung benchmark ladder (UNIT→SMOKE→DEV→STANDARD→FULL) where the *same* five
measurement axes are scored at every rung and a method may not consume rung k+1
compute until it clears rung k's gate (`CLAUDE.md` §3-4); (iii) a
SHA-256-fingerprinted, multi-objective "composite" that prices five axes —
behavior efficacy, capability retention, coherence, safety (Rogue-Scalpel
compliance), selectivity — plus geometry leading-indicators, so a method cannot
win by sacrificing one axis (`eval.py:38-96`). A statistical rigor floor (paired
Wilcoxon + 10k-bootstrap CI + Holm-Bonferroni + empirical seed-noise band)
governs any "winner" claim, with n≤3 hard-classified as SCREENING and n≥7 as
EVALUATION (`autoresearch-paper-rigor/SKILL.md`, `CLAUDE.md` §7). The mechanics
(`hooks.py`, `extract.py`) implement DiffMean/PCA vector extraction, three
residual-stream operations (add / norm-preserving rotate / project_out), exact
hook teardown, and special-token masking. The whole apparatus is also abstracted
into a portable `meta-skills/` pack, with steering as its first instantiation.

The honest framing in the code matters: the offline datasets, the projection
behavior scorer, and the safety responses are all explicitly documented as
SMOKE-tier / Rung-0/1 plumbing stand-ins to be replaced at Rung 2+
(`eval.py:133-137`, `runner.py:78-82`, `datasets.py:11-13`). **As an
infrastructure-and-first-screening artifact this is unusually disciplined. As a
source of any quantitative claim about steering, it currently measures nothing
external.**

---

## 2. Strengths (genuinely rigorous elements)

These are real and above the norm for autoresearch pipelines:

1. **Pre-registration with anti-HARKing enforcement.** The screening-vs-evaluation
   classification and success criterion are committed to version control *before*
   the sweep, and post-hoc reclassification of a loser as "screening" is named a
   BLOCKER (`autoresearch-paper-rigor/SKILL.md` Pillar 2;
   `IDEA_TABLE.md:6-8` forbids editing a row's hypothesis/metric after the idea
   dir exists). This is the single best feature.

2. **Fingerprinted composite.** `COMPOSITE_FORMULA` is a frozen string,
   `composite_fingerprint()` = sha256[:12] (`eval.py:38-61`), and the fingerprint
   is mandated in every reasoning entry and dashboard footer. Editing the formula
   to crown a favored row is flagged as a BLOCKER (`meta SKILL` §4). This is the
   correct defense against the most common autoresearch self-deception.

3. **Ladder gating.** The cost-ordered rung promotion ("never run an expensive
   benchmark to find a bug a cheap one would catch", `CLAUDE.md` §4) with the same
   five axes measured at every rung is a sound experimental-economics design and
   makes a method's trajectory comparable across rungs.

4. **Dual-track audit + negative controls.** Impl-critic, sci-critic, data-split
   leakage audit, and a shuffle-test negative control are required before any
   external claim (`meta SKILL` §10). The shuffle test in particular is the right
   instrument and is currently *missing from the wired pipeline* (see W6).

5. **Rogue-Scalpel safety gate as a first-class axis.** Safety is priced at the
   dominant weight (`lambda_safe=2.0`, `eval.py:53`), a leak is an automatic
   DISCARD regardless of behavior score (`CLAUDE.md` §10), and the composite is
   explicitly constructed so a gibberish-but-"safe" output cannot win because the
   coherence tax dominates (`eval.py:64-96`). The *design intent* is correct.

6. **Screening-vs-evaluation n-floor with a correct statistical justification.**
   The claim that n=3 cannot reach p<0.05 under paired Wilcoxon (smallest
   two-sided p = 0.25 at W∈{0,6}) is *mathematically correct*, and the
   minimum-seed table (`paper-rigor SKILL` Pillar 1) is derived, not a
   rule-of-thumb. The ordinal gate (worst eval seed beats best baseline seed) is
   a strong, defensible bar.

7. **Mechanics correctness at the unit level.** `hooks.py` is clean: the rotate
   operation is genuinely norm-preserving (Gram-Schmidt basis in the (h,v) plane,
   `hooks.py:59-77`), hooks are torn down in `__exit__` (`hooks.py:211-215`),
   special tokens are masked out of steering (`hooks.py:82-94`), and `project_out`
   is the standard refusal-ablation. `extract.py`'s use of the **uncentered**
   difference SVD for PCA-top1 (`extract.py:158-176`) is the correct choice and is
   correctly justified in the docstring. No mechanical errors found.

---

## 3. Major weaknesses (acceptance-blocking)

### W1 — The behavior "projection proxy" is circular (the crux). **BLOCKER for any efficacy claim.**

`projection_behavior_scorer` (`eval.py:118-162`) defines behavior efficacy as the
delta in mean projection of layer-L activations onto the unit target vector `v̂`,
between a steered and an unsteered run. But the steered run *is* `h ← h + alpha·v`
(operation "add", `hooks.py:50-51`), and `v̂ = v/‖v‖`. So the measured quantity is

    Δproj = mean[(h + alpha·v)·v̂] − mean[h·v̂] = alpha·(v·v̂) = alpha·‖v‖,

i.e. the score is a **deterministic, monotone function of the very edit being
applied**, before a logistic squash (`eval.py:160-162`). It cannot *fail* for any
non-degenerate v and positive alpha — it measures that addition adds, not that the
*behavior* changed. This is not an efficacy measurement; it is a tautology. For
the "rotate" operation it is only marginally less circular (it measures the
angle you imposed). The docstring honestly flags it as a Rung-0/1 stand-in
(`eval.py:133-137`), which is acceptable for plumbing — **but no number produced
by this scorer can appear in any efficacy, monotonicity (E3 cliff), or
Pareto claim, and several hypotheses (E1 knee, E3 cliff, E7 norm-relative,
E27 rotation-vs-add) are defined in terms of "behavior" and would be silently
satisfied by this proxy.** The SMOKE-rung promotion gate "monotone effect"
(`CLAUDE.md` §4) is trivially passed by an identity tautology. Until behavior is
scored by an *independent* instrument (LLM-judge concept-incorporation on real
generations, or AxBench's scorer), the ladder's rung-1 gate is not a real gate.

### W2 — The safety and selectivity axes are constants, not measurements. **BLOCKER for the Rogue-Scalpel claim.**

`_fake_responses` returns the literal string `"I can't help with that."` for
*every* harmful prompt unconditionally (`runner.py:84`), and the harmless
responses are the hardcoded literal `"Sure, here is how you can do that safely."`
for every benign prompt (`runner.py:212`). Therefore `compliance_rate ≡ 0` and
`harmless_refusal_rate ≡ 0` *by construction*, regardless of the model, the
vector, alpha, or the operation. The safety axis (Axis 4) and the selectivity
axis (Axis 5) **do not respond to the independent variable at all** in the wired
pipeline. The dominant `lambda_safe=2.0` penalty and the "automatic DISCARD on
leak" mandate are therefore inert — they can never fire. The marquee safety
contribution (the five-layer guard, the 20-vector universal-attack red-team,
`CLAUDE.md` §10) has **zero measurement support** in the current code. The
comment claims steering "can in principle flip this" (`runner.py:78-79`), but the
function never reads the model output — it returns a constant list. This is the
second crux: an ICML safety claim resting on a stubbed constant is
non-existent.

### W3 — TINY synthetic datasets invalidate every quantitative/construct claim.

The wired slices are: axbench_mini = **10** hand-written contrast pairs (concept
"ocean"), mmlu_tiny = **20** synthetic MCQs that are *not real MMLU*,
wikitext_ppl_mini = **10** synthetic passages that are *not WikiText*,
jailbreak_mini = **10** deliberately-vague placeholders, xstest_mini = **8**
(`datasets.py:1-13` and the per-file `_doc` fields). Consequences:
- **Construct validity fails.** mmlu_tiny is not MMLU, wikitext_ppl_mini is not
  WikiText — so "capability retention" and "coherence" measure performance on
  10–20 toy items, not the constructs the axes name. The corpus suite itself
  specifies the *real* benchmarks (`datasets-suite.md` §3-4) — the wired data is
  not them.
- **No statistical power.** A 10-pair extraction cannot test E1's "knee at ≥50
  pairs" — the hypothesis's own falsifier needs 50+ pairs the dataset cannot
  supply (`IDEA_TABLE.md:22`). Per `paper-rigor` Pillar 3 this is
  `UNTESTED_ON_RIGHT_DATASET`, not a result.
- **Single concept.** One concept ("ocean") cannot support DEV-rung's
  "generalizes on held-out concepts" gate (`CLAUDE.md` §4, rung 2). There are no
  held-out concepts.

These are appropriate for UNIT/SMOKE plumbing and are honestly labeled as such,
but they cannot underwrite any axis claim. The program must not let a SMOKE pass
on these slices read as evidence about steering.

### W4 — Capability (Axis 2) is a non-semantic tripwire, not an accuracy measure.

`mcq_accuracy`'s offline path (`eval.py:185-207`) scores an option by
`last_logits[opt_ids % vocab].mean()` — a modular-index hash of token ids into the
logit vector. The docstring is candid that this is "NOT a real MMLU score … a
reproducible capability *tripwire*" (`eval.py:175-183`). Fine as a forward-pass
corruption sensor; useless as the "MMLU_drop_pp" term the composite prices
(`eval.py:81`). The composite's capability tax is therefore currently driven by a
hash artifact.

### W5 — Single-seed (n=1), FakeLM/0.5B-class screening supports no external claim.

The default model is `google/gemma-3-1b-it` but all wired, runnable evaluation is
on the offline `FakeResidualLM` (`model.py`, `datasets.py`, `runner.py`); real
Gemma is gated and not exercised by the pipeline. A first screening run is
therefore (a) on a non-Gemma synthetic LM, (b) at n=1, (c) on toy data. By the
program's *own* rules this is SCREENING and "a statement about the default
config, not about the prior" (`meta SKILL` §5). That is internally consistent —
the danger is only if any downstream artifact (dashboard headline, FINDINGS,
abstract) reads a SMOKE/FakeLM number as evidence. The rigor floor forbids this;
the reviewer flags it as the highest-risk failure mode for the write-up.

### W6 — The composite's λ-weights are asserted, not justified, and the axes are not commensurate.

The weights `lambda_cap=1.0, lambda_coh=0.5, lambda_coh_rep=0.5, lambda_safe=2.0,
lambda_sel=1.0, lambda_geo=0.25` (`eval.py:49-56`) are pinned but **nowhere
derived**. Three concrete problems:
- **Unit incommensurability.** `behavior_efficacy` is a logistic-squashed
  projection in (0,1); `mmlu_drop_pp` is a probability-point drop in (0,1);
  `dppl_norm` is an *unbounded* relative PPL rise (`eval.py:331-333`);
  `offshell_displacement` is in activation-norm units of unknown scale. Subtracting
  these with fixed λ implicitly assumes a common scale that does not exist. A
  single large-PPL run can dominate the composite by orders of magnitude.
- **No sensitivity analysis.** With arbitrary λ, the ranking of methods is a free
  parameter. ICML will ask: does the champion ordering survive a λ-perturbation?
  There is no such robustness check specified.
- **Fingerprinting freezes an unjustified choice.** Fingerprinting correctly
  prevents *mid-project* tampering, but it does not make the *initial* weights
  defensible. The degenerate-row test (`meta SKILL` §4) checks one corner
  (gibberish can't win); it does not validate the trade-off rates between
  non-degenerate methods.

### W7 — No prompting baseline; the AxBench apples-to-apples comparison is absent.

AxBench's central finding is that **prompting and finetuning beat steering**; the
benchmark exists to compare steering against prompting/probing/SAE on equal
footing (`datasets-suite.md` §1, §2). The wired pipeline compares steered vs
*unsteered* only. With no prompting baseline, even a perfectly-measured behavior
lift cannot be positioned as a contribution — the obvious reviewer question
("does a one-line prompt do better?") is unanswerable. This is a required
baseline, not optional.

### W8 — Geometry leading-indicators feed the composite but their validity is unestablished.

`offshell_displacement` is priced in the composite (`lambda_geo`, `eval.py:95`)
and used as a behavior-free cliff predictor (`AUTORESEARCH_PROCESS.md` §2). But
its claimed predictive validity (N17, N20 — that off-shell displacement / local
curvature predicts incoherence and rogue-fragility) is itself one of the
*hypotheses under test*. Using an unvalidated predictor as a *scored penalty term*
in the metric that adjudicates those same hypotheses is mildly circular: the axis
that is supposed to be tested is baked into the judge.

---

## 4. Minor issues

- **M1 — Naming overstates content.** `mmlu_tiny`, `wikitext_ppl_mini`,
  `axbench_mini` read as subsets of the named benchmarks; they are unrelated
  synthetic stand-ins. The `_doc` fields are honest, but the loader names invite
  misreading in dashboards. Prefer `synthetic_mcq_tripwire`, etc.
- **M2 — `behavior_prompts` uses only the positive members of the contrast
  pairs** (`runner.py:184`, `[p for p,_ in pairs][:6]`), so the projection proxy
  is evaluated on the very prompts adjacent to extraction — a leakage-flavored
  choice even setting W1 aside.
- **M3 — Reproducibility gaps in the offline path.** 4-bit vs fp16 invariance
  (E5) is a *precondition* for the whole "4090 results transfer" story but is
  itself only a PENDING hypothesis (`IDEA_TABLE.md:26`); until E5 passes, every
  4-bit number carries an unquantified transfer risk that should be stated up
  front.
- **M4 — Repetition rate computed on input passages, not generations**
  (`runner.py:206` passes the wikitext passages, not steered output), so it
  measures the fixed corpus, not degeneration under steering.
- **M5 — Judge calibration unaddressed.** The suite mandates validating LLM-judge
  precision against a human slice (94% target, `datasets-suite.md` §7), but no
  judge is wired and no calibration harness exists yet.
- **M6 — Missing baselines beyond prompting:** random-direction control (the
  Rogue-Scalpel "≥1000 random vectors" standard, `datasets-suite.md` §7) and a
  mean-ablation control are not wired; both are cheap and expected.

---

## 5. Required experiments before ANY external claim

Concrete, ordered, each gating the claim it supports:

1. **Replace the projection proxy with an independent behavior judge** (W1).
   Score concept-incorporation on *real generated text* via an LLM-as-judge (or
   AxBench's released scorer), validated against a human-annotated slice. No
   efficacy/monotonicity/Pareto claim ships on the projection scorer.
2. **Wire a real safety + selectivity measurement** (W2). Generate with the real
   model on JailbreakBench (100 prompts, 10 categories) and XSTest, judge
   SAFE/UNSAFE with a calibrated judge, confirm baseline CR≈0%, and demonstrate
   the `lambda_safe` penalty and the auto-DISCARD can actually fire. Replace
   `_fake_responses` entirely.
3. **Load the real datasets per the suite** (W3): real AxBench concept set with a
   held-out-concept split (for the DEV-rung generalization gate), real MMLU (≥
   the suite's 500), real WikiText-103 perplexity. Retire the synthetic slices to
   UNIT/SMOKE only.
4. **n≥7 seeds with the full rigor contract** (W5): paired Wilcoxon + 10k
   bootstrap CI + Holm-Bonferroni + empirical 2σ_seed band + ordinal gate, on the
   pre-registered evaluation split — exactly as the program's own
   `paper-rigor SKILL` already specifies. (The contract is well-designed; it just
   has to be *run* on real instruments.)
5. **Prompting baseline (and random-direction / mean-ablation controls)** (W7,
   M6): the AxBench apples-to-apples comparison. A steering "win" is only a
   contribution relative to a tuned prompt.
6. **Gemma reproduction + λ-sensitivity** (W5, W6): reproduce on real
   Gemma-2-2B-it (not FakeLM, not 1B-only), and report that the champion ordering
   survives a λ-weight perturbation and that E5 (4-bit↔fp16 invariance) holds, so
   the composite ranking and the precision-transfer story are both defensible.
7. **Held-out concept generalization + shuffle-test negative control**: confirm
   the method works on concepts not used in extraction (W3) and *fails* under
   label/condition shuffling (the missing negative control, W6/§3 strength 4).

---

## 6. Score + recommendation

**Recommendation: Reject (in its current state) — but as a methods/infrastructure
contribution it is a credible Borderline once instruments 1–5 above are real.**

Numerically, on an ICML 1–10 scale: **3 (Reject)** for the program *as a source
of steering claims today*; **6 (Borderline / Weak Accept)** for the
*methodology-and-harness* framing *if and only if* the projection-proxy and
stubbed-safety instruments are replaced with real measurements and the synthetic
slices are demoted to plumbing. The gap between those two numbers is exactly the
gap between "infrastructure + first screening run" (what exists) and "a defensible
body of evidence" (what is claimed as the North Star).

**One-sentence meta-review:** *A genuinely well-engineered, honestly-labeled
rigor harness — pre-registration, a fingerprinted multi-objective composite, a
cost-ordered ladder, and a real statistical floor — that currently measures
behavior with a tautological projection-of-the-edit-onto-itself proxy and
"measures" safety with a hardcoded constant string on 10–20 synthetic items, so
no quantitative steering conclusion is yet supportable; accept the process, reject
the (non-existent) results, and re-evaluate once the five named instruments are
real.*

---

## 7. Circularity disclosure (same-model-family reviewer)

This review was produced by a Claude-family model. The program under review is
itself authored, implemented, critiqued, and (per its own design) audited largely
by agents sharing that model family (`CLAUDE.md` §14; `meta SKILL` §8). Per the
program's own mandated disclosure, **this verdict is an internal-QA-grade filter,
not an external seal of approval**: it has not been calibrated against a
known-good external reference codebase, so the non-PASS findings above should be
read as *descriptive, not diagnostic*. An independent, different-family review —
and the program's own dual-track audit run on a real reference implementation to
establish a false-positive baseline — remains a prerequisite before any of these
findings (or any rebuttal to them) is treated as settled.
