# STA Implementation-Fidelity Audit

Status: COMPLETE. Read-only audit; no lesson file was edited.
Evidence script: `scratchpad/demo_sta.py` (CPU-only, imports the lesson classes unmodified,
no model, no network, no GPU).

## 1. Per-method fidelity table

| method | paper | what the paper does | what we do | verdict |
|---|---|---|---|---|
| `ESNCusum` | 2608.02464 Dubey, *Real-Time Detection and Repair of LLM Agent Failures* | one-class **echo-state-network ENSEMBLE** with CUSUM alarms over raw agent **telemetry**; AUROC 0.872 at 5% FA budget | single FIXED reservoir (spectral-radius-scaled, leaky), per-channel one-sided CUSUM vs a benign-only null, channel-max fusion, over Gemma **embeddings** | **INSPIRED-BY, correctly labelled** — one omission (ensemble→single) |
| `SafetyDriftMonitor` | 2603.27148 Dhodapkar & Pishori, *SafetyDrift* | absorbing Markov chain over hand-designed risk states, fit on trajectories that REACH a violation; finite-horizon absorption probability | k-means clusters of embeddings as states; transient Q from benign-only transitions; absorbing column assigned by a **monotone heuristic** on centroid-distance rank; absorption by real matrix powers | **INSPIRED-BY, labelled — but one departure is UNDECLARED and it is the load-bearing one** (see below) |
| `GRUAggregator` | 2601.00516 Advani, *Trajectory Guard* | **Siamese Recurrent Autoencoder**, task tower + trajectory tower, **hybrid contrastive + reconstruction** loss; F1 0.88–0.94, 32 ms | plain 1-layer GRU + linear head, BCE loss | **INSPIRED-BY, correctly and explicitly labelled** |
| `QueryTokenCompressor` | 2606.00611 Hong et al., *TRACE (Risk-Aware Compression)* | Compressor → compact latent evidence state → **Reader** judges safety, trajectory-level supervision; +12.6 pp | k learnable queries cross-attend over steps → concat → linear head. **No Reader** | **INSPIRED-BY, correctly labelled** (absence of a Reader is stated) |
| `MeanPool` / `MaxPool` / `DeviationWeighted` / `LastStep` / `FirstKSteps` | 2601.00516 (the dilution claim) | claim only, no architecture borrowed | fixed pooling + shared LogisticRegression | N/A — these are the harness's own controls, not reproductions |

### Rule-4 labelling: PASSES on three of four, with one real gap

The docstrings are unusually disciplined. `sequence.py:181` says outright "This class is a plain
GRU classifier, not a reproduction of that Siamese-autoencoder architecture or its loss", and
`sequence.py:243` says "a small standalone approximation of the 'compress via learned queries'
idea, not a reproduction of their Compressor+Reader pipeline". Both add an explicit "do not print
this class's numbers beside the paper's" instruction. `esn_cusum.py:13` and `markov.py:14` both
open a block headed "THIS IS AN ADAPTATION, not a reproduction". Nothing in this lesson is
**mislabelled as a reproduction**.

The gaps are of a different kind — departures that are real but not *listed*:

**(a) SafetyDriftMonitor does not implement drift.** This is the important one.
`markov.py` declares two departures (k-means states; heuristic absorbing column). It does not
declare a third and larger one: **the runtime score has no memory at all.** `update()`
(`markov.py:100-111`) computes `s = self._discretize(step_vec)` and returns
`self._absorb_prob_table[s]` — a pure function of the *current step alone*. `self._current_state`
is written and never read; the trajectory score is `max` over per-step lookups. The Markov
transition structure is used exactly once, at fit time, to build a per-state table, and the
resulting monitor is behaviourally **`max_t f(cluster(step_t))`** — a max-pool over a
step-wise deviation score. The paper's entire thesis is *composition*: "individually benign
actions composing into a policy violation", i.e. risk that accumulates along a path. A monitor
whose score does not depend on the path cannot express that mechanism. It is still an absorbing
chain (Q is genuinely sub-stochastic, row sums `1 - p_absorb[i]`, and
`1 - (Q^h).sum(axis=1)` at `markov.py:160-161` is the correct finite-horizon absorption
probability computed by real matrix powers — that part is faithful), but the chain is a
*state-labelling device*, not a *drift accumulator*.

**(b) ESNCusum drops the ensemble.** The abstract says "one-class echo-state-network **ensemble**".
`esn_cusum.py` builds one reservoir. The docstring's list of departures (embeddings vs telemetry,
our null vs their 2,823-episode corpus) does not mention it.

**(c) Two smaller ESN notes.** The CUSUM is **one-sided positive only**
(`g = max(0, g + z - k)`, `esn_cusum.py:128`), so a channel drifting *below* its healthy mean is
invisible; two-sided CUSUM is the textbook form and the paper's phrasing does not specify. And
`_peak` is a running max, which has a consequence in §3 below.

**What is genuinely faithful (verified in code, not from the docstring):**
- The ESN reservoir IS untrained. `_init_reservoir` (`esn_cusum.py:93`) draws `W_in`/`W_res` once;
  `fit()` (line 140) only estimates `mu`/`sigma`/`threshold` and never touches the weights.
  Spectral-radius rescaling (line 98-102), sparsity mask, leak rate, `input_scale` all present.
- The CUSUM IS a real cumulative sum, not a threshold on the instantaneous score — `self._g`
  carries across steps and is only reset per trajectory.
- Both monitors are genuinely one-class: `fit()` filters `y == 0` before estimating anything
  (`esn_cusum.py:142`, `markov.py:125`) and raises rather than degrading if no benign rows exist.
- `state_bytes()` is genuinely O(1) in steps for both.

**Evidence for (a), measured CPU-only with the class imported unmodified**
(`scratchpad/demo_sta.py`, DEMO 2):

```
score(trajectory)          = 0.618508340514
score(SHUFFLED same steps) = 0.618508340514
score(REVERSED)            = 0.618508340514
absorb_prob_table (the ONLY 8 values any step can score):
  [0.134, 0.224, 0.382, 0.473, 0.569, 0.619, 0.676, 0.724]
corr(risk_rank, absorb_prob) = 0.9817
```

Bit-identical under permutation and reversal: the score is provably path-independent. And
the absorption table correlates 0.98 with centroid-distance rank, so the whole
Markov apparatus reduces to a monotone relabelling of "how far is this step from the benign
centroid" — the monitor is `max_t g(distance(step_t))` with eight quantisation levels. That
is a legitimate baseline, but it is a deviation max-pool, not safety drift, and the module
docstring should say so alongside its other two declared departures.


## 2. Does the experiment test its hypothesis?

### 2.1 F1 AND F2 CANNOT ISOLATE THE HORIZON. The design doc says they can.

`types.py:25-31` states the inference explicitly: "If mean pooling degrades on SHADE but
not AgentDojo, the HORIZON is the cause." **That inference is not licensed by the design.**
SHADE and AgentDojo hold exactly one thing constant — the label machinery (same
ControlArena schema, same Gemini-3.1-Pro `ponr` judge, same admission filter). Everything
else varies together with the horizon: task suite (SHADE-Arena side-task scheming vs
AgentDojo's injected-attack banking/email/slack suites), tool inventory, environment,
generating agent, the prose style of a rendered step, base difficulty, and — critically —
the *kind* of positive (a covert secondary objective vs an injected instruction). A joint
F1-holds/F2-holds result is equally consistent with "AgentDojo's prompt injections are
blatant enough that any pooled representation finds them", which has nothing to do with
horizon length.

`run_sta.py:489-491` prints this same over-strong reading to the console ("F1 holds + F2
holds means the horizon is the cause"). It should read "is *consistent with* the horizon
hypothesis; corpus and horizon are not separated by this contrast".

There is also a plumbing consequence: `CORPUS` is a single env-selected value and
`RESULTS_PATH` is per-corpus, so F1 and F2 are **evaluated in different processes writing
different JSON files**, and the non-active falsifier is stamped `applicable: false` in each.
No artifact in the lesson ever holds both verdicts, and nothing computes the conjunction.
The reader is asked to join two files by hand.

### 2.2 A within-corpus horizon control EXISTS but is built in the wrong shape

`FirstKSteps` is present (`pooling.py:244`) and genuinely swept — `run_sta.py:350` builds
`FirstKSteps(k)` for `k in EARLY_KS = 2,4,8,16,32,64` and plots the curve. So the answer to
"is there a within-corpus horizon control" is *half yes*, and that is worth crediting.

But `FirstKSteps._pool` is `prefix.mean(axis=0)` — it is **MeanPool restricted to a
prefix**. There is no `MaxPool`-at-k, no `GRU`-at-k. The curve therefore answers *"how much
signal sits in an early prefix"* and cannot answer *"does the mean-vs-max gap widen as the
horizon grows"* — which is the actual dilution claim. A falling `FirstKSteps(k)` curve is
what you would get from *less information*, not from *dilution*; the two are
indistinguishable in the curve as built.

**The fix is small and it is the highest-value change in this audit.** Make truncation a
wrapper rather than a pooling function — `Truncate(k, inner)` where `inner` is any
`_PooledAggregator` — then report, on SHADE alone,

```
delta(k) = AUC[MaxPool@k] - AUC[MeanPool@k]   for k in EARLY_KS, and k = all
```

If `delta(k)` grows monotonically with `k` **within one corpus**, the horizon claim is
established with task, generator, tools and label machinery held perfectly fixed, and
AgentDojo is no longer load-bearing at all — it becomes a nice external replication instead
of the sole control.

A second within-corpus control is free and uses data already loaded: SHADE's own step
counts run 30 → 422 (median 131). Split SHADE at its own median into short-SHADE and
long-SHADE and run the full ladder on each half. Same corpus, same tasks, same generator,
different horizon. Two strata of ~250 trajectories each still clear screening tier.

### 2.3 The comparator IS defined — but it is the wrong statistic, and F2 rewards noise

"Materially below" is **not vibes**: `run_sta.py:102` defines
`margin_ok(a,b) = (a - b) > 0.02`, and `config.py`'s F1/F2 text pre-registers the same
0.02 before the run. Credit where due. Three problems with *what* it is:

1. **It is an unpaired point-estimate threshold.** 10,000 bootstrap resamples are computed
   per method (`BOOTSTRAP = 10000`) and then **never used in a verdict**. Meanwhile
   `AggregatorResult.scores` is a per-trajectory, index-aligned array across every method
   (all methods score the same out-of-fold indices), so a *paired* bootstrap on
   `AUC_max - AUC_mean` — which is what CLAUDE.md §7 asks for — costs one extra loop and
   would give the delta its own CI.
2. **F2 accepts the null, so noise makes it easier to pass.**
   `holds = not margin_ok(max,mean) and not margin_ok(gru,mean)` (`run_sta.py:130`). Any
   measurement too imprecise to resolve a 0.02 gap scores F2 as HOLDS. The falsifier that
   carries the entire *control* half of the argument is the one that rewards imprecision.
   It should be an **equivalence** test: the 95% CI on the paired delta contained within
   +/-0.02. As written, a botched AgentDojo run "confirms" the horizon hypothesis.
3. **No multiplicity handling.** F3's `best` is `max` over all eight main-ladder AUCs
   (`run_sta.py:100-101`), which is winner's-curse biased upward, with no Holm correction
   across the family. F3 is structurally biased toward HOLDS.

### 2.4 `LastStep` is a FAIR control against the pooling arm, and an UNFAIR one against the learned arm

`LastStep` (`pooling.py:225`) subclasses `_PooledAggregator` exactly like MeanPool/MaxPool/
MeanMaxStd/DeviationWeighted/FirstK: identical `StandardScaler`, identical
`LogisticRegression(max_iter=2000, random_state=seed)`, identical folds, identical
`evaluate_offline` path. Only `_pool` differs (`trajectory[-1]`). **Against the pooling arm
the comparison is clean** — this is done right.

It is *not* clean against `GRUAggregator` / `QueryTokenCompressor`, which are trained
end-to-end in torch with their own heads, their own capacity and their own optimiser
(`sequence.py` says so, as a deliberate second axis). F3 takes `best` across that boundary,
so a GRU-over-LastStep margin confounds "trajectory structure helps" with "a trained
nonlinear encoder beats a linear model on one 768-vector". F3 should be reported twice:
best-within-the-shared-classifier-arm vs LastStep (apples to apples), and best-overall as a
clearly-labelled secondary.

### 2.5 `MeanMaxStdPool` is genuinely distinct from `MeanPool` — verified

`pooling.py:159-163` returns `concatenate([mean, max, std])`, `3*dim` wide, vs `MeanPool`'s
`mean(axis=0)`, `dim` wide. Different features, separate classifier fits, separate rows in
the ladder. The stated purpose (stopping `cross_trajectory`'s "mean pooling wins" from
hiding a max term) is actually achieved.

### 2.6 The falsifier that decides the lesson never consults the confound bar

`confound_bars` is computed first and is well built — it includes a **step-count** channel
(`units=` is passed at `evaluate.py:351-352`) plus a first-step bag-of-words rival. But only
**F4** compares anything to `binding_bar`. F1, F2 and F3 — the three that decide the lesson's
headline — use raw AUCs and never reference it. The console prints "every claim about it
must still be priced against confound.binding_bar" as prose; the machine-readable verdict
does not.

On this substrate that is not a hypothetical worry. I measured the two streaming monitors'
own scores against pure trajectory length, CPU-only, no model
(`scratchpad/demo_sta.py`, both classes imported unmodified from the lesson):

```
DEMO 1  ESNCusum.score() on trajectories drawn from ONE identical benign distribution,
        differing ONLY in length (30 steps vs 120 steps):
          mean peak  30 steps = 20.373
          mean peak 120 steps = 33.135
          AUC of "longer = positive"  = 0.9222

DEMO 3  SafetyDriftMonitor, same experiment (20 vs 100 steps):
          AUC of "longer = positive"  = 0.9024
```

The cause is structural, not incidental: both monitors score a trajectory as
`self._peak`, a running **maximum** over per-step scores, which is non-decreasing in the
number of steps by construction. Any corpus where positives run longer than negatives hands
these monitors most of an AUC for free. F4's `binding_bar` does gate on this (the step-count
channel is in `confound_report`), which is the right instinct — but F1/F2/F3 do not, and
`MaxPool` has the same monotone-in-length property in a milder form.

### 2.7 F0 does not certify the ladder, and there is a leakage channel it cannot see

`F0_shuffle_control` reads `confound.full_trajectory.shuffle.auc` (`run_sta.py:108`) — the
shuffle is run **inside `common.confound`, on bag-of-words text features**. It certifies the
confound module, not the embedding→aggregator ladder. No shuffle control is run on the
ladder itself.

That matters because of a concrete leakage channel in the loader.
`corpora/__init__.py:_build_steps_from_judge` builds step text as:

```python
if pos < len(raw_texts):
    text = raw_texts[pos]
else:
    rec = priv_by_step.get(idx) or nonpriv_by_step.get(idx) or {}
    text = clean_text(rec.get("rationale", "")) or ("[unrendered step %d]" % idx)
```

When the judge log has more steps than the row's rendered messages, the tail steps are
embedded as the **privileged judge's own rationale** — text written *with knowledge of the
side task*, in prose that names the scheming ("unprompted autodelete rule aimed at GitHub
security alerts", per `DATASETS_VERIFIED.md`'s worked example). Embedding the judge's
explanation of the label and then predicting the label is training on the label. The
docstring frames this as "still a meaningful per-step document for an embedder", which
understates it considerably.

Worse, it is **silent**: `aligned = len(raw_texts) == n` is computed and then discarded on
line 338 (`del aligned`), so a partial fallback is never counted, never warned about, and
never reaches `results.json`. And a label-shuffle control *cannot* detect this — shuffling
destroys the association, so shuffle AUC stays at 0.5 while the real run is inflated.

Minimum fix: count `n_steps_from_rationale` per class, put it in `Corpus` /
`results.json`, warn loudly when it is non-zero, and prefer `[unrendered step N]` over the
privileged rationale. Better: use the `nonpriv_streaming` rationale only (causal, prefix-only)
if any rationale must be used at all, and treat any run with non-zero positive-class
rationale steps as contaminated.

## 3. Rigor plumbing

**Tier labelling — honest in intent, broken in the print, absent from the JSON.**
`_print_summary` declares `SCREENING TIER` and the design comments repeatedly say screening.
But the format string is
`"... (SCREENING TIER, n=%d/%d, one embedder, one seed)" % (results["seed"], results["n_folds"])`
(`run_sta.py:446-447`) — it prints the **seed** in the `n=` slot, so at the default it reads
`n=0/5`. And `results.json` has **no `tier` field at all**: `falsifier_verdicts` carries bare
`holds: true/false`. Any dashboard that reads that dict gets a verdict with no tier chip,
which is exactly what CLAUDE.md §11 forbids ("no bare numbers").

**`n` and `n_distinct_groups` — both reported. GOOD.** `Corpus.shortfall()` returns
`achieved_pos`, `achieved_neg`, `n_distinct_groups`, `shortfall`, `clears_rule_1`; all five
are printed and written to `results["sizes"]`. `step_count_summary` and `pool_fingerprint`
travel with them. This part is done properly.

**Bootstrap unit is WRONG — trajectories, not groups.** `bootstrap_auc_ci`
(`evaluate.py:64-88`) resamples `rng.integers(0, m, m)` over **trajectory** indices. The
folds are group-aware (`StratifiedGroupKFold` on `group_id`) but the CI is not. SHADE and
AgentDojo group by a hash of `main_task_description`, and several rows/epochs share one
task — the brief's own measurement is 50 AgentDojo trajectories → 16 groups. Within-group
trajectories are near-replicates; resampling them as independent understates the variance
and narrows the 95% CI by roughly `sqrt(n_traj / n_groups)` (~1.8x at 50/16). Fix is a block
bootstrap: resample **groups** with replacement and take all their trajectories. Same cost.

**The §7 four-part contract is not attempted, which is fine for screening but must be
stamped.** No paired Wilcoxon, no CI on any delta, no Holm across the 8-method family, one
seed. §7 permits that at screening tier — but F1/F2/F3 verdicts are phrased as findings
("HOLDS"), so each verdict needs `tier: SCREENING` and `n_seeds: 1` beside it, and no
`FINDINGS.md` line should be drawn from this run.

**One seed is doing more work here than usual.** `SEED` simultaneously seeds the corpus
sample, the fold split, the bootstrap, the LogisticRegression, and `torch.manual_seed`. Two
of the three methods deciding F1 (`gru`) and one deciding F3 are stochastically-initialised
neural nets given exactly one draw. A GRU AUC at n=1 initialisation is not comparable to a
convex model's, and F1 requires `gru - mean_pool > 0.02` on that single draw.

**Correct and worth crediting:**
- The lead-time arm calibrates the threshold on **TRAIN benign only**, evaluates on a
  disjoint group-aware TEST split, and streams the **uncapped** trajectory
  (`run_sta.py:371` passes `embeddings`, not `capped`) — deliberate and documented.
- The degenerate-threshold diagnostic (`evaluate.py:227-271`) is genuinely good work: it
  detects a threshold tied to the observed benign peak max, or fewer than four distinct peak
  values, prints a banner, and appends the warning **into `LeadTimeResult.oracle_note`** so it
  survives a JSON round-trip that drops the sibling dict.
- `MAX_STEPS` capping applies to the offline ladder only, is counted (`n_trajectories_capped`),
  and is written to the JSON.
- `evaluate_offline` reusing one aggregator instance across folds is safe: every `fit()`
  overwrites state, and `ESNCusum._init_reservoir` is guarded by `if self._W_res is None`, so
  the reservoir is drawn once and shared across folds — which is the *desired* fixed-reservoir
  behaviour, and only `mu`/`sigma`/`threshold` are refit per fold.

**Two smaller reporting issues:**
- `evaluate_lead_time` counts a detection when `alarm_step <= ponr`, so flagging *at* the
  point of no return counts as "detected" with lead 0. `DATASETS_VERIFIED.md` recommends the
  headline be "fraction detected **strictly before** PoNR". Report both, or use `<`.
- `false_alarm_rate` inherits the same length monotonicity as the score (§2.6), so on a corpus
  with long benigns it is inflated for reasons unrelated to the monitor's discrimination.

## 4. Citation hygiene

**The good news first: this is the strongest part of the lesson.** Every arXiv id that
appears in code — `2601.00516`, `2606.00611`, `2608.02464`, `2603.27148` — is in
`CITATIONS_VERIFIED`'s **SAFE TO CITE** table. No id outside that table is used anywhere.

Every one of the banned system-names-as-titles appears **only inside an explicit warning**:
`esn_cusum.py:5` ("the brief ... called the system 'AgentTrajectorySentinel'; that string is
NOT this paper's title and must never be printed as one"). `HORMA` and `LatentMAS` do not
appear at all. Every not-in-abstract number is disclaimed at its use site:
- `K=16` → made a **required config knob** with the `[NEEDS VERIFICATION]` note quoted
  (`sequence.py:259-266`, `config.py:132`).
- `<0.001 ms/step` → explicitly named as not-in-abstract and **not claimed** (`markov.py:10`).
- COLM → "SUBMITTED ... NOT accepted ... must never be described as a COLM paper".
- "booking agent" scoping of 52%→73% → disclaimed (`esn_cusum.py:10`).
- Both smoke tests print measured latency with "this run, this machine, NOT the paper's number".

### VIOLATION 1 (real, and it is the exact predicted failure) — `types.py:35`

```
Published TRACE numbers (F1 0.713 / 0.790) use Claude Sonnet 4.6 / Opus 4.7 as the
Reader backbone.
```

Three defects in one sentence:
1. **`0.790` appears nowhere in `CITATIONS_VERIFIED`.** The verified figures are F1 **0.713**
   and recall **0.844** (2606.07054), and **+12.6 pp** (2606.00611). `0.790` is unattributed.
2. **It conflates the two TRACEs.** F1 0.713 belongs to *TRACE: Trajectory Reasoning through
   Adaptive Cross-Step Evidence Aggregation* (**2606.07054**), which has a Triage-Inspect-Judge
   loop and **no Reader**. "Reader backbone" belongs to *TRACE: Trajectory Risk-Aware
   Compression* (**2606.00611**). `CITATIONS_VERIFIED` flags this precise hazard and instructs
   "Disambiguate by full title + id **everywhere in the lesson**" — this line does neither.
3. **"Claude Sonnet 4.6 / Opus 4.7" as the backbone is not in `CITATIONS_VERIFIED`** for
   either paper.

### VIOLATION 2 (minor) — mechanism details asserted about papers, beyond what was verified

- `markov.py:8-9` attributes to 2603.27148 a "hand-designed (data exposure, tool escalation,
  reversibility) risk-state triple". That triple is not in `CITATIONS_VERIFIED`'s record of the
  abstract. Mark `[NEEDS VERIFICATION]` or drop.
- `esn_cusum.py:16-17` claims "Only the mechanism — a fixed random reservoir feeding a
  **per-channel** CUSUM over a healthy-only null, fused by **channel-max** — is reproduced".
  The abstract records only "one-class echo-state-network ensemble with CUSUM alarms";
  per-channel-ness and channel-max fusion are our design, not verified paper content. Rephrase
  as "our reading of the mechanism".

### GAP 3 — §17 rule 4's required surfaces do not exist yet

There is **no `README.md` and no `AUDIT.md`** in the lesson directory. Rule 4 requires each
lesson README to open with a `> **Reference:**` block carrying full TITLE as a clickable
`arxiv.org/abs/XXXXX` link plus authors + venue/date + a one-line relevance note, and an
`AUDIT.md` that re-audits each id and states plainly what is reproduction vs inspired-by. All
the raw material is already written in the docstrings; it has no rule-4-compliant home. Note
also that `2601.00516` is already carried by the `trajguard` lesson —
`CITATIONS_VERIFIED` says to reuse that citation block rather than re-deriving it.

## 5. Ranked fixes

1. **Make the horizon control within-corpus.** Refactor `FirstKSteps` into a
   `Truncate(k, inner_aggregator)` wrapper and report `AUC[MaxPool@k] - AUC[MeanPool@k]` as a
   function of `k` on SHADE alone; add the SHADE-split-at-its-own-median stratification as a
   second control. Without this, F1∧F2 cannot separate horizon from corpus and the lesson's
   central claim is untestable as designed. Also soften `types.py:25-31` and
   `run_sta.py:489-491`, which currently assert the unlicensed inference.
2. **Fix the step-text leakage channel** in `_build_steps_from_judge`: stop falling back to the
   *privileged* judge rationale, restore the discarded `aligned` flag into a counted, warned,
   JSON-reported `n_steps_from_rationale` per class, and treat any positive-class hit as
   contamination. F0 cannot detect this defect.
3. **Declare SafetyDriftMonitor's third departure** — the runtime score is path-independent
   (`max_t g(cluster(step_t))`, verified bit-identical under shuffle/reversal), so the
   *composition* mechanism the paper is about is not implemented. Either say so in the
   docstring's departures list, or make `update()` actually consume `self._current_state`
   (a running absorption probability along the observed path) and restore the mechanism.
4. **Fix the citation at `types.py:35`** — drop the unsourced `0.790` and the unsourced
   backbone attribution, and split the sentence into the two disambiguated TRACE papers by
   full title + id.
5. **Price F1/F2/F3 against `confound.binding_bar`** in the verdict logic, not only in prose —
   and record that both streaming monitors' `_peak` statistic is monotone in trajectory length
   (measured: length-only AUC 0.92 / 0.90), so their AUCs are length-contaminated by
   construction.
6. **Replace the 0.02 point-estimate comparator with a paired bootstrap on the delta**
   (`AggregatorResult.scores` is already index-aligned across methods), and **re-specify F2 as
   an equivalence test** (95% CI on the paired delta inside +/-0.02) so that a noisy run can no
   longer "confirm" the control.
7. **Bootstrap by GROUP, not by trajectory,** in `bootstrap_auc_ci` — the folds are already
   group-aware and the CI is not.
8. **Stamp tier in the JSON** (`tier: SCREENING`, `n_seeds: 1` on every falsifier verdict) and
   fix `run_sta.py:446-447`, which prints the seed in the `n=` slot (`n=0/5`).
9. **Split F3 into two rows** — best-within-the-shared-classifier arm vs `LastStep` (fair), and
   best-overall including the end-to-end torch models (labelled as mixing classifier families).
   Note the max-of-8 winner's-curse bias.
10. **Write the rule-4 `README.md` + `AUDIT.md`** (reference block with linked full titles,
    authors, venue/date; reproduction-vs-inspired-by table). Reuse `trajguard`'s existing
    2601.00516 block. Add the ESN **ensemble→single reservoir** omission to the ESNCusum
    departures list, and note the one-sided-CUSUM choice.
