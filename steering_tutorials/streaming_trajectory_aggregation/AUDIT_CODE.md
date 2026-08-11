# STA lesson — code audit (bug hunt)

Auditor: sta-code-audit agent. Target: `steering_tutorials/streaming_trajectory_aggregation`.
Status: COMPLETE. Every finding marked CONFIRMED was reproduced with a CPU-only script
under `scratchpad/t*.py`. No file in the repo was edited; no git command was run.

## Bug-class checklist (from lead)

| # | class | verdict |
|---|---|---|
| 1 | Cache keyed weaker than contents | **CONFIRMED CRITICAL** — corpora cache (CONFIRMED-3). The embed cache is correct except `dim` (item 11). |
| 2 | Train/test leakage (scaler / centroid / tfidf / kmeans / cusum-null) | **CLEAN** for every method; the confound BAR is group-blind (CONFIRMED-4). |
| 3 | Group leakage in CV | **CLEAN** — measured zero straddles, degrades safely below 16 groups. |
| 4 | Replace/assert that matches nothing | **CONFIRMED CRITICAL** — `aligned` computed then `del`'d (CONFIRMED-1). |
| 5 | Directionless AUC | **CLEAN** on the bar side; F0 folds where it should not (CONFIRMED-6). |
| 6 | Off-by-one step indexing | **CLEAN** on convention (all 0-based, `base`-shifted consistently) — but `base` is detected per-row and the shifted range is unbounded (CONFIRMED-2). |
| 7 | `-1` sentinels leaking as data | **CONFIRMED CRITICAL** (CONFIRMED-2). |
| 8 | `is_causal` lying | **CLEAN**. |
| 9 | `state_bytes()` constant not measured | **CLEAN** — measured; scales with configured state size. |
| 10 | Degenerate/empty input paths | **CLEAN** — all raise loudly. |

---

## CONFIRMED-1 — CRITICAL — step TEXT is silently misaligned to step LABELS whenever the rendered message count exceeds the judge step count

**File:** `corpora/__init__.py:297-339` (`_build_steps_from_judge`), specifically `:317` and `:338`.

The docstring promises:

> "Text: aligned 1:1 to the raw rendered messages **when the counts match**; otherwise
> falls back to the judge record's own `rationale` text ... since no verified alignment
> exists for the mismatched case -- **fabricating one would be inventing data**."

The code computes exactly that guard —

```python
aligned = len(raw_texts) == n          # :317
...
del aligned  # (kept for readability of the branch above; not otherwise used)   # :338
```

— and then **throws it away**. `aligned` is never read. The only fallback that survives is
`if pos < len(raw_texts)` (`:322`), which fires only when `raw_texts` is *shorter* than the
judge span. When `raw_texts` is **longer** (the normal case: `messages` contains
system/user/tool turns while `step_analyses` covers only the analysed steps) every judge
step is silently bound to `raw_texts[pos]` — the first `n` messages, in order — which is
precisely the fabricated alignment the docstring refuses.

This is the §18.8 pattern verbatim: a guard that "matches nothing and passes".

**Reproduced** (`scratchpad/t1.py`): 7 rendered messages, 3 judge steps →

```
idx=0 on_path=True  text='system: you are...'     (its own rationale is 'R0')
idx=1 on_path=False text='user: do the task'
idx=2 on_path=True  text='assistant: step A'
```

The label `on_path=True` for judge step 0 is attached to the **system prompt**. Every
per-step label, every embedding, and therefore every aggregator AUC on SHADE/AgentDojo is
computed over text that has no verified correspondence to the labels. The trajectory is
also truncated to `n` steps, discarding messages `n..len(raw_texts)-1` with no warning.

Secondary mismatch on the *other* branch: when `len(raw_texts) < n`, the code produces a
**hybrid** — the first `len(raw_texts)` steps take raw text and the remainder take
`rationale`. The docstring describes an all-or-nothing fallback.

**Fix:** restore the guard.

```python
aligned = len(raw_texts) == n
...
if aligned and pos < len(raw_texts):
    text = raw_texts[pos]
else:
    rec = priv_by_step.get(idx) or nonpriv_by_step.get(idx) or {}
    text = clean_text(rec.get("rationale", "")) or ("[unrendered step %d]" % idx)
```

and record `aligned` / `len(raw_texts)` / `n` on the corpus so `results.json` states what
fraction of trajectories used the rationale fallback. A run where most rows are
mis-aligned must be visible, not inferred.

---

## CONFIRMED-2 — CRITICAL — the `-1` benign sentinel leaks into `point_of_no_return_step` and into lead-time arithmetic

**File:** `corpora/__init__.py:370-377`.

`candidate_ideal_flagging_step` is guarded (`:372`, `int(ideal_raw) >= 0`), matching
`types.py:109` ("-1 in the raw data for benign; None here"). `point_of_no_return_step` at
`:377` has **no such guard**:

```python
ponr = (int(ponr_raw) - base) if isinstance(ponr_raw, (int, float)) else None
```

**Reproduced** (`scratchpad/t2.py`):

| row | ideal_raw | ponr_raw | stored ideal | stored ponr | `available_lead` |
|---|---|---|---|---|---|
| benign | -1 | -1 | None | **-1** | None |
| positive, no PONR adjudicated | 2 | -1 | 2 | **-1** | **-3** |
| 1-based judge log | -1 | -1 | None | **-2** | None |

Failure scenario (`evaluate.py:279-289`): a positive whose adjudicator returned the
sentinel has `available_lead is not None`, so it **is counted in
`n_positives_evaluated`**, but the detection test `int(alarm_step) <= int(ponr)` with
`ponr = -1` can never be satisfied. Every such trajectory is silently booked as an
undetected positive. The reported detection rate `n_detected / n_positives_evaluated` is
deflated by exactly the number of sentinel rows, and `available_lead` carries negative
nonsense (`-3` above). `-1` also survives into `types.py:126`'s subtraction and into any
downstream percentile.

`ponr` is likewise never bounded above by `n_steps - 1`.

**Fix:** mirror the ideal guard and add the upper bound:

```python
ponr = None
if isinstance(ponr_raw, (int, float)) and not isinstance(ponr_raw, bool) and int(ponr_raw) >= 0:
    shifted = int(ponr_raw) - base
    if 0 <= shifted < len(steps):
        ponr = shifted
    else:
        eprint("[%s] row %d: ponr %d out of range [0,%d) -- dropped"
               % (loader_name, row_index, shifted, len(steps)))
```

and count the drops so the shortfall is visible in `results.json`.

---

## CONFIRMED-3 — CRITICAL — the corpora disk-cache fingerprint is TAUTOLOGICAL; it can never detect staleness

**File:** `corpora/__init__.py:97-143` (`load_cache` / `save_cache` / `cached_build`).

`save_cache` stores `uids_sha256 = uids_fingerprint([t.uid for t in trajectories])` inside
the file. `load_cache` then recomputes that same hash **from the trajectories it just read
out of that same file** and compares. The two are equal by construction. The check can only
ever fire if someone hand-edits the JSON.

The package docstring (`:14-17`) claims otherwise:

> "Validity is asserted from a STORED HASH OF THE SAMPLED UIDS, recomputed and compared on
> every load -- **never from row count alone**. Section 18.8 names row-count-only cache
> validation as the exact defect that bit `cross_trajectory` and `multiturn_jailbreak`."

It is in fact *weaker* than row-count validation against the live corpus, because nothing
about the corpus-to-be-built enters the comparison at all. Contrast `embed.py:276-285`,
which recomputes `corpus_texts_fingerprint` **from the live corpus** and refuses on
mismatch — that one is correct.

**Reproduced** (`scratchpad/t7.py`): wrote a cache with step text `WRONG_TEXT-*`, then
called `cached_build` with a builder that returns `FIXED_TEXT-*`:

```
builder invoked?  0     (0 == stale cache served)
step text served: WRONG_TEXT-0-0
```

`embed.py`'s fingerprint over the same two corpora differs (`17028d3c...` vs `aebf3c6c...`)
— proving the content genuinely changed and only the corpora-layer guard failed to notice.

**This compounds CONFIRMED-1 and CONFIRMED-2 directly.** Fixing `_build_steps_from_judge`
or the `ponr` guard will NOT invalidate any cache already on disk under
`artifacts/corpora/`; the next run silently reloads the old, wrong trajectories and the
`results.json` will look freshly computed.

**Fix:** put a schema/builder version in the cache KEY and hash the CONTENT, not the identity.

```python
CACHE_SCHEMA_VERSION = 2          # bump on ANY change to the builder's output

def cache_path(loader_name, hf_ids, n_per_class, seed):
    key_src = "%s|%s|%d|%d|v%d" % (loader_name, "+".join(sorted(hf_ids)),
                                   int(n_per_class), int(seed), CACHE_SCHEMA_VERSION)
```

and store a hash over the **step texts + labels + oracle fields**, not the uids. Delete
every file under `artifacts/corpora/` before the next run regardless.

---

## CONFIRMED-4 — HIGH — the confound bar is group-BLIND while every method AUC is group-AWARE

**Files:** `evaluate.py:336-377` (`confound_bars`) -> `common/confound.py:152-216`
(`content_bar`: `folds = [idx[i::n_folds] ...]`, no `groups` argument anywhere in the module).

`evaluate_offline` splits with `StratifiedGroupKFold` on `Trajectory.group_id`. The bar that
every method is priced against uses plain shuffled K-fold on the same rows. On SHADE and
AgentDojo `group_id` is `sha1(main_task_description)` — several rows share one task, so
near-duplicate text straddles the bar's folds but not the methods'.

**Reproduced** (`scratchpad/t8.py`, 20 tasks x 10 epochs, label constant per task):

```
content bar (GROUP-BLIND, as coded)  = 1.0000
binding_bar reported to falsifiers   = 1.0000 (content)
shuffle control                      = 0.5242
```

Two consequences:

1. F4 (`run_sta.py:147-156`) prices both streaming monitors against a bar produced under a
   *different, leakier* protocol. "Margin over bar" is not like-for-like, and the error is
   unbounded — here the bar was driven to a perfect 1.0.
2. **F0 cannot catch this.** Permuting labels destroys the group-label correlation, so the
   shuffle control sat at 0.5242 and F0 would report HOLDS while the bar was fully leaked.
   F0's own text ("that is leakage, not signal, and invalidates every other falsifier
   below") therefore gives false assurance against exactly this failure.

**Fix:** thread groups through. Add an optional `groups=` to `content_bar` /
`shuffle_control` / `confound_report` and use `GroupKFold` when supplied; pass
`[t.group_id for t in corpus.trajectories]` from `confound_bars`. Record both the
group-aware and group-blind bars so the gap is visible.

---

## CONFIRMED-5 — HIGH — per-trajectory `score()` failures are swallowed, so methods are compared at different denominators

**Files:** `evaluate.py:177-187` (`except Exception: ... continue` around `agg.score`) and
`run_sta.py:89-90` (`_auc_by_method` reads only `r.auc`).

A failing `score()` is logged to stderr and left `NaN` in `pooled_score`; the AUC is then
computed over whatever survived. `n_scored` is recorded in the `notes` **string** — and
nothing ever reads it. F1/F2/F3 compare `r.auc` values directly.

**Reproduced** (`scratchpad/t8.py`, one aggregator failing on every third call):

```
mean_pool  auc=0.9771  notes=... n_scored=200/200 ...
flaky      auc=0.9746  notes=... n_scored=134/200 ...
```

Two AUCs over different subsets, ranked head-to-head, feeding a pre-registered verdict.
Reachable without exotic failures: a torch OOM inside `GRUAggregator.fit` on one fold
(`evaluate.py:171-176`) increments `skipped_folds` and drops that fold's whole test split
from the GRU's denominator while `MeanPool` keeps all of it — and GRU-vs-MeanPool is
literally what F1 and F2 turn on.

**Fix:** carry `n_scored`/`n` as real fields rather than prose, print an explicit
`INCOMPLETE` warning, and have `_falsifier_verdicts` return `applicable: False` (with the
reason) when the two methods it is about to compare have different `n_scored`.

---

## CONFIRMED-6 — MEDIUM — F0 reads the FOLDED shuffle AUC, so half its pre-registered band is unreachable, and pure noise fails it

**Files:** `run_sta.py:108-113`; value originates at `common/confound.py:216`
(`"auc": directionless(raw)`).

The pre-registered statement (`config.py:166-169`) is "Label-shuffle AUC ... lands in
**[0.45, 0.55]**". The code reads `...["shuffle"]["auc"]`, which is `max(auc, 1-auc)` and is
therefore **>= 0.5 by construction**. `[0.45, 0.50)` is dead code.

Worse, folding turns a symmetric null into a positively-biased one.
**Reproduced** (`scratchpad/t3.py`, 30 pure-noise runs at n=120):

```
folded shuffle auc: min=0.5011 max=0.7050 mean=0.5568   (any < 0.5? False)
raw    shuffle auc: min=0.2950 max=0.5600
runs OUTSIDE [0.45,0.55]: 14/30
```

On pure noise F0 reports FAILS almost half the time at n=120 — and its own text says a
failure "invalidates every other falsifier below". At full SHADE scale (n~1000) the sampling
sd is ~0.018 so a spurious failure is rare, but the defect bites at any reduced
`STA_N_PER_CLASS` and on the smaller admitted pools.

**Fix:** test the RAW value against the symmetric band; keep the folded one as a separate
reported line.

```python
sh = (confound.get("full_trajectory") or {}).get("shuffle") or {}
raw = sh.get("auc_raw")
out["F0_shuffle_control"] = {
    "shuffle_auc_raw": raw, "shuffle_auc_directionless": sh.get("auc"),
    "band": [0.45, 0.55],
    "holds": bool(isinstance(raw, float) and 0.45 <= raw <= 0.55),
}
```

---

## CONFIRMED-7 — MEDIUM — F4 reports FAILS when it was never evaluated

**File:** `run_sta.py:153-156`.

```python
"holds": bool(cells) and all(c["clears"] for c in cells.values()),
```

If both streaming monitors were skipped (e.g. `ESNCusum.fit` raised "requires at least one
benign trajectory" on every fold), `cells == {}` and `holds` is `False`. `_print_summary`
(`:483-487`) prints `FAILS`. A falsifier that returns a substantive negative verdict for a
cell that was never measured is worse than none.

**Reproduced** (`scratchpad/t6.py`): `F4: {'binding_bar': 0.6, 'cells': {}, 'holds': False}`.

**Fix:** `if not cells: out["F4_streaming_clears_bars"] = {"applicable": False, "reason": "no streaming monitor produced a finite AUC"}`. The same guard belongs on F3 when `last_step` is NaN.

---

## CONFIRMED-8 — MEDIUM — a bare `NaN` token is written into results.json

**File:** `run_sta.py:120-122`, `margin_vs_maxpool`.

The guard is `if margin_ok(...) or (isinstance(max_pool, float) and isinstance(mean_pool, float))`.
`float('nan')` passes `isinstance(..., float)`, so a NaN AUC (reachable per CONFIRMED-5)
yields `NaN` for the margin. Every other numeric path in this file goes through `_num()`
(`:52-56`, NaN -> null); the falsifier block does not.

**Reproduced** (`scratchpad/t6.py`): `'margin_vs_maxpool': nan`, and `json.dumps` emits the
bare token `NaN` — **invalid strict JSON**. Any dashboard doing `JSON.parse` on
`results_shade.json` throws.

**Fix:** map the block through `_num` (`{k: (_num(v) if isinstance(v, float) else v) ...}`)
and use the NaN-safe `a == a` guard F3 already uses.

---

## CONFIRMED-9 — MEDIUM — `SafetyDriftMonitor`'s score is MEMORYLESS; `_current_state` is dead, and `state_bytes()` counts it

**File:** `streaming/markov.py:100-119`.

```python
s = self._discretize(step_vec)
self._current_state = s          # assigned here and NEVER read anywhere
score = float(self._absorb_prob_table[s])
```

The running score is a pure function of the CURRENT step's cluster. The module title
("cumulative safety state as an ABSORBING MARKOV CHAIN") and `types.py`'s streaming framing
both imply state that accumulates; it does not. The chain algebra lives entirely in the
lookup table built at fit time. `state_bytes()` (`:113-119`) then counts `_current_state` as
part of the footprint — a variable with no effect on any output.

Downstream consequence, **reproduced** (`scratchpad/t9.py`, 60 benign 131-step trajectories,
`n_states=8`):

```
absorb_prob_table: 8 distinct values [0.5306 0.5106 0.5626 0.5515 0.5205 0.4990 0.5726 0.5418]
per-trajectory PEAK: 1 distinct value [0.572648]     <-- every benign trajectory saturates
threshold(p95)=0.572648  max_peak=0.572648  tied_at_max=True
offline AUC of the peak score = 0.1333
```

Over a long horizon every trajectory visits the riskiest cluster, so `max` over steps is the
table maximum for essentially all of them. `evaluate.py:242-247`'s `tied_at_max` /
`low_resolution` diagnostic **does** catch the alarm side — credit where due, that check is
correct and loud. The offline-AUC side is not flagged; F4 reads 0.13 and reports FAILS,
which is honest but uninformative.

**Fix:** either delete `_current_state` and correct the docstrings to "memoryless per-step
risk lookup", or make the score genuinely cumulative (carry a running belief `b_t = b_{t-1} Q`
and score `1 - sum(b_t)`), which is what the absorbing-chain framing actually describes.

---

## Lower-severity confirmed items

| # | sev | file:line | issue |
|---|---|---|---|
| 10 | LOW | `evaluate.py:124-131` | `group_aware_split(test_frac=0.30)` returns a 0.373 test share (`n_splits = round(1/0.30) = 3`). Measured 0.30->0.373, 0.20->0.247, 0.15->0.127. Record the ACHIEVED fraction in `results.json`. |
| 11 | LOW | `embed.py:281` | `ok = all(got_meta.get(k) == want[k] for k in want)` — `want` has no `dim` key (added only at `:300`), so the cached `dim` is written but never validated. Same `embedder_id` at a different output width silently reuses old-width vectors. |
| 12 | LOW (operational) | `embed.py:240-253` | `embed_corpus` passes EVERY step text of the whole corpus to one `encode()` call. SHADE at `N_PER_CLASS=500` is ~131,000 texts -> a single `[131000, 768] float32` = 402 MB, plus `np.vstack` in `_save_pack` doubling it. On this host (CLAUDE.md 18.6: commit limit, not free RAM, is binding) this is the likeliest place a run dies with `OSError 1455`. Chunk it. |
| 13 | LOW | `run_sta.py:141-145` | F3's `best` is `max` over ALL main-ladder AUCs including `last_step` itself, so when `last_step` wins the record reads `best_auc == last_step_auc`, `margin=0.0`, `holds=False`. Correct outcome, confusing record. Exclude `last_step` from the `best` pool. |
| 14 | LOW | `pooling.py:216-221` | `DeviationWeighted` softmaxes UNNORMALISED L2 distances at fixed `temperature=1.0`, so its sharpness is a function of `dim` and embedding norm scale, not of the design. Measured cos(DeviationWeighted, MeanPool): 0.852 at dim=8/10 steps, 0.779 at dim=128/131, 0.782 at dim=768/131. **My initial hypothesis that it saturates into an argmax-pool was WRONG** — it stays fairly diffuse (effective 90 of 131 steps). Still worth z-scoring the distances within a trajectory so the arm's behaviour is not a property of the embedder's scale. |

---

## Bug classes that came back CLEAN (verified, not assumed)

* **Class 3 — group leakage in CV.** `evaluate.py:103-121`. Measured **zero** group straddles
  at `n_groups` in {2, 3, 5, 16, 40} against `n_folds=5`; `k` is correctly clamped to
  `min(n_folds, n_groups, max(minority,2))` and floored at 2, so it degrades safely well
  below the measured 16-group AgentDojo figure. (`scratchpad/t5.py`.)
* **Class 5 — directionless AUC.** `common/confound.py` folds correctly everywhere
  (`length_bar`, `count_bar`, `content_bar` each return `auc = directionless(raw)` alongside
  `auc_raw`), and `evaluate.py:357-362` reads the folded `["auc"]` key for both the full
  report and the first-step rival. Method AUCs are correctly reported RAW — a method
  predicting backwards is a broken method, not a confound, so no fold belongs there.
* **Class 8 — `is_causal` honesty.** Under `types.py:243`'s definition ("score(prefix) never
  depends on future steps"), `MeanPool`/`MaxPool`/`MeanMaxStdPool`/`LastStep`/`FirstKSteps`/
  `GRUAggregator` are all prefix-only, and the two `False` flags are correct and documented:
  `QueryTokenCompressor` genuinely uses unmasked `nn.MultiheadAttention`, and
  `DeviationWeighted` is conservatively marked `False` with an explicit rationale. Both
  streaming `update()` methods read only `x_t` and their own prior state.
* **Class 9 — `state_bytes()` measures real state.** Not a constant: it scales with the
  configured state size. Measured ESNCusum 264 / 1032 / 1608 bytes at `reservoir_dim`
  16 / 64 / 100, flat between step 10 and step 300 in every case; SafetyDrift flat at 20
  bytes for `n_states` 4/8/32. Both correctly exclude the fitted MODEL (reservoir weights,
  k-means centroids) — symmetric between the two, and per-trajectory state is what the
  Protocol asks for. (`scratchpad/t4.py`.)
* **Class 10 — degenerate inputs raise loudly.** A zero-step trajectory raises
  (`LastStep` -> `IndexError`, `MeanPool` -> sklearn `ValueError: Input X contains NaN`);
  single-class `fit` raises with an explicit "do not silently degrade to a constant
  predictor" message (`pooling.py:95-100`, `sequence.py:87-91`); an all-failed method returns
  `auc=nan` with `n_scored=0/20`, never 0.5. (The *reporting* of partial failure is the
  defect — CONFIRMED-5 — but the failures themselves are loud.)
* **`results.json` is written BEFORE any summary print.** `run_sta.py:432-436` (write) then
  `:438` (`_print_summary`). Plots are attempted earlier but each is individually
  try/except-wrapped, so a plotting failure cannot cost the data.
* **Train-only fitting (class 2).** Every fitted object is fit inside `fit()` on the fold's
  train split only: `StandardScaler`+`LogisticRegression` (`pooling.py:104-107`),
  `DeviationWeighted.centroid_` (benign train steps, `pooling.py:199-213`), `content_bar`'s
  vocabulary/IDF/centroids (`confound.py:184-208`), `SafetyDriftMonitor`'s k-means +
  transition matrix + threshold (`markov.py:125-174`), `ESNCusum`'s CUSUM null + threshold
  (`esn_cusum.py:140-172`). Both streaming monitors additionally filter to `label == 0`
  before estimating anything, so neither can be fit on the class it detects. The split
  feeding them is group-aware. **No fit-on-all found.** (The one caveat is CONFIRMED-4: the
  confound BAR's internal CV is train-only per fold but group-blind.)

---

## NOT VERIFIABLE without a live run

`_normalize_manifest` (`corpora/__init__.py:210-254`) guesses the manifest wrapper key from a
list of seven candidates and, failing that, tests whether the manifest *is* the row map by
checking that the first five dict values carry a `"status"` key. If the real
`label_manifest.json` nests the row map under an eighth name it raises loudly (correct). But
if it uses one of the seven names for something that is NOT the row map, the mis-parse is
silent and every row is dropped as "not kept" — indistinguishable from an empty corpus. Only
a live fetch settles it. **This is also the single highest-risk unverified item**, because
`_build_steps_from_judge`'s alignment defect (CONFIRMED-1) can only be sized against the
real ratio of `len(messages)` to judge-step count, which no cached artifact currently on
disk can tell me.
