# `common/` — the shared ≥500/class steering dataset

Every steering tutorial imports its prompts from **`steering_tutorials.common.data`**
instead of building its own. This guarantees all lessons train, steer, and evaluate
on the **same, larger, leakage-safe** prompt set — a minimum of **500 harmful + 500
benign** prompts, versus the 100/class JailbreakBench toy set the lessons started on.

The module is **CPU-only**: it downloads and parses CSV/parquet and returns Python
lists of strings. It never loads an LLM.

```python
from steering_tutorials.common.data import load_harmful_benign, load_concepts

d = load_harmful_benign(n_per_class=500)   # {"harmful": [str], "benign": [str]}
c = load_concepts(n_per_concept=120)       # per-harm-category concept sets + baseline
```

---

## 1. Sources

| Role | Dataset | Label unit | Ungated? |
|---|---|---|---|
| **Primary (all 500 harmful + all 500 benign)** | `lmsys/toxic-chat` @ `0124`, config `toxicchat0124`, **train + test** | `toxicity` (0/1), **human, on the user input** | yes |
| Harmful top-up (ONLY if `n_per_class` > toxic pool) | `JailbreakBench/JBB-Behaviors` `data/harmful-behaviors.csv` (`Goal`) | curated, **short** harmful behaviors | yes |

In the default `n_per_class=500` config **both classes come 100 % from toxic-chat**
— the JBB top-up does not run. It exists only as a length-windowed fallback if you
request more harmful prompts than toxic-chat's ~693 unique toxic pool holds.

> **`TrustAIRLab/in-the-wild-jailbreak-prompts` is deliberately NOT used.** Those
> are long DAN/roleplay templates; mixing them into a short-prompt harmful class
> lets a probe separate on **length/style, not intent**. An earlier draft blended
> them in and produced exactly that confound — it was removed. The loader has no
> code path to them.

**Why toxic-chat is primary.** A prompt classifier / a steering-contrast set needs a
label on the **prompt's intent**. Toxic-chat's `toxicity` flag is annotated by humans
on the user input itself, so it is already prompt-level — no response→prompt collapse
is required. (Collapsing response-labelled sets like BeaverTails to prompt level is
fatally biased: a *refusal* scores "safe", so every harmful prompt the model refused
leaks into the benign class. We therefore do not use response-labelled sets here.
This is the same reasoning documented in `hello_world/data_large.py`.)

---

## 2. Sizes and blend (toxic-chat-dominant)

`load_harmful_benign(n_per_class=500)` returns **500 harmful + 500 benign** (seed 0),
**both 100 % from toxic-chat**:

| Source | Count in the 500 harmful |
|---|---|
| `lmsys/toxic-chat@0124` (`toxicity==1`) | **500** |
| JBB top-up | 0 (not needed) |

We use the **full toxic-chat set** (train + test, `..._all.csv` = config
`toxicchat0124` both splits, ~10 165 rows / 746 toxic). The toxic class is a ~7 %
minority, so honoring only the official *train* split would cap the unique toxic
pool at ~300 (< 500) and force a top-up. The full set yields **~693 unique toxic**
prompts after dedup, so the harmful class is **toxic-chat toxic ONLY** — no top-up,
no long-DAN-template, no length confound.

Leakage safety is not lost by merging train+test: every row carries a `group_id`
(§3), so any **group-aware** train/eval split a lesson makes still keeps a prompt and
its near-duplicates on one side of the cut.

The **benign** class is drawn from toxic-chat non-toxic (8k+ available) and is
**length-matched** to the harmful class (§3a).

The header saved to `artifacts/dataset_500.json` records `primary_source`,
`per_source_counts_sampled`, `topup_log`, `natural_toxic_rate`, `dedup_dropped`,
`dropped_ambiguous`, `per_category_counts_harmful`, `median_char_length`,
`max_chars`, and `seed`.

---

## 3. Dedup, leakage-safety, and base rate

- **`group_id = sha1(normalized_text)[:16]`** on every row. Normalization lowercases
  and keeps alnum tokens only, so exact **and surface near-duplicates** (differ only
  in case / whitespace / punctuation) collapse to one row. Use `group_id` for
  group-aware train/eval splits so a prompt and its trivial variants never straddle
  the split.
- **No `group_id` spans both classes** — the smoke test asserts this. Any normalized
  text seen under both toxicity labels in toxic-chat is dropped as *ambiguous*.
- **Base rate is preserved for reporting.** We rebalance to 1:1 for training, but the
  natural prior is the **7.34 % minority** toxic rate (746 / 10165 in the full set) —
  read PR-AUC and calibration against that, not the balanced 50/50 sample.
- **Truncation:** prompts longer than **`max_chars = 2000`** are truncated (not
  dropped), so a rare wall-of-text prompt still contributes.

### 3a. Length-confound control

A probe must read **intent, not length**. We therefore draw the benign class
**length-matched** to the harmful class: harmful lengths are binned into deciles and
the benign sample is allocated across those same bins (`_length_matched_sample`). The
JBB top-up (if it ever runs) is additionally **length-windowed** to the [10th,90th]
percentile of the toxic-chat toxic length range. The classes end up statistically
inseparable on length across the WHOLE distribution, not just the median (seed 0):

| Class | mean | median | p10 | p90 | max |
|---|---|---|---|---|---|
| harmful | 382 | 165 | 28 | 1292 | 1536 |
| benign  | 382 | 166 | 28 | 1295 | 1536 |

**Length-AUC = 0.501** (0.5 = length carries zero class signal; a Mann–Whitney/AUC
of "predict class from char length"). `header.median_char_length` records the two
medians so the leakage/length audit can re-check the confound at a glance.

---

## 4. Concepts (`load_concepts`)

For the multi-intent / stacking lessons, `load_concepts` derives **K harm concepts**
from each toxic-chat prompt's `openai_moderation` scores, folded from OpenAI's fine
categories into five coarse buckets:

`sexual` · `harassment` · `violence` · `hate` · `self_harm`

Each toxic prompt is assigned to its **argmax coarse bucket**. Each concept is split
into disjoint **`exemplars` (40 %) / `steer` (30 %) / `eval` (30 %)** so a steering
vector is never graded on the prompts that built it. A single shared **benign
baseline** (the common contrast origin) is returned once.

Per-concept availability is **honestly capped** and reported — the harm categories
are naturally imbalanced (seed 0, full set):

| Concept | Available (unique, deduped) |
|---|---|
| `sexual` | 388 |
| `harassment` | 143 |
| `violence` | 111 |
| `self_harm` | 27 |
| `hate` | 24 |

`sexual` is by far the largest; `hate` and `self_harm` are small — treat their eval
numbers as low-power. The **held-out concept for zero-shot transfer** defaults to the
**second-largest pool** (`harassment` here): large enough to evaluate, while keeping
the richest concept available for training. Override with `held_out=`.

---

## 5. Reproduce / verify

```bash
python -m steering_tutorials.common.data
```

Prints the harmful/benign counts (asserts each ≥ 500), the per-source blend, the
top-up log, the per-category harmful counts, the per-concept availability, and two
example prompts per class; asserts no `group_id` spans both classes; and (re)writes
`artifacts/dataset_500.json`. Deterministic under the fixed seed.
