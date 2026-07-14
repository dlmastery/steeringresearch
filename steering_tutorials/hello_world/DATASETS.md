# Larger dataset survey — choosing the scale-up for the harmful-vs-benign probe

The starter probe (`data.py`) trains on **JailbreakBench**: 100 harmful + 100
benign behaviors (200 prompts). Fine for a smoke test, too small to say anything
real. This note surveys larger **prompt-level** safety datasets, justifies the
pick used by `data_large.py`, and — because this is a teaching artifact — spells
out the *principled sampling* methodology in full.

Requirement: **clear prompt-level binary labels** (harmful/unsafe *intent* vs
benign), both classes ideally from the **same distribution** (so a probe can't
cheat on source style), enough volume for a balanced set, and — critically —
**label-unit integrity** (the label must describe the prompt, not something else).

## Candidates (all verified against the live Hub, 2026-07-13)

| Dataset | Size | Label granularity | Both classes? | Gated | License | Verdict |
|---|---|---|---|---|---|---|
| **lmsys/toxic-chat** (`0124`) | ~10.2k prompts (train 5082 / test 5083) | **prompt-level** (`toxicity` on the user input) | Yes — 746 toxic / 9419 benign | No | CC-BY-NC-4.0 | **CHOSEN** |
| allenai/wildjailbreak | ~262k | prompt-level (`vanilla_harmful`/`vanilla_benign`) — ideal | Yes | **Yes (401)** | AI2 ImpACT | Rejected: gated, no token here |
| PKU-Alignment/BeaverTails (`30k`) | 27k rows / 7.8k prompts | **pair-level** (`is_safe` on prompt+**response**) | No genuine benign | No | CC-BY-NC-4.0 | Rejected: collapse maps refused-harmful prompts into benign (see below) |
| TrustAIRLab/in-the-wild-jailbreak-prompts | 13.2k regular / 1.4k jailbreak | prompt-level (jailbreak vs regular) | Yes | No | MIT | Rejected: "regular" split is NSFW/roleplay templates (not benign); ~2x length gap |
| jackhhao/jailbreak-classification | 1998 (1332 benign / 666 jailbreak) | prompt-level (jailbreak vs benign) | Yes | No | Apache-2.0 | Runner-up: clean but benign = generic NLP tasks (different distribution); caps at 666/class |
| AlignmentResearch/ClearHarm | 179 harmful (neg split empty) | prompt-level | No | No | MIT | Rejected: far too small |
| walledai/{AdvBench,HarmBench,StrongREJECT}, sorry-bench | 0.3–0.5k each | prompt-level, **harmful-only** | No | mixed | mixed | Rejected: single-class |

## Principled sampling (the methodology — read this part)

This is the heart of a defensible safety dataset. Five rules, each implemented in
`data_large.py` and reflected in the `header` block of `large_prompts.json`.

### 1. Label-unit integrity — *label the right thing*

The label must describe **prompt intent**. This is where naive scale-ups break:

- **BeaverTails** annotates `is_safe` on a **(prompt, response) PAIR**. The
  principled collapse to prompt level is: group by normalized prompt, keep only
  **unambiguous** prompts (**all** responses unsafe → `harmful=1`; **all** safe →
  `benign=0`), and **drop** prompts with mixed responses. We implemented and ran
  exactly this. It fails for a subtle but fatal reason: **a refusal is a "safe"
  response.** Every genuinely *harmful* prompt that the models reliably refused
  therefore has all-safe responses and collapses into the **benign** class. A
  20-prompt audit of BeaverTails' all-safe prompts found **~60%** express harmful
  intent (blackmail, harassment, slurs). Because BeaverTails prompts are sourced
  from red-teaming, there is essentially no clean benign class to recover — no
  filter fixes it, since refusals leave no harm signal on the pair. **Rejected.**

- **WildJailbreak** would need *no* collapse — its native `vanilla_harmful` /
  `vanilla_benign` types are already prompt-level. It is the ideal set, but it is
  **gated** and returns 401 without an accepted license (no HF token on this
  machine). **Unavailable here.**

- **Toxic-Chat (chosen)** annotates `toxicity` on the **user input itself** — a
  native prompt-level human label, exactly like WildJailbreak's types. **No
  collapse needed.** As a defense-in-depth guard we still drop any prompt whose
  normalized text appears under **both** labels (genuinely ambiguous); on the
  train split that count is **0** (`n_dropped_ambiguous`).

### 2. Positive-class stratification — *don't let one harm type dominate*

The harmful class is stratified across harm categories taken from each prompt's
`openai_moderation` scores (argmax category): `sexual`, `harassment`, `violence`,
`hate`, `self-harm`, `sexual/minors`, … `_stratified_sample` uses
largest-remainder allocation so a subsample spreads across categories rather than
collapsing onto the majority type. **Per-category counts are reported** in the
header. *Caveat:* at the natural ceiling (below) we take the **entire** harmful
pool, so the reported mix reflects Toxic-Chat's real distribution, which is
**sexual-heavy (~55%)** — a genuine property of this corpus. Category *balancing*
(capping the majority) is only possible by discarding data / shrinking the set;
we prefer to keep all harmful prompts and report the skew honestly.

### 3. Dedup + leakage-safe grouping — *no prompt in both train and test*

Every row carries `group_id = sha1(normalized_text)[:16]`, where normalization
lowercases and reduces to alphanumeric tokens. Exact **and** surface
near-duplicates (differing only in case/whitespace/punctuation) share a
`group_id` and are collapsed to a single row (`n_dropped_duplicates` reported).
Downstream code must do a **group-aware split on `group_id`** so a prompt and its
trivial variants can't straddle train/test. We also sample from the dataset's
**official TRAIN split** by default (`split="train"`), holding out its test set.
*(Surface near-dups only — semantic paraphrase detection needs embeddings and is
out of scope for this CPU/text-only loader.)*

### 4. Base-rate reporting — *a balanced sample is not the real world*

Toxic prompts are a **~7.6% minority** (`natural_base_rate.toxic_fraction`; 384
toxic / 5082 rows on the train split). We deliberately rebalance to **1:1** for
training, but the header records the natural prior so that **PR-AUC, calibration,
and any deployment threshold are interpreted against the real ~7% base rate**,
not the artificial 50%.

### 5. Reproducibility

`seed=0`, sampling **without replacement**, deterministic ordering, and a full
provenance `header` in `large_prompts.json` containing: `dataset`, `source_file`,
`split_used`, `natural_base_rate`, `n_dropped_ambiguous`, `n_dropped_duplicates`,
`per_category_counts`, `truncation_maxchars`, and the requested/actual per-class
counts. Each row is `{prompt, label, category, group_id, source_dataset}`.

## What the default build produces (`split="train"`, seed 0)

- **748 prompts = 374 harmful + 374 benign** (balanced 1:1).
- Natural base rate on the split: **7.56%** toxic (384 / 5082).
- Dropped: **160** surface duplicates, **0** cross-label ambiguous.
- Harm categories in the positive class: **9** (`sexual` 205, `harassment` 69,
  `violence` 63, `hate` 17, `sexual/minors` 9, `self-harm` 8, plus 3 rare tails).
- **Ceiling:** the harmful class caps the balanced set at **374/class** on the
  official train split (toxic is a minority). This is under the 750/class target;
  it is the honest, leakage-safest ceiling. `split="all"` (train+test, group-aware
  split via `group_id`) raises it to ~695/class if more volume is needed.

## Swapping datasets later

Changing the training set is a **one-line import swap** in `train_probe.py`:

```python
# from .data import load_safety_dataset        # JailbreakBench (200 prompts)
from .data_large import load_large_dataset      # Toxic-Chat (748 prompts, train split)
```

Both loaders return the same `(prompts: list[str], labels: list[int])` contract
(1 = harmful, 0 = benign), so nothing downstream changes. A different dataset is
just a different `load_*` function behind the same signature. `build_dataset(...)`
additionally returns the per-row `group_id`/`category` and the provenance header
for group-aware splitting and stratified evaluation.
