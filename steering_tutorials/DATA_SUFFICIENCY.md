# DATA_SUFFICIENCY.md — data-sufficiency audit across all steering tutorial lessons

All numbers below were produced by **calling the loaders on CPU** (`common.data.load_harmful_benign`,
`common.data.load_concepts` at each lesson's `N_PER_CONCEPT`), plus reading every lesson's
`config.py` / `data.py`. No model is loaded; no code was changed. Seed 0, full toxic-chat split.

---

## 0. The ceilings (what is actually available)

Every lesson draws from **one shared pool**: `steering_tutorials.common.data`, sourced from
`lmsys/toxic-chat@0124` (train+test), deduped by `group_id`, length-matched.

| Pool | Unique count (deduped, seed 0) |
|---|---|
| **Harmful (toxic) total** | **693** (from 746 raw toxic / 10 165 rows → natural base rate **7.34 %**) |
| Benign total | **8 889** |
| Harmful + JBB top-up (only if requested > 693) | 693 + up to 99 JBB = 792 |

**Per moderation category** (the concept ceilings — this is where the problem is):

| Concept | Unique available |
|---|---|
| `sexual` | **388** |
| `harassment` | **143** |
| `violence` | **111** |
| `self_harm` | **27** |
| `hate` | **24** |

`hate` and `self_harm` are **hard ceilings of 24 and 27** — no config can extract more; they
sum to <10 % of the toxic pool. After the loader's 40/30/30 exemplar/steer/**eval** split, their
eval slice is **7–8 prompts** at any `N_PER_CONCEPT`.

---

## 1. Sufficiency table (every lesson)

Threshold rationale: a binary harmful/benign **eval split < ~30/class** or a per-concept
**eval < ~30** is too small even for a screening rate (a 7–10-prompt eval has a ±~15 pp
95 % CI — pure noise). "Extract < ~30" also means the diff-of-means / probe vector itself is
built on too little signal.

| Lesson | Source (loader) | Extract n | Eval n | Per-concept counts (extract / **eval**) | Verdict | Fix |
|---|---|---|---|---|---|---|
| **common/** | toxic-chat@0124 + JBB fallback | — | — | pool: sexual 388, harassment 143, violence 111, self_harm 27, hate 24 | n/a (foundation) | Ceilings are real; concept lessons must respect them (below). |
| **hello_world** | `load_harmful_benign(750)` | 750/cls (693 toxic + 99 JBB) | 5-fold CV (~150/cls per fold) | binary | **SUFFICIENT** | None. Strongest data footing in the course. |
| **probe_tuning** | inherits hello_world features | 750/cls | CV | binary | **SUFFICIENT** | None (CV model selection, no test peeking). |
| **hello_world_steering** | `load_harmful_benign(250)` | 200/cls | 50/cls | binary | **SUFFICIENT** (screening) | Optional: raise eval to ≥100/cls (pool allows 300 held-out). |
| **reft_r1** | `load_train_eval(250, n_eval=50)` | 200/cls | 50/cls | binary | **SUFFICIENT** (screening) | Optional bump to 100/cls eval. |
| **realignment** | `load_harmful_benign(250)` | 200/cls | 50/cls | binary | **SUFFICIENT** (screening) | None required. |
| **rogue_scalpel** | `load_harmful_benign(250)` (aligned model) | 200/cls | 50 harmful | binary | **SUFFICIENT** (screening) | ASR on 50 harmful is OK for a screen; raise to 100 for a claim. |
| **stacking** | `load_harmful_benign(250)` | 200/cls | 50 harmful; **norm-budget on 6** | binary | **THIN** | Bump `N_NORM_BUDGET_PROMPTS` 6 → ≥20 (geometry mean on 6 is noise); eval 50 fine. |
| **fine_grained** | `load_harmful_benign(500)` | 200/cls | 60/cls | binary | **SUFFICIENT** | None. |
| **non_identifiability** | `load_harmful_benign(500)` | 150/cls | 40 harmful | binary | **THIN** | Bump `N_EVAL` 40 → ≥50 (huge headroom: 150+40 ≪ 500). |
| **contextual_steering** | `load_harmful_benign(500)` | 200/cls | **25/cls (capped)** | binary | **INSUFFICIENT** | Raise `N_EVAL_PER_CLASS` 25 → ≥50 (300/cls held out; only cost is generation time). |
| **multi_intent** | `load_concepts(n_per_concept=150)` | see cells | see cells | sexual 105/**45**, harassment 100/**43**, violence 77/**34**, self_harm 19/**8**, hate 17/**7** | **INSUFFICIENT** | Drop `self_harm` (27) & `hate` (24) from the K-ladder; keep sexual/harassment/violence (all ≥100 avail, eval ≥34). |
| **flas** | `load_concepts(n_per_concept=120)` | see cells | see cells | sexual 84/**36**, harassment 84/**36**, violence 77/**34**, self_harm 19/**8**, hate 17/**7** | **INSUFFICIENT** | Same: restrict trained + held-out concepts to sexual/harassment/violence; `harassment` (held-out, 36 eval) is fine, but `hate`/`self_harm` rungs are noise. |

---

## 2. The concrete problem — concept lessons stack noise onto the ladder

`load_concepts` caps each concept at `min(N_PER_CONCEPT, available)` then splits **40 % exemplars /
30 % steer / 30 % eval**. For the two smallest harm categories the split is fatal **regardless of
`N_PER_CONCEPT`**, because the pool itself is the binding constraint:

- **`hate`** (24 available): exemplars **10**, steer **7**, **eval 7**.
- **`self_harm`** (27 available): exemplars **11**, steer **8**, **eval 8**.

So in **multi_intent**, the K=1…5 additive ladder's last two rungs build their diff-of-means
vector on **17 / 19 prompts** and grade it on **7 / 8 prompts** — a refusal rate on 7 prompts moves
in 14 pp increments and its 95 % CI spans roughly ±35 pp. The "interference degrades at high K"
finding is **confounded**: the degradation could be small-sample variance in the tiny concepts, not
genuine cross-talk. **flas** has the identical exposure for its `hate`/`self_harm` concepts.

`sexual` / `harassment` / `violence` are all fine (eval 34–45); only the two tiny categories are
insufficient. The cleanest fix is to **drop `hate` and `self_harm`** from the concept ladders and
run a **K=1…3** ladder on the three well-populated concepts (each ≥100 available, ≥34 eval). If a
K=5 demonstration is wanted, merge `hate`+`self_harm`+`harassment` into one coarse
"harassment/hate" bucket (≥190 pooled) rather than steering two 24-prompt vectors.

## 3. Binary lessons — mostly fine; two need a bump

The harmful/benign lessons sit on the full 693/8 889 pool and use eval splits of 40–150/class.
Most are **SUFFICIENT for the screening tier** they claim (single-seed tutorial demos, not the
n≥7 EVALUATION contract). Two exceptions: **contextual_steering** caps eval at **25/class**
(below the ~30 floor — INSUFFICIENT for even a screening rate; fix is free-ish, just slower gen),
and **stacking** averages its norm-budget geometry probe over only **6 prompts** (THIN — bump to
≥20). **non_identifiability**'s 40-prompt harmful eval is borderline; 50 is safer and free.

Note: none of these single-seed tutorial results clear CLAUDE.md §7's EVALUATION bar (n≥7 seeds +
Wilcoxon + bootstrap CI). They are honest **screening** demonstrations, and the sufficiency
verdicts above are calibrated to that screening bar, not the external-ready bar.
