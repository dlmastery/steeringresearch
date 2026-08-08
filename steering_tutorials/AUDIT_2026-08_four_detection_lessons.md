# AUDIT 2026-08 — the four detection lessons, against §17 and against their own artifacts

**Scope.** `biencoder_guard`, `cross_trajectory`, `multiturn_jailbreak`, `trajguard`.
Four independent read-only auditors, disjoint scopes, CPU-only, run 2026-08-08. Every arXiv
id and every HuggingFace dataset id below was **WebFetch-verified on the day**; every in-repo
number was read from the artifact or reproduced by re-running the loader, never taken from a
docstring.

*Internal QA pass — independent external review pending. The auditors share a model family
with the authors (§14 circularity disclosure).*

---

## 0. The one-paragraph finding

> **All four lessons fail rule 1 (≥500/class), and in not one of them is the shortfall
> pool-limited — the data was available and unused in every case.** Underneath that sits a
> deeper problem: the confound instrument is *fragmented*, so each lesson reimplements a
> different subset of the discipline and the course-level documents describe a canonical
> implementation **that exists only in prose**. Two lessons' headline premises are
> contradicted by numbers already inside the course. And three of the four artifacts record
> the **requested** configuration rather than the **achieved** one, so the rule-1 failures are
> invisible to anyone reading the artifact instead of the loader.

---

## 1. Cross-cutting failures (fix once, not four times)

### 1.1 Rule 1 fails everywhere, and never because of the pool

| lesson | achieved | claimed / configured | was more available? |
|---|---|---|---|
| `biencoder_guard` | **6 of 16 policy columns < 500** (`jailbreak` 109, `child_abuse` 185, `self_harm` 205, `terrorism` 293, `animal_abuse` 357, `toxicity` 374); benign capped at **500** vs 5,226 harmful = **91.3 % / 8.7 %** | `results.json` says `"n_per_class": 500` | **Yes** — BeaverTails `330k_train` + `30k_test` clear 4 of the 6; benign cap is arbitrary |
| `cross_trajectory` | `hard` **298 / 298** (the condition that carries the claim) | README states 298 in four places — honest | **Yes** — `SafeMTData_1K`, the *same ungated repo*, ships **1,680 more rows**, never touched |
| `multiturn_jailbreak` | **200 / 200**, both conditions | no pool ceiling stated anywhere | **Yes** — Attack_600 has 600; UltraChat `train_sft` has 207,865. `easy` at 600/600 was free |
| `trajguard` | **300 / 300** | `config.py` says 500, README §8 says 120 — **three numbers in three places** | **Yes** — `common.data` has **693** unique harmful available |

Rule 2's pool-limit exemption applies to **none** of these. It is an exemption for a genuinely
capped pool, and no pool here was capped.

### 1.2 The confound instrument is fragmented, and the docs describe code that does not exist

There is **no shared `confound_report` in `common/`**. Four lessons, four partial
reimplementations:

| lesson | directionless fold `max(auc,1−auc)` | count/turn bar | content bar | shuffle control |
|---|---|---|---|---|
| `trajguard` | **yes** (`_directionless`) | n/a | no | no |
| `multiturn_jailbreak` | yes | yes (`turncount_auc`) | no | no |
| `cross_trajectory` | **NO — returns raw AUC** | yes (`kcount_auc`) | no | no |
| `biencoder_guard` | **NO** | **absent** (`count_auc` missing) | no | no |

`CONFOUND_DISCIPLINE.md` §4 prints a `confound_auc` that folds, and says *"the pattern already
exists as `confound_report()` in `cross_trajectory/data.py` — this is the canonical form."*
**It is not. That function does not fold.** The document describes code that has never
existed, and it is the document every future lesson will copy from.

Consequence in the live numbers: `cross_trajectory`'s `easy` condition reports
`totalchar_auc = 0.10975`, which reads as "clean" and is a **0.890 confound with the sign
flipped**. README §9.1 prints easy AUCs of 0.991–0.998 against **no bar at all**. The true easy
margin is **+0.10**, not +0.99. The folded number already exists in `CONFOUND_DISCIPLINE.md` §3
— the lesson simply never picked it up.

### 1.3 Artifacts record the requested config, not the achieved one

- `biencoder_guard/results.json`: `"n_per_class": 500` is `int(C.N_PER_CLASS)` written straight
  from config at `run_biencoder_guard.py:1477`. The achieved per-column counts are **never
  persisted**. Source distribution is **never persisted**. A reader cannot detect the rule-1
  failure or the single-dataset collapse from the artifact.
- `trajguard`: `load_or_build()` silently ignores `N_PER_CLASS`, `MAX_NEW_TOKENS`, `LAYER`,
  `MODEL_ID`. No fingerprint of any kind.
- `cross_trajectory`: `_embed_samples` keys its cache on `(condition, embedder)` only; the load
  path validates **row count alone**. Change `CT_SEED`, the half-split reshuffles which attacks
  are positive, the count is identical — **cached vectors are silently reused against new
  labels.** This is verbatim §18.8's "the embedding cache returned stale labels under a key
  that ignored them," currently unguarded.
- `multiturn_jailbreak`: `artifacts/.gitignore` excludes `*.npz`; `git ls-files` confirms **no
  npz is tracked**. A cloner cannot reproduce any number without a GPU re-embed — against §17's
  "binary artifacts are force-added so each lesson reproduces from the repo."

### 1.4 OOD is absent or is not OOD in three of four

| lesson | OOD arm | verdict |
|---|---|---|
| `biencoder_guard` | `BeaverTails/30k_test` | **not OOD** — same dataset, annotators, taxonomy and rendering as 93.5 % of train. Split transfer, not distribution transfer |
| `multiturn_jailbreak` | none | **absent** |
| `trajguard` | none | **absent** |
| `cross_trajectory` | `intrinsec-ai/cstm-bench` | **real** — but keeps only the **5 longest of ~26 sessions** per scenario, on a `dilution` split whose entire premise is dilution across many sessions |

`intrinsec-ai/cstm-bench` is ungated, MIT, and **already cached on this host**. It is the
benchmark §17 rule 8 names for this lesson family, and three of four lessons do not use it.

### 1.5 Course-level documents contradict the lessons — in both directions

| document | claim | reality |
|---|---|---|
| `CONFOUND_DISCIPLINE.md` §3 | `cross_trajectory` **fully compliant** | true for `hard`, **false for `easy`** (0.890 bar never reported). Belongs under "measured but not reported" |
| `CONFOUND_DISCIPLINE.md` §5 | `trajguard` **non-compliant, margin unpriced** | **stale** — `confound_report()` now exists, runs before CV, folds correctly, and is rendered |
| `CONFOUND_DISCIPLINE.md` §4 | `cross_trajectory`'s is "the canonical form" | it lacks the fold (§1.2) |
| `DATA_SUFFICIENCY.md` | — | `trajguard` **does not appear at all** |
| `cross_trajectory/AUDIT.md` | "no numbers claimed yet; table is `_pending_`" | README now carries a full results table. Certifies a file version that no longer exists |
| `trajguard/AUDIT.md` | certifies n=80/class, threshold 0.665 | that run no longer exists; audits no confound bar |
| `CLAUDE.md` §18.5 | `google/embeddinggemma-300m` gated, unobtainable | **stale for the model** — the weights are on disk. Several *datasets* remain genuinely gated |

An audit that certifies a version of a file that no longer exists is the `meerkat` staleness
defect relocated to the audit layer.

---

## 2. Two lessons are contradicted by their own course

### 2.1 `trajguard` — the premise does not survive contact with `hello_world`

README §10 and `CLAUDE.md` §18.3 item 4 both assert:

> "prompt char-length is at chance, 0.5032 — **you cannot call this from the prompt alone**"

Both halves fail:

1. **0.5032 is designed in, not measured.** `common/data.py` draws the benign class
   **length-matched to the harmful histogram, decile-bin stratified** (`_length_matched_sample`,
   line 246, invoked at 442). The sampler was written to produce that number.
2. **Length ≠ content.** On the *same toxic-chat pool*, `hello_world` reports a **prompt-only
   probe at AUC 0.965** — above every trajectory detector in this lesson (best 0.945).

So on this dataset a prompt-side classifier **beats** the decoding-time detector. The verified
TrajGuard paper claims hidden states during decoding carry *stronger* risk signals than input
prompts. **The lesson does not exercise that comparison, and its own course's evidence
contradicts it** — while the README asserts the opposite from a number engineered to be 0.5.

The mechanism needs prompts whose **surface is benign** and whose **completion turns harmful**.
toxic-chat harmful prompts are overtly toxic user inputs; the abliterated model complies from
token 1, so the drift a trajectory monitor reads never happens. The lesson's own numbers say so:
`threshold_freeform` — the *drift* detector — lands at **0.638, below the 0.735 confound bar**,
while the stateless per-token probe reaches 0.931. **There is no trajectory signal here, only a
per-token one.** The paper evaluates on 12 jailbreak attacks; this lesson uses **zero** attack
wrappers.

**And the fix is already in the pool it loads.** toxic-chat ships a **`jailbreaking` binary
column** (config `toxicchat0124`); `common/data.py` reads only `toxicity`. The disguised-attack
subset this lesson needs is being discarded at load time.

### 2.2 `biencoder_guard` — the hard-negative miner is a near no-op

`hardneg.n_mined = 240` against a train benign pool of ~350 → the miner selects **69 % of
everything available**. ANCE-style mining is only meaningful when the pool dwarfs the selection.

This is why EXP-F's frozen baseline shows `fpr_at_recall90 = **1.000**` — at the threshold for
90 % recall it flags *every* hard negative. Falsifier (iii) "SURVIVES" against a baseline pinned
at the literal worst possible value, and the README does not say so.

Related: the advertised negative design is **not in the data**. The README states toxic-chat
supplies "the benign hard-negatives that look adversarial but are safe." Benign backfill is
gated behind `deficit = max(0, n_benign - bt_benign)` and BeaverTails fills the 500 quota first,
so `tc_benign` is **0 by construction**. **Every benign row is a BeaverTails safe row.**
`wildguardmix` has never loaded (HTTP 403, gated, no token) — disclosed in a code comment, but
the README prose three sentences later is written in the indicative as though it had.

---

## 3. Datasets — verified availability (this host has NO HF token; gated ⇒ unusable)

### 3.1 Usable now — ungated, verified 2026-08-08

| id | rows | what it fixes | for |
|---|---|---|---|
| **`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`** | **33,416** (30,007 / 1,445 / 1,964) | 12 core unsafe categories + 9 fine-grained subcategories over Safe/Needs-Caution — **a real 3-level taxonomy**, which EXP-H currently hand-templates. Prompt *and* response. Separate human and LLM label columns | `biencoder_guard`: fixes single-dataset collapse, the 6 starved columns, the benign cap, **and** gives a genuine cross-annotator OOD in one move |
| **`SafeMTData/SafeMTData` config `SafeMTData_1K`** | **1,680** | same ungated repo the loader already calls — only the config string + a turn extractor change | `cross_trajectory`, `multiturn_jailbreak`: breaks the self-imposed 600 ceiling |
| **`tom-gibbs/multi-turn_jailbreak_attack_datasets`** | 382 Complete-Harmful · 4,136 Harmful · **1,200 Semi-Benign** · 1,200 Completely-Benign | its premise *is* the lessons' construct: distributing harmful prompts across turns so each looks harmless alone. **Semi-Benign is a purpose-built hard-negative pool** | `cross_trajectory`, `multiturn_jailbreak`: replaces *synthesised* payload-stripped negatives with real ones |
| **`intrinsec-ai/cstm-bench`** | 108 (54 dilution + 54 cross-session) | the OOD benchmark §17 rule 8 names for this family; **already cached on this host** | `biencoder_guard`, `multiturn_jailbreak`, `trajguard` — all three lack a real OOD arm |
| **toxic-chat `jailbreaking` column** | already loaded | disguised attacks, discarded at load | `trajguard` — this is the missing substrate, at zero download cost |

**Honest caveat on `SafeMTData_1K`:** its 1,680 rows carry **multiple actors per `query_id`**,
so distinct groups are likely ~500–600, not 1,680. It fixes the **n** floor; it does **not**
proportionally raise independent groups. Both numbers must be reported separately — inflating
n without inflating information is the COUGHVID mistake.

### 3.2 Verified GATED — unusable on this host

`ScaleAI/mhj` (gated:auto — named by §17 rule 5, local cache holds only a 40-byte `refs/main`),
`allenai/wildjailbreak` (gated:auto), `allenai/wildguardmix` (403), `walledai/HarmBench`
(gated:auto), `lmsys/lmsys-chat-1m` (custom licence; also avg 2.0 turns < `MIN_USER_TURNS=3`).

**§17 rule 5 names `ScaleAI/mhj` as this family's benchmark and it cannot be obtained here.**
That should be recorded in the rule, not rediscovered per lesson.

---

## 4. Citations

| lesson | ids | verdict |
|---|---|---|
| `biencoder_guard` | 7 | **7/7 correct** — exact title and authors. Best in the repo |
| `trajguard` | 3 | **3/3 correct**, characterisation matches |
| `multiturn_jailbreak` | — | format good (title + clickable link); see per-lesson file |
| `cross_trajectory` | 6 | **4 of 6 titles paraphrased or invented**; 0 of 6 carry authors or dates |

`cross_trajectory`'s two inventions matter substantively, not just formally:

- **`2606.09084`** — printed as *"Context-Fractured Decomposition: Distributing Harmful Intent
  Across Cooperating Agents."* Real title: *"Context-Fractured Decomposition Attacks on
  Tool-Using LLM Agents: Exploiting Artifact Provenance Gaps"* (Lin, Yang, Guo, Nale, Fleming,
  Cheng; 2026-06-08). The real paper is about **one agent's artifact-provenance gaps across
  steps**, not K cooperating agents. The lesson's core framing is a stretch of it.
- **`2603.13940`** — printed as *"GroupGuard: Graph-based Detection of Colluding-Agent
  Attacks."* Real title: *"GroupGuard: A Framework for Modeling and Defending Collusive Attacks
  in Multi-Agent Systems"* (Tao, Zheng, Yang, Tao, Wang; 2026-03-14). GroupGuard is
  **training-free** and combines graph monitoring + honeypot inducement + structural pruning;
  `gnn_agg` is a **trained** message-passing classifier with none of that machinery.

All six ids **resolve to real papers** — so the README is simultaneously *over*-hedged
(`[UNVERIFIED]` tags on real papers, in five places) and *under*-verified (printed titles are
not the papers' titles). `AUDIT.md` promised the lead would WebFetch before merge; the lesson
shipped with the tags intact. §18.8: *a check that must be remembered will eventually be
skipped.*

---

## 5. Ranked plan

**Tier A — cross-cutting, do first (CHEAP, no GPU).**

1. **Build `common/confound.py`** with the one true `confound_report`: directionless fold,
   length AUC, count AUC, a content bar (TF-IDF / bag-of-words), and a label-shuffle control.
   Port all four lessons onto it. Delete the four partial copies.
2. **Correct `CONFOUND_DISCIPLINE.md`** — it names a canonical implementation that does not
   exist, clears `trajguard` from "non-compliant", and moves `cross_trajectory` to "measured but
   not reported". Add `trajguard` to `DATA_SUFFICIENCY.md`.
3. **Persist achieved config, not requested.** Every `results.json` records realised per-class /
   per-column counts, the source distribution, and a pool fingerprint (SHA-256 of the sampled
   ids). Fail loudly when requested ≠ achieved.
4. **Key every embedding cache on content**, not row count — store a SHA-256 of the concatenated
   texts plus seed/K/n/dim in the npz and **assert** on load.

**Tier B — per-lesson honesty (CHEAP).**

5. `cross_trajectory`: price the `easy` condition against its **0.890** bar; fix six citations;
   correct the "length-matched" docstring; state the EmbeddingGemma mandate as *blocked*, not as
   a free choice; mark `results_gemma_ablation.json` unreproducible (no code emits it, and
   `RESULTS_PATH` is a constant, so the §9.3 side-by-side **cannot** be produced by this code).
6. `biencoder_guard`: publish the real source distribution; correct the held-out base rates
   (chance macro-AP is **0.1344**, not 0.185 → the result is **2.84× chance, not 2.1×** — this
   correction goes *in the lesson's favour*); state that EXP-F's baseline is pinned at 1.000.
7. `trajguard`: reconcile the three different N values; retract "you cannot call this from the
   prompt alone" and replace it with the honest comparison against `hello_world`'s 0.965.

**Tier C — the dataset work the user asked for (MEDIUM/EXPENSIVE, GPU serialises).**

8. `biencoder_guard` ← `nvidia/Aegis-2.0`. Highest single-move payoff in the set.
9. `cross_trajectory` + `multiturn_jailbreak` ← `SafeMTData_1K` + `tom-gibbs` Semi-Benign.
10. `trajguard` ← toxic-chat's `jailbreaking` column (free) — the only change that makes the
    lesson test its own paper's claim.
11. Add `cstm-bench` OOD to the three lessons lacking one.

**Do not** run Tier C before Tier A: re-embedding on top of an unfingerprinted cache and a
non-folding confound report reproduces the current defects at larger n.

---

## 6. What is genuinely right, and should be kept

- `biencoder_guard` reproduces **exactly** — the auditor re-ran the loader and matched
  `length_auc` to full float precision (`0.5263811710677383`). The 0.72 → 0.52 length fix
  recorded in §18.3 **is real and still holding**. 7/7 citations correct. One falsifier recorded
  as **TRIPPED** without softening — exemplary.
- `trajguard`'s confound machinery is the best in the set: folds correctly, runs **before** the
  CV block, margins computed against the larger of {baseline, confound}, and the paper's own
  `threshold_freeform` is allowed to land **below** the bar and reported that way.
- `cross_trajectory` uses group-aware `GroupKFold` by `query_id` with seeded remapping, skips
  degenerate single-class bootstrap resamples, writes `results.json` **before** the summary
  print, and prints its falsifier from the runner so it cannot be quietly dropped from the README.
- `multiturn_jailbreak` computes a real confound baseline and quotes both bars. It is **not** the
  trajguard defect.
- All four correctly take **no generation judge** and say so.
