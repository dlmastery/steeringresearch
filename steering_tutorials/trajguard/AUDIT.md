# AUDIT — trajguard

**Auditor role:** independent paper + artifact verifier. Scope: does the cited paper
exist, does the code implement what the lesson claims, are the claims/results honest,
and does the lesson comply with CLAUDE.md §17.

**Revised 2026-08-08.** The previous version of this file (2026-07-20) certified a run
that no longer existed — *"n=80/class, 32 tokens, learned models 0.92-0.98, the paper's
training-free projection is the WEAKEST (0.665)"* — against a shipped artifact of
n=300/class, 40 tokens, `threshold_freeform` **0.6378**. It also graded **no confound
bar at all**, because it predated the confound work entirely. An audit that certifies a
version of a file that no longer exists is the `meerkat` staleness defect relocated to
the audit layer, and it is recorded here as a finding against this document rather than
quietly overwritten.

*Internal QA pass — independent external review pending (auditor shares a model family
with the author, CLAUDE.md §14).*

---

## 1. Paper existence

| field | finding |
|---|---|
| arXiv id | **2604.07727 — VERIFIED** (`arxiv.org/abs/2604.07727` resolves) |
| actual title | *TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense* |
| actual authors | **Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin, Kangyi Ding** |
| venue / date | ACL 2026 Findings; submitted 2026-04-09 (arXiv YYMM `2604`) |
| method in abstract | Confirmed: a training-free, streaming, decoding-time defence aggregating hidden-state trajectories via a sliding window; jailbreak-attempted tokens progressively shift toward high-risk latent regions |
| **comparative claim in abstract** | *"hidden states during decoding carry stronger risk signals than input prompts"*, evaluated across **12 jailbreak attacks** — see finding **F-2** |

Other ids cited by the lesson, all WebFetch-verified:

| id | title | authors | date |
|---|---|---|---|
| 2602.16935 | *DeepContext: Stateful Real-Time Detection of Multi-Turn Adversarial Intent Drift in LLMs* | Albrethsen, Datta, Kumar, Rajasekar | 2026-02-18 |
| 2404.01318 | *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models* | Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag, Dobriban, Flammarion, Pappas, Tramèr, Hassani, Wong | NeurIPS 2024 D&B, 2024-03-28 |
| 2310.17389 | *ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation* | Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, Jingbo Shang | EMNLP 2023 Findings, 2023-10-26 |

**No bogus or misresolving id.** 4/4 correct, with full author lists and dates.

---

## 2. Findings

### F-1. The retracted claim (highest value) — **RESOLVED, and recorded as a retraction**

The lesson asserted *"prompt char-length is at chance, 0.5032 — you cannot call this
from the prompt alone."* Both halves were false: the 0.5032 is produced by
`common/data.py`'s length-matched benign sampler **by construction**, and prompt
*content* separates the classes at **0.8779** (overt, 500/class) / **0.9688**
(disguised, 181/class) — measured CPU-only in this lesson — with `hello_world`'s
prompt-only probe independently at **0.965** on the same pool.

README §3 now carries this as an explicit **retraction with the evidence**, not a
softening. `CLAUDE.md` §18.3 item 4 repeats the same error and is **outside this
lesson's edit scope**; it is reported to the lead.

### F-2. The paper's headline comparison was out of scope — **NOW IN SCOPE, untested**

The lesson built no prompt-side classifier, so it could neither support nor refute the
paper's central comparative claim. `data.prompt_confound_report()` now measures it, the
runner prints a `vs PROMPT` column, and **falsifier F2** binds it. Registered
prediction: **F2 fails on both substrates** — i.e. we predict the paper's comparison
does not reproduce on toxic-chat. Predicting against the paper is the point of a
pre-registration.

### F-3. Substrate — **RE-BASED**

toxic-chat ships an unused `jailbreaking` binary column. The lesson now selects it
directly (`overt` = `toxicity==1 ∧ jailbreaking==0`, 512 unique; `disguised` =
`jailbreaking==1`, **181 unique — pool-limited**), through `common/data.py`'s own
primitives, without editing the shared module. The legacy `mixed` arm is retained for
reproducibility only.

### F-4. Rule 1 (≥500/class) — **PASS on `overt`, documented FAIL on `disguised`**

`overt` runs at **500/class** against a 512 pool. `disguised` is capped at **181/class**
by a genuine corpus ceiling (204 `jailbreaking==1` annotations exist in all of
toxic-chat), which rule 2 permits **provided the lesson says so** — it now does, in
`config.py`, in `select_prompts`'s stderr, in `results.json → sizes`, in the runner's
gate print and in README §3.4/§11.2, with every number from that arm labelled
PROVISIONAL.

The superseded 300/class was **not** pool-limited (693 were available) and is recorded
as the defect it was.

### F-5. The three-different-N defect — **CLOSED**

`config.py` 500 / README §8 120 / artifact 300, all simultaneously "true" because
`load_or_build()` checked no knob. Now: one authoritative `N_PER_CLASS`, a sha256
fingerprint over the config **and the sampled prompt group-ids** stored in the npz and
asserted on load (raises `CacheMismatch` with a diff), and an `assert_n_achieved` gate
that runs before any metric and raises on any shortfall not explained by
`pool_limited`. Accounting lands in `results_<substrate>.json → sizes`.

### F-6. Confound discipline — **PASS, and upgraded**

The lesson's own `confound_report` folded correctly (`max(auc, 1−auc)`) and ran
**before** the CV block — the best implementation in the repo at the time, and the
discipline the rule actually asks for. It was missing a **content** bar and a
**label-shuffle** control. It now runs on `common/confound.py` (the shared spine),
keeping the ordering, and adds this lesson's two geometry bars (`mean_norm`,
`final_norm`). The prompt channel is reported **separately and never folded into the
bar**, because a prompt-side classifier is a rival method, not a trivial confound.

Three gaps were left open here and stated in README §12.3 rather than closed: no
multivariate trivial baseline, no matched-bin control within `charlen` quantiles, and no
paired CI when the `content` bar binds.

**UPDATE 2026-08-21 — all three are now in code** (`trajguard/controls.py`, wired into
the runner, `controls` block in `results_<substrate>.json`):

- **Multivariate baseline: MEASURED.** Logistic regression on
  `{charlen, tokencount, mean_norm, final_norm}`, same folds as the methods, scaler fit
  on the training fold only, folded **into** `worst_auc`. On the disguised substrate it
  scores **0.5940** [0.5345, 0.6525] out-of-fold — far under `content` at 0.9103, so the
  binding bar and every §11.2 margin are **unchanged**. The control did not fire; it is
  now priced instead of assumed.
- **Matched-bin: CODE READY**, needs the runner (the per-method within-bin numbers
  require per-item out-of-fold score vectors, which `results_*.json` does not store).
  Its power is bounded by bin width and the self-test quantifies that honestly.
- **Paired CI: the blocker was SOLVABLE and is solved.** The spine's `content_bar` does
  compute a per-item out-of-fold score and merely discards it;
  `controls.content_bar_scores` reproduces the loop through the spine's own primitives,
  returns the vector, and **asserts** equality with the spine to `1e-9`. Paired CIs
  against `content` and `multivariate` are now computable and every CI records
  `against_bar`.

A separate defect surfaced while measuring the above and is written up in README §12.4:
`_save_cache` hardcoded the committed sidecar path, so building the OOD arm silently
overwrote the **in-domain** `trajectory_meta_disguised.json` with the OOD arm's 548
records. Fixed (path derived from the cache, `cache_file` threaded through, and an
assertion that an `ood:` snapshot cannot land in the in-domain sidecar); both sidecars
regenerated from the intact `.npz` caches and verified to reproduce the shipped bars
exactly.

### F-7. Early-detection bar — **CLOSED**

The streaming headline was the lesson's entire pitch and its one unpriced number.
`data.early_k_confound(K)` now computes the bar on the same first-K truncation, the
runner prints it as a row under the early-detection table, it is plotted, and
**falsifier F4** binds it. A K-token prefix has no decoded text, so there is no
completion-character or content bar at K; that is stated, not silently omitted.

### F-8. OOD arm — **ADDED (rule 5)**

Previously **absent**: 100% of the run came from one pool, one model, one layer, one
seed. Now `jackhhao/jailbreak-classification` (ungated, cached; 629 unique jailbreak /
1,323 unique benign), rendered through the same `common/data.py` primitives as the
in-domain set, positives capped at the benign p90 length ceiling before length-matching
— **274/class, length bar 0.5624, prompt-content bar 0.9873**, all measured CPU-only.
The uncapped 400/class arm (length bar **0.6504**) is reported beside it.

`intrinsec-ai/cstm-bench`, which §17 rule 8 names for this lesson *family*, is **not**
used: 108 rows of multi-session agent traces is the wrong granularity for a per-token
lesson, and it is absent from this host's HF cache (no HF token). Stated in README §7,
not quietly skipped. `allenai/wildjailbreak` — the best-matched corpus in existence for
this lesson — is gated and unusable here.

### F-9. Arithmetic and stale cells in the superseded README — **CORRECTED**

`threshold_freeform`'s margin printed **−0.045**; `0.63778 − 0.735439 = −0.09766`, i.e.
**−0.098**. Three F1 cells (0.24 / 0.84 / 0.86) disagreed with the artifact beside them
(0.172 / 0.781 / 0.894). Both fixed in README §11.5, and the §9 banner that declared
the claim "unsupported" while §10 declared it "MEASURED" is gone — those numbers now
live in one place, in a quarantined **superseded** section.

### F-10. Artifact reproducibility — **RESOLVED, with the trade-off stated**

`token_trajectories.npz` was 101.18 MiB, **over GitHub's 100 MiB hard per-file limit**,
so §10's "CPU only, no GPU, no regeneration — the trajectories are cached" instruction
was false for every reader but the original author. The cache is now `savez_compressed`
+ float16 and still **not** committed; the committed artifact is
`trajectory_meta_<substrate>.json` (~100 kB, **text-free**: label, prompt group-id,
token count, completion char length, `mean_norm`, `final_norm`, fingerprint). From a
fresh clone with no GPU: the prompt set reproduces exactly (deterministic
`select_prompts` over a public dataset), the prompt-channel bars recompute in full, and
the numeric completion-channel bars recompute from the meta file. The completion
*content* bar needs the generated text and therefore the re-run — stated, not implied
away. The npz text sidecar stays out of git deliberately: it holds abliterated-model
completions on harmful prompts.

### F-11. Statistical floor — **PARTIALLY CLOSED**

`BOOTSTRAP` raised 2,000 → **10,000** (CLAUDE.md §7). The margin now carries a paired
bootstrap CI over the same resample indices where the binding bar is a scalar. The CI
still treats pooled out-of-fold predictions as i.i.d., which ignores fold correlation
and is optimistic — standard practice, recorded as a known limitation. Screening tier:
one model, one layer, one seed, so §7's `n ≥ 7 + rigor contract` is **not** met and the
word "winner" is not used.

### F-12. Falsifiers — **PRE-REGISTERED, and printed by the runner**

Six falsifiers (F0 leakage, F1 drift-clears-bar, F2 decoding-beats-prompt, F3
substrate-contrast, F4 streaming, F5 OOD) registered **2026-08-08 before any run on the
new substrates**, printed by the runner at startup and evaluated into
`results_<substrate>.json → falsifier_verdicts`. The superseded lesson's binding
confound falsifier was stamped "binding from now on" *after* the AUCs existed — that is
pre-registration after the fact, and it is not repeated.

---

## 3. Verdict table

| check | verdict | note |
|---|---|---|
| Paper ids verified | **PASS** | 4/4, full author lists + dates |
| Rule 1 (≥500/class) | **PASS (`overt`) / documented FAIL (`disguised`)** | 500/512 vs 181/181; pool-limited, said so |
| Rule 2 (pool-limited disclosure) | **PASS** | stated in config, loader stderr, artifact, gate print, README |
| Rule 3 (off-family judge) | **N/A — PASS** | detection lesson; no judge anywhere; `"judge": null` |
| Rule 4 (citations, full detail) | **PASS** | see §1 |
| Rule 5 (real HF benchmark as OOD) | **PASS** | `jackhhao/jailbreak-classification`; cstm-bench exclusion justified |
| Rule 6 (small-N provisional) | **PASS** | `disguised` labelled PROVISIONAL everywhere |
| Rule 7 (confound-matched negatives, margin over the larger bar) | **PASS** | four spine bars + two geometry bars; prompt channel reported as a rival, not folded in |
| Rule 8 (pre-registered falsifier per claim) | **PASS** | six, registered before the run, printed by the runner |
| Artifact regenerable from the code beside it | **PASS (with a stated boundary)** | fingerprint + text-free meta; the content bar needs the GPU re-run |
| `results.json` records achieved, not requested | **PASS** | `sizes` block; gate raises on an unexplained shortfall |
| **Results honesty** | **PASS — both arms measured (2026-08-21)** | the **`disguised`** run executed 2026-08-08 (`artifacts/results_disguised.json`, fingerprint `b34e4b2e85bb…`) and the **`overt`** run 2026-08-21 (`artifacts/results_overt.json`, fingerprint `de4291e620f9…`), so §§11.1–11.5 + §12.2 carry measured numbers with CIs on both. The results that go against the lesson are reported as prominently as the ones that favour it: **F1 FAILS on both arms** (the paper's own `threshold_freeform` at 0.7570 / 0.7424 lands 0.153 / 0.166 *below* the 0.9103 / 0.9083 `content` bar) and **F5 FAILS on both** (OOD best 0.8653 / 0.6772 vs a 0.9873 prompt-content rival). **F2 HOLDS on both, against our own registered prediction** that it would fail. **F3 HOLDS but the artifact refuses to let it read as a win**: `both_margins_negative: true` and a `reading` field record that the contrast is between two failures, +0.0126 of a 15-point deficit. **The one live gap is the `overt` arm's missing controls** — that run predates the wiring of the three confound controls, so it has no `multivariate` bar, no matched-bin check and `null` paired CIs, on the arm whose `length` bar is 0.8155 rather than near chance. Disclosed in §11 status, §12.2, §12.3 and §14. The superseded 2026-07-27 run stays quarantined |
| `AUDIT.md` current | **PASS (this revision)** | the 2026-07-20 version certified a run that no longer existed; recorded above |

---

## 4. Overall verdict

**PASS on process and disclosure; PASS on results — both arms measured as of 2026-08-21,
with one disclosed gap (the `overt` arm's confound controls).**

The lesson's cited papers are real and correctly characterised, its confound machinery
is now the most complete in the course (the shared spine plus two geometry bars, run
before the headline), its cache is fingerprinted and its size accounting is asserted
rather than reported. Its **motivating premise was wrong and is retracted with the
evidence**, and it has been re-based onto the substrate that makes the paper's actual
claim testable.

It now has numbers on **both** substrates — `disguised` 2026-08-08, `overt` 2026-08-21 —
and they cut both ways on each. On both arms three methods clear the `content` bar
(+0.044…+0.075 disguised, +0.061…+0.065 overt) while the paper's own `threshold_freeform`
lands 0.153 / 0.166 **below** it (**F1 FAILS twice**) and nothing transfers OOD (**F5
FAILS twice**, and the rule-1-compliant `overt` arm transfers *worse*: 0.6772 best vs
`disguised`'s 0.8653). The registered prediction was that the paper's headline comparison
**would not reproduce here**; it **did**, on both arms (**F2 HOLDS**, 0.9855 vs 0.9688 and
0.9734 vs 0.8779), so the pre-registration was wrong in the direction that favours the
paper — recorded, not quietly dropped.

**F3 now has a verdict, and the artifact's own framing is the reason it is not oversold.**
The substrate contrast HOLDS by **+0.0126**, but `substrate_comparison` carries
`both_margins_negative: true` and a `reading` field stating that the comparison is between
two failures. §10.2 of the README leads with that framing rather than with the verdict.
This is the right shape for a falsifier that technically passes while the mechanism it
tests is absent, and it is the strongest process item in this revision.

**The remaining gap is narrower but real: the `overt` arm has no confound controls.** That
run predates the wiring of `controls.py` into the runner, so it carries no `multivariate`
bar, no matched-bin control and `null` paired CIs — on the arm whose completion `length`
bar is **0.8155**, not the near-chance 0.5064 the `disguised` arm has. On `disguised` the
matched-bin control was run and showed nothing was riding length (the *bar* moved most,
−0.0142); there is no equivalent assurance for `overt`. Re-running
`TG_SUBSTRATE=overt` under the current runner is the one thing that would close it. The
previous run stays quarantined. That is the honest state of the lesson.

*Internal QA pass — independent external review pending.*
