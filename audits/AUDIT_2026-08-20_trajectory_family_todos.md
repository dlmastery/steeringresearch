# AUDIT 2026-08-20 — outstanding items across the trajectory / latent-aggregation family

Auditor: `loose-ends`-class subagent, 2026-08-20. **READ-ONLY** — no lesson file edited, no git
run, no model loaded, no GPU touched (the 4090 is busy with a live run). Every artifact below was
inspected on CPU with `json.load`; every claim cites a `file:key` or `file:line`.

Scope: `steering_tutorials/{trajguard, cross_trajectory, streaming_trajectory_aggregation,
meerkat, multiturn_jailbreak, biencoder_guard}`.

Classification: **(a) STALE** — marker says pending, artifact shows the result.
**(b) NEEDS GPU** — genuinely unrun (CPU-only-but-unrun items are marked `(CPU)`).
**(c) NEEDS CODE** — no implementation exists. **(d) DOC-ONLY** — text fix, no computation.
**(e) OBSOLETE** — overtaken by events; strike rather than do.

---

## 0. Counts

| class | count |
|---|---|
| **(a) STALE** | **9** |
| (b) NEEDS GPU | 15 |
| (c) NEEDS CODE | 8 |
| (d) DOC-ONLY | 8 |
| (e) OBSOLETE | 3 |
| **total** | **43** |

---

## 1. The STALE list — free wins, each proved from the artifact

### S1–S4. `trajguard`: the whole `disguised` arm ran on 2026-08-08 and the README never noticed

`trajguard/README.md` §11 opens *"Status: NOT YET RUN on the re-based substrates. Every cell below
is `[PENDING RUN]`."* (README.md:527-531). `artifacts/results_disguised.json` (mtime 2026-08-08
09:35, 24,534 bytes, `fingerprint: b34e4b2e85bb…`) contains a complete run.

**S1 — §11.2 disguised results table (README.md:546-549), 4 cells `[PENDING RUN]`. STALE.**

| method | `results_disguised.json` key | AUC | 95% CI | margin vs binding bar (`content` 0.9103) |
|---|---|---|---|---|
| `threshold_freeform` | `methods.threshold_freeform.auc` | **0.7570** | [0.7042, 0.8067] | −0.1533 (`clears: false`) |
| `per_turn_max` | `methods.per_turn_max.auc` | **0.9850** | [0.9732, 0.9941] | +0.0747 (`clears: true`) |
| `trajectory_mlp` | `methods.trajectory_mlp.auc` | **0.9855** | [0.9731, 0.9948] | +0.0752 (`clears: true`) |
| `seq_gru` | `methods.seq_gru.auc` | **0.9544** | [0.9313, 0.9742] | +0.0441 (`clears: true`) |

**S2 — §11.3 early-detection table (README.md:555-557), `[PENDING RUN]`. STALE.**
`early_detection` and `early_confound` are populated for K = 2/4/8/16/32, *each K against its own
bar* — the exact thing AUDIT_2026-08.md fix #14 asked for. E.g. `early_detection.seq_gru["2"] =
0.9684` against `early_confound["2"].worst_auc = 0.6249` (`worst_name: mean_norm`).
`falsifier_verdicts.F4_streaming_clears_its_own_bar.holds = true`, 10/10 cells clearing.

**S3 — §11.4 OOD table (README.md:563), `[PENDING RUN]`. STALE.**
`ood.header`: `jackhhao/jailbreak-classification`, 274/class, `pool_limited: true`,
`length_cap_chars: 1311`. `ood.methods.*.auc` = `threshold_freeform` 0.7832 / `per_turn_max`
**0.8653** / `trajectory_mlp` 0.7854 / `seq_gru` 0.8202, all with CIs.
`ood.confound.worst_auc = 0.9341` and `ood.confound.prompt_channel.content.auc = 0.9873`.

**S4 — §12.2's closing line "Completion-channel bars require the GPU run: `[PENDING RUN]`"
(README.md:642-643). STALE.** All six completion-channel bars are in `confound`:
`length` 0.5064, `count` 0.5515, `content` **0.9103**, `shuffle` 0.5404, `mean_norm` 0.5463,
`final_norm` 0.5753; `worst_name: content`.

> **The result nobody has written up.** §10's pre-registration says *"F2 — we predict it FAILS on
> both substrates"* (README.md:512-517). On `disguised` it **HOLDS**:
> `falsifier_verdicts.F2_decoding_beats_prompt = {best_method_auc: 0.9855, prompt_content_auc:
> 0.9688, holds: true}`. TrajGuard's headline comparative claim **reproduces** on the disguised
> substrate, against the lesson's own registered prediction. That is the single most interesting
> unreported number in this family.

### S5. `trajguard/AUDIT.md:194` — "Results honesty | **PENDING** | the re-based runs have **not** been executed". STALE (for `disguised`); still true for `overt`. Rewrite as half-done.

### S6. `streaming_trajectory_aggregation`: F2 is answered

`README.md:258` — `| F2_mean_pool_survives_short | **PENDING** | — | AgentDojo-only. The AgentDojo
run is in flight as of 2026-08-17 |`. **STALE.** `artifacts/results_agentdojo.json` (mtime
2026-08-18 02:53) `falsifier_verdicts.F2_mean_pool_survives_short = {mean_pool_auc: 0.98668,
max_pool_auc: 0.99504, gru_auc: 0.99398, holds: true}`.

Also unreported from that same file and worth a row in §7: **`F3_sequence_beats_laststep` FAILS on
AgentDojo** — `{best_auc: 0.99880, last_step_auc: 0.99880, margin: 0.0, holds: false}` — and
`F4_streaming_clears_bars` fails there too (`esn_cusum` 0.8150, `safetydrift` 0.6250 vs bar
0.9960). F0 holds (0.5106).

### S7. `meerkat/AUDIT.md:37` — "Results table marked **[PENDING GPU RUN]**". STALE.
`meerkat/artifacts/results.json` (mtime 2026-07-27 22:57) is a complete run: `regimes["0.05"]`
and `["0.5"]` with AP + CIs + ROC-AUC for all three localizers, `clustering.best_k = 16`,
`cluster_purity 0.955`, and a populated `ood` block (`intrinsec-ai/cstm-bench`, 52/56).

### S8. `biencoder_guard/AUDIT.md:64` — "README §10 marks all tables `[PENDING GPU RUN]`". STALE.
Already convicted by `biencoder_guard/AUDIT_2026-08.md:278` ("no longer true"). The §10 tables
carry measured macro-AP (0.240 / 0.575 / 0.169 / 0.658 etc.) — now *suspended*, which is a
different status from *pending*, and `AUDIT.md` records neither.

### S9. `cross_trajectory/data.py:355` + `README.md:408` — CSTM-Bench loader marked
`[SCHEMA UNVERIFIED]`, "no download was performed on this host". **STALE.**
`cross_trajectory/artifacts/results.json:ood` carries measured AUCs over `intrinsec-ai/cstm-bench`
at `n_attack: 52, n_benign: 56` — the schema was exercised. Independently corroborated by
`multiturn_jailbreak/artifacts/results.json:measured_without_a_model_2026_08_08.ood_cstm_bench`
(same 108 rows, 52/56, split into `benign_pristine: 28 / benign_hard: 28`). The `[SCHEMA
UNVERIFIED]` hedge applies to `tom-gibbs/…` (never downloaded), not to cstm-bench.

---

## 2. The prioritised table

Cost key: **XS** < 5 min · **S** ~10–30 min · **M** ~1–3 h · **L** > 3 h.

| # | lesson | item | class | evidence | est. cost |
|---|---|---|---|---|---|
| 1 | trajguard | §11.2 disguised results table `[PENDING RUN]` | **a** | `results_disguised.json:methods.*` (4 AUCs + CIs) | XS |
| 2 | trajguard | §11.3 early-K table `[PENDING RUN]` | **a** | `results_disguised.json:early_detection`, `early_confound` (K=2..32) | XS |
| 3 | trajguard | §11.4 OOD table `[PENDING RUN]` | **a** | `results_disguised.json:ood.methods.*` (274/class) | XS |
| 4 | trajguard | §12.2 completion-channel bars `[PENDING RUN]` | **a** | `results_disguised.json:confound.{length,count,content,shuffle,mean_norm,final_norm}` | XS |
| 5 | trajguard | §10 predicts "F2 FAILS on both"; measured **HOLDS** on disguised — unreported | **d** | `results_disguised.json:falsifier_verdicts.F2_decoding_beats_prompt.holds=true` | S |
| 6 | trajguard | `AUDIT.md:194` "Results honesty PENDING" | **a**/**d** | see S5 | XS |
| 7 | **trajguard** | **`overt` substrate never run** — §11.1 table, and F3 (the substrate contrast, the lesson's whole re-basing rationale) | **b** | no `artifacts/results_overt.json`; `results_disguised.json:substrate_comparison` = *"F3 needs BOTH arms; run the other substrate with TG_SUBSTRATE=overt"*; `config.py:87` default is `overt` | **M** — `TG_SUBSTRATE=overt python -m steering_tutorials.trajguard.run_trajguard`; 1,000 in-domain completions × 40 tok + ~548 OOD completions |
| 8 | trajguard | multivariate trivial baseline (logreg over `{charlen, tokencount, mean_norm, final_norm}`) | **c** | `README.md:647-649` §12.3; AUDIT_2026-08 fix #15 | S (CPU) |
| 9 | trajguard | matched-bin control inside `charlen` quantiles | **c** | `README.md:650-651`; AUDIT_2026-08 fix #13 (half done — shuffle landed, matched-bin did not) | S (CPU) |
| 10 | trajguard | paired margin CI when `content` binds | **c** | `results_disguised.json:methods.*.vs_confound_paired_ci = null` on all 4; `paired_ci_note` explains the centroid bar exposes no per-item score | M (CPU) |
| 11 | trajguard | layer sweep + second model (one fixed layer 12, unjustified) | **b** | AUDIT_2026-08 fix #22; `config.py:79 LAYER=12` | L |
| 12 | trajguard | AUDIT_2026-08 fixes #11/#13(shuffle)/#16/#17/#20/#21 | **e** | all landed: `fingerprint`+`config_snapshot` present; `confound.shuffle` present; `bootstrap: 10000`; disguised npz 23 MB / OOD npz 34 MB (both under 100 MiB); `ood` block exists; disguised = `jailbreaking==1`. Strike them. | XS |
| 13 | trajguard | AUDIT_2026-08 fix #18 (re-source to Aegis 2.0 so the thesis is testable) | **e** | superseded by the `disguised` substrate, which made F2 testable *and* it HOLDS (item 5). Strike or re-scope. | XS |
| 14 | **cross_trajectory** | **`CT_EMBED=embeddinggemma` arm — the mandated encoder, no number in existence** | **b** | `README.md:164`, `:770`; `config.py:71` default is `minilm`; `results_gemma_ablation.json:SUSPENSION_WITHDRAWN…note`. **Newly runnable** — the causal-attention bug is fixed and the weights are on disk | **M** — `CT_EMBED=embeddinggemma`; ~298–500/class × K=5 trajectories ≈ 3–5k embeddings, easy+hard+ood |
| 15 | cross_trajectory | enlarged pool (`Attack_600` **+** `SafeMTData_1K`) never run; every §9 number is 298/class | **b** | `README.md:397`, `:545`; `config.py:85 ATTACK_CONFIGS=["Attack_600","SafeMTData_1K"]` | M |
| 16 | cross_trajectory | `results_gemma_ablation.json` **hand-transcribed, not regenerable** — the live defect | **b** | `README.md:659-681`; the file's own `the_real_and_pre_existing_caveat`; no code emits `hidden`/`margin_over_bar`/`replication_verdict` | M — re-run gemma arm under the new `results_<embedder>.json` path |
| 17 | cross_trajectory | Gemma **OOD** arm `NOT RUN` (reaped mid-load) | **b** | `results_gemma_ablation.json:ood.status = "NOT RUN -- run reaped during CSTM-Bench load"` | S (folds into #16) |
| 18 | cross_trajectory | OOD arm still the `longest`-5 selection; `CT_OOD_SELECT` default already changed to `uniform` but not re-run | **b** | `README.md:627`; `config.py:112 OOD_SELECT="uniform"` | S — 108 scenarios |
| 19 | cross_trajectory | `tom-gibbs` Semi-Benign loader `[SCHEMA UNVERIFIED]`, off by default | **b** (CPU) | `config.py:99-100 CT_TOM_GIBBS`; `data.py:355` | S |
| 20 | cross_trajectory | CSTM-Bench `[SCHEMA UNVERIFIED]` tag | **a** | see S9 | XS |
| 21 | cross_trajectory | `AUDIT.md:25-29,37` still carries `[UNVERIFIED]` on 5 ids + the dataset; README dropped them 2026-08-08 | **d** | `README.md:4`, `:809` vs `AUDIT.md:25-29` | XS |
| 22 | cross_trajectory | AUDIT_2026-08 item 3: correct 6 citations (4 titles paraphrased/invented) | **d** | `AUDIT_2026-08.md:202`; README.md:4 claims done for the README — **AUDIT.md's own paper table was not updated** | XS |
| 23 | STA | §7 `F2` row = PENDING | **a** | see S6 | XS |
| 24 | STA | AgentDojo F3 **FAILS** (margin 0.0) and F4 fails — absent from §7 | **d** | `results_agentdojo.json:falsifier_verdicts.F3…holds=false`, `F4…holds=false` | S |
| 25 | STA | `Truncate(k, inner)` within-corpus horizon control — the fix that would license the horizon inference — **not built** | **c** | `README.md` §7 *"is not built"*; `AUDIT_FIDELITY.md` §2.1 | M (embeddings cached ⇒ mostly CPU) |
| 26 | STA | `SafetyDriftMonitor` does not implement drift (`score == score(SHUFFLED) == score(REVERSED)`) | **c** | `README.md` §8(a); `streaming/markov.py` | M |
| 27 | STA | F1/F2/F3 never consult `binding_bar` in the machine-readable verdict; no shuffle control on the ladder itself | **c** | `README.md` §8(e) | S |
| 28 | STA | 10,000 bootstrap resamples never enter a verdict; no paired `AUC_max − AUC_mean` CI, no Holm on F3's max-over-8 | **c** | `README.md` §8(f) | S |
| 29 | STA | both streaming monitors monotone in length by construction (`self._peak`) | **c** | `README.md` §8(b): length-only AUC 0.9222 / 0.9024 | M |
| 30 | STA | `assebench` / `atbench` corpora written, never run; each carries a `KNOWN GAP AGAINST THE SPINE` header | **b** | `corpora/assebench.py:24`, `corpora/atbench.py:29` | L |
| 31 | STA | `[UNVERIFIED]`: `2605.09863`, Latent Sentinel figures, `QueryTokenCompressor` K=16 | **d** | `CITATIONS_VERIFIED.md:288,322,368-369`; `aggregators/__init__.py:19` | XS (WebFetch) |
| 32 | STA | re-run F1–F5 | **e** | SHADE + AgentDojo both ran; F1/F2/F3(SHADE) HOLD, F3(AgentDojo)/F4 FAIL, F5 hollow. Strike. | — |
| 33 | **meerkat** | **`bge` — the config default AND the paper's own encoder — has never been run; every number is MiniLM** | **b** | `config.py:50 EMBEDDER="bge"` vs `results.json:{"embed_model":"BAAI/bge-base-en-v1.5", "embedder":"minilm"}`; README.md:457-466 documents the field disagreement | **M** — `MK_EMBED=bge`; ~2×400 traces + 108 OOD ≈ 900 embeddings |
| 34 | meerkat | no EmbeddingGemma option at all — `EMBEDDER ∈ {bge, minilm}` | **c** | `config.py:50-51`, `EMB_CACHE` keys `("bge","minilm")`; violates the standing encoder mandate | S then M to run |
| 35 | meerkat | `AUDIT.md:37` "[PENDING GPU RUN]" | **a** | see S7 | XS |
| 36 | meerkat | `[UNVERIFIED]` on 2309.07597 (BGE / C-Pack) | **d** | `README.md:3,591`; `AUDIT.md:25` | XS |
| 37 | **multiturn_jailbreak** | **the entire §9 results table is unmeasured — `results.json` is a deliberate placeholder holding NO metrics** | **b** | `artifacts/results.json:status = "PENDING_RUN"`, `written_by: "…by hand -- NOT by run_multiturn.py"`; `README.md:29`, `:486`. Headline embedder is `embgemma` (`config.py:88-90`) ⇒ **newly unblocked** by the bidirectional fix | **L** — 1,200 conversations (600/600 over 770 goals) × 2 embedders × 5 methods × 5-fold + 108-row OOD |
| 38 | multiturn_jailbreak | F1/F2/F3 (`PREREGISTRATION.md`, `config.PREREGISTRATION`) unevaluated | **b** | `README.md:520-535`; bars already measured CPU-side (`content` 0.8584) | — (folds into #37) |
| 39 | **biencoder_guard** | **full re-run under transformers 5.15.0 — every accuracy number is SUSPENDED** | **b** | `artifacts/results_CAUSAL_ENCODER_SUSPENDED.json:SUSPENDED_CAUSAL_ENCODER.status`; `README.md:5-15`, `:642-652`, `:1362`; `audits/AUDIT_2026-08-17_embeddinggemma_causal.md` | **L** — EXP-A/B/E/F/G/H, 500/class × 21 policy columns, EmbeddingGemma-300M |
| 40 | biencoder_guard | falsifiers **(iv)–(vii)** `[PENDING RUN]`; **(ii)** and **(iii)** suspended | **b** | `README.md:1305-1310` | — (folds into #39) |
| 41 | biencoder_guard | `cross_annotator` arm (Aegis 2.0 `test`) `[PENDING RUN]` — wired, never executed | **b** | `README.md:1141` | M |
| 42 | biencoder_guard | `ood_benchmark` arm (`intrinsec-ai/cstm-bench`) `[PENDING RUN]` — wired, never executed | **b** | `README.md:1142` | S |
| 43 | biencoder_guard | §10.4 falsifier (i) still reads **"SURVIVES — claim upheld … 43.9×"**; the ratio is **CONTESTED** per CLAUDE.md §18.5's withdrawn 64×/40×/43.91× sequence | **d** (+ **b** to settle) | `README.md:1280-1300` records the withdrawal reasoning but the verdict row still asserts the ratio; needs quiet-GPU re-measure on both backbones with `gpu_witness.contended=false` | XS doc / M to settle |

---

## 3. Top 5 NEEDS-GPU items, ranked by how much they would change a stated conclusion

1. **#39 `biencoder_guard` full re-run under 5.15.0.** Every accuracy claim in the lesson is
   currently suspended, including falsifier (ii) ("zero-shot TRIPS", macro-AP 0.382) and (iii)
   ("hard-negative sharpening SURVIVES against a floor"). Both revert to *undecided* until this
   runs. Nothing else in the family has this many live conclusions in limbo.
2. **#37 `multiturn_jailbreak` first real run.** This is the only lesson in the family with **zero
   current metrics** — `results.json` is an honest placeholder. Three pre-registered falsifiers
   (F1 vs the 0.8584 content bar, F2 vs `last_turn_only`+0.02, F3 turn-shuffle) are waiting, and
   F2 is the one that decides whether the lesson's claim is *"escalation is detectable"* or merely
   *"the payload turn is recognisable"*. Its headline embedder is `embgemma`, so the run is
   newly valid rather than newly possible.
3. **#7 `trajguard` `overt` substrate.** F3 — the substrate contrast — is the *entire reason* the
   lesson was re-based, and it is the one falsifier with no verdict on either arm. The disguised
   arm already produced the surprise (F2 HOLDS, against prediction); the overt arm decides whether
   that is a property of disguised attacks or of the pipeline.
4. **#14 `cross_trajectory` EmbeddingGemma arm.** The lesson's headline is MiniLM and its only
   replication is a **hand-transcribed, non-regenerable** Gemma-decoder file (#16). Running the
   mandated encoder both satisfies the standing mandate and replaces inadmissible evidence with
   evidence — two defects with one run. Pair it with #16/#17 in the same sitting.
5. **#33 `meerkat` `bge` arm.** The paper's own encoder has never been run; the config says `bge`
   and the artifact says `minilm`, and the README already concedes the two fields disagree. This
   is the cheapest of the five (~900 embeddings) and it is the difference between "we reconstructed
   the paper" and "we ran a substitute and described the paper."

*Internal QA pass — independent external review pending (same-model-family circularity, CLAUDE.md §14).*
