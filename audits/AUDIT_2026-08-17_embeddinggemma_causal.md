# CRITICAL: EmbeddingGemma has been running CAUSAL, not bidirectional

**Found:** 2026-08-17, while pre-flighting the STA SHADE run.
**Severity:** every EmbeddingGemma number in this repo is affected.
**Status:** CONFIRMED by behavioural test, not by reading a warning.

---

## The finding

`google/embeddinggemma-300m` is a **bidirectional** encoder. Its config sets
`use_bidirectional_attention=True`. The installed **transformers 4.55.0 contains ZERO
references to that parameter** — it does not exist in the library:

```
$ grep -rl "use_bidirectional_attention" site-packages/transformers/ | wc -l
0
```

So the flag binds to nothing and is silently dropped. transformers emits a warning
(*"may be silently ignored … likely producing degraded model outputs"*) which appears
in `_sta_agentdojo.log` and was not acted on.

## The behavioural proof (not the warning — the behaviour)

Prefix-invariance test. Two sequences sharing a 9-token prefix, different suffixes.
Under **causal** attention a prefix token cannot see the suffix, so its hidden state is
identical across both. Under **bidirectional** it must change. Read off the backbone's
last hidden state (not the pooled embedding, which would confound with pooling):

```
config.use_bidirectional_attention = True
config._attn_implementation        = sdpa

max |h_a - h_b| over the 9 SHARED prefix tokens: 0.000000e+00
per-token max abs diff: 0.00e+00 x9
```

**Exactly zero.** Not "small" — bit-identical. The model is running strictly causal.

## Blast radius

Four lessons embed with EmbeddingGemma:

| lesson | what is affected |
|---|---|
| `biencoder_guard` | **both towers**; every AP/AUC in `artifacts/results.json` |
| `streaming_trajectory_aggregation` | all trajectory embeddings; the F4 verdict |
| `cross_trajectory` | embeddings + the Gemma-embedder ablation |
| `multiturn_jailbreak` | embeddings |

**The most consequential implication is for `biencoder_guard`.** Its headline was that
the bi-encoder LOSES to a trained head on seen labels (macro-AP 0.240 frozen / 0.575
contrastive vs 0.658) and that binary harm AUC sits at ~0.59 against a 0.526 length
confound — i.e. near-useless. **A sentence encoder crippled to causal attention is a
sufficient explanation for all of that**, and it was read as a fact about bi-encoders.
That reading is now unsafe and the lesson's conclusion is SUSPENDED pending a re-run.

This does **not** automatically mean the numbers will improve — it means they were
produced by an instrument that was not the instrument claimed, so they cannot be
cited either way until re-measured. Same category as the flas divergence: the defect
is that the artifact does not measure what it says it measures.

## Cached artifacts are now invalid

Every cached `.npz` embedding produced under this install was computed causally:

- `streaming_trajectory_aggregation/artifacts/embeddings/agentdojo_embgemma_*.npz`
- `biencoder_guard/artifacts/emb_train_embeddinggemma.npz`,
  `emb_ood_embeddinggemma.npz`, `policy_bank_embeddinggemma.npz`,
  `opir_taxonomy_embeddinggemma.npz`, `distractor_bank_embeddinggemma.npz`
- `cross_trajectory` / `multiturn_jailbreak` equivalents

The cache keys fingerprint the *data*, not the *attention mode*, so a re-run after the
fix would silently reuse causal vectors. **The caches must be invalidated, and the
cache key should include an encoder-behaviour fingerprint so this cannot recur** —
the same "stamp your inputs" rule that `meerkat`'s `pool_fingerprint` exists for.

## Fix

`pip install -U "transformers>=4.56.2"`. The repo pin is already `transformers>=4.55`
(`requirements.txt:5`, `pyproject.toml:18`), so the upgrade is in-spec and needs no
pin change.

**Upgrade risk to check before trusting anything downstream:** the steering lessons
hook `residual_layers()` on Gemma-3-1B under 4.55.0. A minor-version bump can move
module paths. The upgrade must be followed by (a) re-running the bidirectional test,
(b) re-running `flow.py`/`model_utils` self-tests, (c) confirming a Gemma-3-1B
forward + hook still works — before any GPU experiment is launched on top of it.

## FOLLOW-UP NOT YET DONE — deliberately deferred

**Add an encoder-behaviour fingerprint to every embedding cache key.** The keys
currently fingerprint the *data* only, so a re-run after a library change silently
reuses vectors computed under different attention semantics. That is the mechanism
that would have let this bug survive its own fix.

**Deferred on purpose:** the STA SHADE run was already in flight when this was
identified, and changing the key mid-run would have invalidated the embeddings being
computed and thrown away the run. Do it *after* SHADE lands, not during.

Sketch: fold `transformers.__version__` **and** a cheap behavioural probe (the
prefix-invariance delta from `audits/test_embedder_bidirectional.py`, quantised) into
the key alongside the data fingerprint. Version alone is not enough — the point of
this incident is that a config field and a library version both said "bidirectional"
while the behaviour said otherwise.

## Why this survived

The exact §18.8 pattern: it failed **silently and plausibly**. Nothing crashed. The
warning was printed once per run, in a log, next to numbers that looked reasonable.
Well-formed, confident, wrong.

**Rule this pays for:** *a library warning about a silently-ignored parameter is a
BLOCKER, not noise.* And: verify the behaviour, never the config field — the config
said `True` the whole time.

---

## Addendum 2026-08-18 — the fix is in, and it retro-flags my own SHADE run

The behaviour fingerprint is now folded into both lessons' cache keys
(`biencoder_guard/encoder_behaviour.py`), and its self-test demonstrates the case
that motivated all of this: a cache with the **same library versions** but
**different measured behaviour** is rejected — *"versions alone would have passed
it."*

**It also correctly flags the two caches I created yesterday.** The SHADE (172 MB)
and AgentDojo (24 MB) STA embedding caches were written *before* the fingerprint
existed, so they carry no behaviour block. Under the new rule they are
**unattributable and will be rejected on any re-run**, forcing a re-embed.

That is the right behaviour, and it should not be softened. But the honest
statement about the results already committed is narrower than either "verified"
or "suspect":

- **The SHADE result is correct.** It was run *after* the transformers 5.15.0
  upgrade, after the causal caches were quarantined, and after the prefix
  test measured 5.74 (bidirectional). The encoder was right.
- **The artifact cannot prove that on its own.** The provenance lives in the
  session's commit history, not in `results_shade.json`. By this repo's own
  standard — *an artifact that cannot be regenerated from the code beside it is
  not evidence* — that is a real weakness, just a different one from a wrong
  encoder.

**Do NOT retroactively write a behaviour block into those artifacts.** Fabricating
provenance into an existing file is worse than lacking it. The stamp lands
naturally when the run is next repeated; until then this note is the provenance,
and it is deliberately external to the artifact so it reads as the weaker claim it
is.
