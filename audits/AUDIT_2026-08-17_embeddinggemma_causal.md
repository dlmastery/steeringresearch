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

## Why this survived

The exact §18.8 pattern: it failed **silently and plausibly**. Nothing crashed. The
warning was printed once per run, in a log, next to numbers that looked reasonable.
Well-formed, confident, wrong.

**Rule this pays for:** *a library warning about a silently-ignored parameter is a
BLOCKER, not noise.* And: verify the behaviour, never the config field — the config
said `True` the whole time.
