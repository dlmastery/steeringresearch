# Larger dataset survey — choosing the scale-up for the harmful-vs-benign probe

The starter probe (`data.py`) trains on **JailbreakBench**: 100 harmful + 100
benign behaviors (200 prompts). That is fine for a smoke test but too small to
say anything real. This note surveys larger, **prompt-level** safety datasets and
justifies the pick used by `data_large.py`.

Requirement: **clear prompt-level binary labels** (harmful/unsafe *intent* vs
benign), with both classes ideally drawn from the **same distribution** (so a
probe can't cheat on source style), and enough volume for a balanced ~1.5k set.

## Candidates (all verified against the live Hub, 2026-07-13)

| Dataset | Size | Label granularity | Both classes? | Gated | License | Verdict |
|---|---|---|---|---|---|---|
| **lmsys/toxic-chat** (`0124`) | ~10.2k prompts | **prompt-level** (`toxicity`, `jailbreaking` on the user input) | Yes — 746 toxic / 9419 benign | No | CC-BY-NC-4.0 | **CHOSEN** |
| allenai/wildjailbreak | ~262k | prompt-level (`vanilla_harmful`/`vanilla_benign`) — ideal | Yes | **Yes (401)** | AI2 ImpACT | Rejected: gated, no access on this machine |
| PKU-Alignment/BeaverTails (`30k`) | 27k rows / 7.8k prompts | **pair-level** (`is_safe` on prompt+**response**) | No genuine benign | No | CC-BY-NC-4.0 | Rejected: every prompt is red-team; "all-safe" prompts are ~60% harmful-but-refused |
| TrustAIRLab/in-the-wild-jailbreak-prompts | 13.2k regular / 1.4k jailbreak | prompt-level (jailbreak vs regular) | Yes | No | MIT | Rejected: "regular" split is NSFW/roleplay templates (not benign); ~2x length gap confound |
| jackhhao/jailbreak-classification | 1998 (1332 benign / 666 jailbreak) | prompt-level (jailbreak vs benign) | Yes | No | Apache-2.0 | Runner-up: clean but benign = generic NLP tasks (different distribution); caps at 666/class |
| AlignmentResearch/ClearHarm | 179 harmful (neg split empty) | prompt-level | No | No | MIT | Rejected: far too small |
| walledai/{AdvBench,HarmBench,StrongREJECT}, sorry-bench | 0.3–0.5k each | prompt-level, **harmful-only** | No | mixed | mixed | Rejected: single-class (no benign) |

## Why Toxic-Chat

1. **True prompt-level intent.** The `toxicity` label is on the *user input*, not
   a model response — exactly the signal a prompt classifier needs. (Contrast
   BeaverTails, whose `is_safe` grades the *response*: the same prompt appears as
   both safe and unsafe, and because every BeaverTails prompt is a red-team
   prompt, its "all-safe" prompts are mostly *harmful requests that got refused*,
   not benign ones. There is no real benign class to draw.)
2. **Same distribution for both classes.** Both toxic and benign prompts are real
   user queries from one live-chat stream, so a probe can't win by detecting a
   source/topic shift (the failure mode you get from gluing a red-team set onto a
   benign instruction corpus like Alpaca).
3. **Real, human-annotated, realistic.** ~10k genuine in-the-wild prompts with
   expert human toxicity/jailbreak annotation — the closest thing to production
   safety traffic among ungated options.
4. **Available.** The ideal set (wildjailbreak's matched `vanilla_harmful` /
   `vanilla_benign`) is gated and returns 401 here; Toxic-Chat is ungated and
   loads via the same `hf_hub_download` transport `data.py` already uses.

## Caveats (documented, not hidden)

- **Class ceiling ~700.** Toxic prompts are only ~7% of traffic (695 unique after
  cleaning/dedup). A fully-balanced set therefore caps at **~695/class (≈1390
  prompts)**, not 750/class. `load_large_dataset` requests 750 but transparently
  caps to what's available and logs a NOTE. Realistic safety data *is*
  imbalanced — this is the honest ceiling from one dataset. To reach a true
  750/class you would add a second split/dataset (e.g. the runner-up
  jailbreak-classification), which the loader is structured to make easy.
- **Mild length skew.** Toxic prompts run a little longer (median ~178 vs ~58
  chars) because many are jailbreak attempts; we cap at 4000 chars but do **not**
  otherwise truncate. Worth checking the probe isn't leaning on length (the
  existing leakage audit covers this).
- **`jailbreaking ⊆ toxicity`.** In the `0124` release every `jailbreaking==1`
  prompt is also `toxicity==1`, so `toxicity` alone is the clean single label;
  we don't need to OR the two columns.
- **License is CC-BY-NC-4.0** — research/non-commercial use only.

## Swapping datasets later

Changing the training set is a **one-line import swap** in `train_probe.py`:

```python
# from .data import load_safety_dataset        # JailbreakBench (200 prompts)
from .data_large import load_large_dataset      # Toxic-Chat (~1390 prompts)
```

Both loaders return the same `(prompts: list[str], labels: list[int])` contract
(1 = harmful, 0 = benign), so nothing downstream changes. A different dataset is
just a different `load_*` function behind the same signature.
