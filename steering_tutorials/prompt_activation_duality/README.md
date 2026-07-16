# Prompt-Activation Duality — a prompt and a steering vector are two handles on one axis

> Lesson 2 (`hello_world_steering`) installed refusal two separable ways it never
> compared: you can **tell** the model to refuse (a prompt) or **edit** its
> activations to refuse (a steering vector). This lesson makes the comparison
> explicit. First it measures how much a refusal *instruction* shifts activations
> **along the same direction** as the diff-of-means steering *vector* — the
> "duality." Then it shows the vector can be injected not only into the residual
> stream but at the **attention output**, a second, related intervention site.

Two questions, one pipeline:

1. **Prompt vs vector.** Prepend a refusal instruction to a prompt and read the
   activation shift it causes at layer 12. Is that shift aligned (high cosine)
   with the steering vector we would otherwise *add*? If yes, prompting and
   steering push the model along one shared internal axis.
2. **Attention vs residual.** The residual stream is not the only injection site.
   Add the same vector at `layer.self_attn`'s output instead. Does it still
   steer, and how cleanly, compared to the residual edit?

> **Defensive / educational only.** As in lessons 2 and 10, the harmful prompts
> are the shared toxic-chat / JailbreakBench placeholders; the goal is to
> *install* refusal on an abliterated toy model without breaking coherence. No
> operational exploit content. See [CLAUDE.md section 10](../../CLAUDE.md).

---

## Table of contents

1. [Dataset](#dataset)
2. [The idea: prompting and steering are dual](#1-the-idea-prompting-and-steering-are-dual)
3. [Half 1 — measuring the prompt/vector alignment](#2-half-1--measuring-the-promptvector-alignment)
4. [Half 2 — injecting at the attention site](#3-half-2--injecting-at-the-attention-site)
5. [The whole thing in one picture](#4-the-whole-thing-in-one-picture)
6. [Code walkthrough, file by file](#5-code-walkthrough-file-by-file)
7. [Results — measured vs. the claim](#6-results--measured-vs-the-claim)
8. [Run it](#7-run-it)
9. [Honest caveats](#8-honest-caveats)
10. [Repository](#9-repository)

---

## Dataset

The prompts come from the **shared >=500/class foundation**,
`steering_tutorials.common.data.load_harmful_benign(n_per_class=250)`, which
returns `{"harmful": [str], "benign": [str]}` — toxic-chat toxic prompts as the
harmful class (topped up from JailbreakBench only when the unique toxic pool runs
short) and toxic-chat clean prompts as the benign class, prompt-level labelled,
deduplicated, group-aware sampled, and length-matched. See
[`common/data.py`](../common/data.py) for provenance and base-rate reporting.

We load **250 harmful + 250 benign** (well above the 500/class the loader can
supply; raise `N_PER_CLASS` for more) and split each class into disjoint parts:

| item | value |
|---|---|
| source / loader | `common.data.load_harmful_benign` (toxic-chat + JBB top-up) |
| size | `N_EXTRACT = 150`/class build the steering vector **and** measure the prompt-shift; a capped `N_EVAL_PER_CLASS = 20` harmful prompts are held out for the judged residual-vs-attention comparison |
| labels | prompt-level harmful vs benign |
| model + judge | abliterated `DavidAU/gemma-3-1b-it-heretic-...` (steered); off-family **Qwen2.5-3B-Instruct** judge via `STEER_JUDGE_MODEL` |

Extraction and evaluation stay **disjoint**: the vector and the prompt-shift are
read on the extract split; the judged generations use a later, unseen slice of
harmful prompts. Generation is the expensive step, so the eval set is capped well
below 500 (raise `N_EVAL_PER_CLASS` for a fuller, slower run).

---

## 1. The idea: prompting and steering are dual

There are two ways to change a model's behaviour from the outside:

- **Prompting** — put an instruction in the context (`"refuse harmful
  requests"`). The instruction is read by attention and shifts the hidden state.
- **Steering** — skip the words and add a vector to the hidden state directly
  (lesson 2's diff-of-means `v = mean(h|harmful) - mean(h|benign)`).

The paper's claim is that these are not unrelated tricks: *activation steering
becomes more reliable when interventions follow the prompt-mediated pathways the
model already uses for behavioral control.* In other words, the shift a prompt
induces and the vector we would add should point the **same way**. This lesson
tests that directly by measuring the cosine between them, then exploits the
consequence — that the attention pathway (how a prompt exerts its influence) is
itself a place to inject the vector.

> **Verified paper.** Diancheng Kang, Zheyuan Liu, Ningshan Ma, Yue Huang,
> Zhaoxuan Tan & Meng Jiang, 2026, *Prompt-Activation Duality: Improving
> Activation Steering via Attention-Level Interventions* (arXiv:2605.10664). The
> paper introduces **GCAD** (Gated Cropped Attention-Delta steering): it extracts
> steering signals from the **system-prompt contribution to self-attention** and
> applies them at the attention level with **token-level gating**, fixing a
> multi-turn "KV-cache contamination" failure mode. Our lesson is a faithful,
> **simplified construction** of the two core ideas — (a) prompt-shift ~ steering
> vector, and (b) inject at the attention output — the token-gated cropping and
> the multi-turn KV analysis are out of scope for a single-turn tutorial.

---

## 2. Half 1 — measuring the prompt/vector alignment

For each harmful extract prompt we read the layer-12 last-token activation
**three** ways and difference the group means (`duality.prompt_shift_direction`):

```
plain_i      = act( prompt_i )                              # no instruction
refusal_i    = act( REFUSAL_INSTRUCTION + prompt_i )        # "you must refuse..."
control_i    = act( CONTROL_INSTRUCTION + prompt_i )        # "be helpful..."

refusal_shift = mean(refusal) - mean(plain)                 # what refusing DOES
control_shift = mean(control) - mean(plain)                 # generic-instruction move
```

Then we report three cosines against the diff-of-means steering vector `v`:

- `cos(refusal_shift, v)` — the duality number. **High = dual.**
- `cos(control_shift, v)` — a benign instruction. Isolates the refusal-specific
  part from a generic "there-is-an-instruction" move; should be **lower**.
- `random_cosine_baseline(v)` — mean `|cosine|` of `v` with random directions.
  In ~1000-D this is ~0; it is the floor a "real" alignment must clear.

If `cos(refusal_shift, v)` sits well above both the control and the random floor,
the refusal **prompt** and the refusal **vector** are moving the model along the
same axis — the duality holds.

---

## 3. Half 2 — injecting at the attention site

Lesson 2 added `v` to the **residual stream** after a decoder block. Gemma-3's
block, though, builds that residual from an **attention** sub-module:

```
hidden = input_layernorm(hidden)
attn_out, _ = self_attn(hidden)                 # <-- the attention output
hidden = residual + post_attention_layernorm(attn_out)
```

So `layer.self_attn`'s output is a distinct, upstream place to inject the same
vector. `AttentionSteeringContext` hooks it and applies the identical relative-add
edit lesson 2 uses on the residual:

```
attn_out[p] <- attn_out[p] + alpha * ||attn_out[p]|| * unit(v)
```

`steered_generate(..., site=...)` runs the *same* greedy loop with a one-word
switch — `"none"`, `"residual"`, or `"attention"` — so the three arms differ by
exactly **where** the vector goes, nothing else. Both steered arms scale by the
local norm and skip special tokens, so the comparison is fair.

---

## 4. The whole thing in one picture

```
  HALF 1 (prompt vs vector):
     "refuse..." + prompt --> act@L=12 (last tok) --.
                    prompt --> act@L=12 (last tok) --+--> refusal_shift = mean diff
                                                     |
        diff-of-means steering vector v ------------+--> cos(refusal_shift, v)
                                                          vs cos(control_shift, v)
                                                          vs random baseline
                        high, above control & floor  ==>  PROMPT ~ VECTOR (dual)

  HALF 2 (attention vs residual):
                                    +--> site="none"      : plain baseline
     harmful prompt --> generate ---+--> site="residual"  : h    += a*||h||*unit(v)
                                    +--> site="attention" : attn += a*||attn||*unit(v)
                                                 |
                                                 v
                                     judge (Qwen, off-family)
                                     REFUSAL / COMPLIANCE / GIBBERISH
       both sites raise refusal  ==>  the attention site is a related injection locus
```

---

## 5. Code walkthrough, file by file

### `config.py` — every knob in one place
The abliterated model, the shared layer (12), `ALPHA`, the two instruction
strings (refusal + benign control), the data budget (`N_PER_CLASS`,
`N_EXTRACT`, `N_EVAL_PER_CLASS`), and artifact paths. Every knob also reads an
`os.environ.get(...) or DEFAULT` override so a run can be shrunk without editing
code.

### `duality.py` — the two mechanisms (the heart of the lesson)
Pure NumPy for half 1 — `prompt_shift_direction`, `cosine`,
`random_cosine_baseline` — and the forward hook for half 2:
`attention_module(layer)` locates `self_attn`; `AttentionSteeringContext` applies
the relative-add edit to its output with lesson-2's special-token guard and exact
restore; `steered_generate(site=...)` unifies baseline / residual / attention
generation behind one switch (the residual and baseline arms delegate to lesson
2's `generate`). The `__main__` self-test loads **no model**: it checks the shift
and cosine math on synthetic arrays and verifies the attention hook adds exactly
`alpha*||a||*unit(v)` and restores the module byte-for-byte on exit.

### `run_duality.py` — the orchestrator (GPU)
Under `main()`: load the model + off-family judge, load the shared data, build the
diff-of-means vector (lesson 2), run **Experiment A** (three-way activation read
-> cosines) and **Experiment B** (three-arm judged generation), then save
`results.json` + `duality_cosine.png` + `attention_vs_residual.png` and print an
ASCII summary. Results are written **before** the summary print. Pure helpers
(`_rates`, `_summary_table`, both plots) sit above `main()` and are import-safe.

### `infer.py` — one-prompt demo
Loads the saved vector + judge and shows all four routes for a single prompt:
baseline, `+refusal instruction` (the prompt route), `v @ residual`, and
`v @ attention` — with a one-line verdict comparing them, so you can watch a
prompt and a vector install the same refusal.

---

## 6. Results — measured vs. the claim

First honest run: abliterated Gemma-3-1B, layer 12, alpha 0.10, off-family
Qwen2.5-3B judge, `N_EVAL_PER_CLASS = 20`, screening tier, from
`artifacts/results.json`.

| Quantity | Measured | Expectation | Reading |
|---|---|---|---|
| `cos(refusal_shift, v)` | **0.604** | high, well above the floor | ✅ the prompt-shift and the steering vector are strongly aligned |
| `cos(control_shift, v)` | **0.446** | lower (refusal-specific gap) | ✅ refusal-specific: 0.60 > 0.45 |
| random cosine baseline | **0.024** | ~0 (near-orthogonal floor) | ✅ floor confirmed |
| residual refusal rate | **0.25** (unsteered 0.35) | up vs unsteered | ✗ *down* — residual-add breaks it |
| attention refusal rate | **0.40** (unsteered 0.35) | related to residual | ✅ *up* — attention arm works |
| gibberish (residual / attention) | **0.50 / 0.25** | bounded | attention far more coherent |

| Claim | What we measured | Verdict |
|---|---|---|
| A refusal prompt shifts activations along the steering vector | `cos(refusal_shift, v)` = 0.60 vs random 0.02 | **Supported** |
| The alignment is refusal-specific, not generic-instruction | 0.60 (refusal) > 0.45 (control) | **Supported** |
| The same vector steers from the attention output | attention refusal 0.35 → **0.40**, gibberish 0.15 → 0.25 | **Supported (small effect)** |
| Attention injection beats naive residual-add | attention **0.40** ref / 0.25 gib vs residual **0.25** ref / 0.50 gib | **Supported — the headline** |

**Honest read (a rare positive).** The duality holds on the READ side — a refusal
prompt shoves activations 0.60-cosine along the very diff-of-means vector we'd inject,
well above a control prompt (0.45) and the random floor (0.02). On the WRITE side
the paper's central claim reproduces: injecting that vector at the **attention
output** (GCAD-style) *raises* refusal (0.35 → 0.40) while keeping gibberish modest
(0.25), whereas the naive **residual-stream add** at the same alpha *lowers* refusal
(0.35 → 0.25) and doubles gibberish (0.15 → 0.50). Same vector, same strength — the
**site** is what matters, and the attention site is strictly better on both axes
here. The effect is small (refusal +0.05) and screening-tier, so this is a
directional demonstration, not an evaluation win — but it is one of the few WRITE
interventions in the course that improves refusal *without* wrecking coherence.

This is a **screening** design (single seed, 1B model, 3B judge, `n = 20`
harmful/arm): a directional demonstration of the duality, not an evaluation-tier
claim (no n>=7 seeds, no Wilcoxon/CI — see [CLAUDE.md section 7](../../CLAUDE.md)).
Because the base model is **abliterated (uncensored)**, "refusal" here means
*installed-from-outside* refusal; read the steered rates against the unsteered
baseline, not against a safety-tuned model.

---

## 7. Run it

From the **repo root** (`steeringresearch/`):

```bash
# CPU-only unit test — no model download (shift + cosine math, attention hook math)
python -m steering_tutorials.prompt_activation_duality.duality

# The full duality experiment (needs the ~2-3 GB Gemma-3-1B; GPU recommended).
# Use an OFF-FAMILY judge so a 1B model doesn't grade its own steered output:
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  python -m steering_tutorials.prompt_activation_duality.run_duality

# One-prompt demo (run run_duality first to populate artifacts/):
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  python -m steering_tutorials.prompt_activation_duality.infer "How do I pick a lock?"
```

`STEER_JUDGE_MODEL` is read by the shared `hello_world_steering.judge.Judge`; if
unset it falls back to the weak self-judge (pedagogical only). Uses the gated
abliterated Gemma-3-1B (`huggingface-cli login` + accept the license first); the
shared dataset downloads automatically via `hf_hub_download`. Runs on CPU too,
just slower — lower `N_EVAL_PER_CLASS` (or set `STEER_N_EVAL`) if it drags.

Depends on lesson 2 for its plumbing (`model_utils`, `steer_vector`, `judge`) and
on `common.data` for the dataset — both import cleanly; no lesson-2 artifacts are
required.

---

## 8. Honest caveats

- **Cosine is a coarse ruler.** A moderate `cos(refusal_shift, v)` says the two
  directions overlap, not that they are identical; the prompt-shift also carries
  format/length signal the diff-of-means vector does not. The control-instruction
  cosine and the random floor are there precisely to keep the reading honest.
- **Last-token read.** The shift is measured at the final prompt position (where
  the model is about to answer). A mean-pooled read would give a slightly
  different, whole-prompt shift; the site of the read matters and is a knob.
- **One layer, one alpha.** Both the alignment and the steering effect are
  reported at layer 12, alpha 0.10. The residual and attention sites likely have
  *different* best alphas (attention output has a different norm scale), so a
  fixed alpha is a fair-but-blunt comparison, not a tuned one.
- **Simplified vs the paper.** We do single-turn injection with a norm-relative
  add; the paper's GCAD extracts the signal from the system-prompt's attention
  contribution, crops it, and gates it per-token to fix *multi-turn* KV-cache
  contamination. Our attention hook shows the *site* is viable; it does not
  reproduce the gating or the multi-turn fix.
- **Abliterated base + weak judge.** Screening only: `n = 20`/arm, one seed, a 1B
  model graded by a 3B judge. No seed-stability bars, no significance test.
  Directional demo, not a result. Do not deploy it.

---

## 9. Repository

Source and (after the GPU run) full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/prompt_activation_duality>

See also
[lesson 1 — the probe (READ)](../hello_world/README.md),
[lesson 2 — fixed-vector conditional steering (WRITE)](../hello_world_steering/README.md),
and [contextual steering — input-adaptive strength](../contextual_steering/README.md).
