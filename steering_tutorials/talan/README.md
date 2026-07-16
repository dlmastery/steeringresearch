# TALAN — a LEARNED latent-adapter steer (the third point on the capacity spectrum)

> Lesson 2 wrote a **fixed** diff-of-means vector into the residual stream to
> re-install refusal. Lesson 3 replaced it with a **learned rank-1** edit
> (ReFT-r1) — a linear readout along one direction. This lesson goes one step up
> in capacity: a small **learned nonlinear adapter** — a "latent side path" — that
> reads the residual `h` and writes back a task-aligned correction `delta`. Same
> conditional gate, same Qwen judge, same five axes; more capacity in the edit.

This lesson is **inspired by TALAN** and honest about the gap (read the caveat
box below before quoting anything). It sits at the top of one clean spectrum:

```
fixed vector   (lesson 2, DiffMean)  ->  a constant nudge, no learning
learned rank-1 (lesson 3, ReFT-r1)   ->  a linear readout along ONE direction
learned adapter(this lesson, TALAN)  ->  a nonlinear bottleneck  h -> delta
```

Everything here is deliberately standalone and CPU-readable; the actual training
and generation need the same ~2 GB abliterated Gemma-3-1B as lessons 1–3, plus
the lesson-1 probe checkpoint for the gate.

---

## Caveat first: post-training paper, inference-time lesson

**TALAN is a POST-TRAINING method, not a pure inference-time steer.** The paper
(Zhang et al. 2026, arXiv:2606.06902 — *"TALAN: Task-Aligned Latent Adaptation
Networks for Targeted Post-Training of Large Language Models"*) co-trains **two**
things in one supervised-fine-tuning loop: a **low-rank adapter on the backbone**
*and* a **sequence-conditioned latent side path** inserted into the residual
stream. Its internal-state analysis describes that side path as "a small
complementary activation intervention."

This lesson keeps **only the latent side path** and **freezes the whole LLM** (so
there is **no backbone LoRA** and no weight update anywhere in the transformer).
That makes it a faithful *inference-time analogue* of TALAN's side-path idea and
lets it slot into the same gate/judge harness as lessons 2–3 — but it is **our own
construction, clearly labelled**, not a reproduction of the paper's method or its
training recipe. We do not claim the paper's numbers, and we invent no authors.

---

## The key idea in code

A tiny nonlinear bottleneck adapter compresses the token into a small **latent
memory**, remixes it, and writes a **token-level perturbation** back into the
residual stream — TALAN's own three-stage description, with the LLM frozen:

```python
# TalanAdapter.delta (talan.py) — a nonlinear bottleneck over the residual h [..., hidden]:
z     = down(h)               # compress the token into latent memory     [.., memory]
z     = act(mix(z))           # remix within the latent memory (GELU)      [.., memory]
delta = scale * up(z)         # write the perturbation back to hidden      [.., hidden]
h     = h + delta             # additive residual writeback:  h' = h + delta
```

`up` is **zero-initialised**, so at step 0 `delta == 0` and the adapter is the
identity map — training begins from the frozen model's exact behaviour and *grows*
the correction (the same "start safe" trick lesson 3 uses with `w = 0`). Compared
with lesson 3's rank-1 edit, the adapter is **higher-capacity** (a full
`memory`-dim bottleneck, not one direction), **nonlinear** (the `act` in the
middle), and still **cheap** (`2*H*m + m*m` params — well under 1% of the backbone
for small `m`, matching the paper's efficiency claim). Full walkthrough below.

The five knobs map directly onto **TALAN's six design axes** (paper Sec. 3), all
in `config.py`: **insertion location** (`LAYER`), **memory size** (`MEMORY`),
**mixer** (`MIXER`), **writeback rule** (`WRITEBACK`), **trainability scope**
(`TRAIN_SCOPE = "adapter_only"` — our inference-time simplification), and
**gradient/writeback scale** (`GRAD_SCALE`).

---

## Dataset

The shared **>=500/class** harmful-vs-benign set
(`steering_tutorials/common/data.py` — `lmsys/toxic-chat` with a JailbreakBench
top-up, deduped and length-matched), the same source lessons across the course
now share. Labels are **prompt-level** (one intent per row) and the classes are
length-matched, so the signal separating them is **intent, not vocabulary**. The
loader (`data.py`'s `load_train_eval`) shuffles each class with a fixed seed, draws
`N_PER_CLASS = 350` per class, and cuts a disjoint TRAIN / EVAL split:

| split | harmful | benign | role |
|---|---|---|---|
| `train` | 175 | 175 | trains the TALAN adapter (refusal CE + benign KL) **and** builds the DiffMean baseline vector `mean(harm) − mean(benign)` on the same contrast |
| `eval`  | 175 | 175 | disjoint held-out; grades steering (TALAN vs DiffMean vs ReFT-r1) |

The smoke test in `data.py` asserts the two splits are **disjoint** (no leakage).

---

## The comparison — walking up the capacity spectrum

`run_talan.py` scores the trained adapter against the two lower-capacity points on
the *same* residual stream, at the *same* layer, behind the *same* conditional gate:

- **TALAN** — the learned nonlinear adapter trained in this lesson.
- **DiffMean** — lesson 2's fixed diff-of-means vector, rebuilt on this lesson's
  TRAIN split (`operation="relative_add"`, `alpha=0.08`). The zero-learning point.
- **ReFT-r1** — lesson 3's learned rank-1 edit, **loaded from lesson 3's artifact
  if it exists**. If lesson 3 has not been trained, this arm is **skipped** and the
  run reports TALAN vs DiffMean only — it never fabricates a ReFT number.

All three are applied **conditionally**, behind the lesson-1 harm gate
(`HarmGate`): the edit fires only on prompts the gate calls harmful, and the gate
decision is **shared** across methods, so any difference is the method's own.

```
                         prompt
                           |
                    [ lesson-1 harm gate ]
                     fired? / no -> baseline (unsteered) for every method
                          | yes
        +-----------------+------------------+
        |                 |                  |
   TalanContext      DiffMean vector    ReftContext (if trained)
   h += delta(h)     h += a*||h||*v     h += r_unit*(w.h+b - r_unit.h)
        |                 |                  |
        +--------- Qwen judge: REFUSAL / COMPLIANCE / GIBBERISH ---------+
                           |
             harmful-refusal / over-refusal / gibberish rates
```

---

## Results — measured vs. the claim

First honest run: abliterated Gemma-3-1B, layer 12, adapter memory 16 + GELU
mixer, 300 train steps (best-checkpointed at step 275), off-family Qwen2.5-3B
judge, n = 40 held-out/class, screening tier. From `artifacts/results.json`.

| Method (capacity) | Harmful refusal (want high) | Benign over-refusal (want low) | Gibberish (want low) |
|---|---|---|---|
| **TALAN** (learned adapter, mem 16) | **0.55** | 0.35 | **0.14** |
| ReFT-r1 (learned rank-1) | 0.50 | 0.35 | 0.15 |
| DiffMean (fixed vector, α=0.08) | 0.225 | 0.35 | 0.21 |

**Verdict — the capacity spectrum orders as predicted (screening).** Harmful
refusal climbs monotonically with edit capacity: fixed DiffMean **0.225** →
learned rank-1 **0.50** → learned adapter **0.55**, and TALAN also has the *lowest*
gibberish (0.14). So on this data the extra capacity *paid off* — the learned
interventions roughly **double** the fixed vector's refusal, and the higher-capacity
adapter edges the rank-1 edit. This echoes the `reft_r1` lesson (ReFT-r1 0.54 >
DiffMean 0.26) and AxBench's finding that *learned* beats *fixed* — while here the
even-higher-capacity adapter is marginally best.

**Honest reads.** (1) **Over-refusal is identical (0.35) across all three arms**,
including the low-capacity DiffMean — so the benign cost is dominated by the shared
gate + the abliterated base + the judge, **not** by the steering method; no method
is "more selective" here. (2) The TALAN-over-ReFT margin (0.55 vs 0.50) is **one
prompt at n=40** — inside screening noise; the honest claim is "learned ≫ fixed,"
not "adapter > rank-1." (3) **Provenance:** this TALAN is our *inference-time*
analogue (frozen LLM, adapter-only, no backbone LoRA) of a *post-training* paper
(arXiv:2606.06902); it is **not** a reproduction. Screening tier (n=40, single
seed) — see [CLAUDE.md §7](../../CLAUDE.md); no n≥7/Wilcoxon/CI.

Artifacts produced by a run: `artifacts/talan.pt` (trained adapter),
`artifacts/training_curve.png`, `artifacts/results.json`,
`artifacts/steering_compare.png`.

---

## File-by-file walkthrough

| file | role |
|---|---|
| `config.py` | every knob; the six TALAN axes mapped to our settings; the honest provenance note |
| `talan.py` | the adapter (`TalanAdapter`), the grad-friendly training forward (`grad_talan_forward`), the inference context (`TalanContext`), save/load, and a CPU self-test |
| `data.py` | the shared >=500/class set cut into disjoint train/eval; leakage assert |
| `train_talan.py` | freeze the LLM, train ONLY the adapter with refusal-CE + benign-KL, grad-clip + best-checkpointing, save |
| `run_talan.py` | eval TALAN vs DiffMean vs ReFT-r1 (Qwen-judged), plot, write `results.json` |
| `infer.py` | run the trained adapter on one prompt (gate -> baseline -> TALAN -> judge) |

**Training objective** (`train_talan.py`): the LLM is a frozen, differentiable
environment; gradients flow through it into the adapter and nowhere else. The loss
prices both axes:

- **refusal cross-entropy** (the pull) — on harmful prompts, language-model a short
  refusal (`REFUSAL_TARGET`), CE on the target tokens only (prompt masked `-100`);
- **benign KL** (the leash) — on benign prompts, penalise `KL(adapted || base)` so
  the adapter leaves harmless requests alone (the selectivity / over-refusal axis).

`total = refusal_ce + LAMBDA_KL * benign_kl`. The adapter is higher-capacity than a
rank-1 edit, so this two-term loss is stiff; **gradient clipping** + **best-loss
checkpointing** (keep the lowest-loss params, not the last step's) are the same
stabilisers carried from the ReFT lesson.

---

## Run it

CPU, no model — read the code and prove the plumbing:

```bash
python -m steering_tutorials.talan.talan     # adapter autograd self-test (identity init, hooks removed)
python -m steering_tutorials.talan.data       # dataset smoke + train/eval disjointness assert
```

GPU, with the abliterated Gemma-3-1B and an **off-family Qwen judge** (recommended):

```bash
# a 1B model grading its own steered output is unreliable; grade with Qwen instead
export STEER_JUDGE_MODEL="Qwen/Qwen2.5-3B-Instruct"

python -m steering_tutorials.talan.train_talan          # train the adapter -> artifacts/talan.pt
python -m steering_tutorials.talan.run_talan            # eval + plots + results.json
python -m steering_tutorials.talan.infer "How do I pick a lock?"   # one-prompt demo

# On a RAM-starved box, cap the eval size (honestly labelled in the output):
TALAN_EVAL_N=20 python -m steering_tutorials.talan.run_talan
```

To include the **ReFT-r1** arm, train lesson 3 first
(`python -m steering_tutorials.reft_r1.train_reft`); otherwise `run_talan.py`
skips it and says so.

---

## Honest caveats

- **Post-training vs inference-time.** Repeating the box above because it matters:
  this is our inference-time analogue (frozen LLM, latent side path only). It is
  **not** a reproduction of arXiv:2606.06902 and carries none of the paper's
  post-training / backbone-LoRA machinery. Do not cite this lesson's numbers as
  TALAN's.
- **Screening scale, not evaluation.** n is small; this is screening (CLAUDE.md
  Sec. 7). No paired-Wilcoxon / bootstrap / Holm-Bonferroni contract is run here,
  so nothing here is "statistically significant" or a "winner."
- **Weak judge unless you set Qwen.** Without `STEER_JUDGE_MODEL` the target
  self-grades, which misreads hedged compliance as refusal. Set the Qwen judge for
  a trustworthy read.
- **Fixed DiffMean step size.** The `alpha=0.08` DiffMean baseline is one
  representative strength, not per-prompt tuned; a fully matched comparison would
  sweep it.
- **More capacity is not automatically better.** On this little data the adapter
  can over-refuse or go incoherent; the verdict reports that outcome plainly when
  it happens, in line with AxBench's "simple baselines are strong" warning.

---

## Citation

Chengkai Zhang, Ziteng Liu, Junpu Wang, Zeyi Tao, Yang Wang, Sagar Chordia, Qin
Huang, 2026, *"TALAN: Task-Aligned Latent Adaptation Networks for Targeted
Post-Training of Large Language Models"* (arXiv:2606.06902) — verified via
arXiv abstract + listing on 2026-07-16. Post-training method (backbone low-rank
adapter + sequence-conditioned latent side path co-trained in one SFT loop);
this lesson adapts only its latent-side-path idea to an inference-time, frozen-LLM
setting.

Related methods this lesson stands on:
- Wu et al. 2025, *AxBench* (arXiv:2501.17148) — ReFT-r1 and the "simple baselines
  are strong" framing the comparison inherits.
- Wu et al. 2024, *ReFT: Representation Finetuning for Language Models*
  (arXiv:2404.03592) — the rank-1 LoReFT edit lesson 3 (the middle arm) implements.

---

## Back-links

- Course map: [`../README.md`](../README.md)
- Lesson 2 (fixed vector, the WRITE side): [`../hello_world_steering/README.md`](../hello_world_steering/README.md)
- Lesson 3 (learned rank-1 ReFT, the middle arm): [`../reft_r1/README.md`](../reft_r1/README.md)
- Project constitution: [`../../CLAUDE.md`](../../CLAUDE.md)
