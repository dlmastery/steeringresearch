# Control/data split — the impossibility theorem and ASIDE's rotation, made measurable

> **Reference:**
> - [ASIDE: Architectural Separation of Instructions and Data in Language Models (arXiv:2503.10566)](https://arxiv.org/abs/2503.10566)
>   — Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Alexandra Volkova,
>   Soroush Tabesh, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert;
>   v1 13 Mar 2025, v4 9 Feb 2026, **ICLR 2026**. WebFetch-VERIFIED 2026-08-29.
>   *Relevance:* the isoclinic rotation this lesson reproduces exactly
>   (`aside.py`), and the layer-wise separability + cosine-trajectory claims this
>   lesson measures without any fine-tuning. The `pi/2` ablation-optimum angle is
>   **not stated on the abstract page** — carried here as `[UNVERIFIED-DETAIL]`
>   until someone reads the full text; it is a config knob (`config.ANGLE`), not
>   a hard-coded constant, precisely so "pi/2 is special" stays testable.
> - [On the Inseparability of Instructions and Data in Shared-Embedding Sequence
>   Models (arXiv:2606.27567)](https://arxiv.org/abs/2606.27567) — Pant, Lohani,
>   Kumar; 25 Jun 2026 (cs.CR). WebFetch-VERIFIED 2026-08-29. *Relevance:* the
>   impossibility proof this lesson turns into a measured number
>   (`inseparability.py`): the Bayes-optimal error of ANY provenance detector is
>   `(1 - TV(P_trusted, P_untrusted)) / 2`, estimable on our own corpus rather
>   than cited abstractly.
> - Related 2026 architectural/system defenses this lesson does **not**
>   reproduce, kept here as the neighborhood ASIDE sits in (arXiv ids
>   WebFetch-VERIFIED by the team lead 2026-08-29; full author/venue strings not
>   independently re-fetched by this file, so treat those two fields as
>   provisional pending a direct check):
>   [SecOPD (arXiv:2608.21500)](https://arxiv.org/abs/2608.21500),
>   [CaMeL — Defeating Prompt Injections by Design (arXiv:2503.18813)](https://arxiv.org/abs/2503.18813)
>   (its **"SaTML 2026" venue is `[UNVERIFIED-DETAIL]`** — not on the abstract page),
>   [DRIP (arXiv:2511.00447)](https://arxiv.org/abs/2511.00447),
>   [Meta SecAlign (arXiv:2507.02735)](https://arxiv.org/abs/2507.02735),
>   [COPA (arXiv:2608.19982)](https://arxiv.org/abs/2608.19982),
>   [PICO (arXiv:2504.21029)](https://arxiv.org/abs/2504.21029).

> **Status: geometry built and CPU-verified; the GPU pass has NOT run.**
> `aside.py` (the rotation) and `inseparability.py` (the Bayes floor) are
> pre-existing, verified spine. This drop adds the corpus (`data.py`), the
> analysis layer that turns real activations into the paper's two training-free
> claims (`separability.py`), and this README. Every number below that needs a
> forward pass through Gemma-3-1B is marked **[PENDING RUN]** — nothing here is
> invented.

---

## 1. What this lesson can and cannot show you

`aside.py`'s own docstring is blunt about this and it is worth repeating instead
of burying it:

**CAN reproduce here, with no training at all** (the whole point — the rotation
is a fixed change of basis, an arithmetic fact, not an empirical one):
- the rotation itself: orthogonal, norm-preserving, parameter-free, role-masked
  (`aside._self_test`, already passing);
- layer-wise linear-probe separability of instruction vs. data ROLE on real
  Gemma-3-1B activations (`separability.layer_separability_sweep`);
- the cosine-similarity trajectory of rotated-vs-vanilla hidden states by layer
  (`separability.cosine_trajectory`);
- the same two measurements for a training-free stand-in for ISE's learned
  offset (`aside.learned_offset_baseline` fed a difference-of-means vector from
  `separability.diff_of_means_offset`);
- the surface-level (no model) Bayes floor on our own instruction/data text pools
  (`inseparability.estimate_provenance_bound`, wrapped as
  `separability.provenance_floor`).

**CANNOT reproduce here** — and reporting any of these numbers from this package
would be measuring our own omission, not the method:
- the paper's **SEP scores** (Llama-2-7B 68.7→81.0, Mistral-7B 48.0→92.1,
  Qwen3-8B 45.3→71.4);
- the paper's **ASR reductions** (BIPIA-text 14.7→4.9, StruQ-ID 45.6→28.1);
- anything about **utility** after the rotation.

All three need **SFT on the modified forward pass** — the paper trains the model
*with* the rotation present from the start. A rotation applied to a model that
was never trained with it will **degrade** that model; a Gemma-3-1B forward pass
with mid-stream rotation and no adaptation is expected to look worse than
vanilla on any task-utility metric, and that expected degradation is not a
finding about ASIDE.

---

## 2. The limit that matters more than any number here

The rotation is only as strong as the **role channel** that decides which
documents are "data". If role were inferred from a delimiter or a register cue
inside the text, an attacker who controls the text forges the cue and the split
never happens — this is exactly `arXiv:2606.27567`'s point, and it is why
`aside.rotate_embeddings` takes an explicit boolean mask and has **no path** to
parse roles out of a string. `data.py` enforces the same discipline one level up:
role is assigned by which **loader** a document came from, decided in code
before the text is ever read, and the self-test (`CDS_SELFTEST=1`) checks that
the label survives a shuffle and is recoverable from the construction metadata,
never from a substring of the text.

---

## 3. The corpus, honestly

`data.load_role_corpus()` builds two pools, **>=500 documents per role** (the
CLAUDE.md §17 rubric — no tiny datasets):

| role | source | licence | what it stands in for |
|---|---|---|---|
| **instruction** (`is_data=False`) | [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`train_sft`, first user turn per conversation) | mit | a real user's direct chat request — the textbook TRUSTED channel |
| **data** (`is_data=True`) | [`lmsys/toxic-chat`](https://huggingface.co/datasets/lmsys/toxic-chat) + [`JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), via the shared `common.data.load_harmful_benign` loader | cc-by-nc-4.0 (non-commercial) / mit | content routed through the untrusted/retrieved channel by the application's own decision — **not** a synonym for "harmful": roughly half of this pool is the loader's *benign* class |

Both real sources are already vetted in `common.dataset_export.REDISTRIBUTABLE`;
`corpus_manifest()` calls `assert_redistributable` on each and raises loudly if a
future edit ever adds an unvetted one. `lmsys/toxic-chat` is **non-commercial**
and that restriction travels with any exported slice, per
`dataset_export.py`'s licence gate.

**The honest confound.** These two pools are not the same distribution of
English text. UltraChat openers are long, detailed, multi-clause requests;
toxic-chat/JBB prompts are shorter, blunter, imperative sentences. Any
separability this lesson measures — surface-level (`inseparability.py`) or
activation-level (`separability.py`) — is real separability of **these two
concrete pools**, not proof that "instructions and data always look different."
That is exactly why `separability.provenance_floor` (the surface Bayes floor,
with its same-distribution control) is reported **beside** every activation-level
AUC, never instead of it: a layer-L probe that barely beats the unigram floor is
not telling you much the residual stream added on top of vocabulary alone.

**The document-level simplification.** The paper rotates individual **data
tokens** inside a shared instruction+data template. This lesson's role label is
per **document** (an entire prompt is instruction-role or data-role), which is
the natural unit for two disjoint text pools built from different datasets. When
`separability.extract_hidden_states` runs, it rotates every token of a data-role
document uniformly. This is a real simplification worth stating plainly, not
hiding: it measures "does the model's residual stream treat a whole rotated
document differently", not the paper's finer-grained "does it treat one rotated
token within a mixed instruction+data prompt differently." Both are legitimate
questions; only the first is what this package answers.

---

## 4. What `separability.py` measures, and how

Three training-free claims, each built on `probe_auc` (held-out logistic-
regression ROC-AUC for role) and `cosine_trajectory` (paired cosine similarity by
layer):

1. **Layer-wise separability.** `layer_separability_sweep(activations_by_layer,
   is_data)` runs `probe_auc` at every layer in `config.LAYERS` (0 = embedding
   output, 4..26 = post-block, spaced across Gemma-3-1B's 26 transformer
   blocks). The paper's claim: ASIDE is ~100% separable from the embedding layer
   onward while vanilla only becomes separable later. **[PENDING RUN]**
2. **The cosine trajectory.** `cosine_trajectory(rotated_by_layer,
   vanilla_by_layer)` — mean cosine similarity, per layer, between the
   rotated-forward and vanilla-forward hidden state of the **same** document.
   The paper's claim: this starts near 0 at the embedding layer and **rises**
   toward ~0.7–0.8 by mid-depth — later layers partially, but never fully, undo
   the rotation. **[PENDING RUN]**
3. **The ISE contrast.** The *same two measurements*, run against
   `aside.learned_offset_baseline` instead of the rotation, using a
   training-free stand-in for the "learned" offset:
   `diff_of_means_offset(embeddings, is_data)` (the data-role mean minus the
   instruction-role mean at a chosen layer — the natural offset without an SFT
   run). The paper's claim: ISE collapses toward vanilla (cosine > 0.9) where the
   rotation does not, because a rotation's displacement is **input-dependent**
   (`aside.py`'s own self-test: `resid[mask]` is not constant across rows) and no
   single downstream bias can cancel it, while a constant offset is exactly what
   one bias term *can* cancel. **[PENDING RUN]**

### The GPU boundary

Every function above is pure numpy/sklearn and is fully exercised by
`CDS_SELFTEST=1` on **synthetic** activations — no model, no GPU, no network.
`separability.extract_hidden_states(model, tokenizer, texts, is_data_mask,
layers, mode=...)` is the one function that needs a real loaded model: it
registers a forward hook on the embedding layer, rotates (or offsets) the
data-role documents' token embeddings, forwards with
`output_hidden_states=True`, and mean-pools each layer over non-pad tokens (the
same pooling convention as `hello_world`'s layer-12 probe). It is written in
full but **not called** by this package's self-test or by any CPU-only build
step — it is the function the lead runs on GPU to turn
`data.load_role_corpus()` into the `activations_by_layer` dicts every other
function consumes.

---

## 5. Running it

```bash
# CPU-only, no model, no network: verifies the rotation/offset/probe/cosine
# machinery on synthetic activations. aside.py and inseparability.py self-test
# unconditionally; separability.py and data.py gate their self-test behind
# CDS_SELFTEST=1 (their default __main__ instead builds the real corpus).
python -m steering_tutorials.control_data_split.aside
python -m steering_tutorials.control_data_split.inseparability
python -m steering_tutorials.control_data_split.config
CDS_SELFTEST=1 python -m steering_tutorials.control_data_split.separability
CDS_SELFTEST=1 python -m steering_tutorials.control_data_split.data

# Downloads real HF data, builds the >=500/role corpus, prints licence + fingerprint.
# CPU-only, no model.
python -m steering_tutorials.control_data_split.data

# [PENDING RUN, GPU]: load Gemma-3-1B, call separability.extract_hidden_states
# in vanilla/rotate/offset modes over config.LAYERS, then
# layer_separability_sweep + cosine_trajectory, write config.RESULTS_PATH.
```

---

## 6. Open items for the GPU pass

- Run `extract_hidden_states` in all three modes (`vanilla`, `rotate`,
  `offset`) over `config.LAYERS` and populate `config.RESULTS_PATH`.
- Compare the resulting layer-0 separability against
  `separability.provenance_floor(instruction_texts, data_texts)` — the surface
  floor this corpus's own register difference sets, so any activation-level win
  is read against it rather than against zero.
- Sweep `config.ANGLE` away from `pi/2` (the angle is a config knob exactly so
  this is a one-line change) to test whether the reported optimum is actually
  special on Gemma-3-1B, or whether any angle nearby would do — the steering
  program's own NUMEROLOGY check (CLAUDE.md §7), applied here to ASIDE's own
  reported ablation.
- Everything in Section 1's "CANNOT reproduce" list stays out of scope for this
  package; it would need an SFT run on the modified forward, which this lesson
  deliberately does not undertake.
