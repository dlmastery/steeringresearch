"""data.py — several harm CONCEPTS + a shared benign baseline, from the SHARED set.

Lessons 2-3 learned ONE behaviour ("refuse this"). FLAS learns a SINGLE velocity
field that is *concept-conditioned*, so it must be trained on SEVERAL concepts at
once — otherwise there is nothing for the concept embedding ``c`` to select
between and the field degenerates to a fixed vector. This module supplies those
concepts.

REWIRED to the shared loader ``steering_tutorials.common.data.load_concepts``
--------------------------------------------------------------------------------
This lesson used to carve its concepts from JailbreakBench's ``Category`` column
(10 prompts/category — far too few to hold out a real eval split). It now draws
them from the ONE shared dataset every tutorial imports: toxic-chat's
``openai_moderation`` harm categories folded to coarse concepts, deduped,
group-id'd and length-controlled. The shared loader applies a data-sufficiency
gate (``MIN_CONCEPT_AVAILABLE`` = 100), so only the WELL-POPULATED concepts come
through — ``sexual`` (~388), ``harassment`` (~143) and ``violence`` (~111); the
tiny ``hate`` (~24) and ``self_harm`` (~27) pools are dropped because they cannot
yield a stable field or a disjoint >=30-prompt eval split. Each kept concept is
far larger than JBB's 10, so its ``eval`` split reads honestly (see the
per-concept counts printed at load time).

The shared loader reserves ONE kept concept for zero-shot transfer (its default
held-out = the second-largest kept pool, ``harassment``); we train on the rest
(``sexual`` + ``violence``) and carry the held-out concept through untouched for
the payoff-3 zero-shot test.

Per concept we return three DISJOINT prompt sets:

    {
      "exemplars":     [...],  # encode -> the concept embedding c (identity of c)
      "steer_prompts": [...],  # harmful side of the diff-of-means TARGET shift
      "eval_prompts":  [...],  # held out; grade steering here, never trained on
    }

plus a shared benign ``baseline`` (the common contrast origin for every concept's
diff-of-means, and the source of "unsteered" activations the flow transports).
Keeping ``eval_prompts`` disjoint from ``exemplars``/``steer_prompts`` is what
stops us from grading a steered activation on the very prompts that defined it.

This module is CPU-ONLY: importing it or calling ``load_concepts()`` downloads +
parses CSVs and builds Python lists of strings. It NEVER loads an LLM.
"""
from __future__ import annotations

import sys

from steering_tutorials.common.data import load_concepts as _load_common_concepts

# How many prompts to draw PER concept before the shared loader's 40/30/30
# exemplars/steer/eval split. These concepts are POOL-LIMITED (sexual ~388,
# harassment ~143, violence ~111), so a per-concept count cannot reach 500 — 120
# just requests the available pool: ~36 eval prompts for the big categories and ~34
# for violence (the smallest KEPT concept). The loader already drops any concept
# with < 100 available, so every returned concept clears a >=30-prompt eval split.
N_PER_CONCEPT = 120
# Size of the shared benign baseline (contrast origin + unsteered h0 pool). Raised
# to 120 to match the per-concept pool so the selectivity arm's benign contrast is
# not the bottleneck; the concept pools above remain the binding limit.
N_BENIGN_BASELINE = 120  # was 40


def load_concepts(
    seed: int = 0,
    n_per_concept: int = N_PER_CONCEPT,
    n_benign_baseline: int = N_BENIGN_BASELINE,
    held_out: str | None = None,
) -> dict:
    """Return train concepts + a held-out concept + a shared benign baseline.

    Sourced from ``common.data.load_concepts`` (toxic-chat coarse harm categories)
    and reshaped to the contract ``train_flas.py`` and ``run_flas.py`` consume.

    Structure::

        {
          "concepts": {
              "sexual":   {"exemplars":[...], "steer_prompts":[...],
                           "eval_prompts":[...], "heldout": False},
              ...,
              "harassment": {..., "heldout": True},   # zero-shot, untrained
          },
          "baseline": [ ...benign prompts... ],   # shared contrast origin + h0 pool
          "train":            { train-only concepts (heldout == False) },
          "held_out":         {"name", "exemplars", "eval_prompts"},
          "train_concepts":   [ names with heldout == False ],
          "holdout_concept":  "harassment",
        }

    The trainer iterates the ``heldout == False`` concepts only; the held-out
    concept is carried through so inference can encode its embedding and steer a
    concept the field never trained on. ``held_out`` overrides the shared loader's
    default held-out concept (its second-largest pool) if you pass a name.
    """
    common = _load_common_concepts(n_per_concept=n_per_concept, seed=seed,
                                   held_out=held_out)
    src = common["concepts"]            # {name: {exemplars, steer, eval, n_available}}
    hold_name = common["held_out"]      # a concept name string (e.g. "harassment")

    # Keep only concepts that actually carry data for every role we need.
    usable = [
        name for name in common["concept_names"]
        if src[name]["eval"] and (src[name]["exemplars"] or src[name]["steer"])
    ]
    train_names = [name for name in usable if name != hold_name]

    concepts: dict[str, dict] = {}
    for name in usable:
        s = src[name]
        concepts[name] = {
            "exemplars": list(s["exemplars"]),
            "steer_prompts": list(s["steer"]),
            "eval_prompts": list(s["eval"]),
            "heldout": name == hold_name,
        }

    baseline = list(common["baseline"])[:n_benign_baseline]

    train_only = {name: concepts[name] for name in train_names}
    held = concepts.get(hold_name)
    held_out_rec = None
    if held is not None:
        held_out_rec = {
            "name": hold_name,
            "exemplars": held["exemplars"],
            "eval_prompts": held["eval_prompts"],
        }

    for name in usable:
        split = concepts[name]
        tag = "HOLDOUT" if split["heldout"] else "train  "
        print(f"[data] {tag} {name:12s} exemplars={len(split['exemplars'])} "
              f"steer={len(split['steer_prompts'])} eval={len(split['eval_prompts'])}",
              file=sys.stderr)
    print(f"[data] shared benign baseline = {len(baseline)}", file=sys.stderr)

    return {
        "concepts": concepts,
        "train": train_only,
        "held_out": held_out_rec,
        "baseline": baseline,
        "train_concepts": list(train_names),
        "holdout_concept": hold_name,
    }


if __name__ == "__main__":  # smoke: python -m steering_tutorials.flas.data
    d = load_concepts()
    print(f"train_concepts = {d['train_concepts']}")
    print(f"holdout_concept = {d['holdout_concept']}")
    print(f"baseline = {len(d['baseline'])} benign prompts")
    for name, split in d["concepts"].items():
        role = "HELD OUT (zero-shot)" if split["heldout"] else "train"
        ex = split["exemplars"][0] if split["exemplars"] else "(none)"
        print(f"  [{role}] {name}: {len(split['exemplars'])} exemplars / "
              f"{len(split['eval_prompts'])} eval  e.g. {ex[:60]!r}")
