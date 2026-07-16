"""data.py — K distinct harm concepts + a shared benign baseline, SHARED loader.

Lesson 2 needed ONE harmful set and ONE benign set. Lesson 9 needs K *separate*
harmful concepts (so we can steer each independently and watch them interfere)
plus ONE shared benign baseline (the common origin every concept is contrasted
against).

REWIRED to the shared loader ``steering_tutorials.common.data.load_concepts``
--------------------------------------------------------------------------------
This lesson used to carve K concepts from JailbreakBench's ``Category`` column
(10 prompts/category — far too few for a stable diff-of-means AND a disjoint eval
split). It now draws them from the ONE shared dataset every tutorial imports:
toxic-chat's ``openai_moderation`` harm categories folded to five coarse concepts
(``sexual`` / ``harassment`` / ``violence`` / ``self_harm`` / ``hate``), deduped,
group-id'd and length-controlled. Each concept is much larger than JBB's 10, so
the eval split (used to measure steering success and cross-talk) is big enough to
read honestly — see the per-concept counts printed at load time.

We return a nested structure:

    {
      "concepts": {
          "sexual":     {"extract": [...], "eval": [...]},
          "harassment": {"extract": [...], "eval": [...]},
          ...
      },
      "baseline": [ ...benign prompts... ],   # the shared contrast origin
    }

``extract`` prompts build each concept's diff-of-means vector (the shared loader's
exemplars + steer halves pooled); ``eval`` prompts (disjoint) measure steering
success and cross-talk. Keeping them disjoint stops us from grading a vector on
the very prompts that defined it.

This module is CPU-ONLY: it downloads + parses CSVs and builds Python lists of
strings. It NEVER loads an LLM.
"""
from __future__ import annotations

import sys

from steering_tutorials.common.data import load_concepts as _load_common_concepts


def load_multi_intent(
    concepts: list[str],
    n_per_concept: int = 150,
    n_eval_per_concept: int | None = None,
    n_benign_baseline: int = 40,
    seed: int = 0,
) -> dict:
    """Return K concept prompt-sets (extract/eval split) + a shared benign baseline.

    Parameters
    ----------
    concepts : the shared coarse harm-concept names to use (see ``config.CONCEPTS``:
        ``sexual`` / ``harassment`` / ``violence`` / ``self_harm`` / ``hate``). Each
        must exist among the shared loader's concepts.
    n_per_concept : prompts to draw PER concept before the shared loader's 40/30/30
        exemplars/steer/eval split. ``extract`` = exemplars + steer (~70%); ``eval``
        = the disjoint 30%. Larger categories give more; small ones return what they
        have. 150 yields ~40-45 eval prompts for the big concepts.
    n_eval_per_concept : ACCEPTED for backward compatibility but IGNORED — the shared
        loader owns the split, and its disjoint eval share is larger than the old
        fixed 5, which is the point of this rewiring.
    n_benign_baseline : size of the shared benign baseline (from the shared set).

    The shared loader is seeded, deduped and group-aware, so the split is
    reproducible and leakage-safe.
    """
    common = _load_common_concepts(n_per_concept=n_per_concept, seed=seed)
    src = common["concepts"]            # {name: {exemplars, steer, eval, n_available}}

    available = sorted(src)
    out_concepts: dict[str, dict[str, list[str]]] = {}
    for name in concepts:               # preserve config order for the K-ladder
        if name not in src:
            raise ValueError(f"concept {name!r} not among shared concepts: {available}")
        s = src[name]
        extract = list(s["exemplars"]) + list(s["steer"])
        evalset = list(s["eval"])
        out_concepts[name] = {"extract": extract, "eval": evalset}

    baseline = list(common["baseline"])[:n_benign_baseline]

    for name, split in out_concepts.items():
        print(f"[data] {name:12s} extract={len(split['extract'])} "
              f"eval={len(split['eval'])}", file=sys.stderr)
    print(f"[data] shared benign baseline = {len(baseline)}", file=sys.stderr)
    return {"concepts": out_concepts, "baseline": baseline}


if __name__ == "__main__":  # smoke: python -m steering_tutorials.multi_intent.data
    from . import config as C

    d = load_multi_intent(
        C.CONCEPTS,
        n_per_concept=C.N_PER_CONCEPT,
        n_eval_per_concept=C.N_EVAL_PER_CONCEPT,
        n_benign_baseline=C.N_BENIGN_BASELINE,
        seed=C.SEED,
    )
    print(f"concepts={list(d['concepts'])}  baseline={len(d['baseline'])}")
    for name, split in d["concepts"].items():
        ex = split["extract"][0] if split["extract"] else "(none)"
        print(f"[{name}] extract={len(split['extract'])} eval={len(split['eval'])} "
              f"extract[0]={ex[:70]}")
