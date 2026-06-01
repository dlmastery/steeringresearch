"""datasets.py — loaders for the tiny pinned OFFLINE dataset slices.

These are deliberately small, hand-written, clearly-synthetic slices bundled as
JSON under `src/steering/data/`. They wire the five axes (datasets-suite §0):

    axbench_mini      — concept contrast pairs (Axis 1 / extraction source)
    mmlu_tiny         — MCQ capability tripwire (Axis 2)
    wikitext_ppl_mini — coherence/perplexity passages (Axis 3)
    jailbreak_mini    — harmful-intent prompts (Axis 4 safety)
    xstest_mini       — benign-but-scary prompts (Axis 5 over-refusal)

No copyrighted text is used. These are SMOKE-tier slices (datasets-suite §8); the
real ladder loads the full community benchmarks at run time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load(name: str) -> Any:
    """Load a bundled JSON slice. Returns whatever the file holds (dict or list)."""
    with open(_DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _require(data: dict, key: str, name: str) -> Any:
    """Fetch `key` from a loaded slice with a clear 'malformed slice' message."""
    if not isinstance(data, dict) or key not in data:
        raise KeyError(f"dataset slice '{name}' is malformed: missing key '{key}'")
    return data[key]


def load_axbench_mini() -> list[tuple[str, str]]:
    """Concept contrast pairs as a list of (pos_prompt, neg_prompt)."""
    data = _load("axbench_mini.json")
    return [(p["pos"], p["neg"]) for p in _require(data, "pairs", "axbench_mini")]


def axbench_concept() -> str:
    """The concept name the axbench_mini pairs are built around."""
    return _require(_load("axbench_mini.json"), "concept", "axbench_mini")


def load_mmlu_tiny() -> list[dict]:
    """List of MCQ dicts: {question, options[4], answer_idx}."""
    return _require(_load("mmlu_tiny.json"), "questions", "mmlu_tiny")


def load_wikitext_ppl_mini() -> list[str]:
    """List of short passages for perplexity/coherence."""
    return _require(_load("wikitext_ppl_mini.json"), "passages", "wikitext_ppl_mini")


def load_jailbreak_mini() -> list[str]:
    """List of harmful-intent prompts (safety tripwire)."""
    return _require(_load("jailbreak_mini.json"), "prompts", "jailbreak_mini")


def load_xstest_mini() -> list[str]:
    """List of benign-but-scary prompts (over-refusal dual)."""
    return _require(_load("xstest_mini.json"), "prompts", "xstest_mini")


def load_all() -> dict:
    """Convenience: every slice in one dict."""
    return {
        "axbench_mini": load_axbench_mini(),
        "axbench_concept": axbench_concept(),
        "mmlu_tiny": load_mmlu_tiny(),
        "wikitext_ppl_mini": load_wikitext_ppl_mini(),
        "jailbreak_mini": load_jailbreak_mini(),
        "xstest_mini": load_xstest_mini(),
    }


def load_wikitext2_real():
    """Real WikiText-2 test passages (40, >120 chars) for rigorous perplexity.

    Replaces the synthetic wikitext_ppl_mini for the rung-3 N17/N5 escalation —
    real held-out English prose is required before any external coherence claim.
    Source: Salesforce/wikitext, wikitext-2-raw-v1 test split (CC-BY-SA).
    """
    import json as _json
    from pathlib import Path as _P
    p = _P(__file__).parent / "data" / "wikitext2_real.json"
    return _json.loads(p.read_text(encoding="utf-8"))
