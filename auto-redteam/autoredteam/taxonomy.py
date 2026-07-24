"""taxonomy.py -- load and query the harm taxonomy the judges map findings onto.

A taxonomy is a small hierarchical tree (top -> mid -> leaf). Attack goals tag
themselves with a LEAF key (e.g. `direct_prompt_injection`); the evaluator maps a
finding back to that leaf, and the reporter groups ASR per category. Keeping this
in a data file (config/taxonomies/*.yaml) means adding a category is a YAML edit,
not a code change.

Tree shape (see owasp_asi_2026.yaml)::

    name: owasp_asi_2026
    categories:
      <top_key>:
        title: "..."
        subcategories:
          <mid_key>:
            title: "..."
            leaves:
              <leaf_key>: "one-line description"

Only pure dict/list work here -- no model, no network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Resolve bare names like "owasp_asi_2026" to config/taxonomies/<name>.yaml.
BASE_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_DIR = BASE_DIR / "config" / "taxonomies"


def _resolve_path(name_or_path: str) -> Path:
    """Map a taxonomy name OR an explicit path to a YAML file on disk."""
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    named = TAXONOMY_DIR / f"{name_or_path}.yaml"
    if named.exists():
        return named
    if p.exists():
        return p
    raise FileNotFoundError(
        f"taxonomy '{name_or_path}' not found (looked for {named} and {p})"
    )


def load_taxonomy(name_or_path: str) -> dict[str, Any]:
    """Load a taxonomy tree by name ('owasp_asi_2026') or explicit YAML path."""
    path = _resolve_path(name_or_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict) or "categories" not in data:
        raise ValueError(f"taxonomy {path} must have a top-level 'categories' mapping")
    return data


def list_categories(taxonomy: dict[str, Any]) -> list[str]:
    """Flatten the tree to its LEAF category keys (what goals/findings reference)."""
    leaves: list[str] = []
    for top in (taxonomy.get("categories") or {}).values():
        subs = (top or {}).get("subcategories") or {}
        for mid in subs.values():
            for leaf_key in ((mid or {}).get("leaves") or {}).keys():
                leaves.append(leaf_key)
    return leaves


def describe_category(taxonomy: dict[str, Any], leaf_key: str) -> dict[str, str] | None:
    """Return {top, mid, leaf, description} for a leaf key, or None if unknown.

    Handy for the reporter, which wants the human-readable path of a finding.
    """
    for top_key, top in (taxonomy.get("categories") or {}).items():
        subs = (top or {}).get("subcategories") or {}
        for mid_key, mid in subs.items():
            leaves = (mid or {}).get("leaves") or {}
            if leaf_key in leaves:
                return {
                    "top": top_key,
                    "mid": mid_key,
                    "leaf": leaf_key,
                    "description": str(leaves[leaf_key]),
                }
    return None


if __name__ == "__main__":
    tax = load_taxonomy("owasp_asi_2026")
    cats = list_categories(tax)
    print("taxonomy   :", tax.get("name"))
    print("leaf count :", len(cats))
    print("leaves     :", cats)
    sample = cats[0] if cats else None
    if sample:
        print("describe   :", describe_category(tax, sample))
