"""axbench.py — loader for the REAL AxBench steering benchmark (Wu/Zhong et al.,
arXiv:2501.17148; data at huggingface.co/pyvene/axbench-concept{10,500,16k}).

This replaces the project's hand-written synthetic concepts + invented eval
prompts with the community benchmark: AxBench supplies the CONCEPTS, the
contrast data that builds the DiffMean vector, AND the held-out evaluation
instructions — none of it authored by us. AxBench's own finding is that
difference-in-means is the strongest concept method, so DiffMean is exactly the
primitive worth evaluating here.

Schema (per the `pyvene/axbench-concept500` parquets, 2b/l20 split — the text is
model-agnostic; we extract from OUR model's activations on it):
  train: per concept_id (0..N-1) 72 `category=='positive'` rows whose `output`
         exhibits the concept; plus shared `concept_id==-1` `category=='negative'`
         neutral rows. -> DiffMean(positive, negative).
  test:  `input` = held-out, concept-agnostic instructions (the steering eval
         prompts); positive/negative/hard-negative rows for the rubric.

Downloads via huggingface_hub (retry/resume; robust to the SSL middlebox that
breaks plain urllib on large files); truststore is injected for the OS trust
store. Datasets are PUBLIC (no token needed).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, TypedDict

try:  # OS trust store for the SSL-intercepting middlebox (same as hf_fetch.py)
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional
    pass


class AxConcept(TypedDict):
    concept_id: int
    description: str
    pos_texts: list[str]
    neg_texts: list[str]


def _download(name: str, split: str, model_dir: str = "2b/l20"):
    """Fetch + cache one AxBench parquet, return a DataFrame. Robust to flaky SSL."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=f"pyvene/axbench-{name}",
        filename=f"{model_dir}/{split}/data.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(path)


@lru_cache(maxsize=8)
def _frames(name: str):
    return _download(name, "train"), _download(name, "test")


def load_axbench_concepts(name: str = "concept500",
                          n_concepts: Optional[int] = None) -> list[AxConcept]:
    """Per-concept contrast data for DiffMean extraction.

    pos_texts = the concept's `output` responses (category=='positive');
    neg_texts = the SHARED neutral `output` responses (concept_id==-1). Returns
    the first ``n_concepts`` concepts (all of them when None).
    """
    train, _ = _frames(name)
    neg_texts = (train[(train["concept_id"] == -1) & (train["category"] == "negative")]
                 ["output"].astype(str).tolist())
    ids = sorted(i for i in train["concept_id"].unique() if i >= 0)
    if n_concepts is not None:
        ids = ids[:n_concepts]
    out: list[AxConcept] = []
    for cid in ids:
        rows = train[(train["concept_id"] == cid) & (train["category"] == "positive")]
        if rows.empty:
            continue
        out.append(AxConcept(
            concept_id=int(cid),
            description=str(rows["output_concept"].iloc[0]),
            pos_texts=rows["output"].astype(str).tolist(),
            neg_texts=neg_texts,
        ))
    return out


def load_axbench_eval_instructions(name: str = "concept500", k: int = 10) -> list[str]:
    """A fixed set of k held-out, concept-agnostic steering-eval instructions.

    These are AxBench's real `input` prompts (the steering protocol applies the
    same generic instructions to every concept and asks whether the steered
    continuation exhibits the concept). Deterministic (first k unique).
    """
    _, test = _frames(name)
    seen: list[str] = []
    for s in test["input"].astype(str).tolist():
        if s not in seen:
            seen.append(s)
        if len(seen) >= k:
            break
    return seen
