"""aggregators -- the offline aggregator ladder for streaming_trajectory_aggregation.

Eight `Aggregator`-Protocol implementations (`../types.py`), split by how they turn
`[n_steps, dim]` into one trajectory score:

  pooling.py  (fixed feature + a SHARED sklearn LogisticRegression head)
    1. MeanPool           -- true mean; the literature's claimed failure mode.
    2. MaxPool             -- per-dimension max; the literature's preferred alternative.
    3. MeanMaxStdPool      -- the [mean, max, std] concat cross_trajectory's `mean_agg`
                              actually is, kept separate from #1 so the two are not
                              conflated.
    4. DeviationWeighted   -- softmax-weighted mean by distance from the train benign
                              centroid.
    5. LastStep            -- THE CONTROL: final step alone.
    6. Truncate(k, inner)  -- the WITHIN-CORPUS horizon control: restrict every
                              trajectory to its first k steps and delegate to ANY
                              aggregator, so `AUC[Truncate(k,MaxPool)] -
                              AUC[Truncate(k,MeanPool)]` is measurable as a function of
                              k on ONE corpus (see ../horizon.py).
       FirstKSteps(k)      -- `Truncate(k, MeanPool)` under its historical name; the
                              prefix-coverage curve, NOT dilution evidence on its own.

  sequence.py (learned end-to-end; own linear head as part of the network)
    7. GRUAggregator            -- Trajectory Guard family (arXiv:2601.00516).
    8. QueryTokenCompressor(k)  -- TRACE family (arXiv:2606.00611); k is UNVERIFIED
                                    against the abstract, not hardcoded to 16 as fact.

See each module's docstring for the full citation and causality rationale. Every class
is fit on TRAIN ONLY inside its own `fit()` -- no aggregator here reads test labels or
test features during fitting.
"""
from __future__ import annotations

from .pooling import (
    DeviationWeighted,
    FirstKSteps,
    LastStep,
    MaxPool,
    MeanMaxStdPool,
    MeanPool,
    Truncate,
)
from .sequence import GRUAggregator, QueryTokenCompressor

__all__ = [
    "MeanPool",
    "MaxPool",
    "MeanMaxStdPool",
    "DeviationWeighted",
    "LastStep",
    "Truncate",
    "FirstKSteps",
    "GRUAggregator",
    "QueryTokenCompressor",
]
