"""streaming/ -- genuine causal, bounded-state monitors for the streaming ladder.

Both monitors here implement `types.StreamingAggregator`: `is_causal=True`, `update()`
is O(1) in trajectory length, `state_bytes()` is O(1) in steps seen (constant-size
arrays, never a growing buffer of past steps). See each module's smoke test for a
measured demonstration that state size does not grow between step 10 and step 500.
"""
from .esn_cusum import ESNCusum
from .markov import SafetyDriftMonitor

__all__ = ["ESNCusum", "SafetyDriftMonitor"]
