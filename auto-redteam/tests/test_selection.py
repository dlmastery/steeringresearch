"""test_selection.py -- selector factory + bandit convergence.

Depends on the AR-STRATEGIES sibling (`autoredteam.selection`); SKIPS until it
lands. The learning bandits must converge toward the higher-reward arm; the
static weighted selector must at least select a valid arm and expose
JSON-serializable stats. Everything is seeded for determinism.
"""
from __future__ import annotations

import json
import random

import pytest

from autoredteam.models import SelectionConfig
from .conftest import require

ARMS = ["good", "bad"]
LEARNING_MODES = ["epsilon_greedy", "ucb", "thompson"]


def _make(selection, mode, **kw):
    """Build a selector via the contracted factory, seeded for reproducibility."""
    sel_cfg = SelectionConfig(mode=mode, **kw)
    return selection.get_selector(sel_cfg, list(ARMS), None, seed=0)


def _train(selector, rng, n=200):
    """Select+update loop where 'good' pays off 0.9 vs 'bad' 0.1 (Bernoulli)."""
    for _ in range(n):
        choice = selector.select(ARMS)
        if choice == "good":
            reward = 1.0 if rng.random() < 0.9 else 0.0
        else:
            reward = 1.0 if rng.random() < 0.1 else 0.0
        selector.update(choice, reward, 1)


@pytest.mark.parametrize("mode", LEARNING_MODES)
def test_bandit_converges_to_better_arm(mode):
    selection = require("autoredteam.selection")
    try:
        selector = _make(selection, mode)
    except (ValueError, KeyError):
        pytest.skip(f"selection mode {mode!r} not implemented yet")

    rng = random.Random(0)
    _train(selector, rng, n=200)

    # After learning, an exploitation sweep should favour 'good' >50% of the time.
    counts = {"good": 0, "bad": 0}
    for _ in range(200):
        counts[selector.select(ARMS)] += 1
    assert counts["good"] > counts["bad"], (mode, counts)
    assert counts["good"] > 100, (mode, counts)


@pytest.mark.parametrize("mode", LEARNING_MODES + ["weighted", "fixed"])
def test_selector_select_returns_valid_arm_and_serializable_stats(mode):
    selection = require("autoredteam.selection")
    try:
        selector = _make(selection, mode)
    except (ValueError, KeyError):
        pytest.skip(f"selection mode {mode!r} not implemented yet")

    assert selector.select(ARMS) in ARMS
    stats = selector.stats()
    assert isinstance(stats, dict)
    json.dumps(stats)  # must be JSON-serializable for persistence/resume


def test_selector_is_reproducible_under_same_seed():
    selection = require("autoredteam.selection")
    s1 = _make(selection, "thompson")
    s2 = _make(selection, "thompson")
    rng1, rng2 = random.Random(1), random.Random(1)
    _train(s1, rng1, n=50)
    _train(s2, rng2, n=50)
    # identical seeds + identical reward stream -> identical learned stats
    assert s1.stats() == s2.stats()
