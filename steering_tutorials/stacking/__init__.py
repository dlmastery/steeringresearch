"""Lesson 12 — Stacking: which steering priors STACK vs COMPETE.

A standalone tutorial that builds an ADDITIVE 2->N ladder of steering priors and
reads the MARGINAL effect of adding one prior at a time, demonstrating:

  * STACK      — two priors on DIFFERENT sites (layers) → gains add.
  * COMPETE    — two priors on the SAME site + same direction, different
                 operation → the second double-counts and cancels/degrades.
  * OVER-STACK — the forbidden "everything-on" hybrid → the norm budget is
                 spent, coherence collapses, gibberish rises.

Reuses lesson 2's mechanical core (``hello_world_steering``) verbatim: the model
loader + steering hook, the diff-of-means CAA vector, and the self-grading judge.

See ``README.md`` for the decision rule, the methodology, and run commands, and
CLAUDE.md section 9 for the project-level stacking discipline this instantiates.
"""
