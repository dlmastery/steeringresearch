"""Contextual Linear Activation Steering — a standalone steering tutorial lesson.

The steering *strength* is no longer a single fixed number. Instead it is scaled
per input by how much that input already points along the steering direction, so
harmful-looking prompts are steered hard and benign prompts are barely touched —
an *implicit* gate that needs no separate probe (contrast with lesson 2's
explicit lesson-1-probe gate). See ``README.md``.
"""
