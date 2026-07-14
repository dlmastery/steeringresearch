"""Lesson 10 — the Rogue Scalpel: red-teaming a steering intervention, and defending it.

Steering is DUAL-USE. The very same residual-stream write that lesson 2 uses to
*install* refusal can be turned around to *strip* refusal — a jailbreak. This
lesson (a) demonstrates that attack on an aligned Gemma-3-1B, then (b) implements
and ablates a layered GUARD and shows it neutralizes the attack, bringing the
attack-success-rate back down.

Standalone and CPU-readable; the live run needs the same small Gemma as lessons
1-3. See ``README.md`` and CLAUDE.md section 10 (the Rogue-Scalpel mandate).
"""
