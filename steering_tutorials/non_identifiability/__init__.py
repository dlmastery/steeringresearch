"""Lesson: the non-identifiability of steering vectors.

Many DIFFERENT directions in activation space produce the SAME behavioral
effect, so "the refusal direction" is a misnomer — there is a whole family of
them. This lesson extracts several refusal-ish directions by different recipes,
shows they have LOW pairwise cosine similarity, yet reach a SIMILAR steering
effect (refusal rate). See ``README.md``.

Standalone, like the other tutorials: it reuses lesson 2's plumbing
(``hello_world_steering``) and the shared ``common.data`` loader, but does not
touch the research harness in ``src/steering``.
"""
