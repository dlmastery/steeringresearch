"""realignment — lesson 11: RESTORE refusal in an abliterated model.

Abliteration deletes a model's ability to refuse by projecting the refusal
direction out of its weights. This lesson EXTRACTS that direction from the
still-aligned base model and TRANSPLANTS it back into the abliterated model as
a steering vector, measuring how much refusal it restores and at what cost.

The package is split into two importable-but-inert modules so that neither one
loads a model at import time (all model work lives under ``main()``):

  * ``extract_refusal``  — phase 1: load ONLY the aligned base model, compute the
                           Arditi refusal direction, save it, and exit.
  * ``run_realignment``  — phase 2: load ONLY the abliterated model, load the
                           saved direction, sweep alpha, measure ASR / over-
                           refusal / coherence.

See ``README.md`` for why the two phases must run as separate processes.
"""
