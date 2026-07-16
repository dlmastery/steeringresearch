"""fine_grained — lesson: FINE-GRAINED (sparse) activation steering.

The idea in one line: you do not have to perturb the WHOLE steering direction.
Keep only the top-k% highest-magnitude coordinates of the diff-of-means vector,
zero the rest, and steer with that SPARSE edit at matched strength. The claim
("Steering Less, Achieving More") is that a sparse edit can match dense steering
on the target behavior (refusal) while doing LESS collateral damage — lower
benign over-refusal and less gibberish.

Everything model-touching lives under ``main()`` in ``run_fine_grained`` so that
importing this package never loads torch or a model. The pure ``sparse.py``
sparsifier is unit-tested on CPU with no model at all.

  * ``sparse``            — ``sparsify(v, keep_frac)`` + ``SparseSteeringContext``.
  * ``run_fine_grained``  — extract the dense refusal vector, sweep sparsity, and
                            measure refusal / over-refusal / gibberish, judged.
  * ``infer``             — steer a single prompt at a chosen sparsity, for a feel.

See ``README.md`` for the recipe, the frontier experiment, and honest caveats.

Reference (a simplified reconstruction inspired by this paper):
  'Fine-Grained Activation Steering: Steering Less, Achieving More' (AUSteer;
  Feng et al., ICLR 2026, arXiv:2602.04428). The paper selects units by an
  activation-momentum discriminativeness metric with adaptive per-input
  strength; we use a simpler top-k magnitude mask.
"""
