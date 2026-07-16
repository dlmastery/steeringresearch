"""Lesson (TALAN-inspired): a learned latent-adapter steer of the residual stream.

Inference-time analogue of TALAN (arXiv:2606.06902). The paper is a POST-TRAINING
method; here we freeze the LLM and train only a small latent side-path adapter,
placing the "full adapter" point on the fixed-vector -> rank-1 -> adapter spectrum.
See README.md for the honest post-training-vs-inference-time note.
"""
