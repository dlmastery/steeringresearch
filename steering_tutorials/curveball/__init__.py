"""curveball — lesson: the right direction to steer isn't always a straight line.

Every steering lesson so far moved the residual stream along a STRAIGHT line:
lesson 2 added a fixed diff-of-means vector (``h + alpha*v``), lesson 3 learned a
one-shot rank-1 edit. This lesson asks whether a CURVED path — one that bends to
follow the activation manifold instead of shooting off it — reaches the same
behavioural target (refusal) at a lower coherence cost.

Motivated by Raval, Song, Wu, Harrasse, Phillips, Barez & Abdullah 2026,
'Curveball Steering: The Right Direction To Steer Isn't Always Linear'
(arXiv:2603.09313). The curved-path construction here is our own (a great-circle
geodesic on the local activation-norm shell), NOT the paper's polynomial-kernel-PCA
method; see README.md and curveball.py for the honest relationship.
"""
