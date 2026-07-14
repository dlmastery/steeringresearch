"""train_flas.py — train the concept-conditioned velocity field by RECTIFIED FLOW.

This is the learning loop for the FLAS lesson. Where lesson 2 hand-built a fixed
steering vector and lesson 3 learned a one-shot rank-1 edit, FLAS learns a
*velocity field* ``v_theta(h, t, c)`` whose flow TRANSPORTS an unsteered
activation to its steered position. At inference you integrate the ODE
``dh/dt = v_theta(h, t, c)`` from ``t=0`` to a chosen flow-time ``T`` (the
zero-shot strength dial); here we only teach ``v_theta`` what that transport is.

The training signal is a classic **rectified-flow / flow-matching** regression,
and the elegant part is that it needs NO sampling of the frozen LLM during the
loop — after one up-front pass to cache activations, everything below is pure
regression on tensors, so it is fast and stable on CPU-sized tensors.

The transport we regress onto
-----------------------------
For a concept ``c`` we already know, from the same contrastive diff-of-means used
in lesson 2, WHERE a steered activation should end up: adding the concept's
target shift ``delta_c = mean(act | steer_c) - mean(act | benign)`` moves an
activation toward "refuse category c". So for an unsteered activation ``h0`` we
define its steered target as::

    h1 = h0 + delta_c

Rectified flow connects ``h0`` and ``h1`` by a STRAIGHT line and asks the field
to reproduce that line's (constant) velocity everywhere along it:

    t   ~ U(0, 1)                        # a random point in flow-time
    h_t = (1 - t) * h0 + t * h1          # = h0 + t * delta_c  (linear interpolant)
    target velocity  = h1 - h0 = delta_c # constant along the straight path
    loss = MSE( v_theta(h_t, t, c) , h1 - h0 )

Because the interpolant is a straight line, the ground-truth velocity is the SAME
vector ``delta_c`` at every ``t`` — that is exactly what makes few-step Euler
integration accurate (Liu et al. 2023, rectified flow). Integrating the learned
field then recovers the transport: summing ``v_theta * dt`` along the path
accumulates back to ``≈ delta_c``, carrying ``h0`` to ``h1``. Conditioning the
field on the concept embedding ``c`` is what lets ONE field encode a different
straight-line transport for each concept — and, at test time, for a concept it
never trained on, purely from that concept's embedding.

  Lipman et al. 2023, 'Flow Matching for Generative Modeling' (arXiv:2210.02747)
    [UNVERIFIED] — the conditional flow-matching objective (regress the field
    onto the interpolant's velocity) this loop is a rectified special case of.
  Liu et al. 2023, 'Flow Straight and Fast: Rectified Flow' (arXiv:2209.03003)
    [UNVERIFIED] — straight-line transport => constant target velocity => accurate
    few-step Euler integration.
  Rimsky et al. 2023, 'Steering Llama 2 via Contrastive Activation Addition'
    (arXiv:2312.06681) — the diff-of-means ``delta_c`` that defines each concept's
    steered target ``h1 = h0 + delta_c``.

CPU-ONLY NOTE: importing this module runs NOTHING — all work is inside ``main()``
under the ``__main__`` guard. Loading Gemma and running the loop is a GPU job the
lead launches separately; here we only WRITE and import-check the loop.
"""
from __future__ import annotations

import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from steering_tutorials.hello_world_steering.model_utils import (
    last_token_activations,
    load_model,
)
from steering_tutorials.hello_world_steering.steer_vector import extract_caa_vector
from steering_tutorials.flas import config as C
from steering_tutorials.flas.data import load_concepts
from steering_tutorials.flas.flow import (
    VelocityField,
    concept_embedding,
    save_flow,
)


def _as_tensor(x, device) -> torch.Tensor:
    """Coerce a numpy array or tensor to a float32 tensor on ``device``.

    ``concept_embedding`` / ``extract_caa_vector`` may hand back either a numpy
    array or a torch tensor depending on the peer implementation; this keeps the
    trainer agnostic to that choice.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.detach().to(device=device, dtype=torch.float32).reshape(-1)


def main() -> None:
    # ---- reproducibility ----------------------------------------------------
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # ---- frozen model as a fixed activation source --------------------------
    # We never backprop through the LLM here — it only serves cached activations
    # for the concept embeddings, the diff-of-means targets, and the h0 bank.
    model, tok = load_model(C.MODEL_ID)
    for p in model.parameters():  # FREEZE: nothing in the LLM is trained
        p.requires_grad_(False)
    device = next(model.parameters()).device
    hidden = model.config.hidden_size

    # ---- data ---------------------------------------------------------------
    data = load_concepts(seed=C.SEED)
    concepts = data["concepts"]
    baseline = data["baseline"]
    train_names = data["train_concepts"]  # heldout concept is NOT trained on

    # ---- one up-front pass: cache everything the regression needs -----------
    # After this block the frozen model is never queried again — the loop is pure
    # tensor regression, which is what makes FLAS training fast and stable.
    print(f"[train] caching activations at layer {C.LAYER} for "
          f"{len(train_names)} concepts + baseline...", file=sys.stderr)

    concept_vecs: dict[str, torch.Tensor] = {}   # c : the conditioning embedding
    deltas: dict[str, torch.Tensor] = {}         # c : the diff-of-means target shift
    for name in train_names:
        split = concepts[name]
        # (a) concept embedding c — the mean-activation "ConceptEncoder" identity
        #     of the concept, used to CONDITION the velocity field.
        c_vec = concept_embedding(model, tok, split["exemplars"], C.LAYER)
        concept_vecs[name] = _as_tensor(c_vec, device)
        # (b) target shift delta_c = mean(steer_c) - mean(benign) at LAYER — WHERE
        #     a steered activation for this concept should end up (h1 = h0 + delta).
        caa = extract_caa_vector(
            model, tok, harmful=split["steer_prompts"], benign=baseline, layer=C.LAYER
        )
        deltas[name] = _as_tensor(caa["v_raw"], device)

    # (c) the h0 bank: a pool of UNSTEERED activations the flow learns to transport.
    #     We pool the shared benign baseline (the natural unsteered origin) with the
    #     concepts' own steer prompts, so the field sees transport starting from both
    #     off-concept and on-concept points and generalises across the residual
    #     stream rather than memorising one starting cloud.
    h0_prompts = list(baseline)
    for name in train_names:
        h0_prompts += concepts[name]["steer_prompts"]
    h0_bank_np = last_token_activations(model, tok, h0_prompts, C.LAYER)  # [N, hidden]
    h0_bank = torch.from_numpy(h0_bank_np).to(device=device, dtype=torch.float32)
    n_bank = h0_bank.shape[0]

    # Stack the per-concept tensors so a batch can be gathered by concept index.
    name_order = list(train_names)
    delta_mat = torch.stack([deltas[n] for n in name_order])          # [K, hidden]
    cvec_mat = torch.stack([concept_vecs[n] for n in name_order])     # [K, hidden]
    n_concepts = len(name_order)

    # ---- the only trainable module: the velocity field ----------------------
    # concept_dim defaults to hidden (our concept embeddings are [hidden]); pass
    # C.WIDTH explicitly so the field's capacity is config-driven.
    vfield = VelocityField(hidden, width=C.WIDTH).to(device)
    vfield.train()
    opt = Adam(vfield.parameters(), lr=C.LR)

    print(
        f"[train] steps={C.STEPS} batch={C.BATCH} lr={C.LR} grad_clip={C.GRAD_CLIP} "
        f"layer={C.LAYER} concepts={n_concepts} h0_bank={n_bank}",
        file=sys.stderr,
    )

    # Two standard stabilizers, carried as a scar from the sibling ReFT/HyperSteer
    # runs (whose two-term losses oscillated without them):
    #   1. gradient clipping — bound every update so one stiff step can't fling the
    #      field into a bad region;
    #   2. BEST-CHECKPOINTING — keep the params that hit the LOWEST loss, not
    #      whatever the last step landed on (ordinary early-stopping).
    losses: list[float] = []
    best_loss = float("inf")
    best_state = None
    best_step = -1

    for step in range(C.STEPS):
        # -- assemble a rectified-flow minibatch ------------------------------
        # Sample BATCH (h0, concept) pairs: a random unsteered activation paired
        # with a random train concept. Everything below is a straight-line
        # regression, so the whole batch is a few tensor ops — no LLM forward.
        idx_h0 = torch.randint(0, n_bank, (C.BATCH,), device=device)
        idx_c = torch.randint(0, n_concepts, (C.BATCH,), device=device)

        h0 = h0_bank[idx_h0]              # [B, hidden]  unsteered start
        delta = delta_mat[idx_c]          # [B, hidden]  concept target shift
        c_emb = cvec_mat[idx_c]           # [B, hidden]  concept conditioning
        h1 = h0 + delta                   # [B, hidden]  steered target

        # a random point along each straight path; rectified flow => the target
        # velocity is the SAME (h1 - h0) at every t, which is the whole trick.
        # t is [B,1] to broadcast into the interpolation, but VelocityField wants
        # t shaped like h's LEADING dims ([B]), so we squeeze it for the field.
        t = torch.rand(C.BATCH, 1, device=device)          # [B, 1] ~ U(0,1)
        h_t = (1.0 - t) * h0 + t * h1                      # = h0 + t * delta
        target_v = h1 - h0                                 # = delta (constant in t)

        pred_v = vfield(h_t, t.squeeze(-1), c_emb)         # [B, hidden]
        loss = F.mse_loss(pred_v, target_v)

        # snapshot the params that produced this loss (pre-update) if they are best
        loss_val = float(loss.detach())
        if loss_val < best_loss:
            best_loss = loss_val
            best_step = step
            best_state = {
                k: v.detach().cpu().clone() for k, v in vfield.state_dict().items()
            }

        opt.zero_grad()
        loss.backward()
        # clip BEFORE opt.step() so the update itself is bounded.
        torch.nn.utils.clip_grad_norm_(vfield.parameters(), max_norm=C.GRAD_CLIP)
        opt.step()

        losses.append(loss_val)
        if step % 20 == 0 or step == C.STEPS - 1:
            print(f"[train] step {step:4d}/{C.STEPS}  mse={loss_val:.5f}",
                  file=sys.stderr)

    # restore the best checkpoint before saving (not the last step)
    if best_state is not None:
        vfield.load_state_dict(best_state)
        print(f"[train] restored BEST checkpoint step {best_step} loss {best_loss:.5f}",
              file=sys.stderr)

    # ---- persist the trained field + provenance -----------------------------
    # The concept embeddings travel WITH the field: inference conditions the flow
    # on these to steer each train concept (and encodes the held-out concept the
    # same way for the zero-shot test).
    save_flow(
        C.FLOW_PATH,
        vfield,
        meta={
            "layer": C.LAYER,
            "model_id": C.MODEL_ID,
            "train_concepts": list(train_names),
            "concept_vectors": {
                n: concept_vecs[n].detach().cpu().numpy() for n in train_names
            },
            "n_steps": C.N_STEPS,
        },
    )

    # ---- training curve -----------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG, never open a window
    import matplotlib.pyplot as plt

    curve_path = C.ARTIFACTS / "training_curve.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("rectified-flow MSE  ||v_theta(h_t,t,c) - (h1-h0)||^2")
    ax.set_title("FLAS velocity-field training loss")
    fig.tight_layout()
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    print(f"[train] final loss={losses[-1]:.5f}  best={best_loss:.5f}")
    print(f"[train] saved flow          -> {C.FLOW_PATH}")
    print(f"[train] saved training curve -> {curve_path}")


if __name__ == "__main__":
    main()
