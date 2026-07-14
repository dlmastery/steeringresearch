"""train_hypersteer.py — train the hypernetwork that GENERATES a steering vector.

This is the learning loop for lesson 3. The picture:

    concept exemplars ── concept_embedding ──►  e  (frozen [hidden] vector)
                                                │
                                        H_theta │  (the ONLY thing we train)
                                                ▼
                                          v = H(e)   the steering vector
                                                │
              harmful prompt + refusal target │  added at layer L (alpha_train)
                                                ▼
                         grad_steer_forward(model, ids, v, L, alpha)  ──►  logits
                                                │
                                     loss = refusal_CE + λ · benign_KL

We freeze the whole LLM and train ONLY the hypernetwork ``H_theta``. The model is
just a fixed, differentiable *environment*: gradients flow backward through the
frozen transformer, into the injected steering vector ``v``, and from there into
H's parameters. Nothing in the LLM changes; we are learning a function that emits
the right nudge to the residual stream.

The objective has TWO terms — the whole point of the lesson is that steering must
be *selective*, so we price both "did it work" and "did it break benign prompts":

  1. REFUSAL cross-entropy (the pull).  On harmful prompts we add ``v`` and ask
     the steered model to language-model a short refusal (``C.REFUSAL_TARGET``).
     CE is computed on ONLY the refusal-target token positions (the prompt
     positions are masked with ``-100``), so we teach "given this harmful prompt,
     the steered next tokens are a refusal" — not "reconstruct the prompt".

  2. BENIGN KL divergence (the leash).  A vector that makes *everything* refuse is
     useless. So on a benign prompt we compare the steered next-token distribution
     against the UNSTEERED one (a plain no-grad forward) and penalise KL(steered ‖
     unsteered). This keeps ``v`` from disturbing harmless requests — the
     over-refusal / selectivity axis from the project's five axes.

  total = refusal_ce + LAMBDA_KL * benign_kl

  Sun et al. 2025, 'HyperSteer: Activation Steering at Scale with Hypernetworks'
    (arXiv:2506.03292) [UNVERIFIED] — the amortised hypernetwork objective.

CPU-ONLY NOTE: importing this module runs NOTHING (all work is inside ``main()``,
guarded by ``__main__``). Loading Gemma and running the loop is a GPU job the lead
launches separately; here we only WRITE and import-check the loop.
"""
from __future__ import annotations

import random
import sys

import torch
import torch.nn.functional as F
from torch.optim import Adam

from steering_tutorials.hello_world_steering.model_utils import load_model
from steering_tutorials.hypersteer import config as C
from steering_tutorials.hypersteer.data import load_train_eval
from steering_tutorials.hypersteer.hypernet import (
    HyperSteerNet,
    concept_embedding,
    grad_steer_forward,
    save_hypernet,
)

# Weight on the benign-KL leash relative to the refusal pull. Lives in config if
# the peer exposes it; otherwise defaults to the lesson value (~0.5). A LARGER
# value protects benign prompts harder at the cost of a weaker refusal.
LAMBDA_KL = float(getattr(C, "LAMBDA_KL", 0.5))


def _refusal_ce(model, tok, prompt: str, v: torch.Tensor, device) -> torch.Tensor:
    """Cross-entropy for one harmful prompt: make the STEERED model refuse.

    We build ``input_ids = chat_template(prompt) + tokens(REFUSAL_TARGET)`` and run
    ONE steered, gradient-enabled forward. Labels mask the prompt positions with
    ``-100`` so only the refusal-target tokens contribute to the loss; the usual
    causal shift (logits[t] predicts token t+1) means each refusal token is
    predicted from the prompt + the refusal tokens before it.
    """
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    # add_special_tokens=False: the refusal is a CONTINUATION, so it must not get
    # its own BOS inserted in the middle of the sequence.
    target_ids = tok(
        C.REFUSAL_TARGET, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)  # [1, seq]
    prompt_len = prompt_ids.shape[1]

    # Steered forward with gradient flowing through v (do NOT wrap in no_grad).
    # Upcast to float32 for a numerically clean cross-entropy (logits may be bf16).
    logits = grad_steer_forward(
        model, input_ids, v, C.STEER_LAYER, C.ALPHA_TRAIN
    ).float()

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # ignore everything before the refusal target
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _benign_kl(model, tok, prompt: str, v: torch.Tensor, device) -> torch.Tensor:
    """KL(steered ‖ unsteered) on one benign prompt's next-token distribution.

    The leash: steering a harmless request should barely move its output. We take
    the last-position logits from a STEERED forward (grad flows into v) and from a
    plain UNSTEERED forward (no grad — the reference), and penalise how far the
    steered distribution has drifted from the unsteered one.
    """
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    steered_logits = grad_steer_forward(
        model, ids, v, C.STEER_LAYER, C.ALPHA_TRAIN
    )[:, -1, :].float()  # [1, vocab], differentiable in v
    with torch.no_grad():  # the unsteered reference is a fixed target
        unsteered_logits = model(ids).logits[:, -1, :].float()

    # KL(P ‖ Q) = sum_x P(x) (log P(x) - log Q(x)), P = steered, Q = unsteered.
    logp_steer = F.log_softmax(steered_logits, dim=-1)
    logp_unsteer = F.log_softmax(unsteered_logits, dim=-1)
    p_steer = logp_steer.exp()
    return (p_steer * (logp_steer - logp_unsteer)).sum(-1).mean()


def main() -> None:
    # ---- reproducibility ----------------------------------------------------
    random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # ---- frozen model as a differentiable environment -----------------------
    model, tok = load_model(C.MODEL_ID)
    for p in model.parameters():  # FREEZE the LLM: we train only the hypernetwork
        p.requires_grad_(False)
    device = next(model.parameters()).device

    # ---- data + the frozen concept embedding (computed ONCE) ----------------
    data = load_train_eval(seed=C.SEED)
    harmful_train = data["train"]["harmful"]
    benign_train = data["train"]["benign"]
    concept_exemplars = data["concept_exemplars"]
    # The concept embedding is a fixed INPUT to the hypernetwork. It comes back as
    # a numpy [hidden] float32 vector; move it onto the device as a plain (no-grad)
    # tensor so no gradient tries to flow back into the exemplar activations. Kept
    # in float32 to match the hypernetwork's parameters.
    concept_np = concept_embedding(model, tok, concept_exemplars, C.STEER_LAYER)
    concept_emb = torch.from_numpy(concept_np).to(device)

    # ---- the only trainable module ------------------------------------------
    net = HyperSteerNet(
        hidden_dim=model.config.hidden_size, bottleneck=C.BOTTLENECK
    ).to(device)
    net.train()
    opt = Adam(net.parameters(), lr=C.LR)

    print(
        f"[train] steps={C.STEPS} batch={C.BATCH} lr={C.LR} "
        f"alpha_train={C.ALPHA_TRAIN} lambda_kl={LAMBDA_KL} "
        f"layer={C.STEER_LAYER}",
        file=sys.stderr,
    )

    losses: list[float] = []
    for step in range(C.STEPS):
        # Regenerate v every step: as H's params update, so does the vector.
        v = net(concept_emb)  # [hidden], differentiable in H's params

        # -- term 1: refusal CE over a small batch of harmful prompts ---------
        batch = random.sample(harmful_train, k=min(C.BATCH, len(harmful_train)))
        refusal_ce = torch.stack(
            [_refusal_ce(model, tok, prompt, v, device) for prompt in batch]
        ).mean()

        # -- term 2: benign KL leash on one sampled benign prompt -------------
        benign_prompt = random.choice(benign_train)
        benign_kl = _benign_kl(model, tok, benign_prompt, v, device)

        total = refusal_ce + LAMBDA_KL * benign_kl

        opt.zero_grad()
        total.backward()
        opt.step()

        losses.append(float(total.detach()))
        if step % 10 == 0 or step == C.STEPS - 1:
            print(
                f"[train] step {step:4d}/{C.STEPS}  total={float(total):.4f}  "
                f"refusal_ce={float(refusal_ce):.4f}  benign_kl={float(benign_kl):.4f}",
                file=sys.stderr,
            )

    # ---- persist the trained hypernetwork + provenance ----------------------
    save_hypernet(
        C.NET_PATH,
        net,
        meta={
            "concept": "refusal",
            "layer": C.STEER_LAYER,
            "alpha_train": C.ALPHA_TRAIN,
            "steps": C.STEPS,
            "model_id": C.MODEL_ID,
            # The eval/infer agents look for meta["concept_embedding"] (numpy
            # [hidden]) so they can re-emit v = H(concept_embedding) with no
            # re-extraction pass; ship the exemplar list alongside for provenance.
            "concept_embedding": concept_np,
            "concept_exemplars": concept_exemplars,
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
    ax.set_ylabel("total loss (refusal_ce + λ·benign_kl)")
    ax.set_title("HyperSteer training loss")
    fig.tight_layout()
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    print(f"[train] final loss={losses[-1]:.4f}")
    print(f"[train] saved hypernetwork -> {C.NET_PATH}")
    print(f"[train] saved training curve -> {curve_path}")


if __name__ == "__main__":
    main()
