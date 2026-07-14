"""train_reft.py — train the rank-1 ReFT intervention that RE-INSTALLS refusal.

This is the learning loop for lesson 3 (the ReFT-r1 / AxBench rebuild). The
picture:

    harmful prompt + refusal target ─┐
                                     │  installed at layer L by the intervention
                                     ▼
        grad_reft_forward(model, ids, reft, L)  ──►  logits
                                     │
                          loss = refusal_CE + λ · benign_KL
                                     │
                                     ▼
                        ∂loss/∂(reft.r, reft.w, reft.b)   (the ONLY trainable params)

We freeze the whole LLM and train ONLY the rank-1 intervention ``reft`` (its
direction ``r`` and affine readout ``w, b`` — see reft.py). The model is just a
fixed, differentiable *environment*: gradients flow backward through the frozen
transformer, into the residual-stream edit, and from there into the
intervention's parameters. Nothing in the LLM changes; we are learning a low-rank
edit that emits the right nudge to make the abliterated model refuse again.

Unlike lesson-3-HyperSteer there is NO hypernetwork and NO concept embedding: the
intervention reads the hidden state ``h`` directly through ``w``, so it conditions
on the *input* at run time rather than on a frozen concept vector. There is also
NO ``alpha`` — the intervention carries its own learned magnitude through
``r``/``w``/``b`` (config.py explains this), so nothing here scales the edit.

The objective has TWO terms — the whole point of the lesson is that steering must
be *selective*, so we price both "did it work" and "did it break benign prompts":

  1. REFUSAL cross-entropy (the pull).  On harmful prompts we run the intervened
     forward and ask it to language-model a short refusal (``C.REFUSAL_TARGET``).
     CE is computed on ONLY the refusal-target token positions (the prompt
     positions are masked with ``-100``), so we teach "given this harmful prompt,
     the intervened next tokens are a refusal" — not "reconstruct the prompt".

  2. BENIGN KL divergence (the leash).  An edit that makes *everything* refuse is
     useless. So on a benign prompt we compare the intervened next-token
     distribution against the UNINTERVENED one (a plain no-grad forward) and
     penalise KL(intervened ‖ base). This keeps the edit from disturbing harmless
     requests — the over-refusal / selectivity axis from the project's five axes.

  total = refusal_ce + LAMBDA_KL * benign_kl

  Wu et al. 2025, 'AxBench: Steering LLMs? Even Simple Baselines Outperform
    Sparse Autoencoders' (arXiv:2501.17148) [UNVERIFIED] — the ReFT-r1 steering
    objective (weakly-supervised refusal CE + a KL leash) this loop reproduces.

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
from steering_tutorials.reft_r1 import config as C
from steering_tutorials.reft_r1.data import load_train_eval
from steering_tutorials.reft_r1.reft import (
    ReftR1,
    grad_reft_forward,
    save_reft,
)


def _refusal_ce(model, tok, prompt: str, reft: ReftR1, device) -> torch.Tensor:
    """Cross-entropy for one harmful prompt: make the INTERVENED model refuse.

    We build ``input_ids = chat_template(prompt) + tokens(REFUSAL_TARGET)`` and run
    ONE intervened, gradient-enabled forward. Labels mask the prompt positions with
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

    # Intervened forward with gradient flowing through the rank-1 edit (do NOT wrap
    # in no_grad). Upcast to float32 for a numerically clean cross-entropy (logits
    # may be bf16).
    logits = grad_reft_forward(model, input_ids, reft, C.LAYER).float()

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # ignore everything before the refusal target
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _benign_kl(model, tok, prompt: str, reft: ReftR1, device) -> torch.Tensor:
    """KL(intervened ‖ base) on one benign prompt's next-token distribution.

    The leash: intervening on a harmless request should barely move its output. We
    take the last-position logits from an INTERVENED forward (grad flows into the
    edit) and from a plain UNINTERVENED forward (no grad — the reference), and
    penalise how far the intervened distribution has drifted from the base one.
    """
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    intervened_logits = grad_reft_forward(model, ids, reft, C.LAYER)[
        :, -1, :
    ].float()  # [1, vocab], differentiable in the intervention's params
    with torch.no_grad():  # the base (unintervened) reference is a fixed target
        base_logits = model(ids).logits[:, -1, :].float()

    # KL(P ‖ Q) = sum_x P(x) (log P(x) - log Q(x)), P = intervened, Q = base.
    logp_int = F.log_softmax(intervened_logits, dim=-1)
    logp_base = F.log_softmax(base_logits, dim=-1)
    p_int = logp_int.exp()
    return (p_int * (logp_int - logp_base)).sum(-1).mean()


def main() -> None:
    # ---- reproducibility ----------------------------------------------------
    random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # ---- frozen model as a differentiable environment -----------------------
    model, tok = load_model(C.MODEL_ID)
    for p in model.parameters():  # FREEZE the LLM: we train only the intervention
        p.requires_grad_(False)
    device = next(model.parameters()).device

    # ---- data ---------------------------------------------------------------
    data = load_train_eval(seed=C.SEED)
    harmful_train = data["train"]["harmful"]
    benign_train = data["train"]["benign"]

    # ---- the only trainable module: the rank-1 ReFT intervention ------------
    reft = ReftR1(model.config.hidden_size).to(device)
    reft.train()
    opt = Adam(reft.parameters(), lr=C.LR)

    print(
        f"[train] steps={C.STEPS} batch={C.BATCH} lr={C.LR} "
        f"lambda_kl={C.LAMBDA_KL} grad_clip={C.GRAD_CLIP} layer={C.LAYER}",
        file=sys.stderr,
    )

    # The two-term loss (refusal CE vs benign KL) is stiff on this tiny data, and
    # the rank-1 edit divides by ``||r||`` — so plain SGD OSCILLATES between the
    # two objectives (in the sibling HyperSteer run we observed the total bouncing
    # 0.4 <-> 6+ across steps). Two standard stabilizers make the run reliable:
    #   1. gradient clipping — caps the size of each update so a single stiff step
    #      (or a ||r||-driven gradient spike) can't fling the edit into a
    #      benign-wrecking region;
    #   2. BEST-CHECKPOINTING — we keep the params that achieved the LOWEST loss
    #      seen, not whatever the last step happened to land on. This is ordinary
    #      early-stopping practice; without it the saved edit is a coin flip.
    losses: list[float] = []
    best_total = float("inf")
    best_state = None
    best_step = -1
    for step in range(C.STEPS):
        # -- term 1: refusal CE over a small batch of harmful prompts ---------
        batch = random.sample(harmful_train, k=min(C.BATCH, len(harmful_train)))
        refusal_ce = torch.stack(
            [_refusal_ce(model, tok, prompt, reft, device) for prompt in batch]
        ).mean()

        # -- term 2: benign KL leash on one sampled benign prompt -------------
        benign_prompt = random.choice(benign_train)
        benign_kl = _benign_kl(model, tok, benign_prompt, reft, device)

        total = refusal_ce + C.LAMBDA_KL * benign_kl

        # snapshot the params that produced this loss (current, pre-update) if best
        total_val = float(total.detach())
        if total_val < best_total:
            best_total = total_val
            best_step = step
            best_state = {
                k: t.detach().cpu().clone() for k, t in reft.state_dict().items()
            }

        opt.zero_grad()
        total.backward()
        # clip BEFORE opt.step() so the update itself is bounded (mandatory here).
        torch.nn.utils.clip_grad_norm_(reft.parameters(), max_norm=C.GRAD_CLIP)
        opt.step()

        losses.append(total_val)
        if step % 10 == 0 or step == C.STEPS - 1:
            print(
                f"[train] step {step:4d}/{C.STEPS}  total={total_val:.4f}  "
                f"refusal_ce={float(refusal_ce):.4f}  benign_kl={float(benign_kl):.4f}",
                file=sys.stderr,
            )

    # restore the best checkpoint before saving (not the last, oscillating one)
    if best_state is not None:
        reft.load_state_dict(best_state)
        print(
            f"[train] restored BEST checkpoint: step {best_step} total={best_total:.4f}",
            file=sys.stderr,
        )

    # ---- persist the trained intervention + provenance ----------------------
    save_reft(
        C.REFT_PATH,
        reft,
        meta={
            "layer": C.LAYER,
            "model_id": C.MODEL_ID,
            "concept": "refusal",
            "steps": C.STEPS,
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
    ax.set_title("ReFT-r1 training loss")
    fig.tight_layout()
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    print(f"[train] final loss={losses[-1]:.4f}")
    print(f"[train] saved intervention -> {C.REFT_PATH}")
    print(f"[train] saved training curve -> {curve_path}")


if __name__ == "__main__":
    main()
