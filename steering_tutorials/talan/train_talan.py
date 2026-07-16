"""train_talan.py -- train the latent adapter that RE-INSTALLS refusal (frozen LLM).

The learning loop for the TALAN-inspired lesson. The picture is identical in shape
to lesson 3's ReFT trainer -- only the trainable module changes (a nonlinear
bottleneck adapter instead of a rank-1 edit):

    harmful prompt + refusal target -+
                                     |  installed at layer L by the adapter
                                     v
        grad_talan_forward(model, ids, adapter, L)  -->  logits
                                     |
                          loss = refusal_CE + lambda * benign_KL
                                     |
                                     v
              d loss / d(adapter down, mix, up, scale)   (the ONLY trainable params)

We FREEZE the whole LLM and train ONLY the adapter. The model is a fixed,
differentiable environment: gradients flow backward through the frozen transformer,
into the residual-stream writeback, and from there into the adapter. Nothing in the
LLM changes -- this is the inference-time simplification of TALAN (the paper also
trains a backbone LoRA; we do not).

The objective has TWO terms -- steering must be SELECTIVE, so we price both "did it
refuse the harmful prompt" and "did it leave benign prompts alone":

  1. REFUSAL cross-entropy (the pull).  On harmful prompts we run the adapted
     forward and language-model a short refusal (``C.REFUSAL_TARGET``). CE is on
     ONLY the refusal-target token positions (prompt positions masked with -100).

  2. BENIGN KL divergence (the leash).  On a benign prompt we compare the adapted
     next-token distribution against the UNADAPTED one (a plain no-grad forward) and
     penalise KL(adapted || base) -- the over-refusal / selectivity axis.

  total = refusal_ce + LAMBDA_KL * benign_kl

  Zhang et al. 2026, 'TALAN: Task-Aligned Latent Adaptation Networks for Targeted
    Post-Training of Large Language Models' (arXiv:2606.06902) -- the latent
    side-path adapter this loop reproduces at inference-time scale (frozen LLM).
    NOTE: the paper is a post-training method with a joint SFT objective; our
    weakly-supervised refusal-CE + KL-leash objective is our own construction for
    the inference-time analogue, not the paper's training recipe.

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
from steering_tutorials.talan import config as C
from steering_tutorials.talan.data import load_train_eval
from steering_tutorials.talan.talan import (
    TalanAdapter,
    grad_talan_forward,
    save_talan,
)


def _refusal_ce(model, tok, prompt: str, adapter: TalanAdapter, device) -> torch.Tensor:
    """Cross-entropy for one harmful prompt: make the ADAPTED model refuse.

    Builds ``input_ids = chat_template(prompt) + tokens(REFUSAL_TARGET)`` and runs
    ONE adapted, gradient-enabled forward. Labels mask the prompt positions with
    -100 so only the refusal-target tokens contribute; the usual causal shift
    (logits[t] predicts token t+1) means each refusal token is predicted from the
    prompt + the refusal tokens before it.
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

    # Adapted forward with gradient flowing through the writeback (no no_grad).
    # Upcast to float32 for a numerically clean cross-entropy (logits may be bf16).
    logits = grad_talan_forward(model, input_ids, adapter, C.LAYER).float()

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # ignore everything before the refusal target
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _benign_kl(model, tok, prompt: str, adapter: TalanAdapter, device) -> torch.Tensor:
    """KL(adapted || base) on one benign prompt's next-token distribution.

    The leash: adapting a harmless request should barely move its output. We take
    the last-position logits from an ADAPTED forward (grad flows into the adapter)
    and from a plain UNADAPTED forward (no grad -- the reference), and penalise how
    far the adapted distribution has drifted from the base one.
    """
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    adapted_logits = grad_talan_forward(model, ids, adapter, C.LAYER)[
        :, -1, :
    ].float()  # [1, vocab], differentiable in the adapter's params
    with torch.no_grad():  # the base (unadapted) reference is a fixed target
        base_logits = model(ids).logits[:, -1, :].float()

    # KL(P || Q) = sum_x P(x) (log P(x) - log Q(x)), P = adapted, Q = base.
    logp_adapt = F.log_softmax(adapted_logits, dim=-1)
    logp_base = F.log_softmax(base_logits, dim=-1)
    p_adapt = logp_adapt.exp()
    return (p_adapt * (logp_adapt - logp_base)).sum(-1).mean()


def main() -> None:
    # ---- reproducibility ----------------------------------------------------
    random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # ---- frozen model as a differentiable environment -----------------------
    model, tok = load_model(C.MODEL_ID)
    for p in model.parameters():  # FREEZE the LLM: we train only the adapter
        p.requires_grad_(False)
    device = next(model.parameters()).device

    # ---- data ---------------------------------------------------------------
    data = load_train_eval(n_per_class=C.N_PER_CLASS, n_eval=C.N_EVAL, seed=C.SEED)
    harmful_train = data["train"]["harmful"]
    benign_train = data["train"]["benign"]

    # ---- the only trainable module: the TALAN latent adapter ----------------
    adapter = TalanAdapter(
        model.config.hidden_size, memory=C.MEMORY, mixer=C.MIXER,
        init_scale=C.GRAD_SCALE,
    ).to(device)
    adapter.train()
    opt = Adam(adapter.parameters(), lr=C.LR)

    n_backbone = sum(p.numel() for p in model.parameters())
    n_adapter = adapter.num_params()
    print(
        f"[train] steps={C.STEPS} batch={C.BATCH} lr={C.LR} lambda_kl={C.LAMBDA_KL} "
        f"grad_clip={C.GRAD_CLIP} layer={C.LAYER} memory={C.MEMORY} mixer={C.MIXER}",
        file=sys.stderr,
    )
    print(
        f"[train] adapter params={n_adapter} "
        f"({100.0 * n_adapter / max(1, n_backbone):.4f}% of the {n_backbone} backbone)",
        file=sys.stderr,
    )

    # The two-term loss (refusal CE vs benign KL) is stiff on this tiny data and the
    # adapter is higher-capacity than a rank-1 edit -- so plain SGD OSCILLATES
    # between the two objectives. Two standard stabilisers make the run reliable
    # (the same scar carried from the ReFT lesson):
    #   1. gradient clipping -- caps each update so one stiff step can't fling the
    #      adapter into a benign-wrecking region;
    #   2. BEST-CHECKPOINTING -- we keep the params that achieved the LOWEST loss
    #      seen, not whatever the last step landed on. Ordinary early stopping;
    #      without it the saved adapter is a coin flip.
    losses: list[float] = []
    best_total = float("inf")
    best_state = None
    best_step = -1
    for step in range(C.STEPS):
        # -- term 1: refusal CE over a small batch of harmful prompts ---------
        batch = random.sample(harmful_train, k=min(C.BATCH, len(harmful_train)))
        refusal_ce = torch.stack(
            [_refusal_ce(model, tok, prompt, adapter, device) for prompt in batch]
        ).mean()

        # -- term 2: benign KL leash on one sampled benign prompt -------------
        benign_prompt = random.choice(benign_train)
        benign_kl = _benign_kl(model, tok, benign_prompt, adapter, device)

        total = refusal_ce + C.LAMBDA_KL * benign_kl

        # snapshot the params that produced this loss (current, pre-update) if best
        total_val = float(total.detach())
        if total_val < best_total:
            best_total = total_val
            best_step = step
            best_state = {
                k: t.detach().cpu().clone() for k, t in adapter.state_dict().items()
            }

        opt.zero_grad()
        total.backward()
        # clip BEFORE opt.step() so the update itself is bounded (mandatory here).
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=C.GRAD_CLIP)
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
        adapter.load_state_dict(best_state)
        print(
            f"[train] restored BEST checkpoint: step {best_step} total={best_total:.4f}",
            file=sys.stderr,
        )

    # ---- persist the trained adapter + provenance ---------------------------
    save_talan(
        C.ADAPTER_PATH,
        adapter,
        meta={
            "layer": C.LAYER,
            "model_id": C.MODEL_ID,
            "concept": "refusal",
            "steps": C.STEPS,
            "memory": C.MEMORY,
            "mixer": C.MIXER,
            "writeback": C.WRITEBACK,
            "train_scope": C.TRAIN_SCOPE,
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
    ax.set_ylabel("total loss (refusal_ce + lambda*benign_kl)")
    ax.set_title("TALAN latent-adapter training loss")
    fig.tight_layout()
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    print(f"[train] final loss={losses[-1]:.4f}")
    print(f"[train] saved adapter -> {C.ADAPTER_PATH}")
    print(f"[train] saved training curve -> {curve_path}")


if __name__ == "__main__":
    main()
