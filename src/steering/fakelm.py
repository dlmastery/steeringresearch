"""FakeResidualLM — a tiny OFFLINE decoder stub for unit tests.

Why this exists
---------------
Gemma is gated on HuggingFace and may not be available (no token set). Every
Rung-0 / Rung-1 unit test in this harness must run with NO network and NO Gemma.
`FakeResidualLM` is a minimal, deterministic decoder-only transformer-shaped
module that exposes per-layer residual blocks so the *same* hook/extract/geometry
code paths exercised against real Gemma also work here.

Mechanics mirror the residual-stream picture in
`corpus/steering-first-principles-v2-...md` Step 0:

    h_{l+1} = h_l + attn_l(h_l) + mlp_l(h_l)

Each block is a `ResidualBlock` nn.Module; `model.layers` is the ModuleList that
`get_residual_layers` returns for hooking. Forward returns an object with a
`.logits` attribute (shape [batch, seq, vocab]) so it quacks like a HF causal-LM
output, and also exposes `.hidden_states` (tuple, one per layer + embeddings)
when `output_hidden_states=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FakeLMOutput:
    """Mimics a HF CausalLMOutput closely enough for the harness."""

    logits: torch.Tensor
    hidden_states: Optional[tuple] = None


@dataclass
class FakeLMConfig:
    """Minimal HF-like config shim (built once per model, not per access)."""

    num_hidden_layers: int
    hidden_size: int
    vocab_size: int


class _Attn(nn.Module):
    """A cheap, deterministic causal self-attention surrogate.

    Not a real attention — it is a single linear mixing that respects causal
    structure via a lower-triangular average. Enough to make positions interact
    so steering at one position can move logits at later positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [batch, seq, dim]
        s = h.shape[1]
        v = self.proj(h)
        # Causal running mean: a lower-triangular, row-normalised mixing matrix
        # [seq, seq] applied per batch. dtype/device follow h so the stub also
        # works under a bf16/cuda FakeLM (not exercised today, but kept honest).
        mask = torch.tril(torch.ones(s, s, device=h.device, dtype=h.dtype))
        weights = mask / mask.sum(dim=-1, keepdim=True)  # row-normalised [s, s]
        mixed = torch.einsum("st,btd->bsd", weights, v)  # [batch, seq, dim]
        return self.out(mixed)


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(h)))


class ResidualBlock(nn.Module):
    """One residual block: h <- h + attn(h) + mlp(h).

    This module IS the hook target. A forward hook on this module sees the
    block's *output* — i.e. the post-block residual stream h_{l+1}. The harness
    hooks here to read/edit the residual stream at layer `l`.
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _Attn(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.ln1(h))
        h = h + self.mlp(self.ln2(h))
        return h


class FakeResidualLM(nn.Module):
    """A tiny deterministic decoder-only LM for offline tests.

    Parameters
    ----------
    vocab_size : small vocabulary (default 32)
    dim        : residual-stream width (the "d_model")
    n_layers   : number of residual blocks (exposed as `.layers`)
    hidden     : MLP hidden width
    seed       : determinism seed for weight init
    """

    # Special-token ids the harness must NOT steer (BOS / start_of_turn analogues).
    bos_token_id: int = 1
    start_of_turn_id: int = 2

    def __init__(
        self,
        vocab_size: int = 32,
        dim: int = 16,
        n_layers: int = 4,
        hidden: int = 32,
        seed: int = 0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Build with a fixed generator for full determinism.
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [ResidualBlock(dim, hidden) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(dim)
        self.unembed = nn.Linear(dim, vocab_size, bias=False)

        # Build the HF-like config once (not on every `.config` access).
        self._config = FakeLMConfig(
            num_hidden_layers=n_layers, hidden_size=dim, vocab_size=vocab_size
        )

        # Deterministic init: reset every parameter from the seeded generator.
        self._init_deterministic(g)
        self.eval()

    def _init_deterministic(self, g: torch.Generator) -> None:
        # First, set all 2-D weights to a scaled normal and all 1-D params to 0.
        for p in self.parameters():
            if p.dim() >= 2:
                # scaled normal so signals propagate but don't explode
                p.data = torch.randn(p.shape, generator=g) * 0.1
            else:
                p.data = torch.zeros(p.shape)
        # LayerNorm weights MUST be 1 (zero weight would zero the output and
        # make the unembed produce constant logits insensitive to steering).
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data = torch.ones_like(m.weight.data)
                m.bias.data = torch.zeros_like(m.bias.data)

    @property
    def config(self) -> FakeLMConfig:  # minimal HF-like config shim
        return self._config

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> FakeLMOutput:
        """Run the stack.

        input_ids : [batch, seq] long tensor
        Returns FakeLMOutput with .logits [batch, seq, vocab].
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        h = self.embed(input_ids)
        hidden_states: list[torch.Tensor] = [h] if output_hidden_states else []
        for layer in self.layers:
            h = layer(h)
            if output_hidden_states:
                hidden_states.append(h)
        h = self.final_ln(h)
        logits = self.unembed(h)
        return FakeLMOutput(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )

    def special_token_ids(self) -> list[int]:
        """Ids excluded from steering (BOS / start_of_turn analogues)."""
        return [self.bos_token_id, self.start_of_turn_id]


def make_fake_lm(seed: int = 0, **kwargs) -> FakeResidualLM:
    """Factory: a deterministic FakeResidualLM. All tests use this."""
    return FakeResidualLM(seed=seed, **kwargs)
