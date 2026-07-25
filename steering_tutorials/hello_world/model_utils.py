"""model_utils.py — load the frozen LLM and read its activations.

Two jobs, nothing else:
  1. load_model()      — load the (uncensored) Gemma-3-1B + tokenizer, once.
  2. extract_features()— for a list of prompts, return one activation vector
                          each: the mean of the residual stream at LAYER.

Standalone: no dependency on the research harness. The only third-party pieces
are ``torch`` and ``transformers`` (plus an optional SSL guard for corporate
middleboxes).
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:  # Some networks sit behind an SSL-intercepting proxy; use the OS trust store.
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - optional
    pass


def load_model(model_id: str, device: str | None = None) -> tuple[Any, Any]:
    """Load ``model_id`` in bf16 and put it in eval mode. Returns (model, tokenizer).

    We load in bf16 (not 4-bit) because the model is tiny (~2 GB) and full
    precision gives the cleanest activations to probe. The two guards below
    stop transformers from trying to ``torch.compile`` a Triton CUDA kernel,
    which has no Windows wheel and would crash.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Guard 1: if anything still tries to compile, fall back to eager.
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.suppress_errors = True
    except Exception:  # pragma: no cover
        pass

    tok = AutoTokenizer.from_pretrained(model_id)

    # Cross-scale check: when config.LOAD_4BIT is set, load a LARGER model (e.g.
    # multimodal Gemma-3-4B) in 4-bit so it fits VRAM/RAM. bitsandbytes places the
    # model on the GPU itself (device_map), so we must NOT call .to(device) after.
    load_4bit = False
    try:
        from . import config as _cfg
        load_4bit = bool(getattr(_cfg, "LOAD_4BIT", False))
    except Exception:  # pragma: no cover
        pass

    if load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
            low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model = model.to(device)
    model.eval()

    # Guard 2: never select the compiled static KV cache.
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        try:
            gen_cfg.cache_implementation = "dynamic"
        except Exception:  # pragma: no cover
            pass

    print(f"[model] loaded {model_id} on {device} "
          f"({num_layers(model)} layers, hidden={hidden_size(model)})", file=sys.stderr)
    return model, tok


def residual_layers(model: nn.Module) -> list[nn.Module]:
    """The list of decoder blocks whose forward output is the residual stream.

    Text-only Gemma-3 (1B) exposes them at ``model.model.layers``; the multimodal
    4B (``Gemma3ForConditionalGeneration``) nests the TEXT decoder under a
    ``language_model`` sub-module. We search the common paths, then fall back to
    the first non-empty ``.layers`` ModuleList in the tree.
    """
    candidates = (
        ("model", "layers"),
        ("language_model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "model", "layers"),
    )
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj)
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod) > 0:
            return list(mod)
    raise ValueError("Could not find decoder layers on this model.")


def num_layers(model: nn.Module) -> int:
    return len(residual_layers(model))


def hidden_size(model: nn.Module) -> int:
    """Residual-stream width. Multimodal Gemma-3 keeps it under ``text_config``."""
    cfg = model.config
    if getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None):
        return int(text_cfg.hidden_size)
    return int(residual_layers(model)[0].mlp.gate_proj.in_features)


@torch.no_grad()
def extract_features(
    model: Any,
    tok: Any,
    prompts: list[str],
    layer: int,
    pooling: str = "mean",
    log_every: int = 25,
) -> np.ndarray:
    """Return an ``[n_prompts, hidden]`` float32 matrix of activation features.

    For each prompt we:
      1. wrap it in the model's chat template (so activations match how the
         model actually sees a user turn),
      2. run a single forward pass,
      3. capture the residual stream at ``layer`` via a forward hook,
      4. mean-pool over the token positions -> one vector.

    One prompt at a time keeps the code trivial (no padding / attention-mask
    bookkeeping); a few hundred short prompts take a minute or two on a GPU.
    """
    device = next(model.parameters()).device
    layers = residual_layers(model)
    layer = max(0, min(layer, len(layers) - 1))
    target = layers[layer]

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach()

    handle = target.register_forward_hook(hook)
    feats: list[np.ndarray] = []
    try:
        for i, prompt in enumerate(prompts):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            model(ids)
            h = captured["h"][0]  # [seq, hidden]
            vec = h.mean(0) if pooling == "mean" else h[-1]  # pool over tokens
            feats.append(vec.float().cpu().numpy())
            if log_every and (i + 1) % log_every == 0:
                print(f"[features] {i + 1}/{len(prompts)}", file=sys.stderr)
    finally:
        handle.remove()
    return np.stack(feats).astype(np.float32)
