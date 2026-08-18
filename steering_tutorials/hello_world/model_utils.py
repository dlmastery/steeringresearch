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


class ActivationWidthError(RuntimeError):
    """Raised when the activations we captured are not the width we expected."""


# Sub-module name fragments that mark a NON-text tower. Gemma-3-4B's SigLIP
# vision encoder is 1152-d — the same width as Gemma-3-1B's text residual
# stream — so hooking it by accident produces plausible, wrong features that a
# 1B-trained probe would happily consume. Never fall back onto these.
_NON_TEXT_MARKERS = ("vision", "visual", "image", "audio", "multi_modal")


def residual_layers(model: nn.Module) -> list[nn.Module]:
    """The list of TEXT decoder blocks whose forward output is the residual stream.

    Text-only Gemma-3 (1B) exposes them at ``model.model.layers``; the multimodal
    4B (``Gemma3ForConditionalGeneration``) nests the TEXT decoder under a
    ``language_model`` sub-module. We search the common paths, then fall back to
    the first non-empty ``.layers`` ModuleList that is NOT part of a vision /
    audio tower.
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
        if any(marker in name.lower() for marker in _NON_TEXT_MARKERS):
            continue  # never probe the vision/audio tower — wrong width, wrong stream
        if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod) > 0:
            return list(mod)
    raise ValueError("Could not find TEXT decoder layers on this model.")


def num_layers(model: nn.Module) -> int:
    return len(residual_layers(model))


def config_hidden_size(model: nn.Module) -> int | None:
    """Text residual-stream width according to the model CONFIG, or None.

    ``text_config`` is consulted FIRST: on a multimodal Gemma-3 the top-level
    config can also carry a ``hidden_size``, and that one belongs to the vision
    tower (1152 on the 4B) rather than to the text decoder (2560).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None):
        return int(text_cfg.hidden_size)
    if getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    return None


def chat_ids(tok: Any, prompt: str, device=None) -> torch.Tensor:
    """Chat-template ``prompt`` and return a PLAIN ``[1, seq]`` id Tensor.

    Exists because **transformers 5.x changed the return type of
    ``apply_chat_template``**. Under 4.x, ``return_tensors="pt"`` returned a raw
    Tensor; under 5.x it defaults to ``return_dict=True`` and returns a
    ``BatchEncoding``. Passing that straight into ``model(ids)`` fails deep inside
    the embedding layer with::

        TypeError: embedding(): argument 'indices' (position 2) must be Tensor,
                   not BatchEncoding

    We normalise by KEY rather than by passing ``return_dict=False``, so this
    works on both major versions and cannot silently change behaviour if the flag
    is renamed again. A dict-like result yields ``input_ids``; a Tensor passes
    through untouched.

    Lesson 1 keeps its own copy (rather than importing lesson 2's identical
    helper) because this lesson is deliberately standalone — lesson 2 depends on
    lesson 1, never the other way round.
    """
    out = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids.to(device) if device is not None else ids


def residual_width(model: nn.Module) -> int | None:
    """Text residual width measured from the decoder MODULE we actually hook."""
    try:
        block = residual_layers(model)[0]
    except Exception:  # pragma: no cover - exotic architecture
        return None
    for attr_path in (("mlp", "gate_proj"), ("mlp", "up_proj"), ("self_attn", "q_proj")):
        obj = block
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and getattr(obj, "in_features", None):
            return int(obj.in_features)
    norm_w = getattr(getattr(block, "input_layernorm", None), "weight", None)
    if norm_w is not None and norm_w.ndim == 1:
        return int(norm_w.shape[0])
    return None


def hidden_size(model: nn.Module) -> int:
    """Residual-stream width, measured from the module we hook where possible.

    The module is the ground truth (it is what ``extract_features`` reads); the
    config is the cross-check. If the two disagree we refuse to guess — a silent
    wrong width here is exactly how a 1152-d probe ends up eating 2560-d
    activations three scripts downstream.
    """
    measured = residual_width(model)
    declared = config_hidden_size(model)
    if measured is not None and declared is not None and measured != declared:
        raise ActivationWidthError(
            f"Residual-stream width disagrees with the model config: the decoder "
            f"block we hook is {measured}-d but config says {declared}-d. On a "
            f"multimodal Gemma-3 this means the TEXT decoder was not the module "
            f"found (the SigLIP vision tower is 1152-d). Refusing to guess."
        )
    if measured is not None:
        return measured
    if declared is not None:
        return declared
    raise ActivationWidthError("Could not determine the residual-stream width.")


def check_cache_model(cache, path, model_id: str) -> None:
    """Fail loudly if a feature cache was produced by a DIFFERENT model.

    A cache is only reusable for the model that produced it — 1B activations are
    1152 numbers wide and 4B activations 2560 — so a stamp mismatch is an error,
    not a cache miss to be silently overwritten (or, worse, silently consumed by
    a probe trained at the other width).
    """
    cached_id = str(cache["model_id"])
    if cached_id != model_id:
        raise ActivationWidthError(
            f"[features] {getattr(path, 'name', path)} was extracted from "
            f"{cached_id} ({cache['X'].shape[1]}-d) but this run uses {model_id}. "
            f"Reusing or overwriting it would mix two models' activations. "
            f"Artifact names are model-tagged by default - unset "
            f"STEER_FEATURES_NAME, or choose a name that is not already taken."
        )
    stamped = int(cache["hidden"]) if "hidden" in getattr(cache, "files", []) else None
    if stamped is not None and stamped != int(cache["X"].shape[1]):
        raise ActivationWidthError(
            f"[features] {getattr(path, 'name', path)} is internally inconsistent: "
            f"X is {cache['X'].shape[1]}-d but the stamp says {stamped}-d. Delete "
            f"it and re-extract."
        )


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
            ids = chat_ids(tok, prompt, device)
            model(ids)
            h = captured["h"][0]  # [seq, hidden]
            vec = h.mean(0) if pooling == "mean" else h[-1]  # pool over tokens
            feats.append(vec.float().cpu().numpy())
            if log_every and (i + 1) % log_every == 0:
                print(f"[features] {i + 1}/{len(prompts)}", file=sys.stderr)
    finally:
        handle.remove()
    X = np.stack(feats).astype(np.float32)

    # Stamp check: what we captured must be as wide as the text residual stream.
    # (If a multimodal model's vision tower were hooked instead, this fires.)
    declared = config_hidden_size(model)
    if declared is not None and X.shape[1] != declared:
        raise ActivationWidthError(
            f"Captured {X.shape[1]}-d activations from layer {layer}, but the "
            f"model's text config declares a {declared}-d residual stream. The "
            f"hooked module is not the text decoder (on multimodal Gemma-3 the "
            f"vision tower is 1152-d while the 4B text stream is 2560-d). "
            f"Refusing to return features of an unknown provenance."
        )
    return X
