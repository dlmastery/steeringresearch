"""model.py — real Gemma loading + a residual-layer helper that works on BOTH
FakeResidualLM and real Gemma.

Gemma is GATED on HuggingFace. If weights are unavailable (no `huggingface-cli
login`, no accepted license, no network), `load_model` raises a clear, actionable
error rather than a cryptic stack trace. Tests never require real Gemma — they
use `steering.fakelm.FakeResidualLM`.

VRAM (4090 Laptop, 16 GB): default to 4-bit bitsandbytes quantisation per
CLAUDE.md §2. gemma-3-1b-it ~1-2 GB, gemma-2-2b-it ~2-3 GB.
"""

from __future__ import annotations

import gc
from typing import Any, cast

import torch
import torch.nn as nn

DEFAULT_MODEL = "google/gemma-3-270m-it"  # tiny Gemma (smoke); gemma-3-1b-it = standard
SUPPORTED_MODELS = (
    "google/gemma-3-1b-it",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
)

_GATED_HELP = (
    "HF login required (gated Gemma). Could not load '{name}'.\n"
    "  Gemma is a gated model. To use REAL Gemma:\n"
    "    1. Create a HuggingFace account and accept the Gemma license at\n"
    "       https://huggingface.co/{name}\n"
    "    2. Run:  huggingface-cli login   (or set HF_TOKEN env var)\n"
    "  For OFFLINE work (all unit tests), use steering.fakelm.FakeResidualLM\n"
    "  via load_model(name='fake').\n"
    "  Underlying error: {err}"
)


def load_model(
    name: str = DEFAULT_MODEL, quant: str = "4bit", device: str | None = None
) -> tuple[Any, Any]:
    """Load a model + tokenizer for steering.

    Parameters
    ----------
    name  : HF model id, or the literal "fake" for an offline FakeResidualLM.
    quant : "4bit" (default, bitsandbytes), "8bit", or "none"/"bf16".
    device: torch device string; defaults to cuda if available else cpu.

    Returns
    -------
    (model, tokenizer). For name="fake", tokenizer is a tiny char-less stub that
    returns deterministic token ids (sufficient for plumbing tests).

    Raises
    ------
    RuntimeError with a clear "HF login required (gated Gemma)" message if the
    real weights cannot be loaded.
    """
    if name == "fake":
        from .fakelm import make_fake_lm

        fake = make_fake_lm()
        return fake, _FakeTokenizer(fake.vocab_size)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as err:  # pragma: no cover - import guard
        raise RuntimeError(
            f"transformers is required to load real Gemma: {err}"
        ) from err

    quant_kwargs: dict[str, Any] = {}
    if quant in ("4bit", "8bit"):
        try:
            from transformers import BitsAndBytesConfig

            if quant == "4bit":
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
        except Exception as err:  # pragma: no cover
            raise RuntimeError(
                f"bitsandbytes {quant} quantisation requested for real Gemma but "
                f"bitsandbytes is unavailable: {err}.\n"
                "  Fix: pip install bitsandbytes  (or pass quant='none' for bf16, "
                "needs more VRAM).\n"
                "  For OFFLINE work (all unit tests), use load_model(name='fake')."
            ) from err
    else:
        quant_kwargs["torch_dtype"] = torch.bfloat16

    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model: Any = AutoModelForCausalLM.from_pretrained(
            name,
            device_map=device if quant in ("4bit", "8bit") else None,
            **quant_kwargs,
        )
        if quant not in ("4bit", "8bit"):
            model = model.to(device)
        model.eval()
        _force_eager_generation(model)
        return model, tokenizer
    except Exception as err:  # gated / SSL / connection / OOM / missing weights
        # Honest, mode-specific diagnosis (ICML_REVIEW spirit: do not cry "gated"
        # when the real cause is SSL interception, a dead connection, or OOM).
        msg = str(err).lower()
        if "certificate" in msg or "ssl" in msg:
            hint = (
                f"SSL certificate verification failed loading '{name}' — an "
                f"SSL-intercepting proxy/middlebox is in the path. Fix: "
                f"`pip install truststore` then inject the OS trust store, or set "
                f"REQUESTS_CA_BUNDLE to your corporate root CA. NOT a gating issue."
            )
        elif "couldn't connect" in msg or "max retries" in msg or "connection" in msg or "timed out" in msg:
            hint = (
                f"Network could not reach huggingface.co to download '{name}' "
                f"(connection blocked/offline). If the model is gated, also accept "
                f"its license + `huggingface-cli login`. Pre-download it where the "
                f"network works: `huggingface-cli download {name}` — once in the HF "
                f"cache the harness loads it offline automatically."
            )
        elif (
            "paging file" in msg
            or "out of memory" in msg
            or "1455" in msg
            or ("cuda" in msg and "memory" in msg)
        ):
            hint = (
                f"Out-of-memory / paging-file exhaustion loading '{name}'. Load ONE "
                f"model per process (don't reload N times in a loop), or quantise "
                f"(pip install bitsandbytes; quant='4bit'). NOT a gating issue."
            )
        elif "gated" in msg or "401" in msg or "403" in msg or "authoriz" in msg or "access" in msg:
            hint = _GATED_HELP.format(name=name, err=err)
        else:
            hint = (
                f"Could not load '{name}'. Underlying error: {err}\n"
                f"For OFFLINE work use load_model(name='fake')."
            )
        raise RuntimeError(hint) from err


def _force_eager_generation(model: nn.Module) -> None:
    """Force eager (uncompiled, dynamic-cache) generation.

    WHY: on Windows, transformers' default ``cache_implementation`` for several
    recent architectures (Gemma-3 included) is a *static* cache that HF routes
    through ``torch.compile`` -> TorchInductor. Inductor then tries to JIT a CUDA
    kernel via Triton, and Triton has no Windows wheel, so generation crashes with
    "Cannot find a working triton installation". We never want compilation here:
    the harness generates a handful of short greedy continuations, so an eager
    dynamic cache is both correct and faster to warm up.

    Two independent guards (belt-and-braces):
      1. ``torch._dynamo.config.suppress_errors = True`` — if anything still tries
         to compile, dynamo falls back to eager instead of raising.
      2. ``generation_config.cache_implementation = "dynamic"`` — stops HF from
         selecting the compiled static cache in the first place.
    """
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.suppress_errors = True
    except Exception:  # pragma: no cover - dynamo always present with torch>=2
        pass
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:  # FakeResidualLM has no generation_config — guard.
        try:
            gen_cfg.cache_implementation = "dynamic"
        except Exception:  # pragma: no cover - frozen/odd config objects
            pass


# Process-level model+tokenizer cache keyed by (name, quant, device). Loading a
# real Gemma is expensive (VRAM + paging file); reloading it every experiment is
# the documented "reload N times -> OOM/paging-file exhaustion" footgun. We keep
# exactly ONE loaded model per key and free the previous one when the key changes.
_MODEL_CACHE: dict[tuple[str, str, str | None], tuple[Any, Any]] = {}


def load_model_cached(
    name: str = DEFAULT_MODEL,
    quant: str = "4bit",
    device: str | None = None,
):
    """Load (or reuse) a model+tokenizer, caching one handle per (name, quant, device).

    On a *fresh* key, any previously cached models are freed first
    (``del`` + ``gc.collect()`` + ``torch.cuda.empty_cache()``) so an in-process
    alpha-sweep over many configs never accumulates VRAM/paging-file pressure.
    The ``"fake"`` path is cached too (cheap, but keeps the contract uniform).
    """
    key = (name, quant, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # New key: free everything currently held before loading the next model.
    if _MODEL_CACHE:
        free_model_cache()

    model, tokenizer = load_model(name, quant=quant, device=device)
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def free_model_cache() -> None:
    """Drop all cached models and reclaim memory (VRAM + host)."""
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_to_device(tokenizer, text: str, model: nn.Module | None = None) -> torch.Tensor:
    """Encode ``text`` -> input_ids ``[1, seq]``, moved to the model's device.

    Single source of truth for the tokenize->input_ids->to(device) dance that was
    previously duplicated in ``extract._encode``, ``eval._ids`` and ``runner``.

    Handles:
      * real HF ``BatchEncoding`` (NOT a dict subclass in current transformers) via
        the ``.input_ids`` attribute,
      * the offline ``_FakeTokenizer`` which returns a plain ``dict``,
      * a bare tensor.
    Falls back to CPU (leaves ids put) when the model is parameter-less
    (``StopIteration``) or ``None``.
    """
    out = tokenizer(text, return_tensors="pt")
    if hasattr(out, "input_ids"):
        ids = out.input_ids
    elif isinstance(out, dict):
        ids = out["input_ids"]
    else:
        ids = out
    if model is not None:
        try:
            ids = ids.to(next(model.parameters()).device)
        except StopIteration:
            pass
    return cast(torch.Tensor, ids)


def get_residual_layers(model: nn.Module) -> list[nn.Module]:
    """Return the list of per-layer residual-block modules to hook.

    Works on BOTH FakeResidualLM and real Gemma. The returned modules each
    produce the post-block residual stream h_{l+1} as their forward output, so a
    forward hook on layers[l] reads/edits the residual stream at depth l.

    Resolution order:
      1. FakeResidualLM:        model.layers
      2. Gemma-2 / Gemma-3:     model.model.layers
      3. generic decoder:       first ModuleList of decoder blocks found
    """
    # FakeResidualLM and many HF decoders expose `.layers` directly.
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return list(model.layers)

    # HF wrapping: AutoModelForCausalLM -> .model -> .layers (Gemma, Llama, ...)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        layers = inner.layers
        if isinstance(layers, nn.ModuleList):
            return list(layers)

    # Some Gemma-3 variants nest one level deeper.
    if inner is not None:
        inner2 = getattr(inner, "model", None)
        if inner2 is not None and hasattr(inner2, "layers"):
            return list(inner2.layers)

    # Fallback: find the largest ModuleList of repeated blocks.
    candidates: list[nn.ModuleList] = [
        m for m in model.modules()
        if isinstance(m, nn.ModuleList) and len(m) >= 2
    ]
    if candidates:
        largest = max(candidates, key=lambda ml: len(ml))
        return list(largest)

    raise ValueError(
        "Could not locate residual layers on this model. Pass a FakeResidualLM "
        "or a standard HF decoder (Gemma/Llama-style) with model.model.layers."
    )


def num_layers(model: nn.Module) -> int:
    """Number of residual layers available for hooking."""
    return len(get_residual_layers(model))


class _FakeTokenizer:
    """A minimal deterministic tokenizer for the offline FakeResidualLM.

    Maps text -> a reproducible id sequence (hash-based), prefixed with the BOS
    id. Sufficient for plumbing/Rung-0 tests; never used for real evaluation.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.start_of_turn_id = 2
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for ch in text:
            # deterministic, avoid the reserved special ids 0,1,2
            ids.append(3 + (ord(ch) % (self.vocab_size - 3)))
        return ids

    def __call__(self, text, return_tensors: str | None = None, add_special_tokens: bool = True):
        if isinstance(text, str):
            text = [text]
        seqs = [self.encode(t, add_special_tokens) for t in text]
        maxlen = max(len(s) for s in seqs)
        ids = [s + [self.pad_token_id] * (maxlen - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs]
        out: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
        if return_tensors == "pt":
            out = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out
