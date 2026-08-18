"""local_judge.py — an OFF-FAMILY, fully-local LLM judge (no external API).

Replaces the Gemini judge when API credits/network are unavailable. Runs an
open-weight instruct model from a DIFFERENT family than the steered Gemma model
(default Qwen2.5-7B-Instruct, Apache-2.0, 4-bit) so the same-model-family
circularity the project keeps disclosing is still broken — the judge just lives
on the 4090 instead of behind an API. Free, offline, unlimited, deterministic
(greedy), cached.

Same interface as ``judge.GeminiJudge.score_axbench`` (AxBench concept 0-2 +
fluency 0-2 rubric, via ``judge.build_axbench_prompt``), so a driver swaps
instruments with one flag. ``JudgeUnavailable`` is raised on load/parse failure
so callers can fall back.

VRAM: the judge model is loaded INDEPENDENTLY of the steered model (not through
``model.load_model_cached``, which evicts), so both stay resident — a tiny Gemma
(0.5-3 GB) plus a 7B 4-bit judge (~5 GB) fits in 16 GB.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from .judge import AXBENCH_RUBRIC_VERSION, JudgeUnavailable, build_axbench_prompt

# The default judge must be a model that is (a) present in the local HF cache and
# (b) validated by H0 (see hypotheses/A_foundations/H0_instrument_validity.md).
# It was previously "Qwen/Qwen2.5-7B-Instruct", whose cache entry on this host is a
# 16 KB stub with zero safetensors — so any run that forgot to pass --judge-model
# silently began a ~15 GB download mid-experiment. Never let the default be a model
# that is not on disk.
_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "autoresearch_results" / "judge_cache"
_LOCAL_RUBRIC = f"{AXBENCH_RUBRIC_VERSION}-local"
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _chat_ids(tok: Any, msgs: list) -> Any:
    """Chat-template ``msgs`` and return a plain ``[1, seq]`` id Tensor.

    transformers 5.x changed ``apply_chat_template``: 4.x with
    ``return_tensors="pt"`` returned a raw Tensor, 5.x defaults to
    ``return_dict=True`` and returns a ``BatchEncoding``, which blows up inside
    the embedding layer. Normalise BY KEY rather than by passing
    ``return_dict=False``, so this works under both major versions.

    Mirrors ``steering_tutorials/hello_world_steering/model_utils.chat_ids``.
    Duplicated deliberately: ``src/steering`` is the research harness and the
    tutorials are standalone by design, so neither may import the other.
    """
    out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt")
    return out["input_ids"] if hasattr(out, "keys") else out


def _clamp2(v: Any) -> float:
    return max(0.0, min(2.0, float(v)))


class LocalJudge:
    """Local open-weight instruct model scoring AxBench's concept+fluency rubric."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        quant: str = "4bit",
        cache_dir: Optional[str | Path] = None,
        max_new_tokens: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover
            pass

        kwargs: dict[str, Any] = {}
        if quant == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            self.model.eval()
        except Exception as exc:  # pragma: no cover - load failure
            raise JudgeUnavailable(f"could not load local judge {model_id}: {exc}") from exc
        self.device = next(self.model.parameters()).device

    # -- cache (shares the on-disk judge cache; key namespaced by local rubric) --
    def _key(self, text: str, concept: str, instruction: str) -> str:
        h = hashlib.sha256()
        for part in (self.model_id, _LOCAL_RUBRIC, concept, instruction, text):
            h.update(b"\x00")
            h.update(part.encode("utf-8"))
        return h.hexdigest()

    def _read(self, key: str) -> Optional[dict]:
        p = self.cache_dir / f"{key}.json"
        try:
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover
            return None
        return None

    def _write(self, key: str, payload: dict) -> None:
        try:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:  # pragma: no cover
            pass

    def _generate(self, prompt: str) -> str:
        import torch

        msgs = [{"role": "user", "content": prompt}]
        ids = _chat_ids(self.tok, msgs).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def score_axbench_expected(self, text: str, concept: str, instruction: str) -> float:
        """Continuous concept score in [0,2] read from the token distribution (H0-2).

        The AxBench rubric emits an INTEGER in {0,1,2}. That quantization throws away
        the judge's own uncertainty and saturates the score with ties, which depresses
        ROC-AUC against ground-truth labels (H0-1 measured AUC 0.665 with a large mean
        separation of +0.58 — the signature of a coarse score, not of no signal).

        Instead of generating, we teacher-force the JSON prefix so the very next token
        IS the concept digit, then take the expected value under the model's own
        distribution restricted to {"0","1","2"}:

            score = Σ_d p(d) · d   /   Σ_d p(d),      d ∈ {0,1,2}

        This is strictly more informative than the argmax (it recovers "0.8 vs 1.4"
        where the rubric could only say "1"), needs ONE forward pass instead of a
        16-token generate, and is therefore also faster. Returns a float in [0,2].
        """
        import torch

        msgs = [{"role": "user", "content": build_axbench_prompt(text, concept, instruction)}]
        ids = _chat_ids(self.tok, msgs).to(self.device)
        prefix = self.tok('{"concept": ', add_special_tokens=False,
                          return_tensors="pt").input_ids.to(self.device)
        full = torch.cat([ids, prefix], dim=1)
        with torch.no_grad():
            logits = self.model(full).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)

        # First token id of each digit string (handles tokenizers that prefix a space).
        cand = []
        for d in ("0", "1", "2"):
            tid = self.tok.encode(d, add_special_tokens=False)
            cand.append(tid[0])
        p = probs[cand]
        tot = float(p.sum())
        if tot <= 0:
            raise JudgeUnavailable("digit tokens carry no probability mass")
        p = p / p.sum()
        return float(p[0] * 0.0 + p[1] * 1.0 + p[2] * 2.0)

    @staticmethod
    def _parse(raw: str) -> tuple[float, float]:
        m = _JSON_RE.search(raw)
        if not m:
            raise JudgeUnavailable(f"local judge returned no JSON: {raw!r}")
        try:
            obj = json.loads(m.group(0))
            return _clamp2(obj["concept"]), _clamp2(obj["fluency"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise JudgeUnavailable(f"bad local judge JSON: {raw!r}") from exc

    def score_axbench(self, text: str, concept: str, instruction: str) -> dict:
        """Concept 0-2 + fluency 0-2 (+ normalized fields); cached, deterministic."""
        key = self._key(text, concept, instruction)
        hit = self._read(key)
        if hit is not None and "concept" in hit and "fluency" in hit:
            concept_s, fluency_s, raw, cached = (
                float(hit["concept"]), float(hit["fluency"]), hit.get("raw", ""), True)
        else:
            raw = self._generate(build_axbench_prompt(text, concept, instruction))
            concept_s, fluency_s = self._parse(raw)
            self._write(key, {"concept": concept_s, "fluency": fluency_s, "raw": raw})
            cached = False
        return {
            "concept": concept_s, "fluency": fluency_s,
            "behavior": concept_s / 2.0,
            "axbench": (concept_s / 2.0) if fluency_s >= 1.0 else 0.0,
            "cached": cached, "model": self.model_id, "raw": raw,
        }

    def _result(self, concept_s: float, fluency_s: float, raw: str, cached: bool) -> dict:
        return {
            "concept": concept_s, "fluency": fluency_s, "behavior": concept_s / 2.0,
            "axbench": (concept_s / 2.0) if fluency_s >= 1.0 else 0.0,
            "cached": cached, "model": self.model_id, "raw": raw,
        }

    def score_axbench_batch(self, items: list[tuple[str, str, str]],
                            batch_size: int = 16) -> list[dict]:
        """Batched ``score_axbench`` — many (text, concept, instruction) tuples in
        one GPU generate() call. Cache hits are served free; only misses are
        batched. ~10x faster than per-call for bnb-4bit (amortizes dequant
        overhead). A per-item parse failure degrades that ONE item to concept 0
        (it never aborts the batch)."""
        import torch

        out: list[Optional[dict]] = [None] * len(items)
        misses: list[tuple[int, str, str]] = []   # (index, key, prompt)
        for i, (text, concept, instruction) in enumerate(items):
            key = self._key(text, concept, instruction)
            hit = self._read(key)
            if hit is not None and "concept" in hit and "fluency" in hit:
                out[i] = self._result(float(hit["concept"]), float(hit["fluency"]),
                                      hit.get("raw", ""), True)
            else:
                prompt = self.tok.apply_chat_template(
                    [{"role": "user", "content": build_axbench_prompt(text, concept, instruction)}],
                    add_generation_prompt=True, tokenize=False)
                misses.append((i, key, prompt))

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"  # decoder-only generation needs left padding
        for s in range(0, len(misses), batch_size):
            chunk = misses[s:s + batch_size]
            enc = self.tok([p for _, _, p in chunk], return_tensors="pt",
                           padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    num_beams=1, pad_token_id=self.tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for k, (i, key, _) in enumerate(chunk):
                raw = self.tok.decode(new[k], skip_special_tokens=True)
                try:
                    concept_s, fluency_s = self._parse(raw)
                except JudgeUnavailable:
                    concept_s, fluency_s = 0.0, 0.0   # degrade this one item, don't abort
                self._write(key, {"concept": concept_s, "fluency": fluency_s, "raw": raw})
                out[i] = self._result(concept_s, fluency_s, raw, False)
        return [r for r in out if r is not None]
