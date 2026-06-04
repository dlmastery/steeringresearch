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

_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "autoresearch_results" / "judge_cache"
_LOCAL_RUBRIC = f"{AXBENCH_RUBRIC_VERSION}-local"
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


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
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

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
