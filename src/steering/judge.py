"""judge.py — the rung-3+ LLM-as-judge for behavior efficacy AND coherence.

WHY THIS EXISTS: the cheap Rung-0/2 proxies in ``eval.py``
(``projection_behavior_scorer``, ``generation_behavior_scorer`` /
``concept_rate``, ``lexicon_from_pairs``) score behavior with lexical-overlap
heuristics. Those are fast and offline, but they are gameable: "fluent
gibberish" that happens to name concept words scores high, and incoherent-but-
on-topic text passes the concept gate. At Rung 3+ we replace the proxy with a
REAL LLM-as-judge that reads the STEERED generation and scores, on two separate
0..10 scales, (a) whether the target behavior is actually exhibited and (b)
whether the text is coherent.

OFF-FAMILY BY DESIGN (CLAUDE.md §10/§14, the same-family circularity rule): the
generator is Gemma; the judge is Google **Gemini** (``gemini-2.5-flash-lite`` by
default). Using a different model family means the judge cannot simply rubber-
stamp its own generations — the audit-circularity disclosure that applies to
same-family critics does not apply here.

DETERMINISTIC + FREE RERUNS: every judged (model, rubric, behavior, text) tuple
is disk-cached (sha256 key), so reruns are deterministic and cost nothing. The
judge calls Gemini at ``temperature=0`` and demands compact integer JSON.

OFFLINE-SAFE FALLBACK: if there is no API key, the project's billing credits are
depleted (the live key currently returns HTTP 429 "prepayment credits are
depleted"), or the network fails, the judge raises ``JudgeUnavailable`` instead
of crashing the experiment. The runner catches that and falls back to the
offline proxy in ``eval.py`` — the harness never hard-depends on the network.

SSL: this machine has an SSL-intercepting middlebox, so we
``truststore.inject_into_ssl()`` (guarded) before any urllib HTTPS call, exactly
as ``scripts/hf_fetch.py`` and ``src/steering/...`` do. We use stdlib ``urllib``
(no ``requests`` / ``google-generativeai`` dependency).

The API key is read from the ``GEMINI_API_KEY`` env var, or — only if the env is
unset — from a gitignored ``.gemini_key`` file at the repo root. The key is
never written to any file, never printed, and never placed in the disk cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

try:  # SSL-intercepting middlebox: trust the OS store (degrade if absent).
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional
    pass

# ---------------------------------------------------------------------------
# Constants / configuration.
# ---------------------------------------------------------------------------
# Bump RUBRIC_VERSION whenever the prompt or scale changes — it is part of the
# cache key, so a rubric edit transparently invalidates stale cached scores
# rather than silently mixing rubrics.
RUBRIC_VERSION = "v1"

DEFAULT_MODEL = "gemini-2.5-flash-lite"
_ENV_MODEL = "GEMINI_JUDGE_MODEL"
_ENV_KEY = "GEMINI_API_KEY"
_KEY_FILE = ".gemini_key"  # gitignored repo-root fallback; read ONLY if env unset.

_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)

# Repo root = two parents up from this file (src/steering/judge.py -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / "autoresearch_results" / "judge_cache"

# Backoff base (seconds) for 429/5xx retries; total waits ≈ base * (1,2,4,...).
_BACKOFF_BASE = 8.0


class JudgeUnavailable(Exception):
    """Raised when the judge cannot produce a real score.

    Covers: no API key, depleted credits / auth failure (4xx), exhausted
    retries on transient 429/5xx, malformed judge output, and network/timeout
    errors. Callers catch this and fall back to the offline proxy in eval.py so
    a missing/over-budget judge never crashes an experiment.
    """


# ---------------------------------------------------------------------------
# The strict rubric prompt.
# ---------------------------------------------------------------------------
def build_rubric_prompt(text: str, behavior_name: str, behavior_description: str) -> str:
    """Build the strict two-axis rubric prompt.

    The prompt scores BEHAVIOR and COHERENCE on SEPARATE 0–10 integer scales so
    the two failure modes are decoupled: fluent gibberish that merely mentions
    the topic cannot earn high BEHAVIOR, and on-topic word-salad cannot earn
    high COHERENCE. It demands a compact JSON object ``{"behavior":int,
    "coherence":int}`` and nothing else, which keeps parsing robust.
    """
    return (
        "You are a strict, calibrated evaluator of text produced by a language "
        "model whose internal activations were steered toward a target behavior.\n"
        "Score the TEXT on TWO INDEPENDENT axes, each an INTEGER from 0 to 10.\n\n"
        "AXIS 1 — BEHAVIOR (does the text actually exhibit the target behavior?):\n"
        f"  Target behavior: {behavior_name}\n"
        f"  Description: {behavior_description}\n"
        "  10 = the behavior is strongly, unambiguously present throughout.\n"
        "   5 = the behavior is weakly or partially present.\n"
        "   0 = the behavior is absent.\n"
        "  Do NOT reward merely naming the topic, keyword-stuffing, or fluent "
        "text that does not genuinely exhibit the behavior. Judge the behavior "
        "itself, not surface vocabulary.\n\n"
        "AXIS 2 — COHERENCE (is the text fluent, grammatical, on-topic, and "
        "non-repetitive, independent of the behavior?):\n"
        "  10 = fully coherent, fluent, non-degenerate prose.\n"
        "   5 = partially coherent (some fluency but broken/looping/off-topic).\n"
        "   0 = gibberish, repetition loops, or word salad.\n"
        "  Score COHERENCE independently of BEHAVIOR: on-topic word-salad must "
        "still score LOW coherence.\n\n"
        "Respond with ONLY a compact JSON object and nothing else, exactly:\n"
        '{"behavior": <int 0-10>, "coherence": <int 0-10>}\n\n'
        "TEXT TO EVALUATE:\n"
        "<<<\n"
        f"{text}\n"
        ">>>\n"
    )


AXBENCH_RUBRIC_VERSION = "axb-v1"


def build_axbench_prompt(text: str, concept: str, instruction: str) -> str:
    """AxBench's steering rubric: CONCEPT (0-2) and FLUENCY (0-2), faithful to
    arXiv:2501.17148 (concept expression conditioned on a fluent, instruction-
    following response). Scored separately so keyword-stuffing cannot win."""
    return (
        "You are evaluating the OUTPUT of a language model that was STEERED toward "
        "a target concept while answering an instruction. Score on AxBench's two "
        "axes, each an INTEGER 0, 1, or 2.\n\n"
        "CONCEPT — how strongly the OUTPUT genuinely expresses the target concept:\n"
        f"  Target concept: {concept}\n"
        "  2 = clearly and substantially expressed.\n"
        "  1 = weakly or partially present.\n"
        "  0 = absent.\n"
        "  Do NOT reward bare keyword mentions or stuffing; judge genuine expression.\n\n"
        "FLUENCY — is the OUTPUT fluent, grammatical, and a coherent response "
        "(independent of the concept)?\n"
        "  2 = fluent, coherent, non-repetitive.\n"
        "  1 = partially fluent (some breakage/repetition/off-topic).\n"
        "  0 = gibberish or repetition loops.\n\n"
        f"INSTRUCTION the model was answering:\n<<<\n{instruction}\n>>>\n\n"
        f"OUTPUT to evaluate:\n<<<\n{text}\n>>>\n\n"
        'Respond with ONLY compact JSON: {"concept": <int 0-2>, "fluency": <int 0-2>}\n'
    )


# ---------------------------------------------------------------------------
# Key / model / cache resolution helpers.
# ---------------------------------------------------------------------------
def _resolve_key(api_key: Optional[str]) -> Optional[str]:
    """Resolve the API key from arg → env → gitignored .gemini_key file.

    Never logs or returns anything but the raw key (or None). The file is read
    ONLY when the env var is unset, matching the security contract.
    """
    if api_key:
        return api_key.strip() or None
    env = os.environ.get(_ENV_KEY)
    if env and env.strip():
        return env.strip()
    key_path = _REPO_ROOT / _KEY_FILE
    try:
        if key_path.is_file():
            contents = key_path.read_text(encoding="utf-8").strip()
            return contents or None
    except OSError:  # pragma: no cover - unreadable key file
        return None
    return None


def _resolve_model(model: Optional[str]) -> str:
    """Resolve the judge model from arg → GEMINI_JUDGE_MODEL env → default."""
    if model:
        return model
    env = os.environ.get(_ENV_MODEL)
    if env and env.strip():
        return env.strip()
    return DEFAULT_MODEL


def _clamp_score(value: Any) -> float:
    """Coerce a judge integer to a float in [0, 10], raising on garbage."""
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise JudgeUnavailable(f"judge returned non-numeric score: {value!r}") from exc
    if f != f:  # NaN
        raise JudgeUnavailable("judge returned NaN score")
    return max(0.0, min(10.0, f))


# ---------------------------------------------------------------------------
# The judge.
# ---------------------------------------------------------------------------
class GeminiJudge:
    """Off-family (Gemini) LLM-as-judge for behavior efficacy + coherence.

    See the module docstring for the contract. Public methods: ``score``,
    ``score_batch``, ``behavior_efficacy``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        api_key: Optional[str] = None,
        max_retries: int = 4,
        timeout: int = 60,
    ) -> None:
        self.model = _resolve_model(model)
        self._api_key = _resolve_key(api_key)
        self.max_retries = max(1, int(max_retries))
        self.timeout = int(timeout)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - read-only fs; cache just disabled
            pass

    # -- cache -------------------------------------------------------------
    def _cache_key(self, text: str, behavior_name: str, behavior_description: str) -> str:
        """sha256 over (model, rubric_version, behavior_name, description, text).

        The KEY is what gives deterministic, free reruns. It deliberately
        excludes the API key (never cache secrets) and includes RUBRIC_VERSION
        so a rubric change invalidates the cache.
        """
        h = hashlib.sha256()
        for part in (
            self.model,
            RUBRIC_VERSION,
            behavior_name,
            behavior_description,
            text,
        ):
            h.update(b"\x00")
            h.update(part.encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        try:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt cache
            return None
        return None

    def _cache_write(self, key: str, payload: dict) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:  # pragma: no cover - read-only fs
            pass

    # -- HTTP --------------------------------------------------------------
    def _http_post(self, prompt: str) -> str:
        """POST the rubric prompt to Gemini; return the raw candidate JSON text.

        Single responsibility: perform ONE generateContent call and extract
        ``candidates[0].content.parts[0].text``. Retries/backoff live in
        ``_call_with_retries``. This method is what tests monkeypatch.
        """
        if not self._api_key:
            raise JudgeUnavailable("no GEMINI_API_KEY (env or .gemini_key)")

        url = _ENDPOINT.format(model=self.model, key=self._api_key)
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 2000,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        raw = urllib.request.urlopen(req, timeout=self.timeout).read()
        resp = json.loads(raw.decode("utf-8"))
        try:
            return resp["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeUnavailable(f"unexpected Gemini response shape: {exc}") from exc

    def _call_with_retries(self, prompt: str) -> str:
        """Call ``_http_post`` with exponential backoff on 429/5xx.

        - 429 / 5xx → sleep base*2**i, retry up to ``max_retries`` times.
        - 4xx other than 429 (auth, depleted credits surfaced as 4xx) → give up
          immediately with JudgeUnavailable (retrying a permanent error is
          pointless and burns wall-clock).
        - Network / timeout → JudgeUnavailable.
        """
        last_msg = "judge call failed"
        for attempt in range(self.max_retries):
            try:
                return self._http_post(prompt)
            except urllib.error.HTTPError as exc:
                code = getattr(exc, "code", None)
                last_msg = f"HTTP {code} from judge"
                if code == 429 or (code is not None and 500 <= code < 600):
                    if attempt < self.max_retries - 1:
                        time.sleep(_BACKOFF_BASE * (2**attempt))
                        continue
                    # Persistent 429/5xx (e.g. depleted credits) → unavailable.
                    raise JudgeUnavailable(
                        f"{last_msg} after {self.max_retries} attempts "
                        "(credits depleted or server error)"
                    ) from exc
                # Other 4xx (auth/bad key) — do not retry.
                raise JudgeUnavailable(f"{last_msg} (auth/request error)") from exc
            except urllib.error.URLError as exc:
                raise JudgeUnavailable(f"network error contacting judge: {exc}") from exc
            except (TimeoutError, OSError) as exc:
                raise JudgeUnavailable(f"timeout/IO contacting judge: {exc}") from exc
        raise JudgeUnavailable(last_msg)  # pragma: no cover - loop always returns/raises

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _parse_scores(raw_text: str) -> tuple[float, float]:
        """Parse the judge's JSON into (behavior, coherence), each clamped 0..10.

        The response is expected to be ``{"behavior":int,"coherence":int}``.
        Malformed / missing-field output raises JudgeUnavailable so the caller
        falls back to the proxy rather than trusting a garbage score.
        """
        try:
            obj = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise JudgeUnavailable(f"judge output is not valid JSON: {raw_text!r}") from exc
        if not isinstance(obj, dict):
            raise JudgeUnavailable(f"judge output is not a JSON object: {raw_text!r}")
        if "behavior" not in obj or "coherence" not in obj:
            raise JudgeUnavailable(f"judge output missing behavior/coherence: {raw_text!r}")
        return _clamp_score(obj["behavior"]), _clamp_score(obj["coherence"])

    # -- public API --------------------------------------------------------
    def score(self, text: str, behavior_name: str, behavior_description: str) -> dict:
        """Judge one generation; return behavior/coherence on 0..10 scales.

        Returns ``{"behavior": float, "coherence": float, "model": str,
        "cached": bool, "raw": str}``. A cache hit returns instantly with
        ``cached=True`` and never touches the network. On any unrecoverable
        failure raises ``JudgeUnavailable``.
        """
        key = self._cache_key(text, behavior_name, behavior_description)
        hit = self._cache_read(key)
        if hit is not None:
            return {
                "behavior": float(hit["behavior"]),
                "coherence": float(hit["coherence"]),
                "model": self.model,
                "cached": True,
                "raw": hit.get("raw", ""),
            }

        prompt = build_rubric_prompt(text, behavior_name, behavior_description)
        raw_text = self._call_with_retries(prompt)
        behavior, coherence = self._parse_scores(raw_text)

        self._cache_write(
            key,
            {"behavior": behavior, "coherence": coherence, "raw": raw_text},
        )
        return {
            "behavior": behavior,
            "coherence": coherence,
            "model": self.model,
            "cached": False,
            "raw": raw_text,
        }

    def score_batch(self, items: Sequence[tuple[str, str, str]]) -> list[dict]:
        """Score many ``(text, behavior_name, behavior_description)`` tuples.

        Sequential (one 4090, one judge); each item is independently cached so a
        partially-cached batch only pays for the cache misses.
        """
        return [self.score(text, name, desc) for (text, name, desc) in items]

    def behavior_efficacy(self, text: str, behavior_name: str, behavior_description: str) -> float:
        """Convenience: behavior score normalized to [0, 1] (harness Axis 1).

        Divides the 0..10 BEHAVIOR score by 10 so it slots directly into the
        ``behavior_efficacy`` field consumed by ``eval.composite``.
        """
        return self.score(text, behavior_name, behavior_description)["behavior"] / 10.0

    # -- AxBench rubric (concept 0-2 + fluency 0-2) ------------------------
    def score_axbench(self, text: str, concept: str, instruction: str) -> dict:
        """Score one steered OUTPUT with AxBench's rubric (arXiv:2501.17148).

        Returns ``{"concept": 0..2, "fluency": 0..2, "behavior": concept/2,
        "axbench": fluency-gated concept in [0,1], "cached": bool, "raw": str}``.
        ``behavior`` (concept/2) slots into the harness Axis-1 field; ``axbench``
        zeroes the concept score when the output is non-fluent (a degenerate
        output is not successful steering). Cached + retried like ``score``.
        """
        h = hashlib.sha256()
        for part in (self.model, AXBENCH_RUBRIC_VERSION, concept, instruction, text):
            h.update(b"\x00")
            h.update(part.encode("utf-8"))
        key = h.hexdigest()
        hit = self._cache_read(key)
        if hit is not None and "concept" in hit and "fluency" in hit:
            concept_s, fluency_s, raw, cached = (
                float(hit["concept"]), float(hit["fluency"]), hit.get("raw", ""), True)
        else:
            raw = self._call_with_retries(build_axbench_prompt(text, concept, instruction))
            try:
                obj = json.loads(raw)
                concept_s = max(0.0, min(2.0, float(obj["concept"])))
                fluency_s = max(0.0, min(2.0, float(obj["fluency"])))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
                raise JudgeUnavailable(f"bad AxBench judge output: {raw!r}") from exc
            self._cache_write(key, {"concept": concept_s, "fluency": fluency_s, "raw": raw})
            cached = False
        return {
            "concept": concept_s, "fluency": fluency_s,
            "behavior": concept_s / 2.0,
            "axbench": (concept_s / 2.0) if fluency_s >= 1.0 else 0.0,
            "cached": cached, "model": self.model, "raw": raw,
        }


def make_judge_or_none(
    model: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    api_key: Optional[str] = None,
    max_retries: int = 4,
    timeout: int = 60,
) -> Optional[GeminiJudge]:
    """Return a ``GeminiJudge`` iff a key is resolvable, else ``None``.

    Lets the runner decide proxy-vs-judge WITHOUT a try/except: ``None`` means
    "no judge configured, use the offline proxy". A returned judge may still
    raise ``JudgeUnavailable`` at call time (e.g. depleted credits) — that is a
    separate, recoverable condition.
    """
    if _resolve_key(api_key) is None:
        return None
    return GeminiJudge(
        model=model,
        cache_dir=cache_dir,
        api_key=api_key,
        max_retries=max_retries,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Judge validation — "validate the judge before you trust it".
# ---------------------------------------------------------------------------
def calibration_agreement(
    judge_scores: Sequence[float],
    reference_scores: Sequence[float],
) -> dict:
    """Agreement between judge scores and a human/reference set.

    Returns ``{"pearson", "spearman", "mae", "n"}``. Pearson = linear
    correlation; Spearman = rank correlation (computed dependency-free via the
    Pearson of ranks); MAE = mean absolute error. Use this on a held-out
    reference set to certify the judge before relying on it (CLAUDE.md §7's
    "validate the instrument" discipline). Constant input ⇒ correlation is
    undefined and reported as ``float('nan')``.
    """
    if len(judge_scores) != len(reference_scores):
        raise ValueError("judge_scores and reference_scores must be the same length")
    n = len(judge_scores)
    if n == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "mae": float("nan"), "n": 0}

    j = [float(x) for x in judge_scores]
    r = [float(x) for x in reference_scores]
    mae = sum(abs(a - b) for a, b in zip(j, r)) / n

    return {
        "pearson": _pearson(j, r),
        "spearman": _pearson(_rank(j), _rank(r)),
        "mae": mae,
        "n": n,
    }


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation; ``nan`` if either input is constant or n<2."""
    n = len(a)
    if n < 2:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    denom = (va * vb) ** 0.5
    if denom == 0.0:
        return float("nan")
    return cov / denom


def _rank(values: Sequence[float]) -> list[float]:
    """Average (tie-corrected) ranks of ``values`` — for Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0  # 0-based average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks
