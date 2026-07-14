"""app.py — the standalone demo webapp for lesson 2 (conditional steering).

Run:  python -m steering_tutorials.hello_world_steering.app
      (then open http://127.0.0.1:8001  — note 8001, not lesson 1's 8000)

Where lesson 1 only READ harm out of Gemma's activations, lesson 2 WRITES a
refusal direction back in — but only when a lightweight gate says the prompt is
harmful. This page lets you watch that whole conditional pipeline run live:

    read the gate  ->  (fires?)  ->  add the steering vector  ->  judge the result

Endpoints:

  GET  /                    -> the dashboard (static/index.html)
  GET  /results             -> artifacts/results.json, or 404 if not built yet
  GET  /artifacts/<png>     -> a PNG that lives directly in the artifacts dir
  POST /steer   {prompt}    -> run the conditional pipeline live and return every
                               intermediate (gate decision, baseline vs steered
                               generation, both judge verdicts, a conclusion)

The heavy objects — the abliterated Gemma, the steering vector, the gate probe
and the judge — load ONCE on the first /steer call (see ``_Pipeline``). Importing
this module does NOT touch the GPU, so it is safe to import for a plumbing check.
Nothing here mutates disk; it only reads pre-computed artifacts.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config as C

app = FastAPI(title="Conditional steering — hello world (lesson 2)")

# The dashboard lives next to this file. config.py owns the artifact paths but
# not the static dir, so we resolve it here relative to the lesson root.
STATIC = C.ROOT / "static"

# Fallback conditional strength if results.json hasn't picked one yet. Kept in
# the middle of config.ALPHAS: enough to install refusal, below the gibberish
# cliff. The real run overrides this via results.json["conditional"]["alpha"].
_DEFAULT_CONDITIONAL_ALPHA = 0.10


class _Pipeline:
    """Lazily-loaded bundle of every heavy object the /steer endpoint needs.

    Built exactly once, on the first request. Holds the model + tokenizer, the
    unit steering vector, the harm gate (lesson-1 probe), the self-judge, and
    the single conditional alpha we apply when the gate fires.
    """

    def __init__(self):
        # Deferred imports: these pull in torch / transformers and the sibling
        # lesson modules. Keeping them inside __init__ is what lets ``import
        # app`` stay model-free (the CPU import check relies on this).
        from .model_utils import load_model
        from .steer_vector import load_vector
        from .gate import HarmGate
        from .judge import Judge

        self.model, self.tok = load_model(C.MODEL_ID)
        self.vector = load_vector(C.VECTOR_PATH)
        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)
        self.alpha = self._resolve_alpha()

    @staticmethod
    def _resolve_alpha() -> float:
        """Prefer the alpha the offline run selected; else the config default."""
        try:
            results = json.loads(C.RESULTS_PATH.read_text(encoding="utf-8"))
            alpha = results.get("conditional", {}).get("alpha")
            if alpha is not None:
                return float(alpha)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
        return _DEFAULT_CONDITIONAL_ALPHA


# Module-level handle so the bundle is reused across requests.
_pipeline: _Pipeline | None = None


def get_pipeline() -> _Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = _Pipeline()
    return _pipeline


class SteerRequest(BaseModel):
    prompt: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/results")
def results() -> JSONResponse:
    """Return artifacts/results.json, or 404 if the offline run hasn't produced it.

    Graceful-by-design: results.json is written by the separate build/eval
    script, so before that runs the file simply isn't there. The dashboard
    hides its entire results section when this 404s, leaving the live demo.
    """
    if not C.RESULTS_PATH.exists():
        raise HTTPException(404, "results.json not generated yet")
    try:
        return JSONResponse(json.loads(C.RESULTS_PATH.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(500, f"could not read results.json: {exc}")


@app.get("/artifacts/{name}")
def artifact(name: str) -> FileResponse:
    """Serve a PNG that lives DIRECTLY in the artifacts dir (rates/conditional plots).

    Path-traversal guard: reject any separator or parent ref, and require the
    name to end in .png so the endpoint stays pinned to a single flat directory.
    """
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(404, "invalid artifact name")
    if not name.endswith(".png"):
        raise HTTPException(404, "only .png artifacts are served")
    path = C.ARTIFACTS / name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "artifact not generated yet")
    return FileResponse(path)


def _conclude(fired: bool, baseline_verdict: str, steered_verdict: str) -> str:
    """One-line human verdict on what the conditional pipeline actually did.

    Priority mirrors the causal chain:
      * gate did not fire -> we never touched the output, so nothing changed;
      * steering broke coherence -> "caused corruption" (the failure the judge
        catches first, before REFUSAL-vs-COMPLIANCE);
      * we turned a non-refusal into a refusal -> "steering worked";
      * anything else -> "no change".
    """
    if not fired:
        return "gate did not fire"
    if steered_verdict == "GIBBERISH":
        return "caused corruption"
    if steered_verdict == "REFUSAL" and baseline_verdict != "REFUSAL":
        return "steering worked"
    return "no change"


@app.post("/steer")
def steer(req: SteerRequest) -> dict:
    """Run the full conditional-steering pipeline on one prompt, live.

    1. Ask the gate whether to intervene (and log its P(harmful)).
    2. Always generate a baseline (alpha=0, no vector) for comparison.
    3. If the gate fired, generate again WITH the steering vector added at the
       steer layer; otherwise the steered output is the untouched baseline.
    4. Judge both generations (REFUSAL / COMPLIANCE / GIBBERISH).
    5. Summarise the outcome in one line.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")

    p = get_pipeline()
    from .model_utils import generate  # deferred; cheap once torch is loaded

    fired, prob = p.gate.is_harmful(prompt)

    baseline = generate(p.model, p.tok, prompt, alpha=0.0)

    if fired:
        steered = generate(
            p.model, p.tok, prompt,
            vector=p.vector["v_unit"], layer=C.STEER_LAYER,
            alpha=p.alpha, operation="relative_add",
        )
    else:
        # Gate stayed quiet: leave the output exactly as the model produced it.
        # This is the whole point of CONDITIONAL steering — benign prompts pass
        # through untouched, so there's nothing to break.
        steered = baseline

    baseline_verdict = p.judge.verdict(prompt, baseline)
    steered_verdict = baseline_verdict if not fired else p.judge.verdict(prompt, steered)

    return {
        "prompt": prompt,
        "gate_fired": fired,
        "prob_harmful": prob,
        "alpha": p.alpha if fired else 0.0,
        "layer": C.STEER_LAYER,
        "baseline_response": baseline,
        "steered_response": steered,
        "baseline_verdict": baseline_verdict,
        "steered_verdict": steered_verdict,
        "conclusion": _conclude(fired, baseline_verdict, steered_verdict),
    }


def main() -> None:
    import uvicorn

    # Port 8001 on purpose: lesson 1's app owns 8000, so both demos can run
    # side by side without a clash.
    print("Serving on http://127.0.0.1:8001  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")


if __name__ == "__main__":
    main()
