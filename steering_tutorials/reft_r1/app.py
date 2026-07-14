"""app.py — the standalone demo webapp for lesson 3 (AxBench ReFT-r1).

Run:  python -m steering_tutorials.reft_r1.app
      (then open http://127.0.0.1:8004  — 8004, so it can sit next to lessons
       1-2 on 8000/8001 without a port clash)

Lesson 2 installed refusal with a *fixed* diff-of-means vector: one direction,
read once, added with a hand-tuned alpha. Lesson 3 swaps that for a *learned*
rank-1 ReFT-r1 intervention (AxBench) — a direction ``r`` plus an affine readout
``(w, b)`` trained end-to-end, so the edit is input-dependent and carries its own
magnitude. This page lets you watch the learned edit run live and compare it, on
pre-computed offline metrics, against the fixed DiffMean vector and raw prompting.

    read the gate  ->  (fires?)  ->  apply the learned rank-1 edit  ->  judge

Endpoints:

  GET  /                    -> the dashboard (static/index.html)
  GET  /results             -> artifacts/results.json, or 404 if not built yet
  GET  /artifacts/<png>     -> a PNG that lives directly in the artifacts dir
  POST /steer   {prompt}    -> run the conditional ReFT-r1 pipeline live and
                               return every intermediate (gate decision, baseline
                               vs ReFT-steered generation, both judge verdicts,
                               a one-line conclusion)

The heavy objects — the abliterated Gemma, the trained ReFT-r1 intervention, the
harm gate and the judge — load ONCE on the first /steer call (see ``_Pipeline``).
Importing this module does NOT touch the GPU or the sibling ``reft`` module, so
it is safe to import for a plumbing check. Nothing here mutates disk; it only
reads pre-computed artifacts.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config as C

app = FastAPI(title="Learned rank-1 steering — ReFT-r1 (lesson 3)")

# The dashboard lives next to this file. config.py owns the artifact paths but
# not the static dir, so we resolve it here relative to the lesson root.
STATIC = C.ROOT / "static"


class _Pipeline:
    """Lazily-loaded bundle of every heavy object the /steer endpoint needs.

    Built exactly once, on the first request. Holds the model + tokenizer, the
    trained ReFT-r1 intervention, the harm gate (lesson-1 probe reused) and the
    self-judge. Unlike lesson 2 there is NO steering alpha here — the learned
    rank-1 edit carries its own magnitude through ``r``, ``w`` and ``b``.
    """

    def __init__(self):
        # Deferred imports: these pull in torch / transformers, the sibling
        # ``reft`` module, and the lesson-2 plumbing. Keeping them inside
        # __init__ is what lets ``import app`` stay model- and reft-free (the
        # CPU import check relies on this).
        from steering_tutorials.hello_world_steering.model_utils import load_model
        from steering_tutorials.hello_world_steering.gate import HarmGate
        from steering_tutorials.hello_world_steering.judge import Judge
        from .reft import load_reft

        self.model, self.tok = load_model(C.MODEL_ID)
        # The trained rank-1 intervention (direction r + affine readout w, b).
        self.reft = load_reft(C.REFT_PATH)
        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)


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

    Graceful-by-design: results.json is written by the separate ``run_reft``
    eval script, so before that runs the file simply isn't there. The dashboard
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
    """Serve a PNG that lives DIRECTLY in the artifacts dir (the two compare plots).

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


def _conclude(fired: bool, baseline_verdict: str, reft_verdict: str) -> str:
    """One-line human verdict on what the conditional ReFT-r1 pipeline did.

    Priority mirrors the causal chain:
      * gate did not fire -> we never touched the output, so nothing changed;
      * the learned edit broke coherence -> "caused corruption" (the failure the
        judge catches first, before REFUSAL-vs-COMPLIANCE);
      * we turned a non-refusal into a refusal -> "steering worked";
      * anything else -> "no change".
    """
    if not fired:
        return "gate did not fire"
    if reft_verdict == "GIBBERISH":
        return "caused corruption"
    if reft_verdict == "REFUSAL" and baseline_verdict != "REFUSAL":
        return "steering worked"
    return "no change"


@app.post("/steer")
def steer(req: SteerRequest) -> dict:
    """Run the full conditional ReFT-r1 pipeline on one prompt, live.

    1. Ask the gate whether to intervene (and log its P(harmful)).
    2. Always generate a baseline (no intervention) for comparison.
    3. If the gate fired, generate again INSIDE a ``ReftContext`` so the learned
       rank-1 edit is applied at every position; otherwise the ReFT output is
       the untouched baseline.
    4. Judge both generations (REFUSAL / COMPLIANCE / GIBBERISH).
    5. Summarise the outcome in one line.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")

    p = get_pipeline()
    # Deferred; cheap once torch is already loaded by the pipeline.
    from steering_tutorials.hello_world_steering.model_utils import generate
    from .reft import ReftContext

    fired, prob = p.gate.is_harmful(prompt)

    # Baseline: plain greedy generation, no intervention of any kind.
    baseline = generate(p.model, p.tok, prompt, alpha=0.0)

    if fired:
        # The learned rank-1 edit is applied by the ReftContext hook — note we
        # pass NO CAA vector/alpha to generate(); all the steering lives in the
        # trained intervention installed on layer C.LAYER.
        with ReftContext(p.model, p.reft, C.LAYER):
            reft_response = generate(p.model, p.tok, prompt, alpha=0.0)
    else:
        # Gate stayed quiet: leave the output exactly as the model produced it.
        # This is the whole point of CONDITIONAL steering — benign prompts pass
        # through untouched, so there's nothing to break.
        reft_response = baseline

    baseline_verdict = p.judge.verdict(prompt, baseline)
    reft_verdict = baseline_verdict if not fired else p.judge.verdict(prompt, reft_response)

    return {
        "prompt": prompt,
        "gate_fired": fired,
        "prob_harmful": prob,
        "layer": C.LAYER,
        "baseline_response": baseline,
        "reft_response": reft_response,
        "baseline_verdict": baseline_verdict,
        "reft_verdict": reft_verdict,
        "conclusion": _conclude(fired, baseline_verdict, reft_verdict),
    }


def main() -> None:
    import uvicorn

    # Port 8004 on purpose: lessons 1-2 own 8000/8001, so all demos can run
    # side by side without a clash.
    print("Serving on http://127.0.0.1:8004  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8004, log_level="info")


if __name__ == "__main__":
    main()
