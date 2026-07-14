"""app.py — the standalone demo webapp for the FLAS lesson (flow-based steering).

Run:  python -m steering_tutorials.flas.app
      (then open http://127.0.0.1:8005  — 8005, so it sits next to lessons
       1-3 on 8000/8001/8004 without a port clash)

Lessons 2-3 pushed the residual stream with a single one-shot edit: lesson 2
added a *fixed* diff-of-means vector at a hand-tuned strength, lesson 3 applied a
*learned* rank-1 edit. FLAS reframes steering as **transport**: it learns a
concept-conditioned velocity field ``v(h, t, c)`` and STEERS by integrating a
flow ODE from the unsteered activation to its steered position. Two things fall
out that neither one-shot edit gives:

    * flow-time ``T`` is a continuous, zero-shot STRENGTH dial — integrate less
      far along the SAME learned trajectory, no retraining, no alpha sweep;
    * one field handles many concepts, because the velocity is conditioned on a
      concept embedding ``c``.

This page lets you turn the ``T`` dial live and watch the whole pipeline run:

    read the gate  ->  (fires?)  ->  integrate the flow to time T  ->  judge

Endpoints:

  GET  /                    -> the dashboard (static/index.html)
  GET  /results             -> artifacts/results.json, or 404 if not built yet
  GET  /artifacts/<png>     -> a PNG that lives directly in the artifacts dir
  POST /steer   {prompt, T, concept, exemplars?}
                            -> run the conditional flow pipeline live and return
                               every intermediate (gate decision, baseline vs
                               flow-steered generation at strength T, both judge
                               verdicts, a one-line conclusion)

The heavy objects — the abliterated Gemma, the trained velocity field, the harm
gate and the judge — load ONCE on the first /steer call (see ``_Pipeline``).
Importing this module does NOT touch the GPU or the sibling ``flow`` module, so
it is safe to import for a plumbing check. Nothing here mutates disk; it only
reads pre-computed artifacts.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config as C

app = FastAPI(title="Flow-based activation steering — FLAS")

# The dashboard lives next to this file. config.py owns the artifact paths but
# not the static dir, so we resolve it here relative to the lesson root.
STATIC = C.ROOT / "static"


# ---------------------------------------------------------------------------
# Small, defensive helpers for the flow bundle. flow.py is built alongside this
# app; these normalise the two shapes ``load_flow`` might hand back so the app
# never cares which one it got.
# ---------------------------------------------------------------------------
def _unpack_flow(loaded):
    """Normalise ``load_flow(...)`` -> ``(vfield, meta)``.

    ``load_flow`` may return the velocity field together with its metadata as a
    ``(vfield, meta)`` tuple (mirroring lesson-3's ``load_reft``), or just the
    field with the metadata hung off it as ``.meta``. We accept either.
    """
    if isinstance(loaded, tuple):
        vfield = loaded[0]
        meta = loaded[1] if len(loaded) > 1 else {}
    else:
        vfield = loaded
        meta = getattr(loaded, "meta", None)
    return vfield, (meta or {})


def _concept_vectors(meta) -> dict:
    """Pull the ``name -> concept vector`` map out of ``meta`` (dict OR object)."""
    if isinstance(meta, dict):
        cv = meta.get("concept_vectors")
    else:
        cv = getattr(meta, "concept_vectors", None)
    return cv or {}


class _Pipeline:
    """Lazily-loaded bundle of every heavy object the /steer endpoint needs.

    Built exactly once, on the first request. Holds the model + tokenizer, the
    trained velocity field and its concept-vector table, the harm gate (lesson-1
    probe reused) and the self-judge. Unlike lesson 2 there is NO steering alpha
    here — the strength dial is the flow-time ``T`` passed per request; the
    velocity field carries the trajectory.
    """

    def __init__(self):
        # Deferred imports: these pull in torch / transformers, the sibling
        # ``flow`` module, and the lesson-2 plumbing. Keeping them inside
        # __init__ is what lets ``import app`` stay model- and flow-free (the
        # CPU import check relies on this).
        from steering_tutorials.hello_world_steering.model_utils import load_model
        from steering_tutorials.hello_world_steering.gate import HarmGate
        from steering_tutorials.hello_world_steering.judge import Judge
        from .flow import load_flow

        self.model, self.tok = load_model(C.MODEL_ID)
        # The trained velocity field v(h, t, c) plus the metadata table of
        # per-concept embeddings the field was conditioned on.
        self.vfield, self.meta = _unpack_flow(load_flow(C.FLOW_PATH))
        self.concept_vectors = _concept_vectors(self.meta)
        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)

    # -- concept selection --------------------------------------------------
    def resolve_concept(self, name, exemplars):
        """Return ``(concept_vec, concept_name)`` for this request.

        Priority mirrors "most specific wins":
          1. explicit ``exemplars`` -> rebuild the embedding on the fly via
             ``concept_embedding`` (the zero-shot, bring-your-own-concept path);
          2. a named concept present in the trained table -> use that vector;
          3. otherwise fall back to a default refusal vector (the ``refusal``
             entry if present, else the first concept in the table).
        ``concept_vec`` may be ``None`` only if the field shipped with no concept
        table AND no exemplars were supplied — the caller degrades gracefully.
        """
        if exemplars:
            from .flow import concept_embedding

            vec = concept_embedding(self.model, self.tok, exemplars, C.LAYER)
            return vec, (name or "custom")

        cv = self.concept_vectors
        if name and name in cv:
            return cv[name], name
        if "refusal" in cv:
            return cv["refusal"], "refusal"
        if cv:
            first = next(iter(cv))
            return cv[first], first
        return None, (name or "refusal")


# Module-level handle so the bundle is reused across requests.
_pipeline: _Pipeline | None = None


def get_pipeline() -> _Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = _Pipeline()
    return _pipeline


class SteerRequest(BaseModel):
    prompt: str
    # Flow-time strength dial. Defaults to the config default (T=1.0 = the full
    # learned trajectory); the slider on the page sends anything in [0, 2].
    T: float = C.T_DEFAULT
    # Optional named concept to steer toward (indexes the trained concept table).
    concept: str | None = None
    # Optional exemplars to rebuild a concept embedding on the fly (zero-shot).
    exemplars: list[str] | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/results")
def results() -> JSONResponse:
    """Return artifacts/results.json, or 404 if the offline run hasn't produced it.

    Graceful-by-design: results.json is written by the separate ``run_flas`` eval
    script, so before that runs the file simply isn't there. The dashboard hides
    its entire results section when this 404s, leaving the live demo.
    """
    if not C.RESULTS_PATH.exists():
        raise HTTPException(404, "results.json not generated yet")
    try:
        return JSONResponse(json.loads(C.RESULTS_PATH.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(500, f"could not read results.json: {exc}")


@app.get("/artifacts/{name}")
def artifact(name: str) -> FileResponse:
    """Serve a PNG that lives DIRECTLY in the artifacts dir (the two sweep plots).

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


def _conclude(fired: bool, baseline_verdict: str, flow_verdict: str) -> str:
    """One-line human verdict on what the conditional flow pipeline did.

    Priority mirrors the causal chain:
      * gate did not fire -> we never touched the output, so nothing changed;
      * integrating the flow broke coherence -> "caused corruption" (the failure
        the judge catches first, before REFUSAL-vs-COMPLIANCE);
      * we turned a non-refusal into a refusal -> "steering worked";
      * anything else -> "no change".
    """
    if not fired:
        return "gate did not fire"
    if flow_verdict == "GIBBERISH":
        return "caused corruption"
    if flow_verdict == "REFUSAL" and baseline_verdict != "REFUSAL":
        return "steering worked"
    return "no change"


@app.post("/steer")
def steer(req: SteerRequest) -> dict:
    """Run the full conditional flow-steering pipeline on one prompt, live.

    1. Ask the gate whether to intervene (and log its P(harmful)).
    2. Always generate a baseline (no intervention) for comparison.
    3. If the gate fired, generate again INSIDE a ``FlowContext`` that integrates
       the velocity field to flow-time ``T`` toward the chosen concept; otherwise
       the flow output is the untouched baseline.
    4. Judge both generations (REFUSAL / COMPLIANCE / GIBBERISH).
    5. Summarise the outcome in one line.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")
    # Clamp T into the dial's range so a hand-crafted request can't ask the
    # integrator for a wild extrapolation the field was never trained for.
    T = max(0.0, min(float(req.T), 2.0))

    p = get_pipeline()
    # Deferred; cheap once torch is already loaded by the pipeline.
    from steering_tutorials.hello_world_steering.model_utils import generate
    from .flow import FlowContext

    fired, prob = p.gate.is_harmful(prompt)

    # Baseline: plain greedy generation, no intervention of any kind.
    baseline = generate(p.model, p.tok, prompt, alpha=0.0)

    concept_vec, concept_name = p.resolve_concept(req.concept, req.exemplars)

    if fired and concept_vec is not None and T > 0.0:
        # All the steering lives in the flow: we pass NO CAA vector/alpha to
        # generate() and instead integrate v(h, t, c) to flow-time T at layer
        # C.LAYER inside the FlowContext hook.
        with FlowContext(p.model, p.vfield, concept_vec, C.LAYER, T=T):
            flow_response = generate(p.model, p.tok, prompt, alpha=0.0)
    else:
        # Gate stayed quiet (benign prompt), T dialled to zero, or no concept
        # vector available: leave the output exactly as the model produced it.
        flow_response = baseline

    steered = fired and concept_vec is not None and T > 0.0
    baseline_verdict = p.judge.verdict(prompt, baseline)
    flow_verdict = p.judge.verdict(prompt, flow_response) if steered else baseline_verdict

    return {
        "prompt": prompt,
        "gate_fired": fired,
        "prob_harmful": prob,
        "T": T,
        "concept": concept_name,
        "layer": C.LAYER,
        "baseline_response": baseline,
        "baseline_verdict": baseline_verdict,
        "flow_response": flow_response,
        "flow_verdict": flow_verdict,
        "conclusion": _conclude(steered, baseline_verdict, flow_verdict),
    }


def main() -> None:
    import uvicorn

    # Port 8005 on purpose: lessons 1-3 own 8000/8001/8004, so all demos can run
    # side by side without a clash.
    print("Serving on http://127.0.0.1:8005  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8005, log_level="info")


if __name__ == "__main__":
    main()
