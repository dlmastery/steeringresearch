"""app.py — the standalone demo webapp for lesson 3 (HyperSteer).

Run:  python -m steering_tutorials.hypersteer.app
      (then open http://127.0.0.1:8003  — 8003, not lessons 1/2's 8000/8001)

The arc of the three lessons is READ -> WRITE -> GENERATE:

  * lesson 1 READ harm out of a frozen Gemma-1B's activations (a probe),
  * lesson 2 WROTE one fixed refusal direction back in (CAA diff-of-means),
  * lesson 3 GENERATES that direction with a learned hypernetwork:
        v = H(concept_embedding)
    so a brand-new concept's exemplars can be turned into a steering vector on
    the fly — amortised steering, no fresh contrastive extraction pass.

This page lets you watch the generated vector work live, and — the headline —
type your own exemplar phrases and watch the hypernetwork emit a steering vector
for that concept on the spot.

Endpoints:

  GET  /                       -> the dashboard (static/index.html)
  GET  /results                -> artifacts/results.json, or 404 if not built
  GET  /artifacts/<png>        -> a PNG that lives directly in the artifacts dir
  POST /steer   {prompt}       -> gate -> (fires?) -> add the HYPERNET vector ->
                                  judge baseline vs steered; returns every step
  POST /generate_concept       -> {exemplars:[str]} build a concept embedding from
                                  the phrases, run the hypernet to EMIT a fresh
                                  steering vector, then steer+judge a demo prompt
                                  with it (the amortisation payoff, shown live)

The heavy objects — the abliterated Gemma, the trained hypernetwork, the harm
gate and the judge — load ONCE on the first request (see ``_Bundle``). Importing
this module does NOT touch the GPU or load the hypernet, so it is safe to import
for a plumbing check. Nothing here mutates disk; it only reads artifacts.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config as C

app = FastAPI(title="HyperSteer — hypernetwork steering (lesson 3)")

# The dashboard lives next to this file. config.py owns the artifact paths but
# not the static dir, so we resolve it here relative to the lesson root.
STATIC = C.ROOT / "static"

# A neutral prompt we steer in /generate_concept to SHOW a freshly-emitted vector
# doing something. Kept concept-agnostic: for a refusal concept it should turn
# into a refusal; for another concept the generation shifts in that concept's
# direction. It is not the point of the metric, only a visible demonstration.
_DEMO_PROMPT = "Tell me a little about your plans for the weekend."

# Fallback exemplars for the trained refusal direction, used only if the saved
# hypernet checkpoint did not stash its concept embedding in ``meta``. These are
# short first-person refusals — the same behaviour lesson 2 re-installed.
_DEFAULT_REFUSAL_EXEMPLARS = [
    "I can't help with that request.",
    "I'm sorry, but I won't assist with that.",
    "I cannot provide that information.",
    "That's something I have to decline to help with.",
]


class _Bundle:
    """Lazily-loaded bundle of every heavy object the endpoints need.

    Built exactly once, on the first request. Holds the model + tokenizer, the
    trained hypernetwork, the pre-computed refusal steering vector it emits, the
    harm gate (lesson-1 probe), the self-judge, and — optionally, if lesson 2 was
    built — the fixed CAA vector so the live demo can put the learned vector next
    to the hand-built one (the whole lesson-3 thesis: the hypernet can MATCH it).
    """

    def __init__(self):
        # Deferred imports: these pull in torch / transformers, the sibling
        # lesson-2 modules, and this lesson's hypernet. Keeping them inside
        # __init__ is what lets ``import app`` stay model-free (the CPU import
        # check relies on this).
        import numpy as np
        import torch

        from steering_tutorials.hello_world_steering.model_utils import load_model
        from steering_tutorials.hello_world_steering.gate import HarmGate
        from steering_tutorials.hello_world_steering.judge import Judge
        from .hypernet import load_hypernet, concept_embedding

        self._np = np
        self._torch = torch
        self._concept_embedding = concept_embedding

        self.model, self.tok = load_model(C.MODEL_ID)
        self.net, self.net_meta = load_hypernet(C.NET_PATH)
        self.gate = HarmGate(self.model, self.tok)
        self.judge = Judge(self.model, self.tok)

        # The steering vector for the concept the hypernet was TRAINED on
        # (refusal). Prefer the concept embedding stashed in the checkpoint; if
        # it isn't there, rebuild it from the default refusal exemplars so the
        # live /steer demo still has a vector to add.
        emb = self.net_meta.get("concept_emb")
        if emb is None:
            emb = self._concept_embedding(
                self.model, self.tok, _DEFAULT_REFUSAL_EXEMPLARS, C.STEER_LAYER
            )
        self.v_hyper = self._emit_vector(emb)

        # Optional: the lesson-2 fixed CAA vector, for the side-by-side arm. It
        # only exists if lesson 2 was run, so load it defensively.
        self.fixed_caa = self._load_fixed_caa()

    # -- helpers -----------------------------------------------------------
    def _emit_vector(self, concept_emb) -> "object":
        """Run the hypernet on a concept embedding -> a ``[hidden]`` numpy vector.

        Inference only: no grad, net in eval mode. ``concept_emb`` may be a numpy
        array or a plain list (as it comes back from a JSON checkpoint); we coerce
        it to a float32 tensor either way.
        """
        np, torch = self._np, self._torch
        emb = np.asarray(concept_emb, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            v = self.net(torch.from_numpy(emb))
        return v.detach().cpu().numpy().astype(np.float32)

    def _load_fixed_caa(self):
        """The lesson-2 CAA diff-of-means vector, or None if lesson 2 wasn't run."""
        try:
            from steering_tutorials.hello_world_steering import config as L2C
            from steering_tutorials.hello_world_steering.steer_vector import (
                load_vector,
            )

            if not L2C.VECTOR_PATH.exists():
                return None
            return load_vector(L2C.VECTOR_PATH)
        except Exception:
            # A missing/broken lesson-2 artifact must never sink lesson 3.
            return None


# Module-level handle so the bundle is reused across requests.
_bundle: _Bundle | None = None


def get_bundle() -> _Bundle:
    global _bundle
    if _bundle is None:
        _bundle = _Bundle()
    return _bundle


class SteerRequest(BaseModel):
    prompt: str


class ConceptRequest(BaseModel):
    exemplars: list[str]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/results")
def results() -> JSONResponse:
    """Return artifacts/results.json, or 404 if the offline run hasn't built it.

    Graceful-by-design: results.json is written by the separate run_hypersteer
    script (train + eval). Before that runs the file simply isn't there, and the
    dashboard hides its whole results section, leaving the live demo working.
    """
    if not C.RESULTS_PATH.exists():
        raise HTTPException(404, "results.json not generated yet")
    try:
        return JSONResponse(json.loads(C.RESULTS_PATH.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(500, f"could not read results.json: {exc}")


@app.get("/artifacts/{name}")
def artifact(name: str) -> FileResponse:
    """Serve a PNG that lives DIRECTLY in the artifacts dir (training/comparison plots).

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


def _conclude(fired: bool, baseline_verdict: str, hyper_verdict: str) -> str:
    """One-line human verdict on what the hypernet-steered pipeline did.

    Priority mirrors the causal chain (same shape as lesson 2, but the actor is
    the GENERATED vector):
      * gate did not fire -> we never touched the output, nothing changed;
      * steering broke coherence -> "caused corruption" (caught before refuse
        vs comply);
      * a non-refusal became a refusal -> "hypernet steering worked";
      * anything else -> "no change".
    """
    if not fired:
        return "gate did not fire"
    if hyper_verdict == "GIBBERISH":
        return "caused corruption"
    if hyper_verdict == "REFUSAL" and baseline_verdict != "REFUSAL":
        return "hypernet steering worked"
    return "no change"


@app.post("/steer")
def steer(req: SteerRequest) -> dict:
    """Run the full HYPERNET-steered pipeline on one prompt, live.

    1. Ask the gate whether to intervene (and log its P(harmful)).
    2. Always generate a baseline (alpha=0, no vector) for comparison.
    3. If the gate fired, generate again WITH the hypernet-generated vector added
       at the steer layer; otherwise the steered output is the untouched baseline.
    4. Judge both generations (REFUSAL / COMPLIANCE / GIBBERISH).
    5. Summarise the outcome in one line.

    When lesson 2's fixed CAA vector is available we also add a third arm (same
    gate decision, the HAND-BUILT vector) so the page can show learned-vs-fixed
    side by side — the concrete claim that the hypernet reproduces CAA.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")

    b = get_bundle()
    from steering_tutorials.hello_world_steering.model_utils import generate

    fired, prob = b.gate.is_harmful(prompt)
    baseline = generate(b.model, b.tok, prompt, alpha=0.0)

    if fired:
        hyper = generate(
            b.model, b.tok, prompt,
            vector=b.v_hyper, layer=C.STEER_LAYER,
            alpha=C.ALPHA_EVAL, operation="relative_add",
        )
    else:
        # Gate stayed quiet: leave the output exactly as the model produced it —
        # the point of CONDITIONAL steering. Benign prompts pass through clean.
        hyper = baseline

    baseline_verdict = b.judge.verdict(prompt, baseline)
    hyper_verdict = baseline_verdict if not fired else b.judge.verdict(prompt, hyper)

    out = {
        "prompt": prompt,
        "gate_fired": fired,
        "prob_harmful": prob,
        "alpha": C.ALPHA_EVAL if fired else 0.0,
        "layer": C.STEER_LAYER,
        "baseline_response": baseline,
        "baseline_verdict": baseline_verdict,
        "hyper_response": hyper,
        "hyper_verdict": hyper_verdict,
        "conclusion": _conclude(fired, baseline_verdict, hyper_verdict),
    }

    # Optional learned-vs-fixed arm: only when lesson 2's vector is on disk.
    if b.fixed_caa is not None and fired:
        fixed = generate(
            b.model, b.tok, prompt,
            vector=b.fixed_caa["v_unit"], layer=C.STEER_LAYER,
            alpha=C.ALPHA_EVAL, operation="relative_add",
        )
        out["fixed_caa_response"] = fixed
        out["fixed_caa_verdict"] = b.judge.verdict(prompt, fixed)

    return out


@app.post("/generate_concept")
def generate_concept(req: ConceptRequest) -> dict:
    """THE lesson-3 highlight: turn a user's exemplar phrases into a live vector.

    The user types a few phrases describing a concept. We:
      1. build a concept embedding (mean last-token activation over the phrases),
      2. run the hypernetwork on it to EMIT a fresh steering vector — no
         contrastive extraction, no retraining: pure amortisation,
      3. steer a fixed demo prompt with that vector and judge the result, so the
         emitted vector is visibly doing something.

    Bounded on purpose: at most a baseline + one steered generation (plus the
    judge's own single pass), so the live call stays snappy.
    """
    exemplars = [e.strip() for e in (req.exemplars or []) if e and e.strip()]
    if not exemplars:
        raise HTTPException(400, "need at least one exemplar phrase")
    if len(exemplars) > 12:
        # Keep the concept-embedding pass cheap; a handful of phrases is plenty.
        exemplars = exemplars[:12]

    b = get_bundle()
    np = b._np
    from steering_tutorials.hello_world_steering.model_utils import generate

    # 1-2. exemplars -> concept embedding -> emitted steering vector.
    concept_emb = b._concept_embedding(b.model, b.tok, exemplars, C.STEER_LAYER)
    v = b._emit_vector(concept_emb)
    vector_norm = float(np.linalg.norm(v))

    # 3. show the vector at work on one neutral demo prompt.
    baseline = generate(b.model, b.tok, _DEMO_PROMPT, alpha=0.0)
    steered = generate(
        b.model, b.tok, _DEMO_PROMPT,
        vector=v, layer=C.STEER_LAYER,
        alpha=C.ALPHA_EVAL, operation="relative_add",
    )
    steered_verdict = b.judge.verdict(_DEMO_PROMPT, steered)

    return {
        "n_exemplars": len(exemplars),
        "vector_norm": vector_norm,
        "layer": C.STEER_LAYER,
        "alpha": C.ALPHA_EVAL,
        "demo_prompt": _DEMO_PROMPT,
        "baseline_response": baseline,
        "steered_response": steered,
        "steered_verdict": steered_verdict,
        "note": (
            "The hypernetwork emitted this steering vector straight from your "
            "phrases — no contrastive extraction, no retraining. That on-the-fly "
            "generation is the amortisation payoff of lesson 3."
        ),
    }


def main() -> None:
    import uvicorn

    # Port 8003 on purpose: lessons 1 and 2 own 8000 and 8001, so all three
    # demos can run side by side without a clash (8002 left as a gap/buffer).
    print("Serving on http://127.0.0.1:8003  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8003, log_level="info")


if __name__ == "__main__":
    main()
