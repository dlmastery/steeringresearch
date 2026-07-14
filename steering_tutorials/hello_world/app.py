"""app.py — the standalone demo webapp (FastAPI).

Run:  python -m steering_tutorials.hello_world.app       (then open http://127.0.0.1:8000)

Endpoints:
  GET  /                 -> the dashboard (static/index.html)
  GET  /metrics          -> the training metrics.json
  GET  /artifacts/<png>  -> the ROC / history / confusion plots
  POST /predict {prompt} -> live classification via the frozen LLM + probe

The model + probe load once at startup, so classification is fast after that.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config as C

app = FastAPI(title="Safety Probe — hello world")

# Lazily-initialised classifier (loads the LLM on first /predict, not at import).
_clf = None


def get_classifier():
    global _clf
    if _clf is None:
        from .infer import Classifier
        _clf = Classifier()
    return _clf


class PredictRequest(BaseModel):
    prompt: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(C.STATIC / "index.html")


@app.get("/metrics")
def metrics() -> JSONResponse:
    if not C.METRICS_PATH.exists():
        raise HTTPException(404, "No metrics yet — run: python -m steering_tutorials.hello_world.train_probe")
    return JSONResponse(json.loads(C.METRICS_PATH.read_text()))


@app.get("/artifacts/{name}")
def artifact(name: str) -> FileResponse:
    # only serve the known PNGs (no path traversal)
    allowed = {"roc_curve.png", "training_history.png", "confusion_matrix.png"}
    if name not in allowed:
        raise HTTPException(404, "unknown artifact")
    path = C.ARTIFACTS / name
    if not path.exists():
        raise HTTPException(404, "artifact not generated yet")
    return FileResponse(path)


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")
    return get_classifier().classify(prompt)


def main() -> None:
    import uvicorn

    print("Serving on http://127.0.0.1:8000  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
