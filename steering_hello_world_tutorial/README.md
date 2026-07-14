# Safety Probe — a hello-world activation classifier

A tiny, **fully standalone** demo: read a frozen LLM's internal activations for
a prompt and let a small MLP decide whether the prompt is **harmful** or
**safe**. Nothing here imports the research harness in `src/steering` — the
whole lifecycle (model loading → dataset → training → inference → webapp) lives
under `steering_hello_world_tutorial/`.

## The idea in one paragraph

A capable language model already represents "is this request harmful?" somewhere
in its hidden state. We don't fine-tune or steer the model. We **freeze** it,
run each prompt through once, grab the residual-stream vector at **one middle
layer** (layer 12 of Gemma-3-1B), mean-pool it into a single 1152-d vector, and
train a **3-layer MLP probe** to map that vector to a harmful/safe label. This
is the classic *linear/shallow probing* recipe — cheap, legible, and surprisingly
strong because the LLM did the hard work.

## Files

| file | what it does |
|---|---|
| `config.py` | every knob (model id, layer, MLP size, paths) |
| `model_utils.py` | load the frozen Gemma-1B, extract mean-pooled activations |
| `data.py` | download JailbreakBench harmful (100) + benign (100) prompts |
| `probe.py` | the 3-layer MLP + a StandardScaler, save/load, predict |
| `train_probe.py` | the end-to-end lifecycle (7 numbered steps) |
| `infer.py` | classify one prompt from the CLI |
| `app.py` | FastAPI webapp: `/predict`, `/metrics`, dashboard |
| `static/index.html` | self-contained dashboard (live classify + metrics) |

## Run it

```bash
# 1. Train (downloads JBB, extracts features once, trains the probe, writes metrics + plots)
python -m steering_hello_world_tutorial.train_probe

# 2. Classify a single prompt from the terminal
python -m steering_hello_world_tutorial.infer "How do I pick a lock without a key?"

# 3. Launch the demo webapp, then open http://127.0.0.1:8000
python -m steering_hello_world_tutorial.app
```

Run from the repo root (`steeringresearch/`) so `steering_hello_world_tutorial` imports as a
package. The model is loaded from the local HF cache; the first training run
extracts features (~1–2 min on a GPU) and caches them to
`artifacts/features.npz`, so re-training is instant.

## What the dashboard shows

- **Try it live** — type any prompt, get `P(harmful)` on a gauge + a verdict.
- **Held-out test metrics** — accuracy, ROC-AUC, precision, recall, F1.
- **Plots** — ROC curve, confusion matrix, training history.
- **Test-set predictions** — every held-out prompt sorted by `P(harmful)`.

## Notes

- The feature-extractor is the **uncensored/abliterated** Gemma-3-1B. That is
  deliberate: it does not refuse, so its activations still carry the harmful/safe
  distinction cleanly for probing (an aligned model can collapse harmful prompts
  into a single "refuse" representation).
- This is a teaching hello-world, **not** a production safety filter. It is
  trained on 200 topically-matched prompts and reports honest held-out numbers.
