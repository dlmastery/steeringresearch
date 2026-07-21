# Safety Probe — a hello-world activation classifier

> **Reference:** linear classifier probes on intermediate activations (Alain &
> Bengio, arXiv:1610.01644); refusal-direction / activation-steering context
> (ActAdd arXiv:2308.10248, Arditi et al. arXiv:2406.11717). Data: JailbreakBench
> (arXiv:2404.01318), XSTest (arXiv:2308.01263).

> Read a **frozen** language model's internal activations for a prompt, and let a
> tiny neural network decide whether that prompt is **harmful** or **safe**. No
> fine-tuning, no steering, no magic — just a small, honest, end-to-end example
> of *reading a concept out of a model's hidden state*.

This is the "hello world" of interpretability and steering research. If you have
never touched activations, probes, or Gemma before, start here. Every idea is
built from zero, every code snippet below is pulled verbatim from the real files
in this folder, and every number we report has been checked with a leakage
audit, cross-validation, and an out-of-domain stress test.

---

## The key idea in code

The whole READ side is one frozen forward pass into one vector, then a tiny head
that turns that vector into `P(harmful)`. That is the entire method:

```python
# 1. Read the concept out: mean-pool the layer-12 residual over tokens -> one [1152] vector.
vec = h.mean(0) if pooling == "mean" else h[-1]      # from extract_features (model_utils.py)

# 2. A 3-layer MLP head on that frozen activation is all the "learning" there is.
class MLPProbe(nn.Module):
    def __init__(self, in_dim, h1=128, h2=32, dropout=0.3):
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),  # 1152 -> 128
            nn.Linear(h1, h2),     nn.ReLU(), nn.Dropout(dropout),  # 128  -> 32
            nn.Linear(h2, 1),                                       # 32   -> 1 harmful-logit
        )
```

The LLM did the hard representational work; a small, heavily-regularized head is
plenty to read a decision surface off it. Full file-by-file walkthrough below.

---

## Table of contents

1. [What you'll build](#1-what-youll-build)
2. [Concepts from zero](#2-concepts-from-zero)
3. [Architecture & data flow](#3-architecture--data-flow)
4. [The dataset](#4-the-dataset)
5. [Code walkthrough, file by file](#5-code-walkthrough-file-by-file)
6. [Run the training](#6-run-the-training)
7. [Understand the metrics](#7-understand-the-metrics)
8. [Did we cheat? Rigor & honesty](#8-did-we-cheat-rigor--honesty)
9. [Reproducible install](#9-reproducible-install)
10. [Other uncensored models for a 16 GB 4090](#10-other-uncensored-models-for-a-16-gb-4090)
11. [Repository](#11-repository)
12. [Disclaimer](#12-disclaimer)

---

## 1. What you'll build

A complete pipeline that takes any English prompt, runs it once through a frozen
1-billion-parameter Gemma model, grabs a single 1152-dimensional "thought vector"
from a middle layer, and feeds that vector to a small 3-layer classifier (a
"probe") that outputs `P(harmful)` — the probability that the prompt is a harmful
request. You will train it, measure it seven different ways, audit it for
cheating, test whether it generalizes to a dataset it never saw, and serve it in
a live web dashboard.

**Results teaser** (all verified — see [Section 8](#8-did-we-cheat-rigor--honesty)):

| Test | What it measures | Headline |
|---|---|---|
| In-domain single split (n=40) | one held-out slice of JailbreakBench | accuracy **0.950**, ROC-AUC **0.980** |
| **5-fold cross-validation (n=200)** | **the trustworthy number** | **accuracy 0.870 ± 0.026, ROC-AUC 0.941 ± 0.014** |
| Linear-probe reference (logreg) | is the signal linearly decodable? | accuracy 0.860, ROC-AUC 0.944 |
| Leakage audit | is the score real or an artifact? | 0 split overlaps; shuffled labels collapse to 0.425 (chance) |
| Out-of-domain (XSTest, zero-shot, n=300) | does it transfer? | ROC-AUC **0.888**, but recall drops to 0.427 |

The one-line takeaway: **a frozen LLM genuinely encodes "is this harmful?" in its
activations, and a tiny probe can read it — but a probe trained on one dataset
ranks well on another yet needs recalibration to keep its accuracy.**

---

## 2. Concepts from zero

### 2.1 What is Gemma, and what does "frozen" mean?

**Gemma** is a family of open-weight language models from Google. Like GPT-style
models, a Gemma model is a stack of *transformer layers* trained to predict the
next token (word-piece) in a sequence. The specific one we use is a small
1-billion-parameter variant with 26 layers.

We use the model as a **frozen feature extractor**. "Frozen" means we never
change a single weight — we do not fine-tune it, and we do not *steer* it. We
only run prompts through it and *read* the numbers that flow through its
internals. Think of the LLM as an expensive, pre-built sensor: we point it at a
prompt and record what it "sees."

The config file says this in one sentence:

```python
# config.py
# The idea in one sentence:
#     A big language model already "knows" whether a prompt is harmful — that
#     knowledge is written into its internal activations. We freeze the model,
#     read the activation vector for a prompt at ONE middle layer, and train a
#     tiny neural network (a "probe") to map that vector to harmful / safe.
```

### 2.2 The residual stream and activations

Inside a transformer, information flows through what researchers call the
**residual stream**: a running vector, one per token, that each layer reads from
and writes back to. After layer `l` processes a token, the residual stream holds
that layer's current "understanding" of the token in context — a list of numbers
(a vector). For our model that vector is **1152 numbers wide** (the model's
`hidden_size`). That vector is an **activation**.

Why read a **middle** layer? Empirically:

- **Early layers** are busy with surface features — tokens, spelling, syntax.
- **Late layers** are specializing toward the very next-token prediction.
- **Middle layers** hold the most abstract, task-relevant *meaning* — exactly
  where a concept like "is this request harmful?" is most cleanly represented.

So we tap **layer 12 of 26**:

```python
# config.py
# Gemma-3-1B has 26 layers; 12 is a touch past the middle. Clamped at runtime.
LAYER = 12
```

A prompt has many tokens, so we get one 1152-d vector *per token*. To get a
single fixed vector for the whole prompt, we **mean-pool** — average the vectors
across all token positions. Simple, robust, and order-agnostic.

### 2.3 What is a probe?

A **probe** is a small, simple classifier trained *on top of* a frozen model's
activations to test what information those activations contain. If a linear (or
shallow) probe can read "harmful vs. safe" straight out of layer 12, then that
concept is *linearly present* in the representation — the model already computed
it. This is the classic technique introduced by Alain & Bengio, who attached
linear classifiers to a frozen network's layers to see what each layer had
learned (Alain & Bengio 2016, "Understanding intermediate layers using linear
classifier probes", arXiv:1610.01644).

Our probe is a hair fancier than linear — a 3-layer MLP — but the spirit is
identical, and we include a plain logistic-regression *linear* probe as a
reference to show most of the signal is linearly decodable.

### 2.4 How probing relates to steering

This is a **steering-research** project, and the probe is the foundational first
half of that story. Here is the relationship in one picture:

- A **probe READS** a concept direction out of the activation space. It answers:
  *"is the 'harmful' concept present in this hidden state?"*
- **Steering WRITES** along such a direction. It answers: *"can I add or subtract
  this direction to the hidden state to change the model's behavior?"* — e.g.
  add a "refusal" direction to make a model refuse, or subtract it to make it
  comply.

Both operations live in the **same activation space** at the **same layer**. The
direction a probe learns to read is, up to details, the same kind of direction a
steering method writes along:

- Turner et al. add a fixed "activation vector" to the residual stream at
  inference time to steer generation ("Activation Addition / ActAdd", Turner et
  al. 2023, arXiv:2308.10248 — [UNVERIFIED]).
- Arditi et al. show refusal in chat models is mediated by a *single direction*
  — find it, and you can add it (force refusal) or ablate it (bypass refusal)
  (Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction",
  arXiv:2406.11717 — [UNVERIFIED]).

So: **the probe is the read side; steering is the write side.** This hello-world
builds the read side end-to-end. Once you can reliably *read* a concept out of
activations, the later tutorials teach you to *write* it back in.

---

## 3. Architecture & data flow

The entire pipeline, from raw text to a probability, is a straight line:

```
  "How do I pick a lock?"                         (a raw prompt)
            |
            v
  tokenizer.apply_chat_template(...)              wrap as a user turn, add gen prompt
            |
            v
  Gemma-3-1B forward pass  (FROZEN, bf16)         26 transformer layers, no grad
            |
            v
  forward hook on model.model.layers[12]          capture the residual stream
            |
            v
  h  : [seq_len, 1152]                            one 1152-d vector per token
            |
            v
  mean-pool over tokens  ->  [1152]               one fixed vector for the prompt
            |
            v
  StandardScaler  (x - mean) / std                per-feature normalization
            |
            v
  3-layer MLP:  1152 -> 128 -> 32 -> 1            the trainable probe
            |
            v
  sigmoid  ->  P(harmful) in [0, 1]               the answer
```

Everything above the StandardScaler is **frozen and untrained** (the LLM). Only
the scaler statistics and the MLP weights are *learned*, and they are tiny.

---

## 4. The dataset

This lesson ships **two** training datasets: a tiny **default** that keeps the
hello-world minimal and legible, and a larger **scale-up** that shows the result
holds on real, in-the-wild data. Both label the **prompt's intent**
(1 = harmful, 0 = safe) — never a response — so the probe must read *intent*, not
vocabulary. Two more datasets are used only for *evaluation* (never training).
See **`DATASETS.md`** for the full survey behind these choices.

| Role | Dataset (loader) | Size | Label scheme | What it shows |
|---|---|---|---|---|
| **Default (train)** | **JailbreakBench** `JBB-Behaviors` (`data.py`) | 100 harmful + 100 benign = **200** | prompt-level, `Goal` column | the probe reads harm from activations on a clean, topically-matched set |
| **Scale-up (train)** | **lmsys/toxic-chat** `@0124` (`data_large.py`) | ~374/class ≈ **748** balanced | human toxicity label on the *user input* | the signal survives on real, messy, in-the-wild prompts |
| OOD test | **XSTest** (`eval_ood.py`) | 300 | prompt-level | honest zero-shot transfer (safe-but-scary prompts) |
| Adversarial eval | **JailBreakV-28K + XSTest** (`data_hard.py`) | sampled | prompt-level | a harder red-team stress test |

**Default — JailbreakBench** (Chao et al. 2024, arXiv:2404.01318 —
[UNVERIFIED]) ships two matched CSVs, `data/harmful-behaviors.csv` (label **1**)
and `data/benign-behaviors.csv` (label **0**), both using a `Goal` column. The
two sets are **topically matched** — for many harmful goals there is a benign
twin on a similar subject. This matters enormously: a classifier *cannot cheat*
by keying on surface words. If the harmful set were all about "bombs" and the
benign set all about "recipes", a dumb bag-of-words model would ace the task
without any real understanding. Topical matching forces the probe to read
*intent*, not vocabulary.

```python
# data.py
JBB_REPO = "JailbreakBench/JBB-Behaviors"
HARMFUL_CSV = "data/harmful-behaviors.csv"
BENIGN_CSV = "data/benign-behaviors.csv"
PROMPT_COLUMN = "Goal"
```

200 prompts is small — perfect for a fast, legible hello-world, but small enough
that a single train/test split is noisy (which is why we cross-validate in
[Section 8](#8-did-we-cheat-rigor--honesty)).

**Scale-up — Toxic-Chat.** 200 toy prompts is not proof a probe works in the
wild, so `data_large.py` is a drop-in loader (same `(prompts, labels)` contract)
for **lmsys/toxic-chat** `@0124` (Lin et al. 2023, arXiv:2310.17389 —
[UNVERIFIED]): genuine user inputs from a live chatbot demo, each **hand-labeled
by humans** for toxicity on the *prompt itself* — already prompt-level, so no
response→prompt collapse is needed. Toxic is a **~7% natural minority**, so the
loader records that base rate, then rebalances to 1:1 for training — capping the
balanced set at ~374/class (**≈748 prompts**) because the smaller harmful pool is
the limiter. It is built to an elite-data-scientist bar: category-stratified
harmful sampling, group-aware dedup (surface near-duplicates share a `group_id`
so they can't straddle a split), and honest base-rate reporting. `train_large.py`
reaches **test accuracy 0.875 / 5-fold CV 0.95** with the length-confound audit
passing.

---

## 5. Code walkthrough, file by file

### `config.py` — every knob in one place

All hyperparameters live here so you can read the whole design at a glance and
change one thing without hunting through code.

```python
# config.py
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
LAYER = 12            # which residual-stream layer to read
POOLING = "mean"      # mean-pool the activations over prompt tokens
HIDDEN1 = 128
HIDDEN2 = 32
DROPOUT = 0.30        # strong dropout — we have only a few hundred examples
WEIGHT_DECAY = 1e-3   # L2 regularization, same reason
LR = 1e-3
EPOCHS = 200
PATIENCE = 30         # early-stop if val loss hasn't improved in this many epochs
TEST_FRACTION = 0.20
VAL_FRACTION = 0.15   # carved out of the training portion for early stopping
SEED = 0
```

Note we deliberately use an **uncensored / abliterated** Gemma-3-1B. An
abliterated model has had its refusal behavior surgically removed, so it does not
clam up on harmful prompts — which means its activations still represent the
harmful/safe distinction *cleanly* instead of collapsing every harmful prompt
into a single "I refuse" blob. That makes it a better sensor for probing.

### `model_utils.py` — load the frozen LLM and read its activations

Two jobs: `load_model()` and `extract_features()`.

Loading is deliberately in **bf16, not 4-bit** — the model is tiny (~2 GB), and
full precision gives the cleanest activations to probe:

```python
# model_utils.py  (inside load_model)
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = model.to(device)
model.eval()
```

Two **Windows-friendly guards** stop `transformers` from trying to JIT-compile a
Triton CUDA kernel (which has no Windows wheel and would crash):

```python
# Guard 1: if anything still tries to compile, fall back to eager.
import torch._dynamo as _dynamo
_dynamo.config.suppress_errors = True
...
# Guard 2: never select the compiled static KV cache.
gen_cfg.cache_implementation = "dynamic"
```

The heart of the file is the **forward hook** that captures the residual stream.
A forward hook is a callback PyTorch runs every time a module produces output; we
attach one to layer 12 and stash whatever flows out:

```python
# model_utils.py  (inside extract_features)
def hook(_module, _inputs, output):
    h = output[0] if isinstance(output, tuple) else output
    captured["h"] = h.detach()

handle = target.register_forward_hook(hook)
```

Then, for each prompt, we wrap it in the model's **chat template** (so the
activations match how the model actually sees a user turn), run one forward pass,
and **mean-pool** the captured activation over token positions:

```python
# model_utils.py  (the per-prompt loop)
ids = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)
model(ids)
h = captured["h"][0]                              # [seq, hidden]
vec = h.mean(0) if pooling == "mean" else h[-1]   # pool over tokens
feats.append(vec.float().cpu().numpy())
```

The whole thing runs under `@torch.no_grad()` — no gradients, because the LLM is
frozen. One prompt at a time keeps the code trivial (no padding or attention-mask
bookkeeping).

### `data.py` — the harmful-vs-benign training set

Downloads the two JailbreakBench CSVs directly with `hf_hub_download` (this works
behind corporate SSL proxies where the higher-level `datasets.load_dataset`
often fails) and returns `(prompts, labels)`:

```python
# data.py
harmful_path = hf_hub_download(JBB_REPO, HARMFUL_CSV, repo_type="dataset")
benign_path = hf_hub_download(JBB_REPO, BENIGN_CSV, repo_type="dataset")
harmful = pd.read_csv(harmful_path)[PROMPT_COLUMN].dropna().astype(str).tolist()
benign = pd.read_csv(benign_path)[PROMPT_COLUMN].dropna().astype(str).tolist()
prompts = harmful + benign
labels = [1] * len(harmful) + [0] * len(benign)
```

### `probe.py` — the 3-layer MLP and the scaler-in-checkpoint

The probe is a plain 3-linear-layer MLP with ReLU and dropout between layers.
Because the LLM already did the hard representational work, this tiny network
with heavy regularization is plenty:

```python
# probe.py
self.net = nn.Sequential(
    nn.Linear(in_dim, h1),   # 1152 -> 128
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(h1, h2),       # 128 -> 32
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(h2, 1),        # 32 -> 1 logit
)
```

A subtle but important detail: the **StandardScaler lives inside the checkpoint**.
The scaler stores each feature's training mean and standard deviation, so at
inference time we reproduce training exactly with no external state to lose:

```python
# probe.py — Scaler is saved alongside the weights
torch.save({
    "state_dict": probe.state_dict(),
    "in_dim": probe.in_dim, "h1": probe.h1, "h2": probe.h2,
    "scaler_mean": scaler.mean, "scaler_std": scaler.std,
    "meta": meta,  # model_id, layer, pooling, threshold, ...
}, path)
```

`predict_proba()` applies the scaler, runs the MLP, and returns `sigmoid(logit)` —
`P(harmful)` in `[0, 1]`.

### `train_probe.py` — the whole lifecycle in seven numbered steps

This is the file you run first. Its `main()` is a clean seven-step story:

1. **Load the dataset** — `load_safety_dataset()`.
2. **Extract features (cached)** — run the LLM once, mean-pool layer 12, and cache
   the result to `artifacts/features.npz` so every later run is instant:
   ```python
   X = extract_features(model, tok, prompts, layer, pooling=C.POOLING)
   np.savez(C.FEATURES_CACHE, X=X, y=y, layer=layer, model_id=C.MODEL_ID, ...)
   del model  # free VRAM — we only need X from here on
   ```
3. **Split** — stratified train/val/test with a fixed seed (balanced across both
   classes).
4. **Standardize — fit the scaler on train only** (this is the single most
   important anti-leakage rule; the val and test sets are transformed with the
   *train* statistics, never their own):
   ```python
   scaler = Scaler.fit(X[tr])
   Xtr, Xva, Xte = scaler.transform(X[tr]), scaler.transform(X[va]), scaler.transform(X[te])
   ```
5. **Train with early stopping** — full-batch Adam (lr=1e-3, weight decay=1e-3),
   BCE loss, and we keep the weights from the epoch with the *best validation
   loss*, stopping after 30 epochs of no improvement:
   ```python
   if vloss < best_val - 1e-4:
       best_val, best_state, bad = vloss, {...clone...}, 0
   else:
       bad += 1
       if bad >= C.PATIENCE:
           break
   ```
6. **Evaluate** — the full 12-metric suite on the held-out test set (see
   [Section 7](#7-understand-the-metrics)).
7. **Persist** — save the probe (+ scaler), write `metrics.json`, and render five
   PNGs (ROC, precision-recall, calibration, training history, confusion matrix).

### `infer.py` — classify one prompt from the CLI

Loads the frozen LLM + the saved probe once, then classifies any prompt:

```python
# infer.py
def classify(self, prompt: str) -> dict:
    feats = extract_features(self.model, self.tok, [prompt], self.layer,
                             pooling=self.meta.get("pooling", "mean"), log_every=0)
    prob = float(predict_proba(self.probe, self.scaler, feats, device=self.device)[0])
    return {"prompt": prompt, "prob_harmful": prob,
            "label": "harmful" if prob >= self.threshold else "safe", ...}
```

### `app.py` + `static/index.html` — the live dashboard

A small FastAPI server exposes four endpoints — the dashboard, the metrics JSON,
the plot PNGs, and a live `/predict`:

```python
# app.py
@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "empty prompt")
    return get_classifier().classify(prompt)
```

The classifier is loaded **lazily on the first `/predict`** (not at import), so
startup is instant. `static/index.html` is a single self-contained page (no CDN,
no framework): a live-classify box with a `P(harmful)` gauge, the held-out metric
tiles, the ROC / confusion / training-history plots, and a sortable table of
every test-set prediction — all fetched from `/metrics` and `/artifacts/<png>`.

### The three rigor scripts

These are what separate an honest demo from a misleading one. None of them
retrain the LLM; the two CPU scripts reuse the cached `artifacts/features.npz`.

- **`audit_leakage.py`** (CPU) — rules out the two boring explanations for a high
  score. It reproduces the *exact* trainer split and runs four checks:
  (1) split disjointness + duplicate detection; (2) a **label-shuffle control** —
  if you scramble the training labels, an honest model must collapse to ~chance;
  (3) **trivial text baselines** (length-only and TF-IDF bag-of-words) to see how
  much of the score is a dataset surface artifact vs. real activation signal;
  (4) a restatement that the scaler is fit on train only.

- **`cross_validate.py`** (CPU) — puts a confidence interval under the headline.
  It runs **5-fold stratified cross-validation** so every one of the 200 examples
  is held out exactly once, trains the *same* MLP recipe per fold, and reports
  mean ± 95% CI for all 12 metrics. It also runs a plain `LogisticRegression`
  linear-probe reference to check the signal is linearly decodable (i.e. the MLP
  is not overfitting).

- **`eval_ood.py`** (GPU) — the honest generalization test. It takes the probe
  *exactly as trained on JailbreakBench* — same weights, same scaler, same layer,
  **zero retraining** — and evaluates it on **XSTest** (Röttger et al. 2024,
  arXiv:2308.01263 — [UNVERIFIED]), a deliberately adversarial set of prompts that
  *sound* dangerous but are safe ("How do I kill a Python process?") plus their
  genuinely-unsafe twins. This is a much harder distribution, so it is an honest
  stress test of what the probe actually learned.

---

## 6. Run the training

From the **repo root** (`steeringresearch/`), run:

```bash
python -m steering_tutorials.hello_world.train_probe
```

On the first run you will see (to stderr) the dataset download, the model load,
feature extraction progress, the split sizes, and early-stop, then the results
block on stdout. Roughly:

```
[data] 100 harmful + 100 benign = 200 prompts
[model] loaded DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated on cuda (26 layers, hidden=1152)
[features] 25/200
...
[split] train=130 val=30 test=40
[train] early stop at epoch NN (best val loss 0.xxxx)

=== TEST-SET RESULTS =========================================
  accuracy            0.950
  balanced_accuracy   0.950
  precision           0.950
  recall              0.950
  specificity         0.950
  f1                  0.950
  mcc                 0.900
  cohen_kappa         0.900
  roc_auc             0.980
  pr_auc              0.980
  log_loss            0.172
  brier_score_loss    0.051
  confusion           [[19, 1], [1, 19]]  [[TN,FP],[FN,TP]]
  saved               probe.pt, metrics.json
==============================================================
```

The first run extracts and **caches** features to `artifacts/features.npz`, so a
second `train_probe` run (or the two CPU rigor scripts) is near-instant and does
not touch the GPU.

---

## 7. Understand the metrics

We report a whole suite, not one number, because no single score tells the truth
about a classifier. In beginner terms:

| Metric | What it answers | Why it's here |
|---|---|---|
| **Accuracy** | fraction of all predictions that are correct | intuitive, but misleading on imbalanced data |
| **Precision** | of prompts flagged harmful, how many *are* harmful | high precision = few false alarms |
| **Recall** (sensitivity) | of truly harmful prompts, how many we caught | high recall = few misses (safety-critical) |
| **Specificity** | of truly safe prompts, how many we let through | the "recall" of the safe class |
| **F1** | harmonic mean of precision & recall | one number balancing the two |
| **Balanced accuracy** | average of recall and specificity | fair when classes are imbalanced |
| **ROC-AUC** | can the probe *rank* harmful above safe, at any threshold? | threshold-independent; **1.0 = perfect ranking, 0.5 = coin flip** |
| **PR-AUC** | precision-recall area | more informative than ROC when positives are rare |
| **Log-loss** | penalizes confident *wrong* probabilities | rewards well-calibrated confidence |
| **Brier score** | mean squared error of the probability | calibration: is `P=0.9` right ~90% of the time? |
| **MCC** (Matthews) | correlation of prediction & truth, −1…+1 | robust single score even on skewed data |
| **Cohen's kappa** | agreement above random chance | like accuracy, but discounts luck |

Two families to keep straight:
- **Threshold metrics** (accuracy, precision, recall, F1, specificity, MCC,
  kappa) depend on the 0.5 cutoff — *where* you draw the harmful/safe line.
- **Ranking / calibration metrics** (ROC-AUC, PR-AUC, log-loss, Brier) use the
  raw probabilities and are independent of that cutoff. This distinction is the
  key to the out-of-domain result below.

---

## 8. Did we cheat? Rigor & honesty

A 0.95 accuracy on 40 test prompts is easy to *report* and easy to *fool
yourself* with. Here is the full, honest picture — every number below is verified.

### 8.1 In-domain single split (n_test = 40)

The one held-out slice from `train_probe.py`:

| accuracy | precision | recall | f1 | specificity | balanced_acc | mcc | roc_auc | pr_auc | log_loss | brier |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.950 | 0.950 | 0.950 | 0.950 | 0.950 | 0.950 | 0.900 | 0.980 | 0.980 | 0.172 | 0.051 |

Confusion matrix `[[TN, FP], [FN, TP]]` = **`[[19, 1], [1, 19]]`** — one false
alarm, one miss, out of 40.

### 8.2 The trustworthy headline: 5-fold cross-validation (n = 200)

A single 40-prompt slice is a noisy draw. Cross-validation rotates the held-out
slice through the whole dataset so every example is tested exactly once:

| metric | 5-fold mean ± 95% CI |
|---|---|
| **accuracy** | **0.870 ± 0.026** |
| **roc_auc** | **0.941 ± 0.014** |
| **f1** | **0.875 ± 0.024** |

Linear-probe reference (plain logistic regression, 5-fold): **accuracy 0.860,
ROC-AUC 0.944**. The MLP barely beats a linear model — which is exactly what we
*want* to see: it means the harmful/safe signal is **largely linearly decodable**
from layer 12, and the MLP is **not overfitting** by memorizing quirks.

**Honesty note:** the single-split 0.950 is **optimistic** — it sits *above* the
cross-validated 95% CI upper bound. The number you should trust and quote is the
**CV mean ± CI (accuracy 0.870 ± 0.026)**, not the lucky single-slice 0.95.

### 8.3 Leakage audit — is the score real, or an artifact?

| Check | Result | Reading |
|---|---|---|
| Cross-split prompt overlaps | **0** | no prompt leaks train → test |
| Shuffled-label control | **0.425** (≈ chance) | scramble the labels and the model collapses — no hidden leakage path |
| Length-only baseline | **0.650** | prompt length alone is weak |
| TF-IDF bag-of-words baseline | **0.575** | surface vocabulary alone is weak |

The trivial text baselines (0.575–0.650) trail far behind the probe (0.87 CV /
0.95 single-split). Combined with the shuffle collapsing to chance, this is strong
evidence the score is **real activation-level signal**, not a text artifact or a
leak.

### 8.4 Out-of-domain generalization — XSTest, zero-shot (n = 300)

We freeze the probe and point it at XSTest, a harder, adversarial set it never saw:

| metric | value |
|---|---|
| accuracy | 0.690 |
| precision | 0.901 |
| recall | 0.427 |
| roc_auc | **0.888** |
| pr_auc | 0.881 |

Confusion `[[TN, FP], [FN, TP]]` = **`[[143, 7], [86, 64]]`**.

The honest interpretation: **the ranking transfers, the calibration does not.**
ROC-AUC of 0.888 means the probe still *ranks* harmful prompts above safe ones
well on a distribution it never trained on — the "harm direction" it found is
partly general. But at the fixed 0.5 threshold, **recall collapses to 0.427**: it
misses more than half the harmful prompts because the decision boundary that was
calibrated on JailbreakBench sits in the wrong place for XSTest. In one line:
**generalizes in ranking, degrades in calibration.** The fix (not done here) would
be to re-tune the threshold on a small OOD calibration slice.

### 8.5 The small-n caveat

The single-split test set is only **40 prompts**. At that size, one flipped
prediction moves accuracy by 2.5 points, so *always* report a confidence interval
(as the cross-validation does) rather than trusting a single-slice point estimate.

---

## Results — measured vs. the claim

The lesson's central claim is a **linear-probing** one: a concept like "is this
prompt harmful?" is **largely linearly decodable** from a mid-layer residual
stream (the lineage of Alain & Bengio 2016, "Understanding intermediate layers
using linear classifier probes", arXiv:1610.01644). Here is what
the artifacts actually show, claim by claim.

| Claim | What we measured | Verdict |
|---|---|---|
| Harm is decodable from layer-12 activations | 5-fold CV **accuracy 0.870 ± 0.026**, **ROC-AUC 0.941 ± 0.014** (n=200); single split 0.950 / 0.980 (n=40) | **Reproduced** |
| The signal is *linearly* decodable (not an MLP artifact) | logreg linear probe **accuracy 0.860, ROC-AUC 0.944** — ~matches the MLP | **Reproduced** |
| The score is real, not a text/leak artifact | shuffled-label **0.425** (chance); length-only **0.650**; TF-IDF **0.575** — all far below 0.87 | **Clean** |
| The direction generalizes out-of-domain | XSTest zero-shot **ROC-AUC 0.888** (n=300), but **recall 0.427** at the 0.5 threshold | **Ranks, miscalibrated** |
| It holds on real in-the-wild data | Toxic-Chat scale-up **test accuracy 0.875, ROC-AUC 0.965** (n=748), 5-fold CV **0.95** | **Reproduced** |

**Honest read.** This is **screening-tier** evidence (n=200 for the headline; a
single seed, not the n≥7 the project reserves the word "winner" for), and the
lucky single-split 0.950 sits *above* the cross-validated 95% CI — quote the
**CV mean 0.870 ± 0.026**, not the point estimate. The load-bearing result is
that the plain logistic-regression linear probe (0.860 / 0.944) essentially ties
the MLP: the harm concept really is largely *linear* at layer 12, exactly what
the Alain-Bengio lineage predicts, and the MLP is not overfitting quirks. The
one honest failure is calibration — the probe **ranks** harm well on XSTest
(AUC 0.888) but its fixed 0.5 boundary misses more than half the harmful prompts
there (recall 0.427); the direction transfers, the threshold does not.

---

## 9. Reproducible install

### Step 1 — create a virtual environment and install deps

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r steering_tutorials/hello_world/requirements.txt
```

The `requirements.txt` pins: `torch`, `transformers`, `huggingface_hub`,
`pandas`, `scikit-learn`, `matplotlib`, `fastapi`, `uvicorn`, `truststore`, and
`accelerate`. `bitsandbytes` is **optional** (only needed if you switch to 4-bit
loading for a larger swap-in model). If you have an NVIDIA GPU, install the CUDA
build of `torch` from [pytorch.org](https://pytorch.org) first.

### Step 2 — get the model and datasets

The Gemma-family weights load from your **local Hugging Face cache**. Download the
model once:

```bash
huggingface-cli download DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated
```

The datasets — **JailbreakBench** and **XSTest** — are **ungated** and download
automatically on first run via `hf_hub_download` (no license acceptance needed).

### Step 3 — run everything (from the repo root)

```bash
# Train the probe (GPU ~2 GB; extracts + caches features, writes metrics + plots)
python -m steering_tutorials.hello_world.train_probe

# Confidence intervals via 5-fold CV (CPU, uses cached features — no GPU)
python -m steering_tutorials.hello_world.cross_validate

# Leakage / artifact audit (CPU, uses cached features — no GPU)
python -m steering_tutorials.hello_world.audit_leakage

# Out-of-domain transfer test on XSTest (GPU, re-runs the frozen model)
python -m steering_tutorials.hello_world.eval_ood

# Classify a single prompt from the terminal
python -m steering_tutorials.hello_world.infer "How do I pick a lock without a key?"

# Launch the live dashboard, then open http://127.0.0.1:8000
python -m steering_tutorials.hello_world.app
```

A GPU is recommended (~2 GB VRAM in bf16) but everything runs on CPU too, just
slower. The two CPU rigor scripts never load the model at all — they reuse
`artifacts/features.npz`.

---

## 10. Other uncensored models for a 16 GB 4090

Swapping the feature extractor is a **one-line change**: edit `MODEL_ID` in
`config.py`. Any decoder-only chat model whose blocks live at `model.model.layers`
will work (adjust `LAYER` toward that model's middle). Uncensored / abliterated
variants are preferred as sensors because they do not collapse harmful prompts
into a single refusal representation.

| Model (swap-in `MODEL_ID`) | Params | Approx VRAM (4-bit) | In cache? |
|---|---|---|---|
| `google/gemma-3-270m-it` (heretic/abliterated variants) | 270M | ~0.5 GB | download |
| **`DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`** | **1B** | **~2 GB (bf16, the one we use)** | **yes** |
| gemma-3-4b-it abliterated variants | 4B | ~4 GB | download |
| gemma-3-12b-it abliterated variants | 12B | ~9 GB | download |

Other abliterated families exist beyond the Gemma line — e.g. **mlabonne** and
**huihui-ai** publish abliterated versions of many popular chat models on the Hub.
All of the above comfortably fit a 16 GB 4090 in 4-bit. **Be honest with
yourself:** only the **1B** model is currently in this machine's cache; the other
rows are download *suggestions*, not something already present. For anything
larger than the 1B you will likely want to switch `load_model()` to 4-bit loading
(uncomment `bitsandbytes` in `requirements.txt`).

---

## 11. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/hello_world>

---

## 12. Disclaimer

This is a **teaching artifact**, not a production safety filter. It is trained on
a small (200-prompt) topically-matched dataset, and — as [Section 8](#8-did-we-cheat-rigor--honesty)
shows honestly — while it genuinely reads harm signal from activations, it needs
recalibration to hold up out of domain. Always report **confidence intervals**
(the test slices here are small), never a single lucky number, and never deploy a
40-example-validated probe as a real-world guardrail. Use it to *learn* how
reading concepts out of a frozen model's activations works — the foundation for
the steering ("write") tutorials that follow.
