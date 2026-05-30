---
name: steering-intervention-lib
description: >
  Use when you need to apply, verify, or extend residual-stream activation
  steering on Gemma-3-1B-it or Gemma-2-2B-it via HuggingFace forward hooks
  on a single RTX 4090 (16 GB). Covers the four primitive operations
  (add / rotate / project_out / clamp), correct hook registration and state
  restoration, the SteeringContext manager, special-token exclusion, injection
  layer selection by Fisher ratio, and the Rung-0 plumbing checks that must
  be green before any compute launches.
---

# Skill — steering-intervention-lib

## When to use

Use this skill whenever you are:

- Writing or reviewing `src/steering/hooks.py` or any module that applies
  activations edits to a Gemma model during inference.
- Diagnosing a plumbing bug before Rung-1 SMOKE (activation not changing,
  logits identical before/after, state leak between prompts).
- Adding a new steering operation variant (e.g., geodesic rotate, clamp-to-
  manifold) to the intervention library.
- Running the Rung-0 UNIT tests to confirm basic plumbing before any sweep.
- Composing multiple interventions (multi-layer, multi-vector stacks) and
  need to reason about hook ordering and norm budget.

This skill is NOT for extracting steering vectors (see
`../../skills/steering-vector-extraction/SKILL.md`) or evaluating behavior
change (see `../../skills/steering-eval-bundle/SKILL.md`).

---

## 0. The object being touched: the residual stream

A Gemma decoder applies every block as an additive write to a shared vector h:

```
h_{l+1} = h_l + Attn_l(h_l) + MLP_l(h_l)
```

Steering = one additional additive (or rotational) term inserted at a chosen
layer l, for a chosen subset of token positions:

```
h_l  <-  h_l  +  delta(h_l, v, alpha, op)
```

Because the operation is local and additive, it is fully reversible:
restoring h_l to its pre-hook value undoes the intervention exactly.
This reversibility is the foundation of the SteeringContext contract.

The residual stream for Gemma-2-2B-it has dimension d = 2304;
Gemma-3-1B-it has d = 1152. Both are accessed at module path
`model.layers[l]` in the HuggingFace implementation.

---

## 1. Why HF hooks, not vLLM

The 4090 laptop has 16 GB VRAM. vLLM's continuous-batching engine does not
expose per-layer residual tensors without forking the codebase; it also
reserves significant overhead for the KV-cache pool. HuggingFace
`register_forward_hook` gives direct access to `(module, input, output)`
tuples at every `model.layers[l]`, which is exactly what we need for
single-token and full-sequence interventions. 4-bit BnB quantization
(`load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16`) keeps the model
at ~1-2 GB (3-1B) or ~2-3 GB (2-2B), leaving room for activations, the
steering vectors, and the evaluation batch.

---

## 2. The four primitive operations

All operations are implemented in `src/steering/hooks.py` as functions that
take `(h, v, alpha)` and return `h_modified`. The caller (SteeringContext) is
responsible for token-masking and state restoration.

### 2.1 ADD (the CAA/ActAdd baseline)

```python
h_modified = h + alpha * v          # v is unit-normed at extraction time
```

Moves h along the direction v by alpha units. Changes the norm:
`||h_modified|| ≈ ||h|| + alpha * cos(h, v)`. Coherence risk rises with alpha.
Use for all baseline comparisons.

### 2.2 ROTATE (norm-preserving, spherical)

```python
# v_perp: component of v orthogonal to h, normalized
h_unit = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
v_perp = v - (v * h_unit).sum(dim=-1, keepdim=True) * h_unit
v_perp = v_perp / v_perp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
theta = alpha                        # alpha = rotation angle in radians
h_modified = h.norm(...) * (
    torch.cos(theta) * h_unit + torch.sin(theta) * v_perp
)
```

Keeps `||h_modified|| == ||h||` exactly; slides h along the great circle
toward v. Preferred over ADD for small models (Gemma-3-1B) where norm
inflation is a primary incoherence driver. See corpus/steering-missed-
dimensions… axis 9 (spherical metric) and hypothesis N16.

### 2.3 PROJECT_OUT (subspace erasure, guard layer A)

```python
# P_S = projection matrix onto safety subspace S (pre-computed)
# Used to sanitize ANY v before applying it (guard A).
v_safe = v - P_S @ v
# Or, applied directly to h to ablate a concept:
h_modified = h - (h @ v) / (v @ v) * v
```

Used in two roles: (a) sanitizing a steering vector before ADD/ROTATE
(guard layer A from `../../skills/steering-rogue-scalpel-guard/SKILL.md`),
and (b) ablating a direction from h to study its causal role (refusal
ablation experiments).

### 2.4 CLAMP (manifold norm-budget enforcement, guard layer B)

```python
mu_l = activation_stats[layer_idx]['mean_norm']   # pre-cached
beta = 0.65                                        # default; sweep in [0.5, 0.75]
delta = alpha * v_safe
if delta.norm() > beta * mu_l:
    delta = delta * (beta * mu_l / delta.norm())
h_modified = h + delta
```

The norm cap ensures the steered state stays within the natural activation
shell at layer l. `activation_stats` is populated by
`src/steering/extract.py` during the contrast-pair caching pass. Parameter
`beta` is a swept hyperparameter; start at 0.65. See hypothesis N17 and
corpus/steering-first-principles-v2-with-PSR-and-rogue-scalpel.md §2.4
guard layer B.

---

## 3. Special-token exclusion

Always exclude the following token positions from the intervention:

- BOS token (position 0 for Gemma models).
- EOS / padding tokens (identified via `attention_mask == 0`).
- For instruction-tuned models: the turn-delimiter tokens (`<start_of_turn>`,
  `<end_of_turn>`); Gemma uses token IDs 106 and 107.

Implementation pattern in the hook:

```python
def _make_hook(v, alpha, op, token_mask):
    def hook_fn(module, input, output):
        h = output[0]                      # shape: (batch, seq, d)
        # token_mask: (batch, seq) bool, True = apply intervention
        active = token_mask.unsqueeze(-1)  # broadcast to (batch, seq, 1)
        delta = compute_delta(h, v, alpha, op)
        h_new = torch.where(active, h + delta, h)
        return (h_new,) + output[1:]
    return hook_fn
```

The `token_mask` excludes BOS, EOS, and pad positions. For generation tasks,
also exclude the system-prompt prefix if it contains no behavior-relevant
tokens (optional; adds robustness against position-dependent artifacts).

---

## 4. The SteeringContext manager

All interventions must be applied via the `SteeringContext` context manager
in `src/steering/hooks.py`. It guarantees:

1. **Exact state restoration.** Hooks are registered on entry and removed on
   exit; the model's `_forward_hooks` dict is checked to be empty before and
   after. If an exception fires inside the context, the `__exit__` method
   still removes all hooks.
2. **No persistent state mutation.** The model's weights are never modified;
   only the activation tensors flowing through the registered hooks are edited.
3. **Composable stacking.** Multiple SteeringContexts may be nested (one per
   layer or per vector); hooks fire in registration order, which is layer order
   because `model.layers` is an ordered `nn.ModuleList`.

```python
class SteeringContext:
    def __init__(self, model, interventions):
        # interventions: list of (layer_idx, v, alpha, op, token_mask)
        self.model = model
        self.interventions = interventions
        self.handles = []

    def __enter__(self):
        for layer_idx, v, alpha, op, mask in self.interventions:
            module = self.model.model.layers[layer_idx]
            h = module.register_forward_hook(
                _make_hook(v, alpha, op, mask)
            )
            self.handles.append(h)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        # Verification: no residual hooks on any layer
        for layer in self.model.model.layers:
            assert len(layer._forward_hooks) == 0
```

The Rung-0 UNIT test verifies that a round-trip (apply + remove) leaves
model outputs byte-identical to the no-hook baseline.

---

## 5. Layer selection (where to inject)

The injection layer is NOT a fixed hyperparameter; it is determined
empirically per behavior vector via the Fisher ratio criterion
(Experiment E2). The procedure lives in `src/steering/extract.py` and is
documented in `../../skills/steering-vector-extraction/SKILL.md`.

Practical constraints for Gemma-3-1B (18 layers) and Gemma-2-2B (26 layers):

- **Avoid layers 0–2 (embedding region).** Representations are not yet
  semantically rich; interventions here diffuse quickly.
- **Avoid the fragile mid-layer band for harmful-prompt experiments.** Per
  Rogue Scalpel (arXiv:2509.22067 F3), refusal-formation circuits are most
  fragile in early-middle layers. Guard layer C forbids injection here for
  any experiment where safety is at risk. The exact fragile band is
  determined empirically (see `../../skills/steering-rogue-scalpel-guard/
  SKILL.md`).
- **Prefer the layer with max Fisher ratio** (class separability of the
  contrast-pair activations) in the upper-middle to late band. This is the
  default. Override only with mechanistic justification.
- For Gemma-3-1B: typical best layers are in [8, 14] (subject to E2
  measurement); for Gemma-2-2B: [12, 20].

For multi-layer stacks (hypothesis N19, trajectory interventions), distribute
the intervention budget across 2-3 layers with equal fractional alpha; the
norm budget (N5) is shared across all injection points.

---

## 6. Activation caching discipline

Steering vectors and activation statistics are cached once per model and
reused across the entire experiment ladder. Never recompute them mid-run.

```
src/steering/extract.py  ->  cache/activations/<model_id>/<dataset_id>.pt
src/steering/hooks.py    ->  reads cache/activations/...
```

The cache stores, per layer:
- `mean_positive`: mean activation of positive-class examples (shape: d)
- `mean_negative`: mean activation of negative-class examples (shape: d)
- `mean_norm`: scalar, mean of `||h||` over the full dataset (for clamp B)
- `pca_components`: top-K PCA components of the pooled positive activations

Cache keys are (model_id, dataset_id, layer_idx, token_aggregation_mode).
If the cache file exists and its SHA-256 matches the stored checksum, skip
recomputation. This discipline makes Rung-0 through Rung-2 fast because the
expensive extraction pass runs at most once per model/dataset pair.

---

## 7. The Rung-0 plumbing checks (UNIT gate)

The Rung-0 gate must be green before ANY compute launches. All checks live
in `src/steering/runner.py` and are called by the runner on every experiment.

**Check 0-A: Vector changes logits.**
Apply the steering vector at alpha=1.0 to a single fixed prompt; assert that
the model logits differ from the no-intervention baseline by at least 1e-4
L1 distance. Failure = hook is not firing or v is the zero vector.

**Check 0-B: State restores exactly.**
After the context manager exits, run the same prompt again without a hook;
assert outputs are byte-identical to the pre-intervention run. Failure =
state leak (hook not removed, or module state mutated).

**Check 0-C: Special tokens excluded.**
Verify that the hook does not modify BOS-position activations by checking
`h[0] == h_baseline[0]` for position 0 after intervention.

**Check 0-D: Norm-budget clamp fires correctly.**
Set alpha extremely high (100.0); assert that the clamped delta norm equals
`beta * mu_l`, not the unclamped value. Failure = clamp logic is bypassed.

**Check 0-E: Multi-hook ordering is layer-order.**
Register two hooks at layers l1 < l2; verify the l1 hook fires before l2 by
inspecting a flag set in a closure. This confirms composition is correct for
stacked interventions.

All five checks must be green (no assertion errors, no exceptions) before the
runner proceeds to Rung-1 SMOKE. A failure at 0-A or 0-B is a BLOCKER.

---

## 8. The 12-axis classification of this skill

This skill implements axes 1 (WHERE = residual stream), 4 (HOW = add /
rotate / project_out / clamp), and 6 (WHICH TOKENS = span with BOS/EOS
exclusion) of the 12-axis framework in
`corpus/steering-missed-dimensions-and-highdim-algebra.md`.

The SteeringContext is the mechanical substrate for all seven original axes
and for the geometry-aware meta-axes 8 (path = chord vs geodesic via
ROTATE), 9 (metric = Euclidean vs spherical), and 11 (dynamics = one-shot
vs multi-layer trajectory via nested context managers).

---

## Hard rules

1. NEVER modify model weights inside a hook. Hooks edit activations in-flight
   only; they are removed by `__exit__` unconditionally.
2. ALWAYS verify 0-A and 0-B pass before running any experiment.
3. NEVER apply a steering vector without first running it through the
   PROJECT_OUT sanitizer when guard layer A is active (stacking/guard runs).
4. NEVER use vLLM for activation steering on this project (16 GB constraint).
5. ALWAYS exclude BOS, EOS, and pad tokens from the intervention mask.
6. ALWAYS use 4-bit BnB quantization; do not load in full precision on 16 GB.
7. Cache activations once; do not recompute mid-experiment.
8. The SteeringContext `__exit__` must run even on exception; use try/finally
   or the context manager protocol, never bare hook registration.

---

## Anti-patterns

| Anti-pattern | Consequence | Do instead |
|---|---|---|
| Registering hooks globally (not in context) | State leaks between experiments; results are wrong | Use SteeringContext exclusively |
| Applying intervention to BOS token | Destabilizes embedding propagation; perplexity spikes | Token-mask excludes position 0 |
| Loading model in fp32 on 16 GB | OOM before any experiment runs | Always 4-bit BnB |
| Forgetting to exclude pad tokens in batch inference | Intervention fires on padding; batch results are corrupted | attention_mask drives token_mask |
| Using a single global alpha without clamp | Off-manifold displacement at high alpha; safety leak | CLAMP op enforces norm budget |
| Re-extracting vectors inside a sweep loop | 10× slowdown; reproducibility risk | Cache once, reuse |

---

## Cross-references

- Vector extraction (DiffMean, PCA, layer selection):
  `../../skills/steering-vector-extraction/SKILL.md`
- Five-axis evaluation including geometry probes:
  `../../skills/steering-eval-bundle/SKILL.md`
- Rogue Scalpel guard (project_out safety subspace, dual-forward):
  `../../skills/steering-rogue-scalpel-guard/SKILL.md`
- Rung-0 gate in the ladder:
  `../../skills/steering-tiered-ladder/SKILL.md`
- Meta-process spine (single-axis perturbation discipline):
  `../../meta-skills/autoresearch-meta/SKILL.md`
- Source module: `src/steering/hooks.py`
- Activation cache: `src/steering/extract.py`
- Runner entry point: `src/steering/runner.py`
