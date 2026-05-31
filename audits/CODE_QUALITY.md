# CODE_QUALITY.md — Elite Code-Quality Audit of `src/steering/`

Auditor: principal ML-research SWE (read-only pass). Date: 2026-05-30.
Scope: `fakelm.py, model.py, hooks.py, extract.py, geometry.py, eval.py,
datasets.py, runner.py, dashboard.py, __init__.py` + `tests/`, `conftest.py`,
`pyproject.toml`.

Method: `ruff check`, `mypy --ignore-missing-imports`, full `pytest`, and a
critical line-by-line read of every module. No `.py` file was edited (another
process is importing the package).

---

## 1. Executive summary

This is a **strong, unusually disciplined** harness. The architecture is clean
(extract → hooks → eval → runner → dashboard), the offline `FakeResidualLM`
contract is honoured everywhere, the composite is frozen+fingerprinted, the
model-load error taxonomy is genuinely honest (SSL vs network vs OOM vs gated),
and the dashboard markdown converter is real (no `**`/`|---|` leaks, verified by
tests). The team clearly internalised the ICML_REVIEW non-circularity critique:
`generation_behavior_scorer` and `generate_responses` are the real instruments,
with the projection proxy retained only as a tagged FakeLM fallback.

The defects are almost all **MINOR/NIT polish**: 8 ruff findings (5 auto-
fixable), 16 mypy errors (mostly cosmetic container-typing), a handful of dead
imports/vars, magic numbers, and a duplicated-encode pattern repeated 3×. There
is **one real correctness footgun (MAJOR)**: `FakeResidualLM._Attn.forward` does
a dead `torch.matmul` then overwrites it with an `einsum`, and the whole block is
written assuming fp32 — under a `device='cuda'`/bf16 FakeLM (never exercised) it
is fragile. There is **one real systemic risk the prompt called out (MAJOR)**:
`runner._run_inner` calls `load_model` every invocation with **no model cache and
no explicit free**, so the documented "reload N times in a loop → OOM/paging-file
exhaustion" failure is structurally guaranteed for any multi-experiment loop on a
real model.

No secrets are logged. No token is ever printed. Determinism is well threaded
(seeded generator in FakeLM, `_set_seed` in runner, greedy decoding for all
gates). The biggest *scientific* caveat is the same-model-family circularity
disclosed in §8.

### Per-module letter grades

| Module | Grade | One-line justification |
|---|---|---|
| `__init__.py` | A | Clean, accurate docstring + `__all__`; `__version__` present. |
| `geometry.py` | A | Pure, well-documented, eps-guarded, edge-case-tested. Best file. |
| `eval.py` | A− | Excellent design; 3 dead imports, 1 dead var, scattered magic constants (8.0, 1e-8). |
| `datasets.py` | A− | Tiny, correct, typed. `_load` returns `dict` but some payloads are lists — loose. |
| `hooks.py` | B+ | Correct + invariant-tested; unused `Callable` import, `__exit__` typing, hard-coded `1e-8`. |
| `fakelm.py` | B | Works + deterministic, but dead `matmul` in `_Attn`, fp32 assumption, `config` shim re-defines a class per call. |
| `extract.py` | B+ | Solid; `dict[str, object]` bank type defeats type-checking; `model_tag` collision risk; `_encode` duplicated. |
| `dashboard.py` | B | Huge (1795 lines), does its job well, but single-responsibility violated; `_num` float-cast crash path; f-string nit. |
| `model.py` | B+ | Best-in-class error taxonomy; but `"cuda" in msg and "memory" in msg` precedence bug; no model cache. |
| `runner.py` | B | Correct artifacts + provenance tags; **no model-cache/free (OOM)**, fake/real safety-placeholder duplication, missing-return typing. |

---

## 2. Linter raw counts

### ruff (`ruff check src/steering tests`) — **8 errors, 5 auto-fixable**

| Code | Count | Files |
|---|---|---|
| F401 unused import | 4 | `eval.py` (×3: `offshell_displacement`, `build_position_mask`, `get_residual_layers`), `hooks.py` (`Callable`) |
| F841 unused local | 2 | `eval.py:467` `pred_token`, `tests/test_runner.py:32` `entry` |
| F541 f-string w/o placeholder | 1 | `dashboard.py:1142` |
| E741 ambiguous name `l` | 1 | `tests/test_runner.py:79` |

5 are `--fix`-able (the 4 F401 + the F541). 2 hidden fixes available with
`--unsafe-fixes`.

### mypy (`mypy src/steering --ignore-missing-imports`) — **16 errors in 7 files**

| File:line | Error |
|---|---|
| `fakelm.py:184` | `Item "None" of "list | None" has no attribute "append"` (hidden-states branch) |
| `fakelm.py:189` | `tuple()` arg incompatible `list | None` |
| `model.py:103` | assignment type clash `_BaseModelWithGenerate` vs `FakeResidualLM` |
| `model.py:182` | `max(key=…)` `Callable[[Sized],int]` vs expected |
| `model.py:226` | dict-comprehension value `Tensor` vs `list[list[int]]` |
| `hooks.py:211` | `bool` invalid as `__exit__` return that always returns False |
| `hooks.py:240` | dict-comp value `Tensor | None` vs `Tensor` |
| `extract.py:223` | `max(key=…)` returns `object` (not orderable) ×2 |
| `extract.py:236-237` | `float(object)` (bank `dict[str,object]`) ×2 |
| `eval.py:282` | `"Tensor" not callable` (`model.generate` shadowed by mypy seeing `nn.Module`) |
| `dashboard.py:463` | `float(Any | None)` in `_num` |
| `runner.py:210` | `Missing return statement` (`_steered_mcq` closure) |
| `runner.py:281-282` | `float(object)` from extract bank ×2 |

Root causes cluster into **three** fixes (see §6): (a) `Optional`/`None`-init
lists, (b) the `dict[str, object]` vector-bank type, (c) `nn.Module` not
declaring `.generate`. Fixing those clears ~12 of 16.

---

## 3. Per-module findings

### 3.1 `__init__.py` — A
Accurate module map, correct `__all__`, `__version__ = "0.1.0"` matches
`pyproject`. Nothing to fix. (NIT: `__all__` lists submodule names but they are
not imported here, so `from steering import eval` works only lazily — fine for a
namespace package, but `import steering; steering.eval` would `AttributeError`
until first import. Minor.)

### 3.2 `geometry.py` — A
The cleanest module. Pure torch, every function documents shapes, eps guards
everywhere (`1e-8`, `1e-12`), `effective_rank`/`participation_ratio` correctly
filter zero singular values and return `0.0` on empty. `offshell_displacement`
is relative/scale-free. Edge cases are unit-tested against analytic rank-k
batches. 
- NIT: the eps constants `1e-8`/`1e-12` are repeated literals; promote to module
  constants `_NORM_EPS`, `_SPECTRUM_EPS` for one source of truth.
- NIT (numerics): `singular_values` always upcasts to `.float()` — good — but on
  a huge `[n,dim]` real batch `svdvals` is O(n·dim²); acceptable at this scale,
  worth a docstring note that callers should subsample.

### 3.3 `eval.py` — A−
Design is excellent: frozen `COMPOSITE_FORMULA` + `composite_fingerprint`,
pinned `COMPOSITE_WEIGHTS`, the non-circular `generation_behavior_scorer` with a
**tagged** FakeLM fallback, greedy deterministic decoding, honest provenance.
- **MINOR (dead code, ruff F401):** lines 32-34 import `offshell_displacement`,
  `build_position_mask`, `get_residual_layers` — none are used. Remove.
- **MINOR (dead var, ruff F841):** `eval.py:467 pred_token = int(last.argmax())`
  is computed and never used in `mcq_accuracy`. Remove.
- **MINOR (magic numbers):** the logistic scale `8.0` (line 386) and `-delta`
  scale in `projection_behavior_scorer` (line 163) are unnamed; the `1e-8` unit
  guards recur. Name them: `_CONCEPT_LOGIT_SCALE = 8.0`.
- **NIT:** `_STOPWORDS` (lines 178-189) contains **duplicates** ("the", "below",
  "across", "down", "above") and content words masquerading as stopwords
  ("love", "open", "bright", "hot", "cold", "small") — these will silently
  suppress legitimate concept words. The set dedups, but the content words are a
  correctness smell for `lexicon_from_pairs`. Curate the list.
- **NIT:** `mcq_accuracy` FakeLM surrogate maps options to tokens via
  `opt_ids % last.shape[0]` — a deliberate tripwire (documented), fine, but the
  modulo can alias distinct options to the same vocab id; acceptable for a
  Rung-0 stand-in.
- **NIT (type):** several public fns return bare `dict` (`generation_behavior_
  scorer`, `selectivity`, `evaluate_bundle`). Use `TypedDict` for the eval
  bundle so the dashboard's key access is checked.

`projection_behavior_scorer` is **NOT redundant** — it is the explicitly-tagged
FakeLM/offline fallback that `generation_behavior_scorer` and the runner dispatch
to when `_can_generate` is False. Keep it. (It is the right design; just make the
"this is a proxy" contract a `TypedDict` field, which it already is via
`"scorer": "projection"`.)

### 3.4 `datasets.py` — A−
Small, typed, offline, well-docstringed. 
- NIT: `_load(name) -> dict` is inaccurate — `json.load` may return a list; the
  payloads are dicts so it works, but the annotation lies. `-> Any` or a
  `TypedDict` per file is more honest.
- NIT: no validation that required keys (`pairs`, `questions`, `passages`,
  `prompts`, `concept`) exist — a malformed JSON yields a raw `KeyError` rather
  than a "dataset slice X is malformed" message. Add a one-line guard.

### 3.5 `hooks.py` — B+
Correct and the Rung-0 invariants (Δh≠0, special tokens untouched, exact
restoration, project_out zeros the component, rotate preserves norm) are all
mechanism-asserting tests. `apply_operation` is pure-functional, dtype/device-
casts `v` to `h`. 
- **MINOR (ruff F401):** `Callable` imported (line 23), never used. Remove.
- **MINOR (mypy):** `__exit__ -> bool` always returns `False`; annotate
  `-> Literal[False]` (or `None`) so mypy knows it cannot swallow exceptions.
- **MINOR (typing):** `probe_activations` returns `{li: p.activations}` where
  `activations` is `Optional[Tensor]` (mypy:240). If a hooked layer never fired
  (bad index, generate path) the value is `None` and downstream `.float()`
  crashes opaquely. Assert non-None with a clear message, or type the return
  `dict[int, Tensor]` only after a guard.
- **NIT (magic number):** `_unit` default `eps=1e-8` and the inline `+ 1e-8`
  rotate guards (lines 71, 75) — promote to a shared `_EPS`.
- **NIT (robustness):** `_SteerHook.__call__` only applies the position mask when
  `m.shape[:2] == h.shape[:2]`; if shapes mismatch (e.g. KV-cache single-token
  decode steps during `.generate`) the mask is **silently dropped** and special
  tokens *could* be steered on those steps. This is a real correctness gap for
  the generation path — at minimum log/raise instead of silently steering
  everything. (The generation scorers rely on the hook; verify the mask is
  rebuilt per decode step, or document that steering during decode is unmasked.)

### 3.6 `fakelm.py` — B
Deterministic (seeded `torch.Generator`), LayerNorm weights correctly forced to
1, exposes `.layers`, `.config`, `.logits`, `.hidden_states` — a faithful HF-
shaped stub. 
- **MAJOR (dead + fragile code):** `_Attn.forward` (lines 53-63) computes
  `mixed = torch.matmul(mask, v) / denom` and **immediately overwrites it** with
  an `einsum`. The first `matmul` line plus its "handle batch"/"redo per-batch"
  comments are dead and misleading. Worse, `mask`/`denom` are built with
  `dtype=h.dtype`; if a future caller runs FakeLM in bf16 the `einsum("st,btd")`
  is fine but the dead `matmul` (kept) silently allocates. Delete the dead path;
  keep only the einsum.
- **MINOR (mypy fakelm:184,189):** `hidden_states` is `[h] if … else None`, then
  `.append`-ed inside `if output_hidden_states`. mypy can't narrow. Initialise
  `hidden_states: list = []` and gate the appends, or guard with an assert.
- **NIT:** `config` property rebuilds a throwaway `_Cfg` class **on every
  access** (lines 156-164). Define a module-level dataclass once.
- **NIT (determinism):** `_init_deterministic` sets all 1-D params to 0 then
  LayerNorm weights to 1 — correct, but `unembed` has `bias=False` and embeddings
  get `*0.1` scaled normal; fine. Document that `dim`/`hidden` changes alter the
  RNG stream (tests pin defaults).

### 3.7 `extract.py` — B+
Cache-once design is right; `_pairs_signature` (sha256 of pair content) +
sanitised `model_tag` is thoughtful (the `re.sub` fix for HF "/" is a real bug
they caught). DiffMean/PCA/Fisher are mathematically correct and the **uncentered
SVD** rationale is documented and correct. 
- **MINOR (mypy:223,236,237):** `build_vector_bank` returns `dict[int, dict[str,
  object]]`; `best_layer`'s `max(key=lambda li: bank[li]["fisher"])` is `object`
  (unorderable to mypy), and `float(d["cosine_dm_pca"])` is `float(object)`.
  Replace the bank value type with a `TypedDict{diffmean: np.ndarray, pca:
  np.ndarray, cosine_dm_pca: float, fisher: float}`. Clears 4 mypy errors here +
  2 in runner.
- **MINOR (cache key collision):** `collect_activations_cached` key is
  `acts_{tag}_{sig}_{layer_tag}.npz`. The `model_tag` defaults to `"fake"` but
  **quant/dtype are not in the key** — a bf16 vs 4bit run of the same model id
  silently reuses the wrong cache. Add `quant`/dtype to the signature.
- **MINOR (dup logic):** `_encode` (extract) ≈ `_ids` (eval) ≈ the inline encode
  in `runner._run_inner` (lines 166-171) — three copies of "tokenize → pick
  input_ids → move to model.device → swallow StopIteration". Extract one helper
  (see §6 `_encode_to_device`).
- **NIT:** `collect_activations` means over **all** non-pad positions but the
  docstring says "answer tokens"; for the FakeTokenizer there's no prompt/answer
  split, so it's really mean-over-sequence. Rename or document.
- **NIT (numerics):** `pca_top1_vector` does `np.linalg.svd` on the raw diffs;
  fine, but no guard for `n < 1` or all-zero diffs (would yield NaN sign-align).
  Add a degenerate-input guard returning the diffmean.

### 3.8 `eval`/`runner` generation instruments — design check
`generation_behavior_scorer` (non-circular, measures concept *words* in
generated text, not projection onto `v`) and `generate_responses` (real steered
generation feeding the rule-based refusal detector) directly answer
ICML_REVIEW W1/W2. Both greedy-decode (deterministic gates). Both correctly
guard FakeLM via `_can_generate`. `_fake_responses` as a named symbol does **not
exist** — there is no dead stub; the FakeLM path is inlined (see runner finding
below). Good.

### 3.9 `runner.py` — B
Artifacts are correct (running.json transient, append-only JSONL with auto-inc
`experiment_num`, global-best KEEP/DISCARD, TODO-REWRITE reasoning skeleton that
**refuses to fabricate** pre-run reasoning). Provenance tags (`behavior_scorer`,
`safety_real`) are exactly the right anti-spoofing instrument. Seeds threaded via
`_set_seed` (random+numpy+torch+cuda).
- **MAJOR (memory/OOM — the flagged issue):** `_run_inner` calls
  `load_model(config["model"], …)` **every run** with no cache and no
  `del model; torch.cuda.empty_cache()` in a `finally`. For `--model fake` this
  is cheap, but for any real-Gemma sweep driven in a loop this is precisely the
  "reload N times → paging-file/CUDA OOM" failure `model.py`'s own error text
  warns about. There is **no model cache and no explicit free anywhere**. Fix:
  a process-level `@lru_cache`/handle cache keyed by `(name, quant, device)`, or
  a `load_once` experiment loop (see §6). At minimum free the model in `finally`.
- **MINOR (duplication/inconsistency):** for FakeLM, `run_single_experiment`
  **bypasses** `generate_responses` and hand-rolls placeholders (lines 245-246:
  harmful→"I can't help…", harmless→"Sure, here is how…"), while
  `generate_responses` itself has a *different* FakeLM placeholder (returns a
  refusal for **every** prompt, lines 421-424). So the two offline code paths
  disagree on the harmless response (runner: comply; eval: refuse). Pick one;
  route FakeLM through `generate_responses` so there is a single offline contract.
- **MINOR (mypy:210):** `_steered_mcq` closure has an implicit return path mypy
  flags as "missing return"; the `with` always returns, but annotate `-> float`
  and ensure the body's single return covers it (it does — cosmetic).
- **MINOR (mypy:281-282):** `float(bank[layer]["fisher"])` is `float(object)` —
  fixed by the extract `TypedDict`.
- **NIT (magic):** `behavior_prompts = […][:6]`, `max_new_tokens` defaults
  (24/40), alpha default `4.0`, the layer clamp — all magic; lift to named
  constants / CLI-configurable where they affect results.
- **NIT (config):** `quant` choices include both `"none"` and `"bf16"` (aliases);
  `model.py` treats them identically — fine, but document the alias.
- **NIT:** `_special_ids` reads `start_of_turn_id` off model/tokenizer; real HF
  tokenizers don't expose that attr (it's `convert_tokens_to_ids("<start_of_
  turn>")`), so on real Gemma only BOS is masked. Verify the real-model special-
  token set is complete (EOT/`<end_of_turn>` too) or special tokens leak into
  steering on real runs.

### 3.10 `dashboard.py` — B
Genuinely impressive: a real GFM-ish markdown converter (no `**`/`##`/`|---|`
leaks, test-verified), self-contained HTML, graceful matplotlib degradation,
three linked tiers, composite-breakdown reconciliation with a warn-on-mismatch.
- **MINOR (ruff F541):** `dashboard.py:1142` `… or f'<tr>…No runs yet.…'` is an
  f-string with no placeholder. Drop the `f`.
- **MINOR (mypy:463 / crash path):** `_num` does `float(v)` in a `try/except
  (TypeError, ValueError)` — good — but `_axis_scores`, `plot_*`, and the
  composite-breakdown call `_num` on arbitrary logged values; a logged value of
  `NaN`/`inf` passes `float()` and silently poisons radar/Pareto. Add an
  `isfinite` clamp in `_num`.
- **MINOR (SRP):** 1795 lines spanning I/O, hypothesis resolution, a markdown
  engine, 7 matplotlib plotters, and 3 HTML renderers. Split into
  `dashboard/{io,markdown,plots,render}.py`. Maintainability, not correctness.
- **NIT (security/robustness):** `_git_sha` shells out to `git` with `timeout=5`
  and a bare `except` → `"no-git"`. Fine, but note it runs on every dashboard
  build (every experiment). Cache it.
- **NIT:** `md_to_html` paragraph gather breaks on `"|" in line` — a legitimate
  prose sentence containing a pipe is mis-parsed. Low risk for reasoning text.
- **NIT (markdown-leak coverage):** tests assert no `**`/`##`/`|---|` in OUTPUT,
  but a reasoning field containing a literal `<script>` is escaped by
  `html.escape` first (good) — confirmed safe. No XSS leak.

### 3.11 `model.py` — B+
The error taxonomy is the standout: SSL-intercept vs dead-network vs OOM/paging
vs gated-401/403 vs generic, each with an actionable fix and an honest "NOT a
gating issue" disclaimer — exactly the ICML_REVIEW spirit. `get_residual_layers`
resolves FakeLM / Gemma-2 / Gemma-3-nested / generic-largest-ModuleList. No token
logging anywhere.
- **MINOR (operator-precedence bug):** line 131:
  `"paging file" in msg or "out of memory" in msg or "1455" in msg or "cuda" in
  msg and "memory" in msg`. `and` binds tighter than `or`, so this parses as
  `… or ("cuda" in msg and "memory" in msg)` — which is the intended grouping by
  luck, but it is fragile and unobvious. Parenthesise:
  `… or ("cuda" in msg and "memory" in msg)` explicitly.
- **MINOR (mypy:103,182,226):** `model` var reused for FakeLM vs HF type;
  `get_residual_layers` `max(key=len)`; `_FakeTokenizer.__call__` dict-comp.
  Cosmetic; fixable with `Union`/cast and a `list` re-annotation.
- **MINOR (no cache):** `load_model` is the natural home for a `(name, quant,
  device)`-keyed cache (or document that the *caller* must cache). Pairs with the
  runner OOM fix.
- **NIT (dead-code guards):** `device_map=device` for quantized loads assumes a
  single device string; multi-GPU `"auto"` is not supported — document the
  single-GPU assumption (CLAUDE.md says 4090 laptop, so fine).
- **NIT:** `_FakeTokenizer.encode` maps chars to `3 + ord(ch) % (vocab-3)` —
  deterministic, avoids specials; good. No `decode` method, so the eval generation
  path's `tokenizer.decode` is wrapped in `try/except → ""` (eval.py:294) — that
  is the FakeLM-has-no-decode guard, correct.

---

## 4. Test-quality assessment

**Verdict: above-average, mechanism-asserting (not shape-only) where it counts —
but with real coverage gaps on the new generation instruments and device paths.**
All 32 tests pass in ~76s.

Strong, mechanism-asserting tests:
- `test_geometry`: analytic rank-k batches assert effective-rank ≈ k and
  participation-ratio ≈ k, and norm-budget *accumulation* (3 steps = 3×). Real
  math, not shapes.
- `test_hooks`: Δh>0, **exact** state restoration (`allclose atol=0`), no
  lingering `_forward_hooks`, special-token position untouched while a neighbour
  changes, project_out zeros the v-component, rotate preserves norm. Excellent.
- `test_extract`: planted-direction recovery (cosine>0.9), PCA≈DiffMean,
  Fisher **peaks at the planted layer**. Mechanism-asserting.
- `test_eval`: safety-leak penalty ≥ 2λ, gibberish-cannot-win, fingerprint
  stability, custom-weight override. Good Goodhart coverage.
- `test_dashboard`: **markdown-leak** assertions (`**`/`##`/`|---|` absent),
  3-tier sub-linking resolution, champion highlight, measured stack-overlay,
  samples rendered vs placeholder. Genuinely good HTML-contract tests.
- `test_runner`: append-only JSONL, auto-increment, running.json cleared,
  TODO-REWRITE skeleton, KEEP champion. Solid integration coverage.

Coverage gaps (prioritised):
1. **`generation_behavior_scorer` / `generate_responses` are UNTESTED.** These
   are the W1/W2 non-circular instruments. There is no test that (a) the FakeLM
   fallback returns `scorer:"projection"` with `delta:None`, (b) `concept_rate`
   counts stems correctly, (c) `lexicon_from_pairs` ranks pos-distinctive words,
   (d) `_word_stems` stemming collapses `waves`→`wave`. Add pure-function tests
   (no model needed) for `concept_rate`, `lexicon_from_pairs`, `_word_stems`,
   `is_refusal`, `compliance_rate`, `selectivity`.
2. **No real-model smoke test (guarded/skippable).** There is no
   `@pytest.mark.skipif(not HF_AVAILABLE)` test that loads a tiny real Gemma and
   exercises the generation path end-to-end. Add one, skipped by default, so the
   real `.generate`+hook path has *some* CI signal when a token is present.
3. **Device paths untested.** Everything runs on CPU FakeLM. The `.to(device)`
   branches in `_encode`/`_ids`/runner and the bf16 quant path in `model.py` have
   zero coverage. A CUDA-guarded test (skipif no cuda) would catch device-
   mismatch regressions.
4. **Geometry edge cases:** `effective_rank` on all-zero / single-row / NaN
   input is not tested (the empty-spectrum `return 0.0` branch is uncovered).
5. **Hooks generation/decode masking:** the per-decode-step position-mask
   shape-mismatch branch (hooks.py:126) is untested — exactly the silent-drop
   correctness gap in §3.5.
6. **`mcq_accuracy` / `perplexity`** have no direct unit test (only exercised
   transitively through the runner integration test).
7. **`apply_operation('rotate')` only tested at alpha=0.5; `project_out` only at
   alpha=1.0** — the `alpha`-scaled partial-projection-removal semantics
   (project_out at alpha=0.5) are unverified.

`conftest.py` (src on path) and `pyproject` (`pythonpath=["src"]`,
`testpaths=["tests"]`) are correct. No ruff/mypy config block in `pyproject` —
consider adding `[tool.ruff]`/`[tool.mypy]` so the gates are pinned in-repo.

---

## 5. Prioritised issue table

| Severity | File:line | Issue | Concrete fix |
|---|---|---|---|
| **MAJOR** | `runner.py:137` (`_run_inner`) | No model cache and no explicit free; `load_model` every run → documented reload-loop OOM/paging-file exhaustion on real models. | Add a process `_MODEL_CACHE: dict[(name,quant,device)] → handle`, or a `load_once` outer loop; at minimum `try/finally: del model; torch.cuda.empty_cache()`. |
| **MAJOR** | `fakelm.py:53-63` (`_Attn.forward`) | Dead `torch.matmul` mixing computed then overwritten by `einsum`; misleading "redo per-batch" comments; fp32 assumption. | Delete the dead `matmul`/`denom`-first lines; keep only `mixed = einsum("st,btd->bsd", mask/denom, v)`. |
| **MAJOR** | `hooks.py:126` | Position mask silently dropped when `m.shape[:2] != h.shape[:2]` (decode-step / KV-cache) → special tokens may be steered during generation. | Rebuild the mask per decode step, or raise/log instead of silently steering all positions; add a test. |
| MINOR | `eval.py:32-34` | 3 unused imports (ruff F401). | `ruff --fix` removes them. |
| MINOR | `eval.py:467` | Dead `pred_token` (ruff F841). | Delete the line. |
| MINOR | `hooks.py:23` | Unused `Callable` import (F401). | `ruff --fix`. |
| MINOR | `hooks.py:211` | `__exit__ -> bool` (mypy exit-return). | Annotate `-> Literal[False]`. |
| MINOR | `extract.py:202-218` | `dict[str, object]` bank type → 6 mypy errors here + runner. | Introduce `class BankEntry(TypedDict)`; retype `build_vector_bank`/`load_vector_bank`. |
| MINOR | `extract.py:133` | Cache key omits quant/dtype → bf16 vs 4bit reuse wrong `.npz`. | Add quant/dtype to `_pairs_signature` or the filename. |
| MINOR | `runner.py:245-246 vs eval.py:421-424` | FakeLM safety placeholders disagree (runner complies on harmless; `generate_responses` refuses all). | Route FakeLM through `generate_responses`; single offline contract. |
| MINOR | `model.py:131` | `or "cuda" in msg and "memory" in msg` precedence is fragile. | Parenthesise `or ("cuda" in msg and "memory" in msg)`. |
| MINOR | `dashboard.py:1142` | f-string without placeholder (F541). | Drop `f`. |
| MINOR | `dashboard.py:460-465` (`_num`) | NaN/inf pass `float()` and poison plots/composite. | Add `if not math.isfinite(x): return default`. |
| MINOR | `runner.py` `_special_ids` | Real Gemma special tokens (`<start_of_turn>`,`<end_of_turn>`) not resolved → leak into steering. | Resolve via `tokenizer.convert_tokens_to_ids` for real models. |
| NIT | `eval.py:178-189` | `_STOPWORDS` has duplicates + content words ("love","hot","bright") that suppress concept words. | Curate the stop-list. |
| NIT | `eval.py:163,386` | Magic logistic scales (`8.0`, implicit `1.0`). | Name `_CONCEPT_LOGIT_SCALE`, `_PROJ_LOGIT_SCALE`. |
| NIT | `geometry.py` / `hooks.py` | Repeated `1e-8`/`1e-12` eps literals. | Module-level `_EPS` constants. |
| NIT | `fakelm.py:156-164` | `config` property rebuilds a class per access. | Module-level dataclass. |
| NIT | `tests/test_runner.py:32,79` | Unused `entry`; ambiguous `l` (F841,E741). | `ruff --fix`; rename `l`→`line`. |
| NIT | `datasets.py:24` | `_load -> dict` but payloads include lists; no malformed-slice guard. | `-> Any` + key-existence check. |

---

## 6. Quick wins (ruff `--fix`-able now)

Run `python -m ruff check --fix src/steering tests` to auto-resolve 5:
- `eval.py` ×3 F401 (`offshell_displacement`, `build_position_mask`,
  `get_residual_layers`).
- `hooks.py` F401 (`Callable`).
- `dashboard.py:1142` F541 (`f` prefix).

Manual one-liners (not auto-fixed but trivial):
- `eval.py:467` delete `pred_token`.
- `tests/test_runner.py:32` delete unused `entry` (or assert on it).
- `tests/test_runner.py:79` rename `l` → `line`.
- `--unsafe-fixes` offers 2 more (review before applying).

---

## 7. Recommended refactors

### R1 — shared `_encode_to_device` helper (kills 3 copies)
`extract._encode`, `eval._ids`, and the inline encode in `runner._run_inner`
(lines 166-171) are the same logic. Extract one helper (suggested home:
`model.py` or a new `tokenize.py`):

```python
def encode_to_device(tokenizer, text: str, model: nn.Module) -> torch.Tensor:
    """Encode `text` → input_ids [1, seq] on the model's device.

    Handles HF BatchEncoding (not a dict subclass), the offline FakeTokenizer
    (plain dict), and a bare tensor. Falls back to CPU if the model is
    parameter-less (StopIteration).
    """
    out = tokenizer(text, return_tensors="pt")
    ids = out.input_ids if hasattr(out, "input_ids") else (
        out["input_ids"] if isinstance(out, dict) else out)
    try:
        ids = ids.to(next(model.parameters()).device)
    except StopIteration:
        pass
    return ids
```
Then `extract._encode = eval._ids = encode_to_device`. Removes ~30 duplicated
lines and one drift vector.

### R2 — `BankEntry` TypedDict + `ModelHandle` dataclass (clears ~8 mypy errors)
```python
class BankEntry(TypedDict):
    diffmean: np.ndarray
    pca: np.ndarray
    cosine_dm_pca: float
    fisher: float

@dataclass
class ModelHandle:
    model: nn.Module
    tokenizer: Any
    name: str
    quant: str
    device: str
    can_generate: bool          # cache _can_generate once
    special_ids: list[int]      # resolved once (BOS/EOT/start_of_turn)
```
Returning a `ModelHandle` from `load_model` (or a thin wrapper) centralises the
`_can_generate`/`_special_ids`/`_is_fake_model` logic that is currently
re-derived in eval **and** runner, and fixes the real-Gemma special-token gap
(§3.9 NIT).

### R3 — load-once experiment loop (fixes the OOM, MAJOR)
Today each `run_single_experiment` reloads. Provide a loop entry point that loads
**once** and reuses:
```python
def run_sweep(configs: list[dict]) -> list[dict]:
    cache: dict[tuple, ModelHandle] = {}
    results = []
    for cfg in configs:
        key = (cfg["model"], cfg.get("quant", "4bit"))
        handle = cache.get(key) or _load_handle(cfg)   # load once per model
        cache[key] = handle
        results.append(_run_inner_with_handle(handle, cfg))
    for h in cache.values():
        del h.model
    torch.cuda.empty_cache()
    return results
```
Refactor `_run_inner` to accept a `ModelHandle` instead of calling `load_model`
itself; keep `run_single_experiment` as a one-shot wrapper that builds a handle,
runs, and frees in `finally`. This makes the documented "load ONE model per
process" advice structurally enforced rather than aspirational.

### R4 — split `dashboard.py`
1795 lines → `dashboard/{io.py, markdown.py, plots.py, render_master.py,
render_hypothesis.py, render_experiment.py, build.py}`. Pure maintainability;
the markdown engine and the plot bank are independently testable units.

### R5 — pin lint/type gates in `pyproject`
Add `[tool.ruff]` (select E,F,I,UP; line-length) and `[tool.mypy]`
(`ignore_missing_imports = true`, `check_untyped_defs = true`) so CI runs the
same gates this audit ran.

---

## 8. Same-model-family circularity disclosure

The harness extracts steering vectors from a model's own residual stream
(DiffMean/PCA over Gemma activations) and then **evaluates** behavior efficacy,
capability, coherence, safety, and selectivity **on the same model family
(Gemma)**. Three circularity vectors remain even after the (commendable) W1/W2
fixes:

1. **Extraction↔intervention same model.** The vector is built from and injected
   into the *same* Gemma; `generation_behavior_scorer` mitigates the *tautology*
   (it scores concept *words* in text, not projection onto `v`), but the concept
   lexicon is itself derived from the same contrast pairs that built the vector
   (`lexicon_from_pairs(pairs)`), so a model that simply parrots pair vocabulary
   under any perturbation can inflate the score. **Mitigation:** score with a
   held-out concept lexicon or an independent judge model from a *different*
   family.
2. **Judge = rule-based on the same generations.** Axis 4/5 use a keyword refusal
   detector (`_REFUSAL_MARKERS`) on the model's own text. A model whose refusal
   style isn't in the marker list reads as "compliant" (false safety leak) or
   vice-versa. **Mitigation:** cross-check with a separate-family classifier on a
   sample.
3. **Capability tripwire (FakeLM) is a surrogate, not MMLU.** `mcq_accuracy`'s
   FakeLM path is a deterministic tripwire (documented), not a real capability
   measurement; only the real-Gemma Rung-2+ path is a genuine MMLU logprob score.
   Do not report FakeLM capability numbers as capability retention.

None of these are bugs — they are inherent to single-family activation steering
and are partially disclosed in-code (ICML_REVIEW references throughout). They are
restated here so any "winner" claim carries the caveat: **efficacy and safety are
measured on the same model family that produced the steering vector; external
validity requires a cross-family judge and a held-out concept lexicon.**

---

## Appendix — top 5 fixes (for the caller)

1. **MAJOR — runner OOM:** add a model cache / load-once loop / `finally` free in
   `runner._run_inner` (R3). The reload-every-run pattern guarantees the exact
   paging-file/CUDA OOM that `model.py`'s own error text warns about.
2. **MAJOR — fakelm dead/fragile attention:** delete the dead `torch.matmul`
   path in `_Attn.forward` (keep only the einsum).
3. **MAJOR — hooks silent mask drop:** `hooks.py:126` silently steers *all*
   positions (incl. special tokens) when the mask shape mismatches the hidden
   during decode; raise/rebuild instead, and test it.
4. **MINOR cluster — typing debt:** introduce a `BankEntry` TypedDict +
   `Literal[False]` `__exit__` + `Optional`-list init in fakelm; clears ~12 of 16
   mypy errors and the F401/F841 ruff findings via `ruff --fix`.
5. **MINOR — offline contract drift + test gap:** unify the two disagreeing
   FakeLM safety placeholders (runner vs `generate_responses`) and add pure-
   function tests for the now-untested W1/W2 instruments (`concept_rate`,
   `lexicon_from_pairs`, `is_refusal`, `compliance_rate`, `selectivity`) plus one
   skippable real-model smoke test.

---

## Resolution (2026-05-30) — fixes applied

All gates now CLEAN: `ruff check src/steering tests` → **All checks passed!**;
`mypy src/steering --ignore-missing-imports` → **Success: no issues found in 10
source files**; `pytest tests/ -q` → **46 passed** (was 32; +13 instrument tests,
+1 incremental-mask test, +1 skippable real-Gemma smoke). `COMPOSITE_FORMULA`
fingerprint unchanged (`a9001e87087e`).

**BLOCKER — Triton/inductor crash on Gemma-3 (Windows).** `model.py` now applies
`_force_eager_generation(model)` on every real-model load: it sets
`torch._dynamo.config.suppress_errors = True` AND
`model.generation_config.cache_implementation = "dynamic"` (guarded for models
without `generation_config`). Gemma-3's default is `"hybrid"`, which routes
generation through torch.compile→TorchInductor→Triton (no Windows wheel) and
crashes; forcing the dynamic eager cache avoids it. Covered by the new
`tests/test_real_model_smoke.py` (skipped unless `models/google/gemma-3-270m-it`
exists; it loads, runs a steered generation, and asserts the dynamic cache + no
crash).

**MAJOR — runner OOM / load-once.** Added a process-level `_MODEL_CACHE` keyed by
`(name, quant, device)` plus `load_model_cached()` / `free_model_cache()` in
`model.py`; a fresh key frees the previous model (`del`+`gc.collect()`+
`torch.cuda.empty_cache()`). `runner._run_inner` now calls `load_model_cached`,
so an in-process alpha sweep reuses one loaded Gemma instead of reloading it
every run. FakeLM path still works (cached cheaply).

**MAJOR — fakelm dead attention.** Removed the dead `torch.matmul` line and the
misleading "redo per-batch" comments in `_Attn.forward`; kept the single correct
row-normalised einsum, dtype/device following `h`.

**MAJOR — hooks position-mask robustness.** `_SteerHook` now rebuilds the mask
PER FORWARD from the actual hidden-state seq dim (`_mask_for`): prompt positions
keep their protection, appended/generated positions are steerable, and a len-1
decode step never silently steers a protected prompt special. New test
`test_special_token_positions_unmodified_across_incremental_forwards` asserts BOS
stays untouched across same-length, longer, and single-token forwards.

**MINOR cluster — typing + lint.** Added `BankEntry` TypedDict (extract.py),
`Literal[False]` on `SteeringContext.__exit__`, list-init for FakeLM
hidden-states, module-level `FakeLMConfig` dataclass (built once), `_EPS`/named
logistic-scale constants, removed dead imports/vars, `_num` NaN/inf clamp in the
dashboard, real-Gemma special-token resolution via `convert_tokens_to_ids`, and
parenthesised the `model.py` OOM precedence. `pyproject.toml` pins
`[tool.ruff]`/`[tool.mypy]` (ruff select `E4,E7,E9,F,I`; mypy
`check_untyped_defs`).

**MINOR — dedupe + unify.** One shared `model.encode_to_device` helper now backs
`extract._encode`, `eval._ids`, and the runner's inline encode. One
`eval.fake_safety_responses` helper is the single offline FakeLM safety contract
used by both `generate_responses` and the runner (the runner no longer "complies
on harmless" — both refuse, CR≈0). Cache key now includes `quant`.
