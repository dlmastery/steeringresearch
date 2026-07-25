# Runnability audit — autoresearch harness (`src/steering/`)

**Date:** 2026-07-25
**Scope:** static inspection + CPU-only checks (imports, `--help`, pytest). No model
was loaded and no GPU job was launched — the single 4090 is reserved for other work.
**Repo:** `C:\Users\evija\steeringresearch`

---

## VERDICT: **GO** — the harness code is healthy and runs today.

Zero code drift. All 26 modules in `src/steering/` import cleanly and the full test
suite is **261 passed / 0 failed** under the correct interpreter. Everything that
looks broken is **environment, not code**.

**Must fix before launch (two items, both one-liners):**

1. **Use `C:\Users\evija\anaconda3\python.exe`, never bare `python`.** (See §0.)
2. **Pass `--judge-model Qwen/Qwen3-4B-Instruct-2507`** (or patch
   `src/steering/local_judge.py:29`). The shipped default is `Qwen/Qwen2.5-7B-Instruct`,
   which is **not** in the local cache — every judged run would silently start a
   ~15 GB download.

---

## 0. THE INTERPRETER TRAP (most important finding)

`python` on PATH is **Windows Store Python 3.13**
(`C:\Users\evija\AppData\Local\Microsoft\WindowsApps\...\python.exe`) and it is
**unusable for this project**:

| Symptom | Detail |
|---|---|
| No CUDA | `torch 2.10.0+cpu`, `torch.cuda.is_available() == False` — the 4090 is invisible |
| `transformers` will not import | `ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module` (installed hub is **1.19.0**) |
| Missing deps | `bitsandbytes` — absent; `matplotlib` — absent |
| Test suite hard-crashes | `Windows fatal exception: access violation` inside `pyarrow.__init__` (pulled in by `pandas 3.0.3` ← `datasets 5.0.0`), killing the run at `tests/test_run_safety_eval.py`; `tests/test_real_model_smoke.py` also FAILS |

**The working interpreter is `C:\Users\evija\anaconda3\python.exe`** (Python 3.12.3):

```
torch 2.6.0+cu124   cuda=True      transformers 4.55.0    huggingface_hub 0.36.2
numpy 1.26.4        scipy 1.11.4   scikit-learn 1.8.0     pandas 2.1.4
datasets 2.17.1     matplotlib 3.10.9   bitsandbytes 0.49.2   accelerate 1.12.0
```

`~/venv` is a **WSL/POSIX** venv (`bin/`, `lib64/`, no `Scripts/`) and cannot be used
from Windows. Conda env `grokking` is also CPU-only torch 2.10; `myenv` has an old
torch 2.4.1+cu121. **Anaconda base is the one to use.**

> Every command below is written with `$P` = `C:\Users\evija\anaconda3\python.exe`.
> Substituting bare `python` will fail at the first `transformers` import.

---

## (a) EXACT COMMANDS

Set up the shell once:

```bash
# Git Bash
export PYTHONPATH=src
P=~/anaconda3/python.exe
```
```powershell
# PowerShell
$env:PYTHONPATH = "src"
$P = "C:\Users\evija\anaconda3\python.exe"
```

### (a1) A single experiment

```bash
$P -m steering.runner \
  --model ~/.cache/huggingface/hub/models--DavidAU--gemma-3-1b-it-heretic-extreme-uncensored-abliterated/snapshots/<SHA> \
  --quant none --rung 2 \
  --layer 12 --alpha 1.0 --operation add --source diffmean \
  --behavior ocean --seed 0 --tag t1 --description "..."
```

`--model` also accepts a plain HF id (`DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`)
— the cache is resolved automatically — or a local dir (`models/google/gemma-3-1b-it`).

**Gotcha:** `python src/steering/runner.py` does **not** work —
`ImportError: attempted relative import with no known parent package`. It must be run
as a module (`-m steering.runner`) with `PYTHONPATH=src`.

Full runner CLI: `--model --rung{0..4} --layer --alpha --operation{add,relative_add,rotate,project_out}
--source{diffmean,pca} --behavior --seed --description(required) --tag --quant{4bit,8bit,none,bf16} --normalize`.

### (a2) A sweep (loads the model once, in-process — the workhorse)

```bash
$P scripts/campaign_sweep.py \
  --model <id-or-path> --quant none --rung 2 \
  --hyp E3 --tag-prefix E3-cliff \
  --layers 12 --alphas 0 1 2 4 8 --ops add
```

Other flags: `--behaviors A B C` (loop concepts), `--sources`, `--normalize`, and the
four pre-run reasoning fields `--diagnosis --citation --hypothesis --prediction`.

AxBench drivers (each takes `--judge-model`):
`scripts/run_axbench_e2.py`, `_e3.py`, `_e4.py`, `_e7.py`, `_ops.py`, `_stack.py`,
`run_axbench_conditional.py`. Example:

```bash
$P scripts/run_axbench_e7.py --model <id-or-path> --quant none \
  --dataset concept500 --concepts 0 --prompts 10 --knee 0.1 \
  --judge local --judge-model Qwen/Qwen3-4B-Instruct-2507
```

### (a3) Regenerate the dashboard

```bash
$P -m steering.dashboard          # writes dashboard/ + docs/dashboard/ mirror
$P scripts/verify_dashboard.py    # Rubric B (15/15)
$P scripts/verify_rubrics.py      # Rubrics A/C/D — ruff + mypy + pytest + secrets + fingerprint
```

### (a4) Offline plumbing smoke (no GPU, no weights)

```bash
$P -m steering.runner --model fake --rung 1 --description "plumbing check" --tag smoke
```

---

## (b) BROKEN IMPORTS AND TESTS

### Under `C:\Users\evija\anaconda3\python.exe` — **NOTHING IS BROKEN**

- **Imports:** all 26 modules import clean —
  `adversarial, axbench, baselines, cast, controls, dashboard, datasets, eval, extract,
  fakelm, gate, geometry, hooks, hypersteer, intent_gate, judge, local_judge, model,
  multi_intent, real_metrics, runner, sae, safety_bench, safety_judge, safety_target, stats`.
  **0 failures.**
- **Tests:** `$P -m pytest tests -q -p no:randomly` → **261 passed, 18 warnings, 273.66 s.**
  Warnings only: two SWIG `DeprecationWarning`s and 13 scipy
  `Sample size too small for normal approximation` (expected — the n≤3 screening-tier
  stats tests deliberately exercise that path).

### Under Store Python 3.13 (the default `python`) — broken as follows

| What | Cause |
|---|---|
| `import transformers` | `ImportError: huggingface-hub>=0.34.0,<1.0 required`; hub is 1.19.0. `src.steering.model` still imports OK because transformers is imported lazily inside `load_model` — **the failure only surfaces at run time.** |
| `pytest tests` (whole run) | Native `access violation` in `pyarrow` (via `pandas 3.0.3` ← `datasets 5.0.0`), crashing at `tests/test_run_safety_eval.py::test_dry_run_produces_pareto_and_verdicts`. 261 collect fine; the process dies ~66% through. |
| `tests/test_real_model_smoke.py::test_gemma_steered_generation_does_not_crash` | FAILED (no CUDA + broken transformers) |
| any GPU work | `torch 2.10.0+cpu` |

No source change is warranted for these — the fix is to use the anaconda interpreter.

---

## (c) MODEL + JUDGE AVAILABILITY

Cache root: `C:\Users\evija\.cache\huggingface\hub` (no `HF_HOME` / `HF_HUB_CACHE`
override set). Snapshot contents verified — the `.safetensors` blobs are physically
present, not just refs.

| Model | Role | Cached? | Size on disk |
|---|---|---|---|
| `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated` | **target (1B)** | **YES** — `model.safetensors` present | 1.9 GB |
| `DavidAU/gemma-3-4b-it-heretic-uncensored-abliterated-Extreme` | **target (4B)** | **YES** — 2 shards + index + `preprocessor_config.json` (multimodal) | 8.1 GB |
| `Qwen/Qwen3-4B-Instruct-2507` | **judge (Qwen ~4B)** | **YES** — 3 shards + index | 7.6 GB |
| `Qwen/Qwen2.5-3B-Instruct` | alt judge | YES — 2 shards + index | 5.8 GB |
| `Qwen/Qwen2.5-7B-Instruct` | **shipped default judge** | **NO — metadata only** | 16 KB (~15 GB to fetch) |
| `google/gemma-2-2b-it` | CLAUDE.md legacy default | **NO — metadata only** | 16 KB |
| `models/google/gemma-3-1b-it` | local dir | YES | on disk |
| `models/google/gemma-3-270m-it` | local dir | YES | on disk |

**Nothing needs downloading for the intended abliterated-Gemma + Qwen-4B config.**
The only download risk is the *unchanged* `Qwen2.5-7B` judge default (see §d).

Also cached and relevant to the eval axes: JailbreakBench JBB-Behaviors, AdvBench,
HarmBench, StrongREJECT, XSTest, BeaverTails, sorry-bench, wildjailbreak,
lmsys/toxic-chat, and both `pyvene/axbench-concept10` / `concept500`.

---

## (d) PRECISE FILES + LINES TO RETARGET

### d1. Judge → Qwen-4B

**The single anchor:**

- `src/steering/local_judge.py:29`
  `_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"` → `"Qwen/Qwen3-4B-Instruct-2507"`

  `src/steering/safety_judge.py:34` does `from .local_judge import _DEFAULT_MODEL, LocalJudge`,
  so **both** the AxBench behavior judge and the JailbreakBench safety judge follow this
  one constant. `LocalJudge.__init__` (line 44) and `SafetyJudge.__init__` (line 201)
  both default to it.

**Per-script CLI defaults that must also change (or be overridden every invocation):**

| File:line | Current default |
|---|---|
| `scripts/run_axbench_e2.py:49` | `Qwen/Qwen2.5-7B-Instruct` |
| `scripts/run_axbench_e3.py:57` | `Qwen/Qwen2.5-7B-Instruct` |
| `scripts/run_axbench_e7.py:100` | `Qwen/Qwen2.5-7B-Instruct` |
| `scripts/run_axbench_ops.py:66` | `Qwen/Qwen2.5-7B-Instruct` |
| `scripts/run_axbench_conditional.py:74` | `Qwen/Qwen2.5-3B-Instruct` (cached, so merely off-spec) |

> Leaving the four `7B` defaults in place means any judged run starts a ~15 GB
> download of an uncached model. This is the top launch blocker after the interpreter.

Note the judge model id is stamped into every result row as
`instrument: "local_judge:<name>"` and into `behavior_scorer`, so changing it is
correctly recorded in the ledger — but it also means **old rows judged by 7B are not
comparable to new rows judged by 4B.** Re-baseline rather than mixing.

The Gemini API judge is separate and unaffected: `src/steering/judge.py:65`
(`DEFAULT_MODEL = "gemini-2.5-flash-lite"`, env `GEMINI_JUDGE_MODEL`, line 66).
There is **no** `STEER_JUDGE_MODEL` env var in this harness — that variable belongs to
`steering_tutorials/`, not `src/steering/`. Judge selection here is **CLI-flag driven**.

### d2. Target model → abliterated Gemma

- `src/steering/model.py:21` — `DEFAULT_MODEL = "google/gemma-3-270m-it"`.
  Change only if you want a new implicit default; `--model` overrides it everywhere,
  so a CLI flag is sufficient and lower-risk.

### d3. Hardcoded `gemma-2-2b` — where it is, and whether it matters

- `src/steering/model.py:22-26` — `SUPPORTED_MODELS = ("google/gemma-3-1b-it",
  "google/gemma-2-2b-it", "google/gemma-2-9b-it")`.
  **This tuple is dead code — it is referenced nowhere in `src/`, `scripts/`, or
  `tests/`.** It does *not* gate or validate `--model`, so an arbitrary HF id or local
  path loads fine. Updating it is cosmetic/documentation hygiene only.
- `scripts/build_provenance.py:263-268, 331-336` — a table of gemma-2-2b command
  strings. This is a **provenance record of past runs**, not a runtime code path. Do
  not "fix" it; it would falsify history.
- `src/steering/model.py:10` docstring and CLAUDE.md §2 still name gemma-2-2b as the
  standard. Worth updating for consistency with the new mandated config, but nothing
  executes off it.

**Summary: exactly one line is load-bearing (`local_judge.py:29`), plus four CLI
defaults. No hardcoded gemma-2-2b blocks the retarget.**

---

## (e) HARDWARE

```
GPU : NVIDIA GeForce RTX 4090 Laptop, 16376 MiB total / 1338 MiB used / 14711 MiB free
RAM : 6.3 GB free of 32.5 GB total
```

The GPU has headroom. **System RAM is the binding constraint** — exactly the
CLAUDE.md §17 wall (Chrome holding ~26 GB). A 4B bf16 target plus a 4B judge resident
simultaneously will not fit comfortably in 6 GB of free host RAM during load.

Operational guidance, per §17:
- Ask the user to close Chrome tabs **before** any 4B run.
- Run model jobs in the **foreground** — background GPU jobs get reaped under RAM pressure.
- Prefer `--quant 4bit` (bitsandbytes 0.49.2 is installed in anaconda base) for the 4B
  target, or stage the judge as a second pass rather than co-resident.
- Use the per-script eval caps to fit a run into one foreground window.

---

## Appendix — verification commands actually executed

All CPU-only. No model loaded, no GPU job started.

```bash
# import health, all 26 modules
$P -c "import importlib,os; [importlib.import_module('src.steering.'+m[:-3]) for m in os.listdir('src/steering') if m.endswith('.py')]"

# test suite
$P -m pytest tests -q -p no:randomly        # 261 passed in 273.66s

# CLI surfaces
PYTHONPATH=src $P -m steering.runner --help
PYTHONPATH=src $P scripts/campaign_sweep.py --help

# environment + cache
$P -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ls ~/.cache/huggingface/hub/models--*/snapshots/*/
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

*Internal QA pass — independent external review pending.*
