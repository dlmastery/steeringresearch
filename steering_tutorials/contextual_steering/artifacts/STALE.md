# `results.json` in this directory is STALE as of 2026-08-30

**Do not quote it.** It was produced on 2026-07-24 with a different probe gate
than the lesson now uses, and it has not been regenerated.

## What changed, and why

`config.PROBE_PATH` used to load `hello_world/artifacts/probe.pt`. It now loads
`probe_large.pt`. Not a cosmetic swap — the two checkpoints disagree on **24–29%
of gate decisions**.

Both probes have the same architecture (128×1152 → 32 → 1) and differ only in
training corpus. Scored on identical feature sets, **neither dominates**:

| feature set | n | `probe.pt` | `probe_large.pt` | decision agreement |
|---|---|---|---|---|
| `features.npz` (JailbreakBench) | 200 | **0.9925** | 0.8315 | 0.76 |
| `features_large.npz` (toxic-chat) | 748 | 0.9033 | **0.9904** | 0.71 |

Each wins in its own domain and degrades out of it. This lesson draws prompts
from `common.data` at `N_PER_CLASS=500` — the **toxic-chat** distribution — and
was gating them with the JailbreakBench-trained probe, out of domain here.

So the fix follows from what the *consumer* sees, not from which probe has the
larger validation set. "Use the better-validated probe" would have been wrong:
on JailbreakBench features `probe_large` is worse by 0.16 AUC.

## Why it has not been regenerated: SIX failures, THREE distinct causes

Treating these as one flaky host wasted four attempts. They are different walls
with different symptoms, and the symptom identifies the wall:

| # | invocation | symptom | actual cause |
|---|---|---|---|
| 1 | background, off-family judge | segfault 139 | two models resident |
| 2 | background, off-family judge | reaped | job length |
| 3 | foreground, `n=25`, off-family | **silent exit, no traceback** | physical memory |
| 4 | foreground, `n=25`, 4-bit, off-family | **silent exit, no traceback** | physical memory |
| 5 | background, off-family, both gates checked | **`OSError 1455`** | Windows **commit** limit |
| 6 | background, off-family, both gates passing | segfault 139 | two models resident |
| 7 | foreground, `n=25`, **self-judge (one model)** | ran ~10 min, then reaped | job length only |

**The diagnostic, now recorded in CLAUDE.md §18.5:** a *silent* exit is the
physical wall; an *`OSError 1455` traceback* is the commit wall; a *segfault at
the judge load* is two-models-resident. Attempts 3 and 4 tried 4-bit and smaller
eval caps — both reduce *physical* footprint — against a limit that was not
physical.

Attempt 7 is the one that matters: with a single model the run **generates
normally on the same host in the same minute**, and only fails by exceeding the
harness's background-job window. So this is not "the host lacks memory". It is
that the off-family judge (`Qwen/Qwen2.5-3B-Instruct`, required by §17 rule 3
for any reported number) is a *second* resident model.

## The durable fix, and it already exists in this repo

`gavel/rejudge.py` solves exactly this: **three phases, one model resident at a
time** — generate with the target and cache the generations, exit; load only the
judge and grade the cache; merge. `run_contextual.py` constructs
`Judge(model, tok)` at line ~273, *before* generation, so both models are
resident for the whole run.

Splitting it the same way is the fix, and it makes the off-family judge possible
here rather than dependent on luck.

**A tiny smoke run was deliberately NOT written over `results.json`.** An
`n=10`, self-judged toy would replace a real-if-stale artifact with one that
cannot support any claim, which is worse than the honest marker you are reading.

## To regenerate

Check **both** gates first — §18.5 says both, and reading one of them is how
this rule gets violated:

```
phys >= 7.5 GB  AND  commit >= 8.5 GB
```

Then:

```
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.contextual_steering.run_contextual
```

`CTX_N_EVAL_PER_CLASS=<n>` shrinks it (added during these attempts, because
§18.5's own playbook requires every `run_*.py` to take an env cap and this
lesson had none). A capped run is **screening tier** and must be labelled so,
never quoted as the pre-registered `N_EVAL_PER_CLASS=150`.

To reproduce the OLD numbers rather than regenerate:
`CTX_PROBE_PATH=../hello_world/artifacts/probe.pt`.

Delete this file once `results.json` is regenerated with the off-family judge.
