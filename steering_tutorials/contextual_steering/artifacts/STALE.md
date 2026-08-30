# `results.json` in this directory is STALE as of 2026-08-30

**Do not quote it.** It was produced on 2026-07-24 with a different probe gate
than the lesson now uses, and it has not been regenerated.

## What changed

`config.PROBE_PATH` used to load `hello_world/artifacts/probe.pt`. It now loads
`probe_large.pt`. That is not a cosmetic swap — the two checkpoints disagree on
**24–29% of gate decisions**.

The reason for the change, measured rather than assumed. Both probes have the
same architecture (128×1152 → 32 → 1) and differ only in training corpus, and
**neither dominates**:

| feature set | n | `probe.pt` | `probe_large.pt` | decision agreement |
|---|---|---|---|---|
| `features.npz` (JailbreakBench) | 200 | **0.9925** | 0.8315 | 0.76 |
| `features_large.npz` (toxic-chat) | 748 | 0.9033 | **0.9904** | 0.71 |

Each wins in its own domain and degrades out of it. This lesson draws prompts
from `common.data` at `N_PER_CLASS=500` — the **toxic-chat** distribution — and
was gating them with the JailbreakBench-trained probe, which is out of domain
here and scores 0.9033 against 0.9904 on exactly that data.

So the fix follows from what the *consumer* sees, not from which probe has the
larger validation set. "Use the better-validated probe" would have been wrong:
on JailbreakBench features `probe_large` is worse by 0.16 AUC.

## Why it has not been re-run

Four attempts on 2026-08-30, all failing at the model-load boundary:

1. background — segfault, exit 139
2. background — reaped
3. foreground, `CTX_N_EVAL_PER_CLASS=25` — silent exit, no traceback
4. foreground, `CTX_N_EVAL_PER_CLASS=25 STEER_LOAD_4BIT=1` — silent exit

Host state measured immediately after: **physical 7.47 GB against the 7.5 GB
gate** recorded in `CLAUDE.md` §18.6, with Chrome holding 10.0 GB. A silent exit
with no traceback at this point is that section's documented signature for
physical memory exhausted mid-mmap of the judge shards — the off-family judge
(`Qwen/Qwen2.5-3B-Instruct`, required by §17 rule 3 for any reported number)
loads immediately after the target model.

This is a host block, not a defect in the change. `CTX_N_EVAL_PER_CLASS` was
added to `config.py` during these attempts, because §18.5's own playbook requires
every `run_*.py` to take an env cap so an eval fits one foreground window and
this lesson had none.

## To regenerate

With physical memory above ~7.5 GB (close browser tabs):

```
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.contextual_steering.run_contextual
```

Add `CTX_N_EVAL_PER_CLASS=<n>` to shrink it — but a capped run is **screening
tier** and must be labelled as such, never quoted as the pre-registered
`N_EVAL_PER_CLASS=150`.

To reproduce the old numbers instead of regenerating, pin the old gate
explicitly: `CTX_PROBE_PATH=../hello_world/artifacts/probe.pt`.

Delete this file once `results.json` is regenerated.
