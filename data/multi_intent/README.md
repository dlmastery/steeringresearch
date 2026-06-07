# Multi-intent safety registry

This directory **operationalizes the "multi-intent" safety problem** that the
conditional steering method (`src/steering/cast.py`, `CASTSteerer`) targets. A
single global "refuse-everything" direction is blunt; the project's claim is that
a **gated, per-intent** policy refuses the *right* harmful requests without
over-refusing benign look-alikes (CLAUDE.md §10, the Rogue-Scalpel selectivity
axis). To test that, the method needs a concrete, enumerated set of harm intents.
That set lives in [`intents.json`](./intents.json).

## What an "intent" is here

Each intent is one harm **category** with everything the method needs:

| field | role in the method |
|---|---|
| `name` | stable key used by `CASTSteerer.add_intent(name, ...)` |
| `definition` | human description of the harm category |
| `benchmark_categories` | the matching `category` strings in the live benchmarks |
| `harmful_examples` | seed the **condition vector** (the WHEN) and **safety vector** (the WHAT) via `extract_refusal_direction(harmful, benign, layer)` |
| `benign_examples` | the contrast set for the DiffMean direction **and** the negatives that calibrate the gate's firing threshold (the over-refusal control) |
| `source` | provenance (taxonomy + license trail) |

The five intents — `weapons`, `cyber`, `self_harm`, `illicit_goods`,
`harassment` — are a defensible subset of the **SORRY-Bench**
(arXiv:2406.14598) and **HarmBench** (arXiv:2402.04249) taxonomies, chosen to be
distinct, high-severity, and well-covered by the benchmarks wired in
`src/steering/safety_bench.py`.

## Non-operational content policy

**The `harmful_examples` are REDACTED, non-operational placeholders.** Each one
*references* a benchmark category (e.g. "[BENCHMARK-SOURCED, REDACTED: a request
to write malware that exfiltrates files]") rather than reproducing any attack
recipe. They are sufficient to extract a contrast direction and to seed the
intent gate; they are **not** a source of harmful instructions. This mirrors the
bundled fallback policy in `safety_bench.py`. Real, full-strength evaluation
prompts are loaded at run time from the public benchmarks (JailbreakBench,
StrongREJECT, HarmBench, AdvBench, XSTest, SORRY-Bench) and are never committed
to this repo.

## How the driver consumes this

`scripts/run_safety_eval.py` reads `intents.json` and, per intent:

1. builds a **condition vector** at `--layer-condition` and a **safety vector**
   at `--layer-write` via `safety_target.extract_refusal_direction`;
2. fits an `IntentGate` on the pooled condition features (harmful labeled by
   intent, benign labeled `benign`) and **calibrates a per-intent firing
   threshold** at a target false-positive rate (the over-refusal knob);
3. registers the intent on a `CASTSteerer` (`add_intent`);

then evaluates the assembled method against every baseline on the live
benchmarks. See that script's module docstring for the full pipeline and the
`--dry-run` smoke path.

> STATUS: registry BUILT and consumed by the dry-run pipeline. Real-model
> extraction + gate calibration on Gemma is **PENDING** (needs GPU + HF login).
