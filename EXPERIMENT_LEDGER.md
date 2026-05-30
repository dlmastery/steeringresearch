# EXPERIMENT LEDGER — Promotion / Demotion Log

> **Role:** The append-only record of every experiment run, its rung outcome,
> its five-axis metrics, and its verdict. This file is the audit trail that
> connects hypotheses (IDEA_TABLE.md) to external-ready findings (FINDINGS.md).
>
> **Editing discipline:**
> - Rows are APPEND-ONLY. Never delete or modify a completed row.
> - Pre-register the success criterion in the relevant `ideas/<NN>/IDEA.md`
>   BEFORE adding a row for that experiment.
> - The composite formula is fingerprinted; changing it without updating the
>   fingerprint in `src/steering/eval.py` is a protocol violation.
> - A `failure_reason` is REQUIRED for any verdict that is not KEEP.
> - `n` must be recorded in every numeric cell; bare numbers are not allowed.

---

## Column schema

| Column | Type | Notes |
|--------|------|-------|
| `experiment_num` | integer | Sequential, globally unique across all runs |
| `tag` | string | Short human tag, e.g. `E1-v1-smoke` |
| `hypothesis_id` | string | E1, E2, N5, etc. from IDEA_TABLE.md |
| `rung` | 0/1/2/3/4 | Ladder rung attempted (UNIT/SMOKE/DEV/STANDARD/FULL) |
| `behavior` | string | Which behavior/concept was tested |
| `dMMLU` | float | MMLU delta in percentage points (negative = capability drop); format `X.XX pp (n=Y)` |
| `PPL` | float | Perplexity on held-out set; format `X.XX (n=Y)` |
| `CR_jailbreak` | float | JailbreakBench Compliance Rate in %; 0% is required for KEEP on safety experiments |
| `over_refusal` | float | Harmless-input refusal rate in % (XSTest-style); lower is better |
| `composite` | float | Composite metric to 4 dp; formula: behavior_efficacy - λ_cap*max(0,MMLU_drop) - λ_coh*max(0,ΔPPL_norm) - λ_safe*CR - λ_sel*max(0,harmless_refusal) - λ_geo*max(0,offshell_displacement) |
| `verdict` | enum | KEEP / DISCARD / NEAR-MISS / promote / demote |
| `failure_reason` | string | Required if verdict != KEEP; cite the specific metric and threshold that failed |

**Composite formula fingerprint (SHA-256 of formula string in `src/steering/eval.py:COMPOSITE_FORMULA`):**
`[TO BE FILLED AFTER src/steering/eval.py IS WRITTEN — placeholder a9001e87087e]`

**Off-manifold geometry columns** (always log alongside the five axes):

| Column | Notes |
|--------|-------|
| `delta_norm` | Off-shell displacement Δ\|\|h\|\| (mean over tokens/prompts) |
| `eff_rank_drop` | Effective-rank drop at injection layer vs unsteered |
| `norm_budget` | Cumulative \|\|Δh\|\|/\|\|h\|\| (N5 budget fraction used) |
| `part_ratio` | Participation ratio at injection layer (N3) |

---

## Experiment log

| experiment_num | tag | hypothesis_id | rung | behavior | dMMLU | PPL | CR_jailbreak | over_refusal | composite | delta_norm | eff_rank_drop | norm_budget | part_ratio | verdict | failure_reason |
|----------------|-----|---------------|------|----------|-------|-----|-------------|-------------|-----------|------------|---------------|-------------|------------|---------|----------------|
| 0 | EXAMPLE-DO-NOT-USE | E1 | 1 | refusal | -0.50 pp (n=3) | 12.34 (n=3) | 0.0% | 2.1% | 0.7823 | 0.021 | 0.003 | 0.18 | 412 | KEEP | — |
| 1 | rung1_plumbing_fakelm | INFRA | 1 | happiness (proxy) | -0.05 pp (n=1) | 32.00 (n=1) | 0.0% | 0.0% | 0.4485 | 0.010 | — | — | — | KEEP | — (infra/plumbing on FakeLM, NOT a Gemma steering claim; Rung-0 gate cleared: intervention perturbs stream, state restores, full loop + 3-tier dashboard execute end-to-end) |

> **Row 0 is an EXAMPLE only.** It shows the column format and is NOT a real
> experimental result. **Row 1 is the Rung-0/1 plumbing gate on the offline
> FakeResidualLM** — it validates the harness + loop, not any Gemma behavior.
> The first scientific steering rows (E1–E3 on Gemma-3-1B-it) begin once Gemma
> access is unlocked (see `NEXT_STEPS.md`). All real Gemma rows are tagged with
> their `Ennn`/`Nnn` hypothesis_id, never `INFRA`.

---

## Promotion ladder summary

*(Auto-updated when a method reaches a new rung gate)*

| Method / hypothesis | Best rung reached | Gate cleared | Last composite | Notes |
|--------------------|-------------------|-------------|---------------|-------|
| *(none yet — program initialized 2026-05-30)* | — | — | — | — |

---

## Demotion log

*(Methods discarded at any rung; ordered by experiment_num)*

| experiment_num | tag | hypothesis_id | rung | failure_reason | date |
|----------------|-----|---------------|------|---------------|------|
| *(none yet)* | — | — | — | — | — |

---

> Program initialized 2026-05-30. No experiments run yet.
