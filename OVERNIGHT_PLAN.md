# Overnight autonomous research campaign (Gemma-3-270m-it, RTX 4090 16GB)

> Started 2026-05-30 night. Full autonomy. Each campaign = a pre-registered
> hypothesis tested by an in-process sweep (load-once via load_model_cached),
> logged as canonical experiments, committed, written up. All n=1 SCREENING
> unless escalated. Lean into GEOMETRY/COHERENCE hypotheses — they are testable
> with the current harness and do NOT depend on the weak synthetic behavior
> proxy or a judge.

## Why geometry-first
The behavior scorer is a synthetic-lexicon concept-incorporation proxy (weak,
not judge-calibrated). PPL, off-shell Δ‖h‖, effective-rank, and the cliff are
REAL measurements on real Gemma. The highest-fidelity overnight findings are the
geometric/coherence ones (E2, E3, N5, N17, N20, E27/rotation).

## Campaign queue (sequenced)

| # | id | hypothesis | sweep | primary readout | status |
|---|----|-----------|-------|-----------------|--------|
| C1 | E2 ✓FALSIFIED | optimal layer = max linear separability; but best BEHAVIOR layer may differ from max-Fisher | layer ∈ {2,4,6,8,10,12,14,16}, α=2, add | behavior & PPL & offshell vs layer | DONE |
| C2 | N17/N5 | off-shell Δ‖h‖ (not raw α) predicts incoherence; PPL collapses onto one curve vs ‖Δh‖/‖h‖ | reuse C1+E3 rows | corr(offshell, logPPL); data-collapse | queued |
| C3 | E27/E23 | rotation preserves coherence better than addition on small models at matched displacement | op ∈ {add, rotate, project_out}, layer=best(C1), α grid | PPL @ matched offshell | queued |
| C4 | N20 | per-layer effective-rank drop predicts that layer's fragility (cheap, behavior-free) | layer sweep, measure eff-rank drop vs PPL slope | corr(effrank_drop, cliff steepness) | queued |
| C5 | E1 | DiffMean from ≥N pairs reaches ≥90% asymptote | paircount ∈ {2,5,10} (limited by data) | vector-cosine-to-full vs N | queued |
| C6 | — | gemma-3-1b-it standard-rung reproduction of E3/E4 | download 1b, re-run E3 | cross-scale | queued |

## Discipline
- One config axis per experiment; pre-author reasoning; honest SCREENING tags.
- Commit + push after each campaign; regenerate dashboard.
- Update FINDINGS (S-n) + IDEA_TABLE status + ledger per campaign.
- If a sweep crashes, diagnose (don't retry blindly); record the failure.
- Geometry numbers are real; behavior numbers carry the proxy caveat.


## Completion (all campaigns run)

C1–C6 complete (exp#20–59). Verdicts: 3 SUPPORTED (N5, N16/CRH, N17), 2 FALSIFIED (E2, E27-rotation), 1 DIRECTIONAL (E1), 1 INCONCLUSIVE (N20), plus cross-scale E27/E4 SUPPORTED. All SCREENING (n=1). See `ideas/_campaigns/` + FINDINGS S-1..S-8.
