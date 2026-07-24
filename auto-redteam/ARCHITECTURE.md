# auto-redteam -- Architecture & Design Decisions

This document is the ADR-style companion to the `README.md`. The README tells you how
to *use* the harness; this tells you *why it is shaped the way it is*, so a future
contributor can extend it without re-deriving the design.

---

## 1. Context and forces

Automated red-teaming of an LLM is a control loop: an **attacker** proposes prompts, a
**defender** responds, a **judge** scores whether a policy violation occurred, and a
**strategist** decides what to try next. The forces that shaped the design:

- **Configuration over code.** A red-teaming run is defined almost entirely by *which*
  models, strategies, judges, and stop-conditions you pick. Those choices must live in
  YAML, not in Python, so a non-author can compose a campaign.
- **Pluggability.** New providers (a new frontier model), strategies (a new published
  technique), and judges appear constantly. Adding one must not touch the core loop.
- **Runnable offline.** The package must import, test, and demo on CPU with no API keys
  and no GPU -- otherwise iteration and CI are gated on secrets and hardware.
- **Safety is structural, not advisory.** The tool discovers violations to fix them.
  The authorization gate, the no-egress attacker, and the never-log-keys rule must be
  properties of the architecture, not conventions a user can forget.
- **Reproducibility.** A finding is only actionable if the exact run can be reproduced
  and disclosed -- hence a config hash + non-secret manifest on every run.

---

## 2. Decision: an interface-driven core

**The Orchestrator depends only on the Protocols in `interfaces.py`, never on a
concrete class.** Providers, strategies, selectors, judges, persistence, and reporters
are all resolved through factories that dispatch on a config string:

```
get_provider(ModelConfig)   -> ModelProvider
get_strategy(StrategyConfig) -> AttackStrategy
get_selector(SelectionConfig, names, warm_stats) -> StrategySelector
build_judges(EvaluationConfig, provider_factory) -> list[Judge]
get_persistence(dict)       -> Persistence
```

**Consequence.** Extension is *one class + one registry entry*. The core loop is closed
for modification, open for extension. It also makes the whole thing testable with a
single deterministic `MockProvider` standing in for every model.

**Why Protocols (structural typing) over ABCs.** `typing.Protocol` with
`@runtime_checkable` lets a class satisfy an interface without importing or subclassing
it -- a provider module stays independent of the core, and duck-typed mocks in tests are
first-class. The cost (no enforced method bodies) is acceptable for a small, well-typed
surface.

---

## 3. Decision: typed models as the shared vocabulary

Every component speaks in the Pydantic v2 models of `models.py`. The object graph:

```
CampaignConfig                      CampaignResult
  |- AuthorizationConfig              |- scope (recorded for disclosure)
  |- ModelConfig  (attacker)          |- config_hash
  |- ModelConfig  (defender)          |- metrics {asr, per-category, cost, ...}
  |- SwarmConfig                      +- Trajectory[]
  |- StrategyConfig[]                       |- AttackGoal (the target behaviour)
  |- SelectionConfig                        |- messages: Message[]   (flat convo)
  |- EvaluationConfig                       +- Turn[]
  +- persistence / logging (maps)                |- attacker_prompt
                                                 |- defender_response
AttackProposal  (attacker output)               +- EvalResult {verdict, success, score}
  |- prompt / strategy / confidence
  +- critique / parent_index  (swarm/branching)
```

**Why Pydantic v2.** Validation at the config boundary (a typo in YAML fails loudly),
free (de)serialization for JSONL persistence, and a stable `config_hash()` from a
canonical `model_dump(mode="json")` -- the reproducibility anchor.

---

## 4. Decision: single-attacker is the default; swarm and bandit are optional

The two 2026 upgrades -- the **attacker swarm** (Generator + Critic) and the **learning
strategy bandit** (Thompson/UCB/epsilon-greedy) -- are both **feature-flagged off**.

**Rationale.** The reproducible, explainable baseline is a single attacker with a
static weighted strategy mix. The swarm adds cost and non-determinism; the bandit adds
online state. Making them opt-in means: (a) the baseline is always available for
comparison, (b) a regression can be bisected by flipping one flag, and (c) the simplest
correct thing runs by default. The Orchestrator branches on `attacker_swarm.enabled`
and on `selection.mode`; both paths produce the same `CampaignResult` shape.

**Bandit as a first-class abstraction.** `StrategySelector` unifies static and learning
selection behind `select` / `update` / `stats`. `fixed` and `weighted` ignore `update`;
the bandits learn from it. `stats()` is JSON-serializable so it round-trips through
`Persistence` for a resumable, warm-started bandit across campaigns.

---

## 5. Decision: hybrid, ensembled evaluation

Judging is an ensemble of `Judge`s: a fast **rule-based** judge (refusal-phrase /
compliance heuristics) and an optional **LLM-as-judge**. The `Evaluator` aggregates
their verdicts (weighted vote) into one `EvalResult`, with `success = score >= threshold
and verdict == VIOLATION`.

**Rationale.** The rule-based judge is cheap, deterministic, and keeps the offline
default meaningful; the LLM judge adds nuance when budget allows. Separating *judging*
from *success thresholding* keeps the ASR definition in one place and Goodhart-resistant
(a hedged "partial" is not a success).

---

## 6. Decision: safety as an architectural invariant

- **Authorization gate.** `Orchestrator.run` calls `banner.assert_authorized` before any
  generation. No confirmed scope (>=10 chars describing who authorized what) =>
  `AuthorizationError` => no model is ever contacted. The scope is copied into
  `CampaignResult.scope` for disclosure.
- **Secret hygiene.** Keys are read via `os.environ[api_key_env]` by *name*. Every
  provider's `get_config()` returns provider/model/temperature only. Persistence writes
  trajectories and stats, never keys.
- **No attacker egress.** The local attacker provider talks only to its configured model
  server (`base_url`); it has no other network reach.
- **Mechanics, not payloads.** Strategy modules encode *published* techniques as
  scaffolds seeded from the user's goals YAML. There are no baked-in working exploits,
  and the shipped example goals are benign placeholders.

These are properties of the code paths, not documentation the operator must remember.

---

## 7. Decision: lazy imports and a thin CLI

`cli.py` imports only Typer at module load; config, orchestrator, providers, and yaml
are imported *inside* the command bodies. Providers import their SDKs
(`google-genai`/`openai`/`anthropic`/`ollama`) lazily inside `generate`/`__init__`.

**Rationale.** `import autoredteam.cli`, `--help`, completion, and `version` stay fast
and dependency-free; installing an optional extra is only required for the provider you
actually use. The package imports and tests on a bare CPU environment with just the core
deps.

---

## 8. Data flow of one campaign

```
load_config (default.yaml < --config < env < --set)  -> CampaignConfig
        |
Orchestrator(config): build providers/selector/evaluator/persistence via factories
        |
assert_authorized(config.authorization)  -> scope   (or AuthorizationError)
        |
for each AttackGoal  (up to max_parallel concurrently):
    TrajectoryManager starts a Trajectory
    loop up to max_turns_per_trajectory:
        selector.select(strategies)                    -> strategy name
        attacker/swarm.propose(...)                    -> AttackProposal[]
        [dry-run stops here: emit prompts, no defender call]
        defender.respond(messages)                     -> GenerationResult
        evaluator.evaluate(goal, prompt, response)     -> EvalResult
        record Turn; check stop_on_success / threshold
    persistence.save_trajectory(...)                   (checkpoint)
    selector.update(strategy, reward, turns)           (bandit learns)
compute_metrics(result); persistence.save_result / save_stats
reporter.render(result, out_dir)                       -> md / html / csv
```

`--dry-run` short-circuits after proposal generation: the attacker/strategy machinery
runs and the intended prompts are surfaced, but **no defender is contacted** -- a cheap
preview before spending API budget or touching a live target.

---

## 9. Extension points (summary)

| Extend | Implement (interface) | Register |
|---|---|---|
| Provider | `ModelProvider` | `providers/__init__.py::PROVIDERS` + `get_provider` |
| Strategy | `AttackStrategy` | `strategies/__init__.py::STRATEGIES` + `get_strategy` |
| Selector | `StrategySelector` | `selection.py::get_selector` |
| Judge | `Judge` | `evaluator.py::build_judges` |
| Persistence | `Persistence` | `persistence.py::get_persistence` |
| Reporter | `Reporter` | `reporting.py` |

Adhere to the `models.py` types at the boundaries and the Orchestrator will pick your
component up from config with no core change.

---

## 10. Known limitations / non-goals (this phase)

- Trajectories are per-goal and (for branching) shallow trees; deep agentic tool-use
  loops are Phase 2.
- The rule-based judge is a heuristic; calibrated LLM-judge ensembles with human
  spot-checks are Phase 3.
- Reporting is file-based (md/html/csv); the interactive drill-down dashboard is Phase 5.
- This is a research harness, not a hardened service: it assumes a trusted local
  operator running authorized tests, not a multi-tenant deployment.
