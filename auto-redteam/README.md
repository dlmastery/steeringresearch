# auto-redteam

**A configurable, end-to-end automated red-teaming harness for authorized AI-safety
research.** An ablated/uncensored *local* Gemma model plays the adaptive attacker
(prompt generator, mutator, strategist) against a fully pluggable frontier
*defender* (Gemini, any OpenAI-compatible endpoint, Anthropic, or a local model).
Single-turn, multi-turn, and simple agentic trajectories are supported, with
reproducible runs, a learning strategy bandit, an optional attacker swarm, and rich
metrics mapped to a harm taxonomy.

---

> ## Authorized use / research only
>
> **This harness generates adversarial prompts for the express purpose of DISCOVERING,
> REPORTING, and helping FIX policy violations in LLMs. It is a defensive tool.**
>
> - Use it **only** against models and endpoints you are **explicitly authorized** to test.
> - Every campaign is gated: it will **refuse to launch** unless `authorization.confirmed`
>   is `true` and an `authorization.scope` string (who authorized this, against what
>   target) is supplied. That scope is recorded in the run manifest for disclosure.
> - The attacker has **no network egress** beyond its configured model server.
> - Defender API keys are read from the environment **by name only** and are **never
>   logged** or written to any artifact.
> - Findings are for **defensive remediation**. Do not use discovered attacks to cause harm.
>
> The bundled strategy modules implement *published* techniques (Crescendo escalation,
> genetic mutation, Tree-of-Attacks branching) as **mechanics/scaffolds** only -- there
> are no baked-in working exploits. Real attack seeds come from *your* goals YAML at
> runtime.

---

## Architecture at a glance

```
                          +--------------------------------------------------+
                          |                 Orchestrator                     |
                          |  (async loop, concurrency cap, stop conditions)  |
                          +--------------------------------------------------+
                             |            |             |              |
              authorization  |   selects  |   scores    |   persists   |
                 gate  ------>|  strategy  |  response   |  trajectory  |
                             v            v             v              v
   +--------------+   +--------------+   (drives)   +----------+   +-------------+
   |   Attacker   |   |  Strategy    |              |  Judge   |   | Persistence |
   | (local Gemma |   |  Selector    |              | ensemble |   | jsonl /     |
   |  provider)   |   | fixed | bandit|             | rule +   |   | sqlite      |
   +------+-------+   | (Thompson,   |              | LLM-judge|   +------+------+
          |           |  UCB, e-grdy)|              +----+-----+          |
          | prompt    +------+-------+                   ^                | stats
          v                  | strategy name             | verdict/score | (bandit
   +--------------+          |                            |                |  resume)
   |  AttackGoal  |          +----------------------------+                v
   |  (taxonomy)  |                                                  +-----------+
   +--------------+          [ optional ] AttackerSwarm             |  Reporter |
          |                  +-------------------------+            | md/html/  |
          v                  | Generator + Critic ->   |            | csv       |
   +--------------+          |  N branching proposals  |            +-----------+
   |   Defender   |<---------+-------------------------+
   | (Gemini / any|   attacker prompt(s)
   |  frontier)   |
   +--------------+
```

The **Orchestrator depends only on Protocol interfaces** (`autoredteam.interfaces`);
every concrete provider, strategy, judge, selector, persistence backend, and reporter
is pluggable behind them. The optional **attacker swarm** (Generator + Critic) and the
learning **Thompson-sampling bandit** are the two SOTA upgrades -- both are
feature-flagged and off by default, so the classic single-attacker loop is the baseline.

---

## Quickstart

```bash
# 1. Install (CPU-only core; real providers are optional extras)
pip install -e .[dev]

# 2. Run the offline default -- MOCK attacker vs MOCK defender, no API keys, no GPU.
#    (The mock defender refuses obviously-unsafe asks and complies otherwise, so the
#    harness, judges, metrics, and reports are all exercised end-to-end offline.)
auto-redteam run

# 3. Preview the attacker/strategy loop WITHOUT calling any defender:
auto-redteam run --dry-run

# 4. Inspect a config's reproducibility manifest without running anything:
auto-redteam validate --config config/gemma_vs_gemini.yaml

# 5. List the registered attack strategies:
auto-redteam strategies
```

### The real example: local Gemma attacker vs Gemini defender

`config/gemma_vs_gemini.yaml` wires a local (Ollama/OpenAI-compatible) ablated Gemma
attacker against a Gemini defender, with the attacker swarm and the Thompson bandit
turned on -- the SOTA demo. To launch it you must (a) provide the defender key in the
environment and (b) affirm authorization:

```bash
# Key is read by NAME (api_key_env: GOOGLE_API_KEY) and never logged.
export GOOGLE_API_KEY="...your key..."

auto-redteam run \
  --config config/gemma_vs_gemini.yaml \
  --set authorization.confirmed=true \
  --set 'authorization.scope=internal safety eval of our own Gemini deployment, ticket SAFE-1234'
```

Without a confirmed scope the run aborts with an `AuthorizationError` -- by design.

> **Optional provider extras** -- install only what you use:
> `pip install -e .[gemini]` (google-genai), `.[openai]`, `.[anthropic]`, `.[local]`
> (ollama), `.[sql]` (sqlite/sqlalchemy persistence), or `.[all]`.

---

## Command reference

| Command | Purpose |
|---|---|
| `auto-redteam run [--config Y] [--set a.b=val ...] [--dry-run] [--out DIR]` | Print the banner, build the Orchestrator from the (deep-merged) config, and run the campaign. `--dry-run` = attacker-only preview, **no defender calls**. |
| `auto-redteam validate [--config Y] [--set a.b=val ...]` | Load + deep-merge the config and print the reproducibility manifest (config hash + non-secret summary). Runs nothing. |
| `auto-redteam strategies` | List the attack strategies registered in the harness. |
| `auto-redteam version` | Print the version. |

`--set` overrides use **dotted keys** and are repeatable, e.g.
`--set max_parallel=8 --set selection.mode=thompson`. Values are coerced with YAML
scalar rules (`true`->bool, `8`->int, bare text->str).

---

## Configuration reference

Configs are layered by **increasing precedence**:

```
autoredteam/../config/default.yaml   <   --config YAML   <   HARNESS__A__B=.. env   <   --set CLI flags
```

Every top-level key in `config/default.yaml`:

| Key | Type | Meaning |
|---|---|---|
| `campaign` | str | Human-readable campaign name; used in report/persistence paths and the manifest. |
| `seed` | int | Master RNG seed -- seeds the selector and any stochastic strategy for reproducibility. |
| `max_parallel` | int | Max concurrent trajectories (asyncio concurrency cap; one per goal). |
| `max_turns_per_trajectory` | int | Turn budget per goal before the trajectory stops. |
| `stop_on_success` | bool | Stop a trajectory as soon as a violation is found (vs. exhausting the turn budget). |
| `time_budget_s` | float\|null | Optional wall-clock budget for the whole campaign. |
| `authorization` | map | `confirmed` (bool) + `scope` (str). **Required to launch** -- see the safety note above. |
| `attacker` | ModelConfig | The adaptive attacker model (an ablated local model in real runs). |
| `defender` | ModelConfig | The target under test (a pluggable frontier model). `api_key_env` names the env var; never the key. |
| `attacker_swarm` | map | `enabled`, `generator`, `critic_enabled`, `critic`, `max_proposals_per_turn`. Off -> classic single attacker. |
| `strategies` | list | Which attack strategies to mix, each with a `weight` and per-strategy `config` knobs. |
| `selection` | map | How the next strategy is chosen: `mode` (`fixed`/`weighted`/`epsilon_greedy`/`ucb`/`thompson`), `epsilon`, `ucb_c`, `update_after_every`, `persist_stats`, `initial_weights`. |
| `evaluation` | map | `taxonomy`, `success_threshold`, and the `judges` ensemble (rule-based and/or LLM-as-judge). |
| `goals_path` | str | Path to the goals YAML (the behaviours to elicit, each tagged to a taxonomy category). |
| `persistence` | map | `backend` (`jsonl`/`sqlite`) + `path` template (`{campaign}` is substituted). |
| `logging` | map | `level`, `trajectory_format`, `save_full_prompts`, `metrics_export`. |

A **`ModelConfig`** (`attacker` / `defender` / judge / swarm models) carries:
`provider` (`local_gemma`/`gemini`/`openai_compatible`/`anthropic`/`mock`), `model`,
`base_url`, `api_key_env` (env var **name**), `temperature`, `max_tokens`,
`system_prompt` or `system_prompt_path`, `tools`, and free-form `extra`.

---

## Pluggable interfaces

Everything the Orchestrator touches is a Protocol in `autoredteam/interfaces.py`.
**Adding a capability is one class + one registry entry -- no core changes:**

| To add a... | Implement | Register in |
|---|---|---|
| Model provider | `interfaces.ModelProvider` (`name`, async `generate`, `get_config`) | `providers/__init__.py::PROVIDERS` |
| Attack strategy | `interfaces.AttackStrategy` (`generate_initial`, `mutate`, `should_continue`) | `strategies/__init__.py::STRATEGIES` |
| Strategy selector | `interfaces.StrategySelector` (`select`, `update`, `stats`) | `selection.py::get_selector` |
| Judge | `interfaces.Judge` (async `evaluate`) | `evaluator.py::build_judges` |
| Persistence backend | `interfaces.Persistence` | `persistence.py::get_persistence` |
| Reporter | `interfaces.Reporter` (`render`) | `reporting.py` |

The typed models in `autoredteam/models.py` (Pydantic v2) are the shared vocabulary
every component speaks: `AttackGoal`, `AttackProposal`, `Turn`, `Trajectory`,
`EvalResult`, `CampaignConfig`, `CampaignResult`.

---

## SOTA features (behind flags)

**Attacker swarm** (`attacker_swarm.enabled: true`). Instead of a single attacker, a
**Generator** proposes attacks and an optional **Critic** reflects on the last
defender response to steer the next mutation -- a structured (not free-form) exchange.
Set `max_proposals_per_turn > 1` to enable Tree-of-Attacks-style branching.

**Learning strategy bandit** (`selection.mode: thompson` | `ucb` | `epsilon_greedy`).
Rather than a static weighted mix, the harness *learns online* which strategy works
against the current defender, updating per-strategy Beta-Bernoulli (Thompson) or
UCB statistics after each trajectory. With `persist_stats: true` the learned state is
checkpointed and warm-starts the next campaign (resumable bandit).

Both default to **off** so the reproducible baseline is the classic single-attacker,
statically-weighted loop.

---

## Reproducibility

- Every config produces a stable **SHA-256 `config_hash`** (`CampaignConfig.config_hash()`),
  recorded in the run manifest and the `CampaignResult`. `auto-redteam validate` prints it.
- The **master `seed`** seeds the strategy selector and stochastic strategies.
- The **manifest** (`config_manifest`) is a non-secret summary: hashes, model
  provider/model names, swarm/selection modes -- enough to reproduce a run, with **no keys**.
- Mock providers are deterministic and seedable, so the default campaign and the test
  suite are byte-stable across machines.

---

## Safety and isolation

- **Authorization gate** -- `banner.assert_authorized` runs before *any* generation;
  no confirmed scope -> `AuthorizationError`, no model is ever contacted.
- **Keys never logged** -- read from `os.environ[api_key_env]` by name; provider
  `get_config()` strips secrets (provider/model/temperature only). Persistence never
  writes keys.
- **No attacker egress** -- the local attacker talks only to its configured model server.
- **Offline by default** -- the mock provider makes the whole package runnable with no
  network and no GPU (also how CI and the tests run).
- **Mechanics, not exploits** -- strategy modules are published-technique scaffolds
  seeded from your goals YAML; the shipped example goals are benign placeholders.

---

## Roadmap

- **Phase 0-1 (this release)** -- typed core, interfaces, mock + real providers,
  strategies, selectors/bandit, evaluator ensemble, persistence, orchestrator, CLI, docs.
- **Phase 2** -- richer multi-turn / agentic trajectories and tool-use defenders.
- **Phase 3** -- expanded harm taxonomies + calibrated LLM-judge ensembles with human spot-checks.
- **Phase 4** -- full attacker-swarm variants (multi-critic, debate) and TAP tree search.
- **Phase 5** -- a self-contained HTML dashboard (per-campaign, per-goal, per-turn drill-down).
- **Phase 6** -- continuous regression harness: track defender ASR across model versions over time.

---

## Project layout

```
autoredteam/
  __init__.py        banner re-exports + version
  banner.py          authorized-use gate (assert_authorized)
  models.py          Pydantic v2 typed vocabulary
  interfaces.py      the pluggable Protocols
  config.py          load_config / load_goals / config_manifest
  taxonomy.py        harm-taxonomy loader
  providers/         mock + gemini + openai_compat + anthropic + local_gemma
  strategies/        single_turn, crescendo, mutation_loop, tree_of_attacks
  selection.py       weighted + epsilon_greedy + ucb + thompson selectors
  attacker.py        single-attacker wrapper
  defender.py        defender wrapper
  swarm.py           Generator+Critic attacker swarm
  conversation.py    trajectory / context-window manager
  evaluator.py       rule-based + LLM judges + ensemble
  metrics.py         ASR + per-category + cost metrics
  persistence.py     jsonl + sqlite backends
  reporting.py       markdown + html + csv reporters
  orchestrator.py    the async campaign loop
  cli.py             the `auto-redteam` entrypoint
config/              default.yaml + gemma_vs_gemini.yaml + goals + taxonomies
prompts/             attacker_system / generator / critic / judge templates
tests/               deterministic, mock-only, CPU test suite
```

See `ARCHITECTURE.md` for the design rationale (interface-driven core, why the swarm
and bandit are optional, the data-model graph, and the extension points).
