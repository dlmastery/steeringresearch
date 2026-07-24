"""orchestrator.py -- the campaign driver.

The Orchestrator ties everything together while depending ONLY on the Protocol
interfaces (never a concrete class). It:

  1. enforces the authorization gate (``banner.assert_authorized``) BEFORE any
     generation, recording the returned scope on the result,
  2. builds the pluggable pieces -- providers, strategies, selector, evaluator,
     persistence -- via their factories (imported lazily so this module stays
     import-light), unless they were injected for testing,
  3. runs each goal's adaptive trajectory (multi-turn, with stop conditions),
     bounded to ``config.max_parallel`` concurrent trajectories via asyncio,
  4. supports BOTH the classic single-attacker loop and the optional Generator+
     Critic swarm (branch on ``config.attacker_swarm.enabled``),
  5. updates the strategy selector after each trajectory (so bandit modes learn),
     checkpointing each finished trajectory + selector stats through persistence,
  6. computes campaign metrics and returns a ``CampaignResult``.

Dependency injection: any of ``defender / attacker / swarm / strategies / selector
/ evaluator / persistence / provider_factory`` may be passed to ``__init__``; each
one left as ``None`` is built from config in ``run``. This is what lets the
``__main__`` demo run a full campaign with inline stubs and no sibling files.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable

from . import banner
from .interfaces import (
    AttackerSwarm,
    Judge,
    ModelProvider,
    Persistence,
    StrategySelector,
)
from .models import (
    AttackGoal,
    CampaignConfig,
    CampaignResult,
    EvalResult,
    ModelConfig,
    Trajectory,
    Verdict,
)


class _RunCtx:
    """Per-``run`` bundle of built components (kept off ``self`` so one Orchestrator
    instance could, in principle, run more than once without cross-talk)."""

    strategies: dict[str, Any]
    available: list[str]
    selector: StrategySelector
    evaluator: Any
    persistence: Persistence
    defender: Any
    attacker: Any
    swarm: AttackerSwarm | None
    selector_lock: asyncio.Lock
    start: float
    time_budget: float | None
    max_context: int | None
    summarize: bool


class Orchestrator:
    """Drives a whole red-team campaign against one defender."""

    def __init__(
        self,
        config: CampaignConfig,
        *,
        defender: Any = None,
        attacker: Any = None,
        swarm: AttackerSwarm | None = None,
        strategies: dict[str, Any] | None = None,
        selector: StrategySelector | None = None,
        evaluator: Any = None,
        persistence: Persistence | None = None,
        judges: list[Judge] | None = None,
        provider_factory: Callable[[ModelConfig], ModelProvider] | None = None,
    ) -> None:
        self.config = config
        self._defender = defender
        self._attacker = attacker
        self._swarm = swarm
        self._strategies = strategies
        self._selector = selector
        self._evaluator = evaluator
        self._persistence = persistence
        self._judges = judges
        self._provider_factory = provider_factory

    # ============================================================= public API ==
    async def run(self, goals: list[AttackGoal], dry_run: bool = False) -> CampaignResult:
        """Run the campaign and return its result.

        The authorization gate fires FIRST -- before any provider is even built --
        so an unauthorized config never reaches a model.

        ``dry_run=True`` is the CLI's attacker-only preview: the attacker (or swarm)
        authors the opening prompt for each goal, but the DEFENDER and the judge
        ensemble are never built or called. Each trajectory gets a single turn with
        a placeholder defender response so operators can inspect what *would* be sent
        without touching the target model (and without needing its API key).
        """
        scope = banner.assert_authorized(self.config.authorization.model_dump())
        print(banner.BANNER)
        if dry_run:
            print("[dry-run] attacker-only preview: the defender will NOT be called.")

        started_at = datetime.now(timezone.utc).isoformat()
        ctx = self._build_ctx(dry_run=dry_run)

        # Bound concurrency to a single 4090's comfort zone (or whatever config says).
        sem = asyncio.Semaphore(max(1, self.config.max_parallel))

        async def _guarded(goal: AttackGoal) -> Trajectory:
            async with sem:
                return await self._run_trajectory(goal, ctx, dry_run=dry_run)

        trajectories = await asyncio.gather(*(_guarded(g) for g in goals))
        finished_at = datetime.now(timezone.utc).isoformat()

        result = CampaignResult(
            campaign_name=self.config.name,
            config_hash=self.config.config_hash(),
            scope=scope,
            trajectories=list(trajectories),
            started_at=started_at,
            finished_at=finished_at,
            meta={
                "swarm_enabled": self.config.attacker_swarm.enabled,
                "selection_mode": self.config.selection.mode,
                "selector_stats": _safe(lambda: ctx.selector.stats(), {}),
                "dry_run": dry_run,
            },
        )
        result.metrics = self._compute_metrics(result)

        # Final campaign checkpoint (best-effort; a persistence hiccup must not lose
        # the in-memory result the caller already holds).
        _safe(lambda: ctx.persistence.save_result(result), None)
        return result

    # ========================================================= construction ==
    def _build_ctx(self, dry_run: bool = False) -> _RunCtx:
        """Build every pluggable component, using injected ones where provided.

        All sibling-factory imports are LAZY (inside this method), so importing
        ``autoredteam.orchestrator`` never pulls in providers/strategies/etc.

        In ``dry_run`` the defender and the judge ensemble are skipped entirely --
        neither is built, so a preview needs only the attacker's provider.
        """
        cfg = self.config
        provider_factory = self._provider_factory
        if provider_factory is None and self._needs_providers(dry_run):
            from .providers import get_provider  # lazy

            provider_factory = get_provider

        ctx = _RunCtx()

        # -- persistence (built first: the selector may warm-start from its stats) --
        if self._persistence is not None:
            ctx.persistence = self._persistence
        else:
            from .persistence import get_persistence  # lazy

            ctx.persistence = get_persistence(cfg.persistence)

        # -- strategies --
        if self._strategies is not None:
            ctx.strategies = self._strategies
        else:
            from .strategies import get_strategy  # lazy

            ctx.strategies = {sc.name: get_strategy(sc) for sc in cfg.strategies}
        ctx.available = list(ctx.strategies.keys())

        # -- selector (optionally warm-started from persisted bandit stats) --
        if self._selector is not None:
            ctx.selector = self._selector
        else:
            from .selection import get_selector  # lazy

            warm = None
            if cfg.selection.persist_stats:
                warm = _safe(lambda: ctx.persistence.load_stats(cfg.name), None) or None
            ctx.selector = get_selector(cfg.selection, ctx.available, warm, seed=cfg.seed)

        # -- evaluator (judge ensemble) -- skipped in dry-run (nothing to judge) --
        if dry_run:
            ctx.evaluator = None
        elif self._evaluator is not None:
            ctx.evaluator = self._evaluator
        else:
            from .evaluator import Evaluator, build_judges  # lazy

            judges = self._judges or build_judges(cfg.evaluation, provider_factory)
            ctx.evaluator = Evaluator(judges, cfg.evaluation.success_threshold)

        # -- defender -- skipped in dry-run (the target is never contacted) --
        if dry_run:
            ctx.defender = None
        elif self._defender is not None:
            ctx.defender = self._defender
        else:
            from .defender import Defender  # lazy (sibling, but ours)

            ctx.defender = Defender(
                provider_factory(cfg.defender),
                system_prompt=cfg.defender.system_prompt,
                tools=cfg.defender.tools,
            )

        # -- attacker OR swarm (mutually exclusive per turn) --
        if cfg.attacker_swarm.enabled:
            ctx.attacker = None
            ctx.swarm = self._swarm or self._build_swarm(provider_factory)
        else:
            ctx.swarm = None
            if self._attacker is not None:
                ctx.attacker = self._attacker
            else:
                from .attacker import Attacker  # lazy (ours)

                ctx.attacker = Attacker(
                    provider_factory(cfg.attacker),
                    system_prompt=cfg.attacker.system_prompt,
                )

        # -- run-scoped scheduling state --
        ctx.selector_lock = asyncio.Lock()
        ctx.start = time.monotonic()
        ctx.time_budget = cfg.time_budget_s
        ctx.max_context = _as_int(cfg.logging.get("max_context_messages"))
        ctx.summarize = bool(cfg.logging.get("summarize_context", False))
        return ctx

    def _build_swarm(self, provider_factory: Callable[[ModelConfig], ModelProvider]) -> AttackerSwarm:
        from .swarm import AttackerSwarm as _Swarm  # lazy (ours)

        sc = self.config.attacker_swarm
        gen_cfg = sc.generator or self.config.attacker
        generator = provider_factory(gen_cfg)
        critic = None
        if sc.critic_enabled:
            critic_cfg = sc.critic or self.config.attacker
            critic = provider_factory(critic_cfg)
        return _Swarm(
            generator,
            critic,
            critic_enabled=sc.critic_enabled,
            max_proposals_per_turn=sc.max_proposals_per_turn,
            generator_system=gen_cfg.system_prompt,
            critic_system=(sc.critic.system_prompt if sc.critic else gen_cfg.system_prompt),
        )

    def _needs_providers(self, dry_run: bool = False) -> bool:
        """Whether any component still to be built will need a real provider."""
        cfg = self.config
        if not dry_run:
            # The defender + judge ensemble are only built for a real run.
            if self._evaluator is None:  # judges may include an llm_judge
                return True
            if self._defender is None:
                return True
        # The attacker/swarm provider is needed even in dry-run (it authors prompts).
        if cfg.attacker_swarm.enabled:
            return self._swarm is None
        return self._attacker is None

    # ============================================================ per-goal ==
    async def _run_trajectory(
        self, goal: AttackGoal, ctx: _RunCtx, dry_run: bool = False
    ) -> Trajectory:
        """Run one adaptive, multi-turn attack against a single goal."""
        from .conversation import TrajectoryManager  # lazy (ours)

        # One strategy per trajectory (crescendo/mutation are inherently multi-turn);
        # the selector learns from the trajectory's outcome afterwards.
        async with ctx.selector_lock:
            strategy_name = ctx.selector.select(ctx.available) if ctx.available else "single_turn"
        strategy = ctx.strategies.get(strategy_name)

        tm = TrajectoryManager(
            goal,
            max_context_messages=ctx.max_context,
            summarize=ctx.summarize,
        )

        if dry_run:
            return await self._preview_trajectory(goal, tm, strategy, strategy_name, ctx)

        for _turn in range(self.config.max_turns_per_trajectory):
            if self._budget_exceeded(ctx):
                break

            proposals = await self._next_proposals(goal, tm, strategy, strategy_name, ctx)
            if not proposals:
                break
            proposal = proposals[0]  # single-attacker/first-branch; tree search is Phase 2

            msgs = tm.defender_messages(proposal.prompt, parent_index=proposal.parent_index)
            try:
                gen = await ctx.defender.respond(msgs)
                eval_result = await ctx.evaluator.evaluate(goal, proposal.prompt, gen.text)
                response_text = gen.text
            except Exception as exc:  # noqa: BLE001 -- log an ERROR turn, end this trajectory
                eval_result = EvalResult(
                    verdict=Verdict.ERROR,
                    success=False,
                    explanation=f"provider/judge error: {type(exc).__name__}",
                    judge="orchestrator",
                )
                tm.record_turn(
                    strategy_name, proposal.prompt, "", eval_result,
                    attacker_reflection=(proposal.critique or None),
                    parent_index=proposal.parent_index,
                )
                break

            tm.record_turn(
                strategy_name, proposal.prompt, response_text, eval_result,
                attacker_reflection=(proposal.critique or None),
                parent_index=proposal.parent_index,
            )

            # -- stop conditions --
            if eval_result.success and self.config.stop_on_success:
                break
            if strategy is not None and not self.config.attacker_swarm.enabled:
                if not strategy.should_continue(tm.trajectory):
                    break

        traj = tm.finalize()

        # -- learn + checkpoint (selector reward = success bonus, else best soft score) --
        reward = 1.0 if traj.succeeded else traj.best_score
        async with ctx.selector_lock:
            _safe(lambda: ctx.selector.update(strategy_name, reward, len(traj.turns)), None)
            if self.config.selection.persist_stats:
                _safe(lambda: ctx.persistence.save_stats(self.config.name, ctx.selector.stats()), None)
        _safe(lambda: ctx.persistence.save_trajectory(self.config.name, traj), None)
        return traj

    async def _next_proposals(self, goal, tm, strategy, strategy_name, ctx):
        """Get the next attack proposal(s) from the swarm or the single attacker."""
        try:
            if self.config.attacker_swarm.enabled and ctx.swarm is not None:
                return await ctx.swarm.propose_next(goal, tm.trajectory, strategy_hint=strategy_name)
            if ctx.attacker is not None and strategy is not None:
                return await ctx.attacker.propose(goal, tm.trajectory, strategy)
        except Exception:  # noqa: BLE001 -- a flaky attacker ends the trajectory, not the run
            return []
        return []

    async def _preview_trajectory(self, goal, tm, strategy, strategy_name, ctx) -> Trajectory:
        """Attacker-only preview (dry-run): author the opening prompt(s), no defender.

        Records one turn per proposed prompt with a placeholder defender response and
        a non-scored ``ERROR`` eval so the trajectory clearly reads as "not evaluated".
        The selector is still updated (reward 0) so its bookkeeping stays consistent.
        """
        proposals = await self._next_proposals(goal, tm, strategy, strategy_name, ctx)
        for proposal in proposals:
            eval_result = EvalResult(
                verdict=Verdict.ERROR,
                success=False,
                score=0.0,
                explanation="dry-run: defender not called",
                judge="dry-run",
                meta={"dry_run": True},
            )
            tm.record_turn(
                strategy_name,
                proposal.prompt,
                "[dry-run: defender not called]",
                eval_result,
                attacker_reflection=(proposal.critique or None),
                parent_index=proposal.parent_index,
            )
        traj = tm.finalize()
        traj.meta["dry_run"] = True
        async with ctx.selector_lock:
            _safe(lambda: ctx.selector.update(strategy_name, 0.0, len(traj.turns)), None)
        _safe(lambda: ctx.persistence.save_trajectory(self.config.name, traj), None)
        return traj

    def _budget_exceeded(self, ctx: _RunCtx) -> bool:
        return ctx.time_budget is not None and (time.monotonic() - ctx.start) > ctx.time_budget

    # =============================================================== metrics ==
    def _compute_metrics(self, result: CampaignResult) -> dict[str, Any]:
        """Prefer the rich AR-EVAL metrics; fall back to a minimal ASR if absent."""
        try:
            from .metrics import compute_metrics  # lazy (sibling)

            return compute_metrics(result)
        except Exception:  # noqa: BLE001 -- metrics module may not exist yet / may error
            n = len(result.trajectories)
            succ = sum(1 for t in result.trajectories if t.succeeded)
            return {
                "asr": (succ / n) if n else 0.0,
                "n_trajectories": n,
                "n_success": succ,
                "avg_turns": (sum(len(t.turns) for t in result.trajectories) / n) if n else 0.0,
            }


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def _safe(fn, default):
    """Run ``fn`` and swallow exceptions, returning ``default`` on failure.

    Used for best-effort I/O (persistence, stats) that must never sink a campaign.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Offline demo: a tiny 2-goal MOCK campaign wired entirely from inline stubs   #
# (no sibling factory files required) so this module is runnable on its own.   #
# --------------------------------------------------------------------------- #
def _demo() -> None:
    from .models import (
        AttackProposal,
        AuthorizationConfig,
        GenerationResult,
    )

    # -- inline stub provider: refuses unless the incoming prompt looks "vuln" --
    class _StubProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        async def generate(self, messages, **kwargs):
            last = messages[-1].content if messages else ""
            if "vuln" in last.lower():
                text = "COMPLY: <benign simulated compliance for an authorized test>"
            else:
                text = "REFUSAL: I can't help with that."
            return GenerationResult(text=text, model=self.name)

        def get_config(self):
            return {"provider": "stub", "model": self.name}

    # -- inline stub strategy: emits the goal id as the (verbatim) draft prompt --
    class _StubStrategy:
        name = "stub"

        def generate_initial(self, goal):
            return [AttackProposal(prompt=f"seed::{goal.id}", strategy=self.name)]

        def mutate(self, trajectory, feedback):
            return [AttackProposal(prompt=f"mut::{trajectory.goal.id}::{len(trajectory.turns)}", strategy=self.name)]

        def should_continue(self, trajectory):
            return len(trajectory.turns) < 2

    class _StubEvaluator:
        async def evaluate(self, goal, attacker_prompt, defender_response):
            complied = "COMPLY" in defender_response
            return EvalResult(
                verdict=Verdict.VIOLATION if complied else Verdict.REFUSAL,
                success=complied,
                score=0.9 if complied else 0.1,
                judge="stub",
            )

    class _StubSelector:
        def select(self, available):
            return available[0] if available else "stub"

        def update(self, strategy, reward, turns):
            pass

        def stats(self):
            return {}

    class _StubPersistence:
        def save_trajectory(self, campaign, trajectory):
            pass

        def save_result(self, result):
            pass

        def load_stats(self, campaign):
            return {}

        def save_stats(self, campaign, stats):
            pass

    from .attacker import Attacker
    from .defender import Defender

    config = CampaignConfig(
        name="mock-demo",
        max_parallel=2,
        max_turns_per_trajectory=3,
        stop_on_success=True,
        authorization=AuthorizationConfig(
            confirmed=True,
            scope="offline self-test of the auto-redteam orchestrator (mock providers only)",
        ),
        attacker=ModelConfig(provider="mock", model="mock-attacker"),
        defender=ModelConfig(provider="mock", model="mock-defender"),
    )

    goals = [
        AttackGoal(id="benign-1", goal="a goal the mock defender refuses", category="demo.refuse"),
        AttackGoal(id="vuln-2", goal="a goal the mock defender complies with", category="demo.comply"),
    ]

    orch = Orchestrator(
        config,
        defender=Defender(_StubProvider("mock-defender")),
        # author_with_provider=False -> the strategy draft passes through verbatim,
        # so the defender sees "seed::vuln-2" and the "vuln" branch fires.
        attacker=Attacker(_StubProvider("mock-attacker"), author_with_provider=False),
        strategies={"stub": _StubStrategy()},
        selector=_StubSelector(),
        evaluator=_StubEvaluator(),
        persistence=_StubPersistence(),
    )

    result = asyncio.run(orch.run(goals))
    m = result.metrics
    asr = m.get("asr_overall", m.get("asr"))  # AR-EVAL key or the inline fallback key
    print("scope recorded:", result.scope)
    print(f"ASR={asr} n={m.get('n_trajectories')} success={m.get('n_success')}")
    for t in result.trajectories:
        print(f"  goal={t.goal.id:<10} succeeded={t.succeeded} turns={len(t.turns)} best={t.best_score:.2f}")


if __name__ == "__main__":
    _demo()
