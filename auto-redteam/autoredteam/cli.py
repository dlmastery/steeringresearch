"""cli.py -- the `auto-redteam` command-line entrypoint (Typer).

This is the thin operator surface over the harness. It is deliberately small: it
parses flags, prints the authorized-use banner, and hands off to the factories in
`autoredteam.config` and `autoredteam.orchestrator`. All heavy machinery (Pydantic
config models, the async orchestrator, providers, strategies) is imported LAZILY
inside each command so that `import autoredteam.cli` -- and therefore `--help`,
shell completion, and `version` -- stays fast and dependency-light. Only Typer is
needed at module import time.

Commands
--------
    auto-redteam run       [--config Y] [--set a.b=val ...] [--dry-run]
    auto-redteam validate  [--config Y] [--set a.b=val ...]
    auto-redteam strategies
    auto-redteam version

`run` executes a campaign end-to-end (banner -> build Orchestrator -> asyncio.run).
`--dry-run` exercises the attacker/strategy loop only, with NO defender calls -- a
cheap way to preview the prompts the harness would send before spending API budget
or touching a live target. `validate` loads + deep-merges the config and prints the
reproducibility manifest (config hash + non-secret summary) WITHOUT running anything.
"""
from __future__ import annotations

from typing import Optional

import typer

from . import __version__

app = typer.Typer(
    name="auto-redteam",
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Configurable auto red-teaming harness for AUTHORIZED AI-safety research: "
        "an ablated local Gemma attacker vs a pluggable frontier defender."
    ),
)


# --------------------------------------------------------------------------- #
# --set parsing helpers                                                        #
# --------------------------------------------------------------------------- #
def _parse_overrides(sets: Optional[list[str]]) -> dict:
    """Turn `--set a.b.c=val` flags into a FLAT dotted-key override dict: {"a.b.c": "val"}.

    `config.load_config` accepts flat dotted keys (splitting any key on ".") and does
    the type coercion itself (YAML scalar rules: "8"->int, "true"->bool, "null"->None),
    so the CLI just splits each flag on the FIRST "=" and hands over the RAW string
    value untouched. Later flags win over earlier ones.
    """
    overrides: dict = {}
    for item in sets or []:
        if "=" not in item:
            raise typer.BadParameter(
                f"--set expects KEY=VALUE (dotted key), got: {item!r}"
            )
        key, _, raw = item.partition("=")   # split on FIRST "=" so values may contain "="
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"--set has an empty key: {item!r}")
        overrides[key] = raw
    return overrides


def _echo_manifest(manifest: dict) -> None:
    """Pretty-print the reproducibility manifest (ASCII-safe, no secrets)."""
    import json

    typer.echo(json.dumps(manifest, indent=2, sort_keys=True, default=str))


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #
@app.command()
def run(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to a campaign YAML (deep-merged over defaults)."
    ),
    set_: Optional[list[str]] = typer.Option(
        None, "--set", "-s", help="Override a config key, dotted: --set selection.mode=thompson (repeatable)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Attacker-only preview: build + emit attack prompts, make NO defender calls."
    ),
    out_dir: Optional[str] = typer.Option(
        None, "--out", "-o", help="Directory for reports/artifacts (default: the persistence path)."
    ),
) -> None:
    """Run a red-teaming campaign end-to-end (banner -> orchestrate -> report)."""
    # Banner FIRST, before any config load or generation -- the authorized-use contract.
    from .banner import BANNER

    typer.echo(BANNER)

    # Lazy: pull in the (heavier) config + orchestrator only when actually running.
    import asyncio

    from .config import load_config
    from .orchestrator import Orchestrator

    overrides = _parse_overrides(set_)
    cfg = load_config(config, overrides=overrides or None)

    if dry_run:
        typer.echo("[dry-run] attacker-only preview: NO defender calls will be made.\n")

    # The Orchestrator builds providers/selector/evaluator/persistence via factories
    # from the config; we inject nothing here. It also runs the authorization gate
    # (banner.assert_authorized) before any generation -- so an unconfirmed config
    # fails loudly rather than silently touching a live model.
    orch = Orchestrator(cfg)
    goals = cfg_goals(cfg)
    # Thread --dry-run only if the orchestrator's run() exposes the kwarg; this keeps
    # the CLI robust to the orchestrator's exact signature (attacker-only preview).
    import inspect

    if "dry_run" in inspect.signature(orch.run).parameters:
        coro = orch.run(goals, dry_run=dry_run)
    else:
        if dry_run:
            typer.secho(
                "(--dry-run not supported by this orchestrator build; running fully)",
                fg=typer.colors.YELLOW,
                err=True,
            )
        coro = orch.run(goals)
    try:
        result = asyncio.run(coro)
    except Exception as exc:  # surface a clean one-line error, not a stack dump
        typer.secho(f"campaign failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    _echo_run_summary(result)

    # Render reports if a reporter is available; never fatal if reporting is absent.
    _maybe_report(cfg, result, out_dir)


@app.command()
def validate(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to a campaign YAML (deep-merged over defaults)."
    ),
    set_: Optional[list[str]] = typer.Option(
        None, "--set", "-s", help="Override a config key, dotted (repeatable)."
    ),
) -> None:
    """Load + deep-merge the config and print its reproducibility manifest (no run)."""
    from .config import config_manifest, load_config

    overrides = _parse_overrides(set_)
    cfg = load_config(config, overrides=overrides or None)
    typer.echo(f"campaign  : {cfg.name}")
    typer.echo(f"hash      : {cfg.config_hash()}")
    typer.echo(f"attacker  : {cfg.attacker.provider}:{cfg.attacker.model}")
    typer.echo(f"defender  : {cfg.defender.provider}:{cfg.defender.model}")
    typer.echo(f"swarm     : {'on' if cfg.attacker_swarm.enabled else 'off'}")
    typer.echo(f"selection : {cfg.selection.mode}")
    authorized = bool(cfg.authorization.confirmed) and len(cfg.authorization.scope.strip()) >= 10
    typer.echo(f"authorized: {'yes' if authorized else 'NO (set authorization.confirmed + scope to launch)'}")
    typer.echo("--- manifest ---")
    _echo_manifest(config_manifest(cfg))


@app.command()
def strategies() -> None:
    """List the attack strategies registered in the harness."""
    from .strategies import list_strategies

    names = list_strategies()
    typer.echo("registered attack strategies:")
    for name in names:
        typer.echo(f"  - {name}")


@app.command()
def version() -> None:
    """Print the auto-redteam version."""
    typer.echo(f"auto-redteam {__version__}")


# --------------------------------------------------------------------------- #
# Small local helpers (kept out of the command bodies for readability)         #
# --------------------------------------------------------------------------- #
def cfg_goals(cfg) -> list:
    """Load the campaign's goals from `cfg.goals_path` (empty list if unset)."""
    from .config import load_goals

    if not cfg.goals_path:
        return []
    return load_goals(cfg.goals_path)


def _echo_run_summary(result) -> None:
    """Print a compact ASCII summary of a finished campaign."""
    metrics = getattr(result, "metrics", {}) or {}
    n_traj = len(getattr(result, "trajectories", []) or [])
    asr = metrics.get("asr", metrics.get("attack_success_rate"))
    typer.echo("")
    typer.echo(f"campaign  : {getattr(result, 'campaign_name', '?')}")
    typer.echo(f"hash      : {getattr(result, 'config_hash', '?')}")
    if getattr(result, "scope", ""):
        typer.echo(f"scope     : {result.scope}")
    typer.echo(f"trajectories: {n_traj}")
    if asr is not None:
        typer.echo(f"ASR       : {asr}")
    for key in ("successful_goals", "avg_turns_to_success", "total_tokens"):
        if key in metrics:
            typer.echo(f"{key:<10}: {metrics[key]}")


def _maybe_report(cfg, result, out_dir: Optional[str]) -> None:
    """Render reports if reporting is wired; stay silent (non-fatal) otherwise."""
    try:
        from .reporting import MarkdownReporter
    except Exception:
        return
    target = out_dir or (cfg.persistence.get("path", "./runs/{campaign}") if isinstance(cfg.persistence, dict) else "./runs")
    target = str(target).replace("{campaign}", cfg.name)
    try:
        paths = MarkdownReporter().render(result, target)
        for p in paths:
            typer.echo(f"report    : {p}")
    except Exception as exc:  # reporting is a convenience, never a run-breaker
        typer.secho(f"(reporting skipped: {exc})", fg=typer.colors.YELLOW, err=True)


if __name__ == "__main__":  # `python -m autoredteam.cli`
    app()
