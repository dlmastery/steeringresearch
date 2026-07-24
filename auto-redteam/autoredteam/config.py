"""config.py -- turn YAML (+ env + CLI) into a validated CampaignConfig.

The harness is "driven almost entirely by configuration", so this module is the
front door. It layers four sources of truth, in increasing precedence:

    1. config/default.yaml        the shipped base (mock providers, offline-safe)
    2. a user --config file       the campaign the operator actually wants to run
    3. HARNESS__A__B env vars     ops/CI overrides (double-underscore = path sep)
    4. overrides dict (CLI)       last-word --set flags from the command line

Each layer is DEEP-MERGED onto the one below it (nested dicts merge key-by-key;
scalars and lists replace), then the merged mapping is validated by the Pydantic
models in `models.py`. Validation is where typos and type errors are caught, so
downstream code can trust `CampaignConfig`.

Two small conveniences live here too:
  * `load_goals`  -- read the attack-goal YAML into typed `AttackGoal`s.
  * `config_manifest` -- a hash + NON-SECRET summary recorded in every run so a
    finding is reproducible and auditable (keys are never included).

Nothing here loads a model or touches the network; it is pure I/O + dict work.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import AttackGoal, CampaignConfig

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
# BASE_DIR is the package root (auto-redteam/). default.yaml, config/ and
# prompts/ are resolved relative to it so the loader works from any cwd.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "default.yaml"
ENV_PREFIX = "HARNESS__"


# --------------------------------------------------------------------------- #
# Low-level helpers                                                            #
# --------------------------------------------------------------------------- #
def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict (empty file -> empty dict)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping at the top of {p}, got {type(data).__name__}")
    return data


def _coerce(value: Any) -> Any:
    """Coerce a raw string (from env/CLI) into a typed scalar via YAML rules.

    So "true"->bool, "5"->int, "0.7"->float, "null"->None, "[a, b]"->list.
    Non-strings pass through untouched (a caller may already hand us real types).
    """
    if not isinstance(value, str):
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`. Nested dicts merge; else replace.

    Lists and scalars in `override` REPLACE the value in `base` (we never try to
    element-merge lists -- "strategies: [...]" in a user file wholly wins).
    """
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _set_path(target: dict[str, Any], path: list[str], value: Any) -> None:
    """Set a nested key given a path list, creating intermediate dicts."""
    node = target
    for part in path[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[path[-1]] = value


def _expand_dotted(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Expand a possibly-flat override mapping into a nested dict.

    Supports CLI-style dotted keys ("selection.mode": "thompson") AND keys that
    are already nested dicts. String leaves are coerced to typed scalars.
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        if isinstance(value, dict):
            # already-nested override: recurse so its string leaves coerce too
            merged = _deep_merge(nested.get(key, {}) if isinstance(nested.get(key), dict) else {},
                                 _expand_dotted(value))
            nested[key] = merged
        else:
            path = key.split(".") if "." in key else [key]
            _set_path(nested, path, _coerce(value))
    return nested


def _env_overrides(env: Mapping[str, str], prefix: str = ENV_PREFIX) -> dict[str, Any]:
    """Collect HARNESS__A__B=value env vars into a nested override dict.

    The path separator is the DOUBLE underscore, so ordinary snake_case keys
    (max_parallel, attacker_swarm) survive: HARNESS__ATTACKER_SWARM__ENABLED=true
    -> {"attacker_swarm": {"enabled": True}}.
    """
    nested: dict[str, Any] = {}
    for raw_key, raw_val in env.items():
        if not raw_key.startswith(prefix):
            continue
        rest = raw_key[len(prefix):]
        if not rest:
            continue
        path = [part.lower() for part in rest.split("__") if part != ""]
        if not path:
            continue
        _set_path(nested, path, _coerce(raw_val))
    return nested


def _normalize_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Bridge YAML spelling to the model field names.

    default.yaml uses the friendly top-level key `campaign:`; the model field is
    `name`. We map it here (without clobbering an explicit `name`).
    """
    out = dict(raw)
    if "campaign" in out:
        campaign = out.pop("campaign")
        out.setdefault("name", campaign)
    return out


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def load_config(
    path: str | None = None,
    overrides: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> CampaignConfig:
    """Build a validated CampaignConfig from the four config layers.

    Precedence (low -> high): default.yaml < `path` YAML < HARNESS__ env < overrides.

    Args:
        path:      optional user campaign YAML (its values win over the default).
        overrides: CLI/programmatic overrides -- a nested dict OR flat dotted keys
                   ({"selection.mode": "thompson"}); string leaves are coerced.
        env:       environment mapping for HARNESS__ vars (defaults to os.environ).

    Returns:
        A CampaignConfig; Pydantic raises if the merged mapping is invalid.
    """
    env = os.environ if env is None else env

    merged = _load_yaml(DEFAULT_CONFIG_PATH)
    if path:
        merged = _deep_merge(merged, _load_yaml(path))
    merged = _deep_merge(merged, _env_overrides(env))
    if overrides:
        merged = _deep_merge(merged, _expand_dotted(overrides))

    return CampaignConfig.model_validate(_normalize_keys(merged))


def load_goals(path: str | None) -> list[AttackGoal]:
    """Load the attack-goal YAML into typed AttackGoals.

    The file is either a top-level list of goal mappings, or a mapping with a
    `goals:` key holding that list. Each entry validates against AttackGoal.
    A None/missing path yields an empty list (a campaign may inline no goals).
    """
    if not path:
        return []
    data = _load_yaml(path)
    raw_goals = data.get("goals", data) if isinstance(data, dict) else data
    if isinstance(raw_goals, dict):
        # allow a {id: {...}} mapping too -- fold the id in
        items = []
        for gid, body in raw_goals.items():
            entry = dict(body or {})
            entry.setdefault("id", gid)
            items.append(entry)
        raw_goals = items
    if not isinstance(raw_goals, list):
        raise ValueError(f"expected a list of goals in {path}")
    return [AttackGoal.model_validate(g) for g in raw_goals]


def load_text_asset(path: str | None) -> str | None:
    """Read a text asset (e.g. a prompt .md) resolved relative to the repo root.

    Absolute paths are read as-is; relative paths are tried against the cwd first
    and then BASE_DIR (so `prompts/attacker_system.md` works from anywhere).
    """
    if not path:
        return None
    candidate = Path(path)
    tried = [candidate]
    if not candidate.is_absolute():
        tried = [Path.cwd() / candidate, BASE_DIR / candidate]
    for p in tried:
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"prompt/text asset not found (tried {', '.join(map(str, tried))})")


def resolve_system_prompt(model_cfg: Any) -> str | None:
    """Return a model's system prompt, loading it from `system_prompt_path` if needed.

    Inline `system_prompt` wins; otherwise the file at `system_prompt_path` is read.
    Providers call this so they receive text, not a path.
    """
    inline = getattr(model_cfg, "system_prompt", None)
    if inline:
        return inline
    return load_text_asset(getattr(model_cfg, "system_prompt_path", None))


def config_manifest(cfg: CampaignConfig) -> dict[str, Any]:
    """A hash + NON-SECRET summary of a config, recorded in every run.

    Deliberately narrow: model provider/name (never keys), the swarm + selection
    setup, strategy names, and the eval taxonomy -- enough to reproduce and audit
    a finding without leaking anything sensitive.
    """
    def _model_summary(m: Any) -> dict[str, Any]:
        if m is None:
            return {}
        # api_key_env is only the NAME of an env var, safe to record; the key is not.
        return {
            "provider": m.provider,
            "model": m.model,
            "temperature": m.temperature,
            "api_key_env": m.api_key_env,
        }

    return {
        "config_hash": cfg.config_hash(),
        "name": cfg.name,
        "seed": cfg.seed,
        "authorization_confirmed": cfg.authorization.confirmed,
        "attacker": _model_summary(cfg.attacker),
        "defender": _model_summary(cfg.defender),
        "swarm_enabled": cfg.attacker_swarm.enabled,
        "selection_mode": cfg.selection.mode,
        "strategies": [s.name for s in cfg.strategies],
        "taxonomy": cfg.evaluation.taxonomy,
        "success_threshold": cfg.evaluation.success_threshold,
        "goals_path": cfg.goals_path,
    }


# --------------------------------------------------------------------------- #
# Self-test: load defaults + example goals, print a manifest summary.          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Defaults + a mock override, to show env/CLI layering without a real model.
    cfg = load_config(overrides={"attacker.provider": "mock", "selection.mode": "thompson"})
    print("name          :", cfg.name)
    print("config_hash   :", cfg.config_hash()[:16], "...")
    print("swarm_enabled :", cfg.attacker_swarm.enabled)
    print("selection.mode:", cfg.selection.mode)
    print("strategies    :", [s.name for s in cfg.strategies])

    goals = load_goals(str(BASE_DIR / "config" / "goals_example.yaml"))
    print("example goals :", len(goals), "->", [g.id for g in goals])

    manifest = config_manifest(cfg)
    assert "config_hash" in manifest and "GOOGLE_API_KEY" not in str(manifest)
    print("manifest keys :", sorted(manifest.keys()))
