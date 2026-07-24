"""persistence.py -- durable storage for trajectories + selector stats.

Two backends behind the same `interfaces.Persistence` surface:

    JsonlPersistence   append-only JSONL trajectories + a result.json + stats.json
                       under a per-campaign directory. Zero dependencies, greppable,
                       diff-friendly -- the default.
    SqlitePersistence  the same data in a SQLite file (sqlite3 imported LAZILY so the
                       module imports even where the stdlib build lacks it). Better for
                       large campaigns + concurrent querying.

`get_persistence(cfg)` dispatches on `cfg["backend"]` and expands the `path`
template (which may contain `{campaign}`).

SAFETY: only typed models (Trajectory / CampaignResult / bandit stats) are written.
None of those carry API keys, and this module never reads env vars -- so no secret
can ever land in a run file. The path template is expanded per-campaign only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CampaignResult, Trajectory


# --------------------------------------------------------------------------- #
# Path handling                                                                #
# --------------------------------------------------------------------------- #
def _expand(path_template: str, campaign: str) -> Path:
    """Expand `{campaign}` in the configured path into a concrete directory."""
    # Sanitise the campaign name so it is a safe single path segment.
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in campaign)
    return Path(path_template.replace("{campaign}", safe)).expanduser()


# --------------------------------------------------------------------------- #
# JSONL backend                                                                #
# --------------------------------------------------------------------------- #
class JsonlPersistence:
    """Append-only JSONL trajectories + JSON result/stats -- satisfies Persistence.

    Layout (one directory per campaign):
        <path>/trajectories.jsonl   one Trajectory per line (append-only)
        <path>/result.json          the final CampaignResult (overwritten)
        <path>/stats.json           bandit/selector stats (overwritten, for resume)
    """

    def __init__(self, path_template: str = "./runs/{campaign}") -> None:
        self.path_template = path_template

    def _dir(self, campaign: str) -> Path:
        d = _expand(self.path_template, campaign)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_trajectory(self, campaign: str, trajectory: Trajectory) -> None:
        line = json.dumps(trajectory.model_dump(mode="json"), ensure_ascii=False)
        with (self._dir(campaign) / "trajectories.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save_result(self, result: CampaignResult) -> None:
        d = self._dir(result.campaign_name)
        (d / "result.json").write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_stats(self, campaign: str) -> dict[str, Any]:
        p = _expand(self.path_template, campaign) / "stats.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def save_stats(self, campaign: str, stats: dict[str, Any]) -> None:
        (self._dir(campaign) / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Convenience for round-trip tests / resume: read trajectories back.
    def load_trajectories(self, campaign: str) -> list[Trajectory]:
        p = _expand(self.path_template, campaign) / "trajectories.jsonl"
        if not p.is_file():
            return []
        out: list[Trajectory] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Trajectory.model_validate_json(line))
        return out


# --------------------------------------------------------------------------- #
# SQLite backend                                                               #
# --------------------------------------------------------------------------- #
class SqlitePersistence:
    """SQLite-backed persistence -- satisfies Persistence. sqlite3 imported lazily.

    Tables:
        trajectories(campaign, traj_id, succeeded, best_score, json)
        results(campaign PRIMARY KEY, json)
        stats(campaign PRIMARY KEY, json)
    """

    def __init__(self, path_template: str = "./runs/{campaign}") -> None:
        self.path_template = path_template

    def _db_path(self, campaign: str) -> Path:
        d = _expand(self.path_template, campaign)
        d.mkdir(parents=True, exist_ok=True)
        return d / "campaign.db"

    def _connect(self, campaign: str):
        import sqlite3  # lazy: stdlib but keeps the import surface honest

        conn = sqlite3.connect(str(self._db_path(campaign)))
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trajectories(
                campaign TEXT, traj_id TEXT, succeeded INTEGER,
                best_score REAL, json TEXT,
                PRIMARY KEY (campaign, traj_id)
            );
            CREATE TABLE IF NOT EXISTS results(campaign TEXT PRIMARY KEY, json TEXT);
            CREATE TABLE IF NOT EXISTS stats(campaign TEXT PRIMARY KEY, json TEXT);
            """
        )
        return conn

    def save_trajectory(self, campaign: str, trajectory: Trajectory) -> None:
        conn = self._connect(campaign)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO trajectories VALUES (?,?,?,?,?)",
                (
                    campaign,
                    trajectory.id,
                    int(trajectory.succeeded),
                    trajectory.best_score,
                    json.dumps(trajectory.model_dump(mode="json"), ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_result(self, result: CampaignResult) -> None:
        conn = self._connect(result.campaign_name)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?)",
                (result.campaign_name,
                 json.dumps(result.model_dump(mode="json"), ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def load_stats(self, campaign: str) -> dict[str, Any]:
        conn = self._connect(campaign)
        try:
            row = conn.execute(
                "SELECT json FROM stats WHERE campaign=?", (campaign,)
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else {}

    def save_stats(self, campaign: str, stats: dict[str, Any]) -> None:
        conn = self._connect(campaign)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO stats VALUES (?,?)",
                (campaign, json.dumps(stats, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def load_trajectories(self, campaign: str) -> list[Trajectory]:
        conn = self._connect(campaign)
        try:
            rows = conn.execute(
                "SELECT json FROM trajectories WHERE campaign=?", (campaign,)
            ).fetchall()
        finally:
            conn.close()
        return [Trajectory.model_validate_json(r[0]) for r in rows]


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def get_persistence(cfg: dict[str, Any]) -> Any:
    """Build a Persistence backend from the `persistence` config dict.

    cfg keys:
        backend: "jsonl" (default) | "sqlite"
        path:    directory template, may contain `{campaign}`
    """
    backend = (cfg or {}).get("backend", "jsonl")
    path = (cfg or {}).get("path", "./runs/{campaign}")
    if backend == "jsonl":
        return JsonlPersistence(path)
    if backend == "sqlite":
        return SqlitePersistence(path)
    raise ValueError(f"unknown persistence backend: {backend!r}")


# --------------------------------------------------------------------------- #
# Self-test: round-trip a toy result, no model                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    from .metrics import _toy_result, compute_metrics

    res = _toy_result()
    with tempfile.TemporaryDirectory() as tmp:
        store = get_persistence({"backend": "jsonl", "path": tmp + "/{campaign}"})

        for traj in res.trajectories:
            store.save_trajectory(res.campaign_name, traj)
        store.save_result(res)
        store.save_stats(res.campaign_name, {"crescendo": {"n": 3, "reward": 1.4}})

        back = store.load_trajectories(res.campaign_name)
        assert len(back) == len(res.trajectories), (len(back), len(res.trajectories))
        assert back[0].id == res.trajectories[0].id, back[0].id
        assert back[0].succeeded == res.trajectories[0].succeeded
        stats = store.load_stats(res.campaign_name)
        assert stats["crescendo"]["n"] == 3, stats

        # Render Markdown to prove the reporting hookup works end-to-end.
        from .reporting import MarkdownReporter

        res.metrics = compute_metrics(res)
        paths = MarkdownReporter().render(res, tmp + "/report")
        assert paths and Path(paths[0]).is_file(), paths

    print("[persistence] self-test OK")
    print(f"  JSONL round-trip: {len(back)} trajectories, stats keys={list(stats)}")
    print(f"  Markdown rendered: {[Path(p).name for p in paths]}")
