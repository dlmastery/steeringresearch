"""Test suite for the auto-redteam harness.

CPU-only, offline, deterministic. Tests speak the spine vocabulary
(`autoredteam.models` / `autoredteam.interfaces` / `autoredteam.banner`) and
exercise sibling modules (providers / strategies / selection / config /
evaluator / persistence / orchestrator) through the CONTRACTED factory
signatures. Sibling modules may still be landing while these tests are written,
so every test that touches one guards its import with `pytest.importorskip` --
the suite therefore COLLECTS cleanly and SKIPS (rather than errors) whatever is
not yet present. No network, no model: the mock provider makes it all runnable.
"""
