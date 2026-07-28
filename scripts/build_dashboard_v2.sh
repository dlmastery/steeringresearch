#!/usr/bin/env bash
# Regenerate the 3-tier dashboard from the CLEAN v2 log.
#
# The v2 log (experiment_log_v2.jsonl) is judge-free and schema-light; the renderer
# expects the legacy schema, so we adapt into .dash_v2/ and build from there. The old
# 124-row log is NOT used: it is not one comparable scale (autoresearch_results/PROVENANCE.md).
set -e
cd "$(dirname "$0")/.."
PYTHONIOENCODING=utf-8 PYTHONPATH=src "$PY" scripts/adapt_v2_for_dashboard.py
PYTHONIOENCODING=utf-8 PYTHONPATH=src "$PY" -c "
from pathlib import Path
from steering.dashboard import build_all_dashboards
print(build_all_dashboards(results_dir=Path('.dash_v2').resolve()))
"
