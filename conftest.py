"""Repo-root conftest: put src/ on sys.path so `import steering` works (src layout)."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
