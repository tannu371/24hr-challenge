"""Shared pytest fixtures for the backend test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `backend.app` importable when pytest is invoked from /backend or repo root.
_BACKEND = Path(__file__).resolve().parent.parent
_REPO = _BACKEND.parent
for p in (str(_BACKEND), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
