"""Hardware result cache — read/write JSON snapshots in /artifacts.

A snapshot is a self-contained record describing one hardware run:

    {
        "name": "n8_seed7",
        "params": { ... canonical params ... },
        "results": { ... ingested result ... },
        "meta": {
            "created_at": "2026-04-29T12:34:56Z",
            "backend": "ibm_brisbane",
            "shots": 4096,
            "stand_in": false      # true ⇒ synthetic stand-in for demo
        }
    }

These files are committed to /artifacts so the demo path works without a
live IBM token. Real runs produced by /hw/submit + /hw/job overwrite the
stand-ins as you generate them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import ARTIFACTS_DIR


def _ensure_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def list_cached() -> list[dict[str, Any]]:
    d = _ensure_dir()
    out = []
    for p in sorted(d.glob("hw_*.json")):
        try:
            with p.open() as f:
                blob = json.load(f)
        except Exception:
            continue
        out.append({
            "name": blob.get("name", p.stem),
            "file": p.name,
            "params": blob.get("params", {}),
            "meta": blob.get("meta", {}),
            "results_summary": _summary(blob.get("results", {})),
        })
    return out


def get_cached(name: str) -> Optional[dict[str, Any]]:
    d = _ensure_dir()
    candidates = [d / f"hw_{name}.json", d / f"{name}.json"]
    for p in candidates:
        if p.exists():
            with p.open() as f:
                return json.load(f)
    return None


def save_cached(name: str, params: dict, results: dict, meta: dict) -> Path:
    d = _ensure_dir()
    record = {
        "name": name,
        "params": params,
        "results": results,
        "meta": {**meta, "saved_at": datetime.now(timezone.utc).isoformat()},
    }
    p = d / f"hw_{name}.json"
    with p.open("w") as f:
        json.dump(record, f, indent=2, default=_json_default)
    return p


def _summary(results: dict[str, Any]) -> dict[str, Any]:
    keys = ("cost", "energy", "approx_ratio", "K", "shots", "selected", "selected_names",
            "n_unique_bitstrings")
    return {k: results[k] for k in keys if k in results}


def _json_default(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
