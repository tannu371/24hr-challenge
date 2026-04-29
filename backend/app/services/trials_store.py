"""SQLite-backed trial history (Task 1.4).

Schema is intentionally simple — params and results stay opaque JSON so the
schema doesn't have to evolve with every new solver kind.

    trials(
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        kind         TEXT NOT NULL,
        params_json  TEXT NOT NULL,
        results_json TEXT NOT NULL,
        created_at   TEXT NOT NULL          -- ISO-8601 UTC
    )
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ..config import SETTINGS


VALID_KINDS = {
    "classical_brute",
    "classical_sa",
    "classical_markowitz",
    "qaoa_sim",
    "qaoa_hw",
}


_LOCK = threading.Lock()


class TrialsStore:
    """Thin wrapper around a per-process sqlite3 connection.

    Not thread-safe across uvicorn workers; we use a module-level lock for
    intra-process safety which is enough for the demo footprint.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or SETTINGS.trials_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trials (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind         TEXT NOT NULL,
                    params_json  TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at   TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trials_created ON trials(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trials_kind ON trials(kind)"
            )

    def record(self, kind: str, params: dict[str, Any], results: dict[str, Any]) -> int:
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown trial kind: {kind!r}")
        created_at = datetime.now(timezone.utc).isoformat()
        params_json = json.dumps(params, default=_json_default)
        results_json = json.dumps(results, default=_json_default)
        with _LOCK, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO trials (kind, params_json, results_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (kind, params_json, results_json, created_at),
            )
            return int(cur.lastrowid)

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        with _LOCK, self._connect() as conn:
            rows: Iterable[sqlite3.Row] = conn.execute(
                "SELECT id, kind, created_at, params_json, results_json "
                "FROM trials ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_row_to_summary(r) for r in rows]

    def update_results(self, trial_id: int, new_results: dict[str, Any]) -> None:
        """Replace the results blob for an existing trial (used by hw ingestion)."""
        results_json = json.dumps(new_results, default=_json_default)
        with _LOCK, self._connect() as conn:
            cur = conn.execute(
                "UPDATE trials SET results_json = ? WHERE id = ?",
                (results_json, int(trial_id)),
            )
            if cur.rowcount == 0:
                raise KeyError(f"trial {trial_id} not found")

    def find_one(
        self, kind: Optional[str] = None, where_results: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        """Return the newest trial matching the given kind + simple results filters.

        ``where_results`` performs equality match on top-level keys inside the
        results JSON. Used to look up an in-flight hardware job by job_id.
        """
        with _LOCK, self._connect() as conn:
            params: list[Any] = []
            sql = "SELECT id, kind, created_at, params_json, results_json FROM trials"
            clauses = []
            if kind is not None:
                clauses.append("kind = ?")
                params.append(kind)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY id DESC"
            rows = conn.execute(sql, params).fetchall()

        for row in rows:
            results = json.loads(row["results_json"]) if row["results_json"] else {}
            if where_results:
                if not all(results.get(k) == v for k, v in where_results.items()):
                    continue
            return _row_to_full(row)
        return None

    def delete(self, trial_id: int) -> bool:
        """Remove a trial. Returns True if a row was deleted, False if not found."""
        with _LOCK, self._connect() as conn:
            cur = conn.execute("DELETE FROM trials WHERE id = ?", (int(trial_id),))
            return cur.rowcount > 0

    def get(self, trial_id: int) -> Optional[dict[str, Any]]:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT id, kind, created_at, params_json, results_json "
                "FROM trials WHERE id = ?",
                (int(trial_id),),
            ).fetchone()
        if row is None:
            return None
        return _row_to_full(row)


def _row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
    results = json.loads(row["results_json"]) if row["results_json"] else {}
    return {
        "id": row["id"],
        "kind": row["kind"],
        "created_at": row["created_at"],
        "summary": _summarise_results(row["kind"], results),
    }


def _row_to_full(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "created_at": row["created_at"],
        "params": json.loads(row["params_json"]) if row["params_json"] else {},
        "results": json.loads(row["results_json"]) if row["results_json"] else {},
    }


def _summarise_results(kind: str, results: dict[str, Any]) -> dict[str, Any]:
    """Lightweight projection for the trials list view — keep heavy fields out."""
    keys = (
        "cost",
        "energy",
        "approx_ratio",
        "K",
        "n_iterations",
        "runtime_s",
        "selected",
        "backend",
    )
    return {k: results[k] for k in keys if k in results}


def _json_default(obj: Any) -> Any:
    """Best-effort JSON serialiser for numpy types."""
    try:
        import numpy as np  # local import — keeps cold-start cheap

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
