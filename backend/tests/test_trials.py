"""End-to-end test for the trials SQLite store.

We hit GET /trials on an empty (test) DB, record a synthetic trial directly
through the store API to mimic a future solver writing to it, and verify the
list and detail endpoints return what we wrote.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "trials_test.db"
    monkeypatch.setenv("TRIALS_DB_PATH", str(db_path))

    # Force a fresh import so SETTINGS picks up the env var.
    import importlib
    import app.config
    import app.services.trials_store
    import app.routers.trials
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.services.trials_store)
    importlib.reload(app.routers.trials)
    importlib.reload(app.main)

    return TestClient(app.main.app), app.services.trials_store.TrialsStore()


def test_list_empty(client):
    api, _store = client
    resp = api.get("/trials")
    assert resp.status_code == 200
    assert resp.json() == []


def test_record_then_list_and_detail(client):
    api, store = client
    tid_a = store.record(
        kind="classical_brute",
        params={"N": 6, "K": 3, "lambda": 2.5},
        results={"cost": -1.234, "selected": [0, 2, 4], "K": 3, "runtime_s": 0.01},
    )
    tid_b = store.record(
        kind="qaoa_sim",
        params={"p": 2, "optimizer": "COBYLA"},
        results={"energy": 0.456, "approx_ratio": 0.92, "n_iterations": 17},
    )
    assert tid_b > tid_a

    listing = api.get("/trials").json()
    assert [t["id"] for t in listing] == [tid_b, tid_a]
    assert listing[0]["kind"] == "qaoa_sim"
    assert listing[0]["summary"]["energy"] == 0.456
    assert listing[1]["summary"]["selected"] == [0, 2, 4]

    detail = api.get(f"/trials/{tid_a}").json()
    assert detail["id"] == tid_a
    assert detail["kind"] == "classical_brute"
    assert detail["params"]["N"] == 6
    assert detail["results"]["cost"] == -1.234
    assert "created_at" in detail


def test_detail_404(client):
    api, _ = client
    resp = api.get("/trials/99999")
    assert resp.status_code == 404


def test_record_rejects_unknown_kind(client):
    _api, store = client
    with pytest.raises(ValueError):
        store.record(kind="not_a_real_kind", params={}, results={})


def test_delete_trial_round_trip(client):
    api, store = client
    tid = store.record(
        kind="classical_brute",
        params={"N": 6, "K": 3},
        results={"cost": -0.123, "K": 3},
    )
    # Visible before delete.
    assert api.get(f"/trials/{tid}").status_code == 200
    # Delete.
    r = api.delete(f"/trials/{tid}")
    assert r.status_code == 200
    assert r.json() == {"trial_id": tid, "deleted": True}
    # Gone afterwards.
    assert api.get(f"/trials/{tid}").status_code == 404
    # Idempotent second delete returns 404 with a clean message.
    r2 = api.delete(f"/trials/{tid}")
    assert r2.status_code == 404


def test_delete_unknown_returns_404(client):
    api, _ = client
    r = api.delete("/trials/99999")
    assert r.status_code == 404
