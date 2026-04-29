"""Phase-4 acceptance tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Per-test fresh DB + isolated artifacts dir + no IBM token configured."""
    db_path = tmp_path / "trials_hw.db"
    art_path = tmp_path / "artifacts"
    art_path.mkdir()
    monkeypatch.setenv("TRIALS_DB_PATH", str(db_path))
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)

    import importlib
    import app.config
    import app.services.trials_store
    import app.services.hw
    import app.services.hw_cache
    import app.routers.hw
    import app.main

    importlib.reload(app.config)
    # Point ARTIFACTS_DIR at the per-test directory so we don't trample the
    # repo's real /artifacts.
    monkeypatch.setattr(app.config, "ARTIFACTS_DIR", art_path, raising=False)
    importlib.reload(app.services.hw_cache)  # picks up patched ARTIFACTS_DIR
    monkeypatch.setattr(app.services.hw_cache, "ARTIFACTS_DIR", art_path, raising=False)
    importlib.reload(app.services.trials_store)
    importlib.reload(app.services.hw)
    importlib.reload(app.routers.hw)
    importlib.reload(app.main)

    return TestClient(app.main.app), app, art_path


# ---------------------------------------------------------------------------
# Token-missing path: /hw/backends, /hw/submit, /hw/job all 503
# ---------------------------------------------------------------------------


def test_hw_backends_returns_503_without_token(client):
    api, _, _ = client
    r = api.get("/hw/backends")
    assert r.status_code == 503
    assert "IBM_QUANTUM_TOKEN" in r.json()["detail"]


def test_hw_submit_returns_503_without_token(client):
    api, _, _ = client
    r = api.post(
        "/hw/submit",
        json={"trial_id": 1, "backend_name": "ibm_brisbane", "shots": 1024},
    )
    assert r.status_code == 503


def test_hw_job_returns_503_without_token(client):
    api, _, _ = client
    r = api.get("/hw/job/abc123")
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Cache endpoints work without a token
# ---------------------------------------------------------------------------


def _seed_artifact(art_path: Path, name: str) -> dict:
    blob = {
        "name": name,
        "params": {"N": 8, "K": 3, "lambda": 2.5, "P_K": 5.0, "P_R": 0.5,
                   "p": 2, "mixer": "x", "init_state": "uniform"},
        "results": {
            "kind": "qaoa_hw", "status": "complete",
            "cost": -0.123, "energy": 0.456, "K": 3, "shots": 4096,
            "selected": [0, 2, 5], "selected_names": ["A00", "A02", "A05"],
            "approx_ratio": 0.97, "classical_optimum": -0.130,
            "n_unique_bitstrings": 73,
            "top_bitstrings": [{"bitstring": "00100101", "x": [1,0,1,0,0,1,0,0],
                                 "count": 412, "probability": 0.10, "cost": -0.12, "K": 3}],
        },
        "meta": {"stand_in": True, "backend": "fake_backend",
                  "shots": 4096, "saved_at": "2026-04-29T00:00:00+00:00"},
    }
    p = art_path / f"hw_{name}.json"
    p.write_text(json.dumps(blob))
    return blob


def test_hw_cached_index_and_get(client):
    api, _, art_path = client
    _seed_artifact(art_path, "test_n8")
    _seed_artifact(art_path, "test_n12")

    listing = api.get("/hw/cached").json()["cached"]
    names = sorted(c["name"] for c in listing)
    assert names == ["test_n12", "test_n8"]
    assert listing[0]["meta"]["stand_in"] is True
    assert "cost" in listing[0]["results_summary"]

    detail = api.get("/hw/cached/test_n8").json()
    assert detail["name"] == "test_n8"
    assert detail["results"]["status"] == "complete"


def test_hw_cached_404(client):
    api, _, _ = client
    r = api.get("/hw/cached/does_not_exist")
    assert r.status_code == 404


def test_hw_cached_import_creates_trial(client):
    api, _, art_path = client
    _seed_artifact(art_path, "test_n8")

    r = api.post("/hw/cached/import", json={"name": "test_n8"})
    assert r.status_code == 200
    trial_id = r.json()["trial_id"]
    assert trial_id >= 1

    detail = api.get(f"/trials/{trial_id}").json()
    assert detail["kind"] == "qaoa_hw"
    assert detail["params"]["cached_name"] == "test_n8"
    assert detail["results"]["from_cache"] is True


# ---------------------------------------------------------------------------
# Submit + poll + ingest with the runtime service mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_token(tmp_path, monkeypatch):
    db_path = tmp_path / "trials_hw_token.db"
    monkeypatch.setenv("TRIALS_DB_PATH", str(db_path))
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "fake-token-not-real")

    import importlib
    import app.config
    import app.services.trials_store
    import app.services.hw
    import app.routers.hw
    import app.routers.qaoa
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.services.trials_store)
    importlib.reload(app.services.hw)
    importlib.reload(app.routers.hw)
    importlib.reload(app.routers.qaoa)
    importlib.reload(app.main)

    return TestClient(app.main.app), app


def test_hw_submit_then_poll_round_trip(client_with_token, monkeypatch):
    api, app_mod = client_with_token

    # 1) Seed a real qaoa_sim trial that /hw/submit can pull theta from.
    qaoa_resp = api.post("/qaoa/run", json={
        "N": 6, "K": 3, "lambda": 2.5, "P_K": 5.0, "P_R": 0.5, "seed": 11,
        "p": 1, "mixer": "x", "init_state": "uniform",
        "optimizer": "COBYLA", "max_iter": 30, "n_restarts": 2,
    })
    assert qaoa_resp.status_code == 200, qaoa_resp.text
    src_trial_id = qaoa_resp.json()["trial_id"]

    # 2) Mock the IBM runtime functions — never actually contact IBM.
    fake_job_id = "FAKEJOB-001"
    monkeypatch.setattr(
        app_mod.services.hw, "submit_qaoa_job",
        lambda **kw: {"job_id": fake_job_id, "backend_name": kw["backend_name"]},
    )

    sub = api.post("/hw/submit", json={
        "trial_id": src_trial_id, "backend_name": "ibm_fake_brisbane",
        "shots": 4096, "error_mitigation": {"readout": True},
    })
    assert sub.status_code == 200, sub.text
    body = sub.json()
    assert body["job_id"] == fake_job_id
    hw_trial_id = body["trial_id"]

    # 3) Poll while QUEUED — no counts yet.
    monkeypatch.setattr(
        app_mod.services.hw, "poll_job",
        lambda jid: {
            "status": "QUEUED", "backend": "ibm_fake_brisbane",
            "queue_position": 7, "est_start": "2026-04-29T13:00:00",
        },
    )
    p1 = api.get(f"/hw/job/{fake_job_id}").json()
    assert p1["status"] == "QUEUED"
    assert p1["queue_position"] == 7
    assert p1["trial_id"] == hw_trial_id

    # 4) Poll on completion — counts trigger ingestion + trial update.
    fake_counts = {"010101": 1100, "001011": 950, "100110": 700,
                   "111000": 500, "110001": 300, "010110": 246}
    monkeypatch.setattr(
        app_mod.services.hw, "poll_job",
        lambda jid: {
            "status": "DONE", "backend": "ibm_fake_brisbane",
            "queue_position": None, "est_start": None,
            "counts": fake_counts,
        },
    )
    p2 = api.get(f"/hw/job/{fake_job_id}").json()
    assert p2["status"] == "DONE"
    assert p2["results"]["status"] == "complete"
    assert p2["results"]["shots"] == sum(fake_counts.values())
    assert "top_bitstrings" in p2["results"]
    # Approximation ratio is computed (N=6 ≤ 18)
    assert p2["results"]["approx_ratio"] is not None

    # 5) Trial detail reflects the ingested counts.
    detail = api.get(f"/trials/{hw_trial_id}").json()
    assert detail["results"]["status"] == "complete"
    assert detail["results"]["job_id"] == fake_job_id
