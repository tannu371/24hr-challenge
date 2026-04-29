"""Phase-5 acceptance tests.

Beyond shape checks we *execute* the generated Qiskit and PennyLane scripts in
fresh subprocesses to prove they reproduce the original trial energy.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client_with_trial(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("trials") / "trials_export.db"
    os.environ["TRIALS_DB_PATH"] = str(db_path)

    import importlib
    import app.config
    import app.services.trials_store
    import app.routers.classical
    import app.routers.qaoa
    import app.routers.exports
    import app.routers.trials
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.services.trials_store)
    importlib.reload(app.routers.classical)
    importlib.reload(app.routers.qaoa)
    importlib.reload(app.routers.exports)
    importlib.reload(app.routers.trials)
    importlib.reload(app.main)

    api = TestClient(app.main.app)
    qaoa = api.post("/qaoa/run", json={
        "N": 6, "K": 3, "lambda": 2.5, "P_K": 5.0, "P_R": 0.5, "seed": 11,
        "p": 1, "mixer": "x", "init_state": "uniform",
        "optimizer": "COBYLA", "max_iter": 30, "n_restarts": 2,
    }).json()
    return api, qaoa


# ---------------------------------------------------------------------------
# 5.1 — QASM 3
# ---------------------------------------------------------------------------


def test_export_qasm_text(client_with_trial):
    api, qaoa = client_with_trial
    r = api.get(f"/export/qasm/{qaoa['trial_id']}")
    assert r.status_code == 200
    text = r.text
    assert text.startswith("OPENQASM 3"), text[:80]
    assert "rz" in text or "rx" in text  # cost / mixer rotations present
    assert "measure" in text


def test_export_qasm_404_for_unknown_trial(client_with_trial):
    api, _ = client_with_trial
    r = api.get("/export/qasm/9999")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5.2 — Standalone Qiskit .py reproduces the trial energy
# ---------------------------------------------------------------------------


def test_export_qiskit_script_runs_and_matches(client_with_trial, tmp_path):
    api, qaoa = client_with_trial
    r = api.get(f"/export/qiskit/{qaoa['trial_id']}")
    assert r.status_code == 200
    script = r.text
    assert "from qiskit" in script
    assert "Statevector" in script

    target_energy = float(qaoa["energy_star"])
    script_path = tmp_path / "qaoa_trial.py"
    script_path.write_text(script)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    m = re.search(r"<H_C>\s*=\s*([\-0-9eE.+]+)", proc.stdout)
    assert m is not None, proc.stdout
    energy = float(m.group(1))
    assert abs(energy - target_energy) < 1e-6, (energy, target_energy)


# ---------------------------------------------------------------------------
# 5.3 — Standalone PennyLane .py reproduces the trial energy
# ---------------------------------------------------------------------------


def test_export_pennylane_script_runs_and_matches(client_with_trial, tmp_path):
    api, qaoa = client_with_trial
    r = api.get(f"/export/pennylane/{qaoa['trial_id']}")
    assert r.status_code == 200
    script = r.text
    assert "import pennylane" in script
    assert "qaoa" in script

    target_energy = float(qaoa["energy_star"])
    script_path = tmp_path / "qaoa_pennylane.py"
    script_path.write_text(script)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    m = re.search(r"<H_C>\s*=\s*([\-0-9eE.+]+)", proc.stdout)
    assert m is not None, proc.stdout
    energy = float(m.group(1))
    assert abs(energy - target_energy) < 1e-5, (energy, target_energy)


# ---------------------------------------------------------------------------
# 5.4 — Circuit SVG/PNG, pre + post transpilation
# ---------------------------------------------------------------------------


def test_export_circuit_svg_pre(client_with_trial):
    api, qaoa = client_with_trial
    r = api.get(f"/export/circuit/{qaoa['trial_id']}.svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in r.content[:2000].lower() or b"<?xml" in r.content[:200]


def test_export_circuit_png_post_transpiled(client_with_trial):
    api, qaoa = client_with_trial
    r = api.get(f"/export/circuit/{qaoa['trial_id']}.png?transpiled=true")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG\r\n\x1a\n")


# ---------------------------------------------------------------------------
# 5.5 — Plots: SVG + PNG + CSV for each kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["trajectory", "histogram", "comparison"])
@pytest.mark.parametrize("fmt", ["svg", "png", "csv"])
def test_export_plot(client_with_trial, kind, fmt):
    api, qaoa = client_with_trial
    r = api.get(f"/export/plot/{qaoa['trial_id']}/{kind}.{fmt}")
    assert r.status_code == 200, r.text
    if fmt == "csv":
        assert r.headers["content-type"].startswith("text/csv")
        assert "," in r.text
    elif fmt == "svg":
        assert r.headers["content-type"].startswith("image/svg")
        assert len(r.content) > 1000
    else:
        assert r.headers["content-type"] == "image/png"
        assert r.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_export_plot_landscape_csv_only(client_with_trial):
    """Landscape involves rebuilding QAOA — slow; just check CSV path works."""
    api, qaoa = client_with_trial
    r = api.get(f"/export/plot/{qaoa['trial_id']}/landscape.csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    rows = r.text.strip().split("\n")
    assert rows[0].startswith("gamma,beta,energy")
    # 41 × 41 grid plus header.
    assert len(rows) == 41 * 41 + 1


# ---------------------------------------------------------------------------
# 5.6 — Full bundle zip
# ---------------------------------------------------------------------------


def test_export_bundle_contains_everything(client_with_trial):
    api, qaoa = client_with_trial
    r = api.get(f"/export/bundle/{qaoa['trial_id']}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert f"trial_{qaoa['trial_id']}.json" in names
    assert any(n.endswith(".qasm") for n in names)
    assert any(n.endswith("_pennylane.py") for n in names)
    assert any(n.startswith("circuit_") and n.endswith(".svg") for n in names)
    for k in ("trajectory", "histogram", "comparison"):
        assert any(n.startswith(f"plots/{k}.") for n in names), (k, names)
