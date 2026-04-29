"""Runtime configuration loaded from backend/.env.

The IBM Quantum token is read here ONCE at import time and never returned to
clients. Only the backend process holds it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    ibm_quantum_token: str | None
    ibm_quantum_instance: str
    ibm_quantum_channel: str
    frontend_origin: str
    trials_db_path: Path

    @property
    def has_ibm_credentials(self) -> bool:
        return bool(self.ibm_quantum_token)


def load_settings() -> Settings:
    token = os.getenv("IBM_QUANTUM_TOKEN") or None
    db_path_raw = os.getenv("TRIALS_DB_PATH", "trials.db")
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        db_path = BACKEND_DIR / db_path
    return Settings(
        ibm_quantum_token=token,
        ibm_quantum_instance=os.getenv("IBM_QUANTUM_INSTANCE", "ibm-q/open/main"),
        ibm_quantum_channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum"),
        frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
        trials_db_path=db_path,
    )


SETTINGS = load_settings()
