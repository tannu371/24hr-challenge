"""FastAPI entrypoint.

Run from /backend directory:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-level `portfolio/` package importable when uvicorn is launched
# from the backend/ directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import SETTINGS
from .routers import classical, exports, hw, problem, qaoa, trials


app = FastAPI(
    title="Hybrid Quantum-Classical Portfolio API",
    version="0.1.0",
    description="Backend for the 24hr-challenge portfolio optimiser.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[SETTINGS.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["meta"])
def healthz() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "ibm_credentials_configured": SETTINGS.has_ibm_credentials,
    }


app.include_router(problem.router)
app.include_router(classical.router)
app.include_router(qaoa.router)
app.include_router(hw.router)
app.include_router(exports.router)
app.include_router(trials.router)
