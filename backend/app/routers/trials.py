"""Trial history endpoints (Task 1.4).

GET /trials       — list newest-first (id, kind, created_at, summary)
GET /trials/{id}  — full record (params + results)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..services.trials_store import TrialsStore


router = APIRouter(prefix="/trials", tags=["trials"])


@router.get("")
def list_trials(limit: int = 200) -> list[dict]:
    return TrialsStore().list(limit=limit)


@router.get("/{trial_id}")
def get_trial(trial_id: int) -> dict:
    record = TrialsStore().get(trial_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"trial {trial_id} not found")
    return record


@router.delete("/{trial_id}")
def delete_trial(trial_id: int) -> dict:
    """Hard-delete the trial row. Idempotent — 404 only if it never existed
    *and* hasn't already been deleted in this session."""
    deleted = TrialsStore().delete(trial_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"trial {trial_id} not found")
    return {"trial_id": trial_id, "deleted": True}
