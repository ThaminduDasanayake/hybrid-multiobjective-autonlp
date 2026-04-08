from fastapi import APIRouter, Query

from api.db import get_db
from api.models.ablation import AblationConfig
from api.services.ablation_service import AblationService

router = APIRouter()
service = AblationService()


@router.post("/ablations", status_code=202)
def start_ablation(config: AblationConfig):
    """Submit an ablation study; inherits config from the parent job.

    Idempotent: returns 200 if a non-failed result already exists.
    Duplicate-submission prevention uses a MongoDB atomic update_one so it
    is safe across multiple Uvicorn workers (unlike an in-process set).
    """
    return service.start_ablation(config)


@router.get("/ablations")
def get_ablations(
    parent_job_id: str | None = Query(
        default=None, description="Filter to a single parent job"
    ),
):
    """Return all completed ablation results, optionally filtered to a single parent job."""
    db = get_db()
    result: dict = {}

    query: dict = {"ablations": {"$ne": {}}}
    if parent_job_id:
        query["_id"] = parent_job_id

    for doc in db.jobs.find(query, {"ablations": 1}):
        doc_id = doc["_id"]
        for eff_mode, ablation in doc.get("ablations", {}).items():
            if not ablation:
                continue
            key = f"{eff_mode}_{doc_id}"
            result[key] = {
                "mode": ablation.get("mode", eff_mode),
                "dataset": ablation.get("dataset", "unknown"),
                "parent_job_id": doc_id,
                "disable_bo": ablation.get(
                    "disable_bo", ablation.get("config", {}).get("disable_bo", False)
                ),
                "status": ablation.get("status", "completed"),
                "metrics": ablation.get("metrics", {}),
                "runtime_seconds": ablation.get("runtime_seconds"),
            }

    return result
