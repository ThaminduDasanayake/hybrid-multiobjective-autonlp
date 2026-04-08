import logging

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from api.db import get_db, get_db_name, get_mongo_uri
from api.models.ablation import AblationConfig
from api.workers.job_manager import _executor
from api.workers.paths import BACKEND_ROOT


def _effective_mode(mode: str, disable_bo: bool) -> str:
    """Map (mode, disable_bo) to the canonical ablation key."""
    if mode == "random_search":
        return "random_search"
    return "ga_only" if disable_bo else mode


class AblationService:
    def start_ablation(self, config: AblationConfig):
        """Submit an ablation study; inherits config from the parent job.

        Idempotent: returns 200 if a non-failed result already exists.
        Duplicate-submission prevention uses a MongoDB atomic update_one so it
        is safe across multiple Uvicorn workers (unlike an in-process set).
        """
        from api.workers.worker import run_ablation

        pid = config.parent_job_id
        if not pid or "/" in pid or "\\" in pid or pid.startswith("."):
            raise HTTPException(status_code=400, detail="Invalid parent_job_id")

        db = get_db()
        parent_doc = db.jobs.find_one({"_id": pid}, {"config": 1})
        if parent_doc is None:
            raise HTTPException(status_code=404, detail="Parent job not found")

        eff_mode = _effective_mode(config.mode, config.disable_bo)
        parent_config = parent_doc["config"]

        # Atomically claim the ablation slot: only succeeds when the slot is absent
        # or previously failed. Any process that wins this update may submit to the
        # executor; all others observe modified_count == 0 and back off.
        claim = db.jobs.update_one(
            {
                "_id": pid,
                "$or": [
                    {f"ablations.{eff_mode}": {"$exists": False}},
                    {f"ablations.{eff_mode}.status": "failed"},
                ],
            },
            {
                "$set": {
                    f"ablations.{eff_mode}": {
                        "status": "queued",
                        "mode": config.mode,
                        "disable_bo": config.disable_bo,
                    }
                }
            },
        )

        if claim.modified_count == 0:
            # Slot already taken — determine whether queued or completed.
            doc = db.jobs.find_one({"_id": pid}, {f"ablations.{eff_mode}": 1})
            existing = (doc or {}).get("ablations", {}).get(eff_mode, {})
            if existing.get("status") == "queued":
                return {
                    "status": "already_queued",
                    "mode": config.mode,
                    "parent_job_id": pid,
                    "disable_bo": config.disable_bo,
                }
            return JSONResponse(
                status_code=200,
                content={
                    "status": "completed",
                    "mode": config.mode,
                    "parent_job_id": pid,
                    "disable_bo": config.disable_bo,
                },
            )

        def _on_done(fut):
            exc = fut.exception()
            if exc:
                logging.getLogger("ablation_service").error(
                    f"Ablation {eff_mode}_{pid} crashed: {exc}",
                    exc_info=exc,
                )

        future = _executor.submit(
            run_ablation,
            config.mode,
            pid,
            parent_config["dataset_name"],
            config.disable_bo,
            parent_config.get("max_samples", 2000),
            parent_config.get("population_size", 20),
            parent_config.get("n_generations", 10),
            parent_config.get("bo_calls", 15),
            parent_config.get("seed", 42),
            str(BACKEND_ROOT),
            get_mongo_uri(),
            get_db_name(),
        )
        future.add_done_callback(_on_done)

        return {
            "status": "queued",
            "mode": config.mode,
            "parent_job_id": pid,
            "disable_bo": config.disable_bo,
        }
