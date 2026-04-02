"""
Job manager for the FastAPI backend.

Runs AutoML jobs in a ProcessPoolExecutor so they never block the async event loop.
All job state lives in MongoDB; the filesystem is used only for log files and dataset caches.

Cancellation is two-tier: setting stop_requested=True in MongoDB triggers cooperative exit
inside the worker; future.cancel() cancels jobs that are still queued.
"""

import time
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from api.db import get_db, get_db_name, get_mongo_uri
from utils.logger import get_logger

logger = get_logger("job_manager")

# One executor shared across all HTTP requests.
# max_workers=1: sequential execution ensures each job gets full CPU resources,
# producing fair and comparable runtime/latency metrics for ablation studies.
_executor = ProcessPoolExecutor(max_workers=1)
_futures: dict[str, Future] = {}

# Absolute path to the backend root — passed to worker processes so they can
# reconstruct sys.path under the 'spawn' start method (macOS default, Python 3.12+).
_BACKEND_ROOT = str(Path(__file__).parent.parent.resolve())

# Fields excluded from status reads (large payload fields).
_STATUS_PROJECTION = {"result": 0, "ablations": 0}


class JobManager:
    """Manages background AutoML jobs using MongoDB and a ProcessPoolExecutor."""

    def __init__(self) -> None:
        self.db = get_db()
        self.jobs = self.db.jobs

    # reads

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        """Return status fields for a job (excluding large result/ablations payloads), or None."""
        doc = self.jobs.find_one({"_id": job_id}, _STATUS_PROJECTION)
        if doc is None:
            return None
        doc["job_id"] = doc.pop("_id")
        doc.pop("config", None)          # not needed in status response
        doc.pop("stop_requested", None)  # internal field, not for frontend
        return doc

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        """Return the result dict (with config attached) for a completed job, or None."""
        doc = self.jobs.find_one({"_id": job_id}, {"result": 1, "config": 1})
        if doc is None or doc.get("result") is None:
            return None
        result = doc["result"]
        result["config"] = doc.get("config")
        return result

    def list_jobs(self) -> dict[str, dict[str, Any]]:
        """Return all jobs sorted by start_time descending, with dataset_name flattened in."""
        cursor = self.jobs.find({}, _STATUS_PROJECTION).sort("start_time", -1)

        jobs: dict[str, dict[str, Any]] = {}
        for doc in cursor:
            job_id = doc.pop("_id")
            # Extract dataset_name from nested config and flatten it.
            config = doc.pop("config", {})
            doc["dataset_name"] = config.get("dataset_name", "")
            doc.pop("stop_requested", None)  # internal field, not for frontend
            doc["job_id"] = job_id
            jobs[job_id] = doc
        return jobs

    # process management

    def create_job(self, config: dict[str, Any]) -> str:
        """Create a new AutoML job and submit it to the executor."""
        from api.worker import run_automl_job

        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        document = {
            "_id": job_id,
            "config": config,
            "status": "created",
            "start_time": time.time(),
            "last_updated": time.time(),
            "progress": 0,
            "current_generation": 0,
            "total_generations": config.get("n_generations", 10),
            "message": "Initializing...",
            "best_f1": 0.0,
            "best_latency_ms": 0.0,
            "best_interpretability": 0.0,
            "cache_hit_rate": 0.0,
            "total_evaluated": 0,
            "result": None,
            "ablations": {},
            "stop_requested": False,
        }
        self.jobs.insert_one(document)

        future = _executor.submit(
            run_automl_job,
            job_id,
            _BACKEND_ROOT,
            get_mongo_uri(),
            get_db_name(),
        )
        _futures[job_id] = future
        logger.info(f"Submitted job {job_id} to ProcessPoolExecutor")
        return job_id

    def terminate_job(self, job_id: str) -> bool:
        """Signal cooperative termination via MongoDB; cancel the future if still queued."""
        result = self.jobs.update_one(
            {"_id": job_id},
            {"$set": {
                "stop_requested": True,
                "status": "terminated",
                "message": "Job was manually terminated",
                "last_updated": time.time(),
            }},
        )
        if result.matched_count == 0:
            logger.error(f"No job found for {job_id}")
            return False

        future = _futures.get(job_id)
        if future:
            future.cancel()

        logger.info(f"Job {job_id} marked as terminated")
        return True

    def delete_job(self, job_id: str) -> bool:
        """Permanently delete a terminal job from MongoDB and remove its log file."""
        if not job_id or "/" in job_id or "\\" in job_id or job_id.startswith("."):
            logger.warning(f"Invalid job_id format: {job_id}")
            return False

        status = self.get_status(job_id)
        if not status:
            return False
        if status.get("status") not in ("completed", "failed", "terminated"):
            return False

        fut = _futures.get(job_id)
        if fut and not fut.done():
            try:
                fut.result(timeout=10)
            except Exception:
                logger.warning(f"Worker for {job_id} did not finish cleanly within timeout")
                return False

        self.jobs.delete_one({"_id": job_id})

        log_path = Path(_BACKEND_ROOT) / "logs" / f"run_{job_id}.log"
        try:
            if log_path.exists():
                log_path.unlink()
        except OSError as e:
            logger.warning(f"Failed to delete log file for {job_id}: {e}")

        _futures.pop(job_id, None)
        logger.info(f"Deleted job {job_id} and associated data")
        return True

    def get_logs(self, job_id: str, lines: int = 100) -> list[str]:
        """Return the last *lines* lines from the job's rotating log file."""
        import glob as _glob

        log_dir = Path(_BACKEND_ROOT) / "logs"
        matches = sorted(_glob.glob(str(log_dir / f"run_{job_id}.log")))
        if not matches:
            return []
        try:
            with open(matches[0]) as f:
                all_lines = f.readlines()
            return [line.rstrip() for line in all_lines[-lines:]]
        except Exception as e:
            logger.error(f"Error reading logs for {job_id}: {e}")
            return []
