"""
Job manager for the FastAPI backend.

Runs AutoML jobs in a ProcessPoolExecutor so they never block the async
event loop.  All job state lives on disk (file-based IPC) so the server
can be restarted without losing in-progress job metadata.

IPC:
  - Worker writes progress to  jobs/{job_id}/status.json  on every callback.
  - FastAPI routes read that file on demand (SSE polling from the React client).
  - Worker writes final results to  jobs/{job_id}/result.json  on completion.

Cancellation (2-tier, cooperative):
  1. Write stop.signal file  → progress_callback raises _TerminationRequested.
  2. future.cancel()         → cancels queued-but-not-yet-started jobs.
"""

import json
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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


class JobManager:
    """Manages background AutoML jobs using file-based state and a ProcessPoolExecutor."""

    def __init__(self, jobs_dir: str = "jobs") -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers

    def _get_job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    # ------------------------------------------------------------------ file I/O

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        """Read jobs/{job_id}/status.json. Returns None if missing or unreadable."""
        status_path = self._get_job_dir(job_id) / "status.json"
        if not status_path.exists():
            return None
        try:
            with open(status_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading status for {job_id}: {e}")
            return None

    def update_status(self, job_id: str, status: dict[str, Any]) -> None:
        """Atomically write status to jobs/{job_id}/status.json.

        Uses a temp file + os.replace() so a concurrent reader never sees a
        partial write.
        """
        import tempfile

        status_path = self._get_job_dir(job_id) / "status.json"
        status["last_updated"] = time.time()
        try:
            fd, tmp_path = tempfile.mkstemp(dir=status_path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(status, f, indent=2)
            os.replace(tmp_path, status_path)
        except Exception as e:
            logger.exception(f"Error updating status for {job_id}: {e}")
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        """Read jobs/{job_id}/result.json. Returns None if missing or unreadable."""
        result_path = self._get_job_dir(job_id) / "result.json"
        if not result_path.exists():
            return None
        try:
            with open(result_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading result for {job_id}: {e}")
            return None

    def list_jobs(self) -> dict[str, dict[str, Any]]:
        """List all job directories, sorted by start_time descending."""
        if not self.jobs_dir.exists():
            return {}
        jobs = {}
        for job_dir in self.jobs_dir.iterdir():
            if job_dir.is_dir():
                status = self.get_status(job_dir.name)
                if status:
                    jobs[job_dir.name] = status
        return dict(
            sorted(jobs.items(), key=lambda x: x[1].get("start_time", 0), reverse=True)
        )

    # ------------------------------------------------------------------ process management

    def create_job(self, config: dict[str, Any]) -> str:
        """Create a new AutoML job, write its config, and submit it to the executor."""
        import shutil

        from api.worker import run_automl_job

        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        job_dir = self._get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Clear any stale checkpoint so ResultStore.load_checkpoint() starts fresh.
        # Without this, a reused job_id would silently replay cached evaluations,
        # causing the GA to finish in milliseconds with sentinel objective_ranges.
        checkpoint_dir = job_dir / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Cleared stale checkpoint directory for {job_id}")
        checkpoint_dir.mkdir(parents=True)

        with open(job_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        status = {
            "job_id": job_id,
            "status": "created",
            "start_time": time.time(),
            "progress": 0,
            "current_generation": 0,
            "total_generations": config.get("n_generations", 10),
            "message": "Initializing...",
            "best_f1": 0.0,
        }
        self.update_status(job_id, status)

        future = _executor.submit(
            run_automl_job,
            job_id,
            str(self.jobs_dir.resolve()),
            _BACKEND_ROOT,
        )
        _futures[job_id] = future
        logger.info(f"Submitted job {job_id} to ProcessPoolExecutor")
        return job_id

    def terminate_job(self, job_id: str) -> bool:
        """Stop a job cooperatively, or cancel it if still queued.

        1. Write stop.signal  → worker's progress_callback detects it and exits cleanly.
        2. future.cancel()    → cancels the job if it hasn't started yet.

        Returns True if the job was found and marked terminated.
        """
        stop_file = self._get_job_dir(job_id) / "stop.signal"
        try:
            stop_file.touch()
        except Exception as e:
            logger.warning(f"Could not write stop.signal for {job_id}: {e}")

        future = _futures.get(job_id)
        if future:
            future.cancel()

        status = self.get_status(job_id)
        if not status:
            logger.error(f"No status found for job {job_id}")
            return False

        status["status"] = "terminated"
        status["message"] = "Job was manually terminated"
        self.update_status(job_id, status)
        logger.info(f"Job {job_id} marked as terminated")
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a stopped or failed job from its checkpoint.

        Returns False if the config is missing or the job is already running.
        """
        from api.worker import run_automl_job

        job_dir = self._get_job_dir(job_id)
        if not (job_dir / "config.json").exists():
            logger.error(f"Cannot resume {job_id}: config.json not found")
            return False

        status = self.get_status(job_id)
        if not status:
            logger.error(f"Cannot resume {job_id}: status not found")
            return False
        if status.get("status") == "running":
            logger.warning(f"Job {job_id} is already running; ignoring resume")
            return False

        stop_file = job_dir / "stop.signal"
        if stop_file.exists():
            stop_file.unlink()

        status["status"] = "running"
        status["message"] = "Resuming from checkpoint..."
        self.update_status(job_id, status)

        future = _executor.submit(
            run_automl_job,
            job_id,
            str(self.jobs_dir.resolve()),
            _BACKEND_ROOT,
        )
        _futures[job_id] = future
        logger.info(f"Resumed job {job_id}")
        return True

    def delete_job(self, job_id: str) -> bool:
        """Permanently delete a job's data from disk.

        Only jobs in a terminal state (completed, failed, terminated) can be
        deleted.  Returns True on success, False if the job doesn't exist or
        is still active.
        """
        import shutil

        if not job_id or "/" in job_id or "\\" in job_id or job_id.startswith("."):
            logger.warning(f"Invalid job_id format: {job_id}")
            return False

        status = self.get_status(job_id)
        if not status:
            return False
        if status.get("status") not in ("completed", "failed", "terminated"):
            return False

        job_dir = self._get_job_dir(job_id)
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except OSError as e:
            logger.error(f"Failed to delete job directory for {job_id}: {e}")
            return False

        results_dir = Path(_BACKEND_ROOT) / "results" / job_id
        try:
            if results_dir.exists():
                shutil.rmtree(results_dir)
        except OSError as e:
            logger.warning(f"Failed to delete results directory for {job_id}: {e}")

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
