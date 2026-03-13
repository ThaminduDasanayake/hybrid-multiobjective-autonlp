"""
FastAPI-specific job manager.

Replaces subprocess.Popen from utils/job_manager.py with
concurrent.futures.ProcessPoolExecutor, which is cross-platform and does not
require OS-specific signal groups or start_new_session flags.

IPC strategy — unchanged from the Streamlit era:
  - Worker writes progress to jobs/{job_id}/status.json on every callback.
  - FastAPI routes read that file on demand (polling from the React client).
  - Worker writes final results to jobs/{job_id}/result.json on completion.

Cancellation strategy:
  1. Write stop.signal file  → cooperative shutdown inside progress_callback.
  2. future.cancel()         → cancels queued-but-not-started jobs immediately.
  3. os.kill(pid, SIGTERM/SIGKILL) → force-kills an already-running worker via
     the PID the worker writes to status.json on startup.
"""

import os
import signal
import time
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.job_manager import JobManager
from utils.logger import get_logger

logger = get_logger("api_job_manager")

# Module-level singletons — one executor shared across all HTTP requests.
# max_workers=2: enough to run one job while a second waits in the queue.
# AutoML is extremely CPU-heavy, so more workers would just thrash the CPU.
_executor = ProcessPoolExecutor(max_workers=2)
_futures: dict[str, Future] = {}

# Absolute path to the backend root — passed to worker processes so they can
# reconstruct sys.path in 'spawn' mode (the macOS default since Python 3.12).
_BACKEND_ROOT = str(Path(__file__).parent.parent.resolve())


class APIJobManager(JobManager):
    """
    Job manager for the FastAPI backend.

    Inherits all file-management helpers (get_status, update_status,
    get_result, list_jobs, _get_job_dir) from utils.JobManager and overrides
    only the process-management methods.
    """

    # ----------------------------------------------------------------- create

    def create_job(self, config: dict[str, Any]) -> str:
        import json
        import shutil

        # Import here to avoid loading it in every module that imports this class.
        from api.worker_fn import run_automl_job

        job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
        job_dir = self._get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Wipe any stale checkpoint from a previous run with the same job_id.
        # Without this, ResultStore.load_checkpoint() would silently reuse old
        # cached evaluations, causing the GA to finish in milliseconds and
        # leaving objective_ranges at their float("inf") sentinel values.
        checkpoint_dir = job_dir / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Cleared stale checkpoint directory for {job_id}")
        checkpoint_dir.mkdir(parents=True)

        config_path = job_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Status file initialised here; the worker will overwrite it with its
        # PID and set status → "running" as soon as it starts.
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

    # --------------------------------------------------------------- terminate

    def terminate_job(self, job_id: str) -> bool:
        # 1. Cooperative stop: worker checks this file in its progress callback.
        stop_file = self._get_job_dir(job_id) / "stop.signal"
        try:
            stop_file.touch()
        except Exception as e:
            logger.warning(f"Could not write stop.signal for {job_id}: {e}")

        # 2. Cancel if the job hasn't started yet (still sitting in the queue).
        future = _futures.get(job_id)
        if future:
            future.cancel()

        # 3. Force-kill the worker process via the PID it wrote to status.json.
        status = self.get_status(job_id)
        if not status:
            logger.error(f"No status found for job {job_id}")
            return False

        pid = status.get("pid")
        if pid:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                    logger.info(f"Sent {sig.name} to PID {pid} (job {job_id})")
                except ProcessLookupError:
                    break  # Process already gone — that's fine.
                except Exception as e:
                    logger.warning(f"Could not send {sig.name} to PID {pid}: {e}")
                if sig == signal.SIGTERM:
                    time.sleep(2)  # Give the process a moment to exit cleanly.

        status["status"] = "terminated"
        status["message"] = "Job was manually terminated"
        self.update_status(job_id, status)
        logger.info(f"Job {job_id} marked as terminated")
        return True

    # ----------------------------------------------------------------- resume

    def resume_job(self, job_id: str) -> bool:
        from api.worker_fn import run_automl_job

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

        # Clear any leftover stop signal from the previous run.
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

    # ------------------------------------------------------------------- logs

    def get_logs(self, job_id: str, lines: int = 100) -> list[str]:
        """Return the last *lines* lines from the job's rotating log file."""
        import glob as _glob

        log_dir = Path(_BACKEND_ROOT) / "logs"
        # RotatingFileHandler creates e.g. run_job_xxx.log, run_job_xxx.log.1 …
        pattern = str(log_dir / f"run_{job_id}.log")
        matches = sorted(_glob.glob(pattern))
        if not matches:
            return []
        try:
            with open(matches[0]) as f:
                all_lines = f.readlines()
            return [line.rstrip() for line in all_lines[-lines:]]
        except Exception as e:
            logger.error(f"Error reading logs for {job_id}: {e}")
            return []
