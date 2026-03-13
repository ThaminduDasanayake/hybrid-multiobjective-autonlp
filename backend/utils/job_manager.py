import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import get_logger

logger = get_logger("job_manager")


class JobManager:
    """
    Manages background AutoML jobs.
    Uses file-based state to track progress and results.
    """

    def __init__(self, jobs_dir: str = "jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _get_job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def create_job(self, config: Dict[str, Any]) -> str:
        """Not implemented — use APIJobManager instead."""
        raise NotImplementedError(
            "JobManager.create_job() is not implemented. "
            "Use APIJobManager (api/job_manager.py) to start jobs."
        )

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a job."""
        status_path = self._get_job_dir(job_id) / "status.json"
        if not status_path.exists():
            return None

        try:
            with open(status_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading status for job {job_id}: {e}")
            return None

    def update_status(self, job_id: str, status: Dict[str, Any]):
        """Update status file for a job using atomic write to prevent race conditions."""
        import tempfile

        status_path = self._get_job_dir(job_id) / "status.json"
        status["last_updated"] = time.time()

        try:
            # Write to temp file first, then atomically replace
            fd, tmp_path = tempfile.mkstemp(dir=status_path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(status, f, indent=2)
            os.replace(tmp_path, status_path)
        except Exception as e:
            logger.exception(f"Error updating status for job {job_id}: {e}")
            # Clean up temp file on failure; suppress errors so callers are not affected
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.warning(f"Failed to remove temp file {tmp_path}")

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load result from completed job."""
        result_path = self._get_job_dir(job_id) / "result.json"
        # Note: worker saves result.json (or pickle if complex objects needed, but JSON is safer for inter-process)
        # We'll use JSON for the main results compatible with the UI

        if result_path.exists():
            try:
                with open(result_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading result for job {job_id}: {e}")
                return None
        return None

    def terminate_job(self, job_id: str) -> bool:
        """
        Terminate a running job by killing its worker process group.

        1. Writes a stop.signal file for cooperative shutdown.
        2. Sends SIGTERM to the entire process group (catches joblib workers too).
        3. Waits up to 2 s, then sends SIGKILL to any survivors.
        4. Falls back to psutil for any processes that escaped the group.

        Returns True if termination was handled, False on error.
        """
        status = self.get_status(job_id)
        if not status:
            logger.error(f"Cannot terminate job {job_id}: status not found")
            return False

        pid = status.get("pid")
        if not pid:
            logger.error(f"Cannot terminate job {job_id}: no PID recorded in status")
            return False

        # --- cooperative stop: worker checks this file in its callback ---
        stop_file = self._get_job_dir(job_id) / "stop.signal"
        try:
            stop_file.touch()
        except Exception as e:
            logger.warning(f"Could not write stop.signal for job {job_id}: {e}")

        # --- primary: kill the whole process group atomically ---
        # The worker is launched with start_new_session=True so its pgid == pid.
        # All joblib/sklearn child processes inherit the same pgid.
        try:
            os.killpg(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to process group {pid} for job {job_id}")
        except ProcessLookupError:
            logger.warning(f"Process group {pid} for job {job_id} not found (already exited)")
        except PermissionError as e:
            logger.error(f"Permission denied killing process group {pid}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending SIGTERM to group {pid}: {e}")

        # Give processes a moment to shut down gracefully
        time.sleep(2.0)

        # Force-kill anything still alive in the group
        try:
            os.killpg(pid, signal.SIGKILL)
            logger.info(f"Sent SIGKILL to process group {pid} for job {job_id}")
        except ProcessLookupError:
            pass  # Group is already gone — that's fine
        except Exception as e:
            logger.warning(f"SIGKILL to group {pid} failed: {e}")

        # --- fallback: psutil sweep for any processes outside the group ---
        try:
            import psutil
            try:
                proc = psutil.Process(pid)
                survivors = proc.children(recursive=True) + [proc]
                for p in survivors:
                    try:
                        p.kill()
                        logger.info(f"psutil force-killed PID {p.pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except psutil.NoSuchProcess:
                pass  # Already gone
        except ImportError:
            pass  # psutil not installed; os.killpg above is sufficient

        # Mark job as terminated
        status["status"] = "terminated"
        status["message"] = "Job was manually terminated"
        self.update_status(job_id, status)
        logger.info(f"Job {job_id} marked as terminated")
        return True

    def resume_job(self, job_id: str) -> bool:
        """
        Resume a previously interrupted job from its checkpoint.

        Re-launches the worker with the same job_id and config so that
        EvolutionarySearch.run() picks up the saved population and
        completed_generations from checkpoint.pkl.

        Returns True if the worker was launched, False on error.
        """
        job_dir = self._get_job_dir(job_id)
        config_path = job_dir / "config.json"

        if not config_path.exists():
            logger.error(f"Cannot resume job {job_id}: config.json not found")
            return False

        checkpoint_path = job_dir / "checkpoints" / "checkpoint.pkl"
        if not checkpoint_path.exists():
            logger.warning(
                f"No checkpoint found for job {job_id}; job will restart from scratch"
            )

        # Mark as running again
        status = self.get_status(job_id) or {"job_id": job_id}
        if status.get("status") == "running":
            logger.warning(f"Job {job_id} is already running; resume request ignored")
            return False
        status["status"] = "running"
        status["message"] = "Resuming from checkpoint..."
        self.update_status(job_id, status)

        root_dir = Path(__file__).parent.parent
        worker_script = root_dir / "worker.py"

        cmd = [
            sys.executable,
            str(worker_script),
            "--job-id",
            job_id,
            "--config",
            str(config_path),
            "--jobs-dir",
            str(self.jobs_dir),
        ]

        env = os.environ.copy()
        env["AUTOML_LOG_FILE"] = f"run_{job_id}.log"

        logger.info(f"Resuming worker for job {job_id}: {' '.join(cmd)}")
        try:
            subprocess.Popen(cmd, cwd=str(root_dir), env=env, start_new_session=True)
            return True
        except Exception as e:
            logger.exception(f"Failed to resume worker for job {job_id}: {e}")
            status["status"] = "failed"
            status["message"] = "Failed to launch resume worker"
            self.update_status(job_id, status)
            return False

        # logger.info(f"Resuming worker for job {job_id}: {' '.join(cmd)}")
        # subprocess.Popen(cmd, cwd=str(root_dir), env=env, start_new_session=True)
        # return True

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        """List all known jobs and their basic status."""
        jobs = {}
        if not self.jobs_dir.exists():
            return jobs

        for job_dir in self.jobs_dir.iterdir():
            if job_dir.is_dir():
                status = self.get_status(job_dir.name)
                if status:
                    jobs[job_dir.name] = status

        # Sort by start time desc
        return dict(
            sorted(jobs.items(), key=lambda x: x[1].get("start_time", 0), reverse=True)
        )
