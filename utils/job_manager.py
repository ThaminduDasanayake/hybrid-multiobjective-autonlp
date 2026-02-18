import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
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
        """
        Create a new job and start the worker process.
        
        Args:
            config: Configuration dictionary for the AutoML run
            
        Returns:
            job_id: Unique identifier for the job
        """
        # Generate job_id based on timestamp (e.g., 20240214_153000)
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{job_id}" # Prefix to ensure valid folder/file names
        
        job_dir = self._get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = job_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        # Initialize status
        status = {
            "job_id": job_id,
            "status": "created",
            "start_time": time.time(),
            "progress": 0,
            "current_generation": 0,
            "total_generations": config.get("n_generations", 10),
            "message": "Initializing...",
            "best_f1": 0.0
        }
        self.update_status(job_id, status)
        
        # Start worker process
        # We assume worker.py is in the root directory
        root_dir = Path(__file__).parent.parent
        worker_script = root_dir / "worker.py"
        
        cmd = [
            sys.executable,
            str(worker_script),
            "--job-id", job_id,
            "--config", str(config_path),
            "--jobs-dir", str(self.jobs_dir)
        ]
        
        logger.info(f"Starting worker for job {job_id}: {' '.join(cmd)}")
        
        # Prepare environment for the worker
        env = os.environ.copy()
        env["AUTOML_LOG_FILE"] = f"run_{job_id}.log"
        
        # Start detached process
        subprocess.Popen(
            cmd,
            cwd=str(root_dir),
            env=env,
            # Detach process (platform dependent)
            # On Windows: creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            # On Unix: start_new_session=True
            start_new_session=True
        )
        
        return job_id
        
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
        """Update status file for a job."""
        status_path = self._get_job_dir(job_id) / "status.json"
        status["last_updated"] = time.time()
        
        try:
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating status for job {job_id}: {e}")

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
        return dict(sorted(jobs.items(), key=lambda x: x[1].get("start_time", 0), reverse=True))
