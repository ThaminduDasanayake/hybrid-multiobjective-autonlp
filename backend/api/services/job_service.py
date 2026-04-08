from api.models.job import JobConfig
from api.workers.job_manager import JobManager


class JobService:
    """Service to manage jobs."""

    def __init__(self):
        self.manager = JobManager()

    def create_job(self, config: JobConfig) -> str:
        """Create a new job from a validated JobConfig."""
        return self.manager.create_job(config.model_dump())

    def list_jobs(self):
        return self.manager.list_jobs()

    def get_result(self, job_id: str):
        return self.manager.get_result(job_id)

    def terminate_job(self, job_id: str):
        return self.manager.terminate_job(job_id)

    def delete_job(self, job_id: str):
        return self.manager.delete_job(job_id)

    def get_status(self, job_id: str):
        return self.manager.get_status(job_id)

    def get_logs(self, job_id: str, lines: int = 100) -> list[str]:
        return self.manager.get_logs(job_id, lines)
