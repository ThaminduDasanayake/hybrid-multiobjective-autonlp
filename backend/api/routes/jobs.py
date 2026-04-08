import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.models.job import JobConfig
from api.services.analytics_service import compute_hypervolume_history
from api.services.job_service import JobService

router = APIRouter()
service = JobService()

_TERMINAL_STATES = {"completed", "failed", "terminated"}


@router.post("/jobs", status_code=201)
def create_job(config: JobConfig):
    """Start a new AutoML job and return its ID."""
    job_id = service.create_job(config)
    return {"job_id": job_id}


@router.get("/jobs")
def list_jobs():
    """Return all jobs sorted by start time (newest first), enriched with dataset_name."""
    return service.list_jobs()


@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Return the result for a completed job, with config attached."""
    result = service.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@router.get("/jobs/{job_id}/hypervolume-history")
async def get_hypervolume_history(job_id: str):
    """Return hypervolume indicator at each GA generation for convergence plotting."""
    result = await asyncio.to_thread(service.get_result, job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return await asyncio.to_thread(compute_hypervolume_history, result)


@router.delete("/jobs/{job_id}", status_code=200)
def terminate_job(job_id: str):
    """Terminate a running or queued job."""
    if service.get_status(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    service.terminate_job(job_id)
    return {"message": "Job terminated"}


@router.delete("/jobs/{job_id}/data", status_code=200)
def delete_job_data(job_id: str):
    """Permanently delete a completed/failed/terminated job and its data."""
    status = service.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    if status.get("status") not in ("completed", "failed", "terminated"):
        raise HTTPException(status_code=409, detail="Job is still active")
    if not service.delete_job(job_id):
        raise HTTPException(status_code=500, detail="Failed to delete job data")
    return {"message": "Job deleted"}


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    """Server-Sent Events (SSE) stream for live job progress.

    The frontend subscribes to this endpoint immediately after submitting a job.
    It receives a push notification every time the job status or logs change,
    without needing to poll. The stream closes automatically when the job reaches
    a terminal state (completed / failed / terminated) or when the client disconnects.

    Diff-based emission: we compare each payload to the previous one and only send
    when something actually changed. This avoids flooding the frontend with identical
    frames during long-running BO evaluations where MongoDB hasn't been updated yet.
    """

    async def event_generator():
        last_payload_json: str | None = None

        while True:
            if await request.is_disconnected():
                break

            # pymongo is synchronous; offload to thread pool to avoid blocking the event loop.
            status = await asyncio.to_thread(service.get_status, job_id)

            if status is None:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            logs = await asyncio.to_thread(service.get_logs, job_id, 100)

            payload_json = json.dumps({"status": status, "logs": logs})

            # Only emit when the payload has changed to avoid redundant frames.
            if payload_json != last_payload_json:
                last_payload_json = payload_json
                yield f"data: {payload_json}\n\n"

            # Yield terminal state before closing so the client receives the final snapshot.
            if status.get("status") in _TERMINAL_STATES:
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
