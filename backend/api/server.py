"""
THE RECEPTIONIST — server.py
==============================
This is the FastAPI application that handles all communication between the React
frontend and the backend ML engine. It does two things:

1. REST endpoints: Accept job configuration from the UI, start background AutoML
   jobs, serve results, and handle ablation study requests.

2. SSE stream: Push live generation-by-generation progress updates to the frontend
   while a job is running, without the client needing to poll.

The server itself never touches any ML code directly. Heavy computation runs in
a ProcessPoolExecutor (managed by job_manager.py), keeping the async event loop
free to handle other requests. The server reads job status from MongoDB and forwards
it to whoever is listening.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Ensure the backend root (the directory containing automl/, utils/, etc.) is on
# sys.path so that imports work when the server is started from the repo root.
_BACKEND_ROOT = str(Path(__file__).parent.parent.resolve())
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.db import get_db, get_db_name, get_mongo_uri
from api.job_manager import JobManager, _executor

# Note: duplicate-submission prevention is handled by an atomic MongoDB update_one
# inside start_ablation(), so no in-process set is needed here.


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Shut down the ProcessPoolExecutor gracefully when the server stops.

    Two-tier shutdown strategy:
    - Queued jobs (not yet started): cancelled immediately via cancel_futures=True.
    - Running jobs (in progress): left to finish naturally. We don't want to abruptly
      kill an 8-minute AutoML run because the server is restarting — the user's results
      would be lost. wait=False returns control immediately so uvicorn can exit.
    """
    yield
    # cancel_futures=True cancels queued (not yet running) futures.
    # Running futures (i.e. active AutoML jobs) are left to finish on their own
    # because we don't want to abruptly kill a user's in-progress optimization.
    _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="T-AutoNLP API",
    version="1.0.0",
    description="Multi-Objective AutoML for NLP — FastAPI backend",
    lifespan=lifespan,
)

_CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,https://t-autonlp.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_job_manager = JobManager()


# --- Request/Response Schemas ---
# JobConfig is the contract between the React config form and the backend.
# Every field here corresponds to a slider or dropdown in the frontend UI.
# Pydantic enforces the types and constraints (ge=) before any processing begins.


class JobConfig(BaseModel):
    dataset_name: str
    max_samples: int = Field(default=2000, ge=100, description="Max training samples")
    population_size: int = Field(default=20, ge=5, description="GA population size")
    n_generations: int = Field(default=10, ge=1, description="Number of GA generations")
    bo_calls: int = Field(default=15, ge=10, description="Bayesian optimization calls")
    optimization_mode: str = Field(
        default="multi_3d",
        description="One of: single_f1, multi_2d, multi_3d",
    )
    disable_bo: bool = False
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")


class FeedbackCreate(BaseModel):
    name: str | None = None
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Job statuses that mean no more updates will ever come.
_TERMINAL_STATES = {"completed", "failed", "terminated"}


# routes


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/jobs", status_code=201)
def create_job(config: JobConfig):
    """Start a new AutoML job and return its ID."""
    job_id = _job_manager.create_job(config.model_dump())
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    """Return all jobs sorted by start time (newest first), enriched with dataset_name."""
    return _job_manager.list_jobs()


@app.post("/api/feedback", status_code=201)
def create_feedback(feedback: FeedbackCreate):
    """Save user feedback to the database."""
    db = get_db()
    feedback_dict = feedback.model_dump()
    result = db.feedback.insert_one(feedback_dict)
    return {"message": "Feedback submitted successfully", "id": str(result.inserted_id)}


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Return the result for a completed job, with config attached."""
    result = _job_manager.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


def _compute_hv_history(result: dict) -> list:
    """Compute per-generation hypervolume history (CPU-bound, runs in thread pool).

    This is what feeds the convergence chart in the frontend. For each generation G,
    we take all pipelines discovered in generations 0..G (cumulative), compute their
    Pareto front, and calculate the hypervolume of that front.

    Hypervolume is a scalar that measures how much of the objective space the Pareto
    front "dominates" relative to a worst-case reference point. A rising hypervolume
    over generations is the mathematical proof that the GA is finding better trade-offs.

    Global objective bounds (min/max across ALL pipelines in the run) are used for
    normalisation so that hypervolume values are comparable across generations — the
    denominator doesn't shift as more solutions are discovered.
    """
    pipelines = result.get("pipelines", [])
    if not pipelines:
        return []

    from collections import defaultdict

    import numpy as np

    from automl.pareto import get_pareto_front
    from utils.evaluation import ParetoAnalyzer

    # Group entries by generation (pipelines only contains status==success entries)
    by_gen: dict[int, list[dict]] = defaultdict(list)
    for entry in pipelines:
        gen = entry.get("generation")
        if gen is not None:
            by_gen[gen].append(entry)

    if not by_gen:
        return []

    # Compute global bounds across ALL pipelines for consistent normalisation
    all_f1 = [e["f1_score"] for e in pipelines]
    all_lat = [e["latency"] for e in pipelines]
    all_interp = [e["interpretability"] for e in pipelines]
    bounds = {
        "f1_score": (float(np.min(all_f1)), float(np.max(all_f1))),
        "latency": (float(np.min(all_lat)), float(np.max(all_lat))),
        "interpretability": (float(np.min(all_interp)), float(np.max(all_interp))),
    }

    # Build cumulative solution set and compute HV at each generation
    generations = sorted(by_gen.keys())
    cumulative = []
    hv_history = []

    for gen in generations:
        cumulative.extend(by_gen[gen])
        front = get_pareto_front(cumulative)
        hv = ParetoAnalyzer.calculate_hypervolume(front, bounds=bounds)
        hv_history.append({"generation": gen, "hypervolume": round(hv, 6)})

    return hv_history


@app.get("/api/jobs/{job_id}/hypervolume-history")
async def get_hypervolume_history(job_id: str):
    """Return hypervolume indicator at each GA generation for convergence plotting."""
    result = await asyncio.to_thread(_job_manager.get_result, job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return await asyncio.to_thread(_compute_hv_history, result)


@app.delete("/api/jobs/{job_id}", status_code=200)
def terminate_job(job_id: str):
    """Terminate a running or queued job."""
    if _job_manager.get_status(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    _job_manager.terminate_job(job_id)
    return {"message": "Job terminated"}


@app.delete("/api/jobs/{job_id}/data", status_code=200)
def delete_job_data(job_id: str):
    """Permanently delete a completed/failed/terminated job and its data."""
    status = _job_manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    if status.get("status") not in ("completed", "failed", "terminated"):
        raise HTTPException(status_code=409, detail="Job is still active")
    if not _job_manager.delete_job(job_id):
        raise HTTPException(status_code=500, detail="Failed to delete job data")
    return {"message": "Job deleted"}


# --- Ablation Study Routes ---
# Ablation studies isolate the contribution of each component of the system by
# running controlled variants (e.g., GA without BO, or optimising for F1 only).
# Each ablation inherits its dataset and parameters from a completed parent job
# so that all conditions are fairly compared on identical data.
#
# Idempotency is a key design requirement: if the same ablation is submitted twice
# (e.g., the user clicks the button twice, or the frontend retries after a timeout),
# the system must not run it twice. This is enforced with a MongoDB atomic update_one
# that claims the ablation slot before submitting to the executor. Any concurrent
# request that loses the race observes modified_count == 0 and backs off safely.

# ablation schema


class AblationConfig(BaseModel):
    mode: str = Field(
        default="multi_3d",
        description="Optimization mode: single_f1 | multi_2d | multi_3d | random_search",
    )
    disable_bo: bool = Field(default=False, description="Disable Bayesian optimization")
    parent_job_id: str = Field(..., description="Job ID whose config to inherit")


# ablation routes


def _effective_mode(mode: str, disable_bo: bool) -> str:
    """Map (mode, disable_bo) to the canonical ablation key."""
    if mode == "random_search":
        return "random_search"
    return "ga_only" if disable_bo else mode


@app.get("/api/ablations")
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


@app.post("/api/ablations", status_code=202)
def start_ablation(config: AblationConfig):
    """Submit an ablation study; inherits config from the parent job.

    Idempotent: returns 200 if a non-failed result already exists.
    Duplicate-submission prevention uses a MongoDB atomic update_one so it
    is safe across multiple Uvicorn workers (unlike an in-process set).
    """
    from api.worker import run_ablation

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
            import logging

            logging.getLogger("server").error(
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
        _BACKEND_ROOT,
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


@app.get("/api/jobs/{job_id}/stream")
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
            status = await asyncio.to_thread(_job_manager.get_status, job_id)

            if status is None:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            logs = await asyncio.to_thread(_job_manager.get_logs, job_id, 100)

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
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE.
        },
    )


# Mount the React SPA catch-all after all /api/* routes; only active when static/ exists.
_static_dir = Path(_BACKEND_ROOT) / "static"
if _static_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
