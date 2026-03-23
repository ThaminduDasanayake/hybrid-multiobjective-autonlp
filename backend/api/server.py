"""
FastAPI server for T-AutoNLP.

Start with:
    cd backend
    uv run uvicorn api.server:app --reload --port 8000

The React Vite dev server runs on http://localhost:5173 and is allowed via CORS.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
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

# Track in-flight ablation tasks so we can reject duplicate submissions.
_running_ablations: set[str] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Shut down the ProcessPoolExecutor gracefully when the server stops."""
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


# ---------------------------------------------------------------------- schemas


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


# Job statuses that mean no more updates will ever come.
_TERMINAL_STATES = {"completed", "failed", "terminated"}


# ---------------------------------------------------------------------- routes


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


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Return the result for a completed job, with config attached."""
    result = _job_manager.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


def _compute_hv_history(result: dict) -> list:
    """Compute per-generation hypervolume history (CPU-bound, runs in thread pool).

    Iterates over the job's search_history, building a cumulative solution set
    per generation, computing the Pareto front and its hypervolume at each step.
    Global objective bounds (from all solutions) are used so hypervolume values
    are consistently normalised across generations.
    """
    search_history = result.get("search_history", [])
    if not search_history:
        return []

    from collections import defaultdict

    import numpy as np

    from automl.pareto import get_pareto_front
    from utils.evaluation import ParetoAnalyzer

    # Group entries by generation
    by_gen: dict[int, list[dict]] = defaultdict(list)
    for entry in search_history:
        if entry.get("status") != "success":
            continue
        gen = entry.get("generation")
        if gen is not None:
            by_gen[gen].append(entry)

    if not by_gen:
        return []

    # Compute global bounds across ALL solutions for consistent normalisation
    all_f1 = [e["f1_score"] for e in search_history if e.get("status") == "success"]
    all_lat = [e["latency"] for e in search_history if e.get("status") == "success"]
    all_interp = [
        e["interpretability"] for e in search_history if e.get("status") == "success"
    ]
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


# ---------------------------------------------------------------------- ablation schema


class AblationConfig(BaseModel):
    mode: str = Field(
        default="multi_3d",
        description="Optimization mode: single_f1 | multi_2d | multi_3d | random_search",
    )
    disable_bo: bool = Field(default=False, description="Disable Bayesian optimization")
    parent_job_id: str = Field(..., description="Job ID whose config to inherit")


# ---------------------------------------------------------------------- ablation routes


def _effective_mode(mode: str, disable_bo: bool) -> str:
    """Map (mode, disable_bo) to the canonical ablation key."""
    if mode == "random_search":
        return "random_search"
    return "ga_only" if disable_bo else mode


@app.get("/api/ablations")
def get_ablations(
    parent_job_id: str | None = Query(default=None, description="Filter to a single parent job"),
):
    """
    Return metrics from all completed ablation studies.

    Reads the ``ablations`` nested object from job documents in MongoDB.
    When ``parent_job_id`` is provided, only that single document is queried
    (index lookup) instead of scanning every job.

    Each entry in the returned dict is keyed by
    ``{effective_mode}_{parent_job_id}`` so the frontend can look up
    ablation results scoped to a specific parent run.
    """
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
    """
    Submit an ablation study to the ProcessPoolExecutor.

    The ablation inherits its full configuration (dataset, samples, population
    size, etc.) from the parent job's config stored in MongoDB.

    Idempotent: if a completed result already exists, returns 200 instead of
    re-submitting.  If the same ablation is already in-flight, returns 202
    with status "already_queued".

    Returns immediately (HTTP 202 Accepted).  The worker saves its result to
    the parent job's ``ablations`` field in MongoDB when finished.  Call
    GET /api/ablations to check for new results.
    """
    from api.worker import run_ablation

    pid = config.parent_job_id
    if not pid or "/" in pid or "\\" in pid or pid.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid parent_job_id")

    db = get_db()
    parent_doc = db.jobs.find_one({"_id": pid}, {"config": 1, "ablations": 1})
    if parent_doc is None:
        raise HTTPException(status_code=404, detail="Parent job not found")

    eff_mode = _effective_mode(config.mode, config.disable_bo)
    running_key = f"{eff_mode}_{config.parent_job_id}"

    # Check for an existing ablation result in MongoDB.
    existing = parent_doc.get("ablations", {}).get(eff_mode)
    if existing and existing.get("status", "completed") != "failed":
        return JSONResponse(
            status_code=200,
            content={
                "status": "completed",
                "mode": config.mode,
                "parent_job_id": config.parent_job_id,
                "disable_bo": config.disable_bo,
            },
        )

    # Reject if this exact ablation is already in-flight.
    if running_key in _running_ablations:
        return {
            "status": "already_queued",
            "mode": config.mode,
            "parent_job_id": config.parent_job_id,
            "disable_bo": config.disable_bo,
        }

    parent_config = parent_doc["config"]

    _running_ablations.add(running_key)

    def _on_done(fut):
        _running_ablations.discard(running_key)
        exc = fut.exception()
        if exc:
            import logging
            logging.getLogger("server").error(
                f"Ablation {running_key} crashed: {exc}", exc_info=exc,
            )

    future = _executor.submit(
        run_ablation,
        config.mode,
        config.parent_job_id,
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
        "parent_job_id": config.parent_job_id,
        "disable_bo": config.disable_bo,
    }


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    """
    Server-Sent Events stream for real-time job progress.

    Each event is a JSON payload containing the full status snapshot and the
    latest 100 log lines:

        data: {"status": {...}, "logs": ["line1", ...]}\n\n

    The stream:
      - Only emits an event when either the status or the logs have changed,
        so the React client can safely update state on every message without
        comparing old vs new itself.
      - Closes automatically once the job reaches a terminal state
        (completed / failed / terminated).
      - Closes immediately if the client disconnects, preventing orphaned
        server-side generators from accumulating.

    Connect from JavaScript with:
        const es = new EventSource('/api/jobs/<id>/stream');
        es.onmessage = (e) => {
            const { status, logs } = JSON.parse(e.data);
        };
    """

    async def event_generator():
        last_payload_json: str | None = None

        while True:
            # Client-disconnect check comes first — no point querying if
            # nobody is listening on the other end.
            if await request.is_disconnected():
                break

            # pymongo is synchronous; run it in the default thread-pool so we
            # never block the asyncio event loop.
            status = await asyncio.to_thread(_job_manager.get_status, job_id)

            if status is None:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            logs = await asyncio.to_thread(_job_manager.get_logs, job_id, 100)

            payload_json = json.dumps({"status": status, "logs": logs})

            # Only push when something has actually changed to avoid sending
            # identical frames every second during quiet periods.
            if payload_json != last_payload_json:
                last_payload_json = payload_json
                yield f"data: {payload_json}\n\n"

            # Close the stream *after* yielding the final state so the client
            # always receives the terminal event before the connection drops.
            if status.get("status") in _TERMINAL_STATES:
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx / proxies not to buffer the response — critical for
            # SSE to work correctly behind a reverse proxy.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------- static files
# Mount the React SPA as a catch-all AFTER all /api/* routes so API requests
# are never swallowed.  Only active in production (when static/ exists); in
# development, Vite's dev server handles the frontend.
_static_dir = Path(_BACKEND_ROOT) / "static"
if _static_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
