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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.job_manager import APIJobManager, _executor

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Shut down the ProcessPoolExecutor gracefully when the server stops."""
    yield
    # cancel_futures=True cancels queued (not yet running) futures.
    # Running futures (i.e. active AutoML jobs) are left to finish on their own
    # because we don't want to abruptly kill a user's in-progress optimisation.
    _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="T-AutoNLP API",
    version="1.0.0",
    description="Multi-Objective AutoML for NLP — FastAPI backend",
    lifespan=lifespan,
)

_CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_job_manager = APIJobManager()


# ---------------------------------------------------------------------- schemas


class JobConfig(BaseModel):
    dataset_name: str
    max_samples: int = Field(default=2000, ge=100, description="Max training samples")
    population_size: int = Field(default=20, ge=5, description="GA population size")
    n_generations: int = Field(default=10, ge=1, description="Number of GA generations")
    bo_calls: int = Field(default=15, ge=0, description="Bayesian optimisation calls")
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
    """Return all jobs sorted by start time (newest first), enriched with dataset_name from config.json."""
    jobs = _job_manager.list_jobs()
    jobs_dir = Path(_BACKEND_ROOT) / "jobs"
    for job_id, status in jobs.items():
        config_path = jobs_dir / job_id / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                status["dataset_name"] = cfg.get("dataset_name", "")
            except Exception:
                pass
    return jobs


@app.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str):
    """Poll the current status of a job."""
    status = _job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Return the result.json for a completed job, enriched with config.json."""
    result = _job_manager.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    config_path = Path(_BACKEND_ROOT) / "jobs" / job_id / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                result["config"] = json.load(f)
        except Exception:
            pass
    return result


@app.delete("/api/jobs/{job_id}", status_code=200)
def terminate_job(job_id: str):
    """Terminate a running or queued job."""
    if _job_manager.get_status(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    _job_manager.terminate_job(job_id)
    return {"message": "Job terminated"}


@app.post("/api/jobs/{job_id}/resume", status_code=200)
def resume_job(job_id: str):
    """Resume a terminated or failed job from its checkpoint."""
    success = _job_manager.resume_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Could not resume job — check that config.json exists and the job is not already running",
        )
    return {"message": "Job resumed"}


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(
    job_id: str,
    lines: int = Query(default=100, ge=1, le=1000, description="Number of tail lines"),
):
    """Return the last N lines of the job's log file."""
    return {"logs": _job_manager.get_logs(job_id, lines)}


# ---------------------------------------------------------------------- ablation schema


class AblationConfig(BaseModel):
    mode: str = Field(
        default="multi_3d",
        description="Optimization mode: single_f1 | multi_2d | multi_3d",
    )
    disable_bo: bool = Field(default=False, description="Disable Bayesian Optimisation")
    dataset: str = Field(default="ag_news", description="Dataset identifier")
    max_samples: int = Field(default=2000, ge=100, description="Max training samples")
    population_size: int = Field(default=20, ge=5, description="GA population size")
    n_generations: int = Field(default=10, ge=1, description="Number of GA generations")
    bo_calls: int = Field(default=15, ge=0, description="Bayesian optimisation calls")
    seed: int = Field(default=42, ge=0, description="Random seed")


# ---------------------------------------------------------------------- ablation routes


@app.get("/api/ablations")
def get_ablations():
    """
    Return metrics from all completed ablation studies found in results/ablations/.

    Each entry in the returned dict is keyed by ``{mode}_{dataset}`` (or
    ``ga_only_{dataset}`` when disable_bo=True) so the frontend can look up
    a specific combination cheaply.
    """
    ablations_dir = Path(_BACKEND_ROOT) / "results" / "ablations"
    result: dict = {}

    if not ablations_dir.is_dir():
        return result

    for json_file in sorted(ablations_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            mode = data.get("mode", "unknown")
            dataset = data.get("dataset", "unknown")
            # disable_bo may be at the top level (API worker) or nested inside
            # config (run_ablations.py CLI).
            disable_bo = data.get(
                "disable_bo", data.get("config", {}).get("disable_bo", False)
            )

            key = f"{'ga_only' if disable_bo else mode}_{dataset}"
            result[key] = {
                "mode": mode,
                "dataset": dataset,
                "disable_bo": disable_bo,
                "metrics": data.get("metrics", {}),
                "runtime_seconds": data.get("runtime_seconds"),
            }
        except Exception:
            # Skip malformed files silently.
            continue

    return result


@app.post("/api/ablations", status_code=202)
def start_ablation(config: AblationConfig):
    """
    Submit an ablation study to the ProcessPoolExecutor.

    Returns immediately (HTTP 202 Accepted).  The worker saves its result to
    results/ablations/ when finished.  Call GET /api/ablations to check for
    new results.
    """
    from api.ablation_worker import run_ablation

    _executor.submit(
        run_ablation,
        config.mode,
        config.dataset,
        config.disable_bo,
        config.max_samples,
        config.population_size,
        config.n_generations,
        config.bo_calls,
        config.seed,
        _BACKEND_ROOT,
    )
    return {
        "status": "queued",
        "mode": config.mode,
        "dataset": config.dataset,
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
            # Client-disconnect check comes first — no point reading files if
            # nobody is listening on the other end.
            if await request.is_disconnected():
                break

            # File I/O is synchronous; run it in the default thread-pool so we
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
