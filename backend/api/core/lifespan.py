from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.workers.job_manager import _executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gracefully shut down the executor on server exit, cancelling queued jobs but allowing running jobs to complete."""
    yield
    _executor.shutdown(wait=False, cancel_futures=True)
