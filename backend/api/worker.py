"""
Worker functions executed in separate processes via ProcessPoolExecutor.

Both entry points (run_automl_job and run_ablation) are module-level functions
so Python's pickle mechanism can locate them under the 'spawn' start method
(the macOS/Python 3.12+ default).

All heavy ML imports are deferred inside the function bodies so they are only
loaded in the worker process, never in the main FastAPI process.

IPC: file-based (status.json / result.json) — no multiprocessing queues.
Cancellation: cooperative via stop.signal file checked in progress_callback.
"""

from contextlib import contextmanager


def _ensure_sys_path(backend_root_str: str) -> None:
    """Insert backend root into sys.path so local packages are importable.

    Required in 'spawn' mode (macOS default since Python 3.12) because worker
    processes start fresh and do not inherit the parent's sys.path.
    # TODO: remove once the project adopts a proper src/ package layout.
    """
    import sys

    if backend_root_str not in sys.path:
        sys.path.insert(0, backend_root_str)


@contextmanager
def _job_file_logging(job_id: str, backend_root_str: str):
    """Attach a per-job RotatingFileHandler to the root logger.

    All sub-component loggers (automl, evaluator, search_engine, …) use
    get_logger() without a log_file, so their output goes to the spawned
    process's stdout — which is invisible to the server.  Attaching one shared
    handler to the root logger captures every logger in the worker process into
    a single, readable job log file.

    The context manager guarantees cleanup even if an unexpected exception
    escapes the try/except in run_automl_job, preventing file-descriptor leaks
    across reused ProcessPoolExecutor workers.
    """
    import logging
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_dir = Path(backend_root_str) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        str(log_dir / f"run_{job_id}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def _format_metrics(raw: dict | None) -> dict:
    """Normalise ParetoAnalyzer output into the flat shape the UI expects."""
    if raw:
        return {
            "best_f1": raw["f1_score"]["max"],
            "best_latency_ms": raw["latency"]["min"] * 1000,
            "best_interpretability": raw["interpretability"]["max"],
            "pareto_front_size": raw["pareto_front_size"],
            "total_solutions": raw["total_solutions"],
            "hypervolume": raw.get("hypervolume", 0.0),
            "knee_point": raw.get("knee_point"),
        }
    return {
        "best_f1": 0.0,
        "best_latency_ms": 0.0,
        "best_interpretability": 0.0,
        "pareto_front_size": 0,
        "total_solutions": 0,
        "hypervolume": 0.0,
        "knee_point": None,
    }


# ---------------------------------------------------------------------------
# Tracked job entry point
# ---------------------------------------------------------------------------


def run_automl_job(job_id: str, jobs_dir_str: str, backend_root_str: str) -> None:
    """Run a tracked AutoML job inside a ProcessPoolExecutor worker.

    Writes progress to jobs/{job_id}/status.json on every generation callback
    so the FastAPI SSE endpoint can stream live updates to the React client.
    Writes final results to jobs/{job_id}/result.json on completion.

    Cancellation is cooperative: the progress_callback raises _TerminationRequested
    when jobs/{job_id}/stop.signal exists.  The caller writes that file via
    JobManager.terminate_job().

    Args:
        job_id:           Unique job identifier.
        jobs_dir_str:     Absolute path to the jobs directory.
        backend_root_str: Absolute path to the backend package root.
    """
    _ensure_sys_path(backend_root_str)

    import json
    import os
    import time
    import traceback
    import warnings
    from pathlib import Path

    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from automl import HybridAutoML
    from utils.evaluation import ParetoAnalyzer
    from utils import DataLoader, to_python_type
    from utils.logger import get_logger

    jobs_dir = Path(jobs_dir_str)
    job_dir = jobs_dir / job_id
    stop_file = job_dir / "stop.signal"
    status_path = job_dir / "status.json"
    result_path = job_dir / "result.json"

    def _read_status() -> dict:
        try:
            with open(status_path) as f:
                return json.load(f)
        except Exception:
            return {"job_id": job_id}

    def _write_status(status: dict) -> None:
        import tempfile

        status["last_updated"] = time.time()
        try:
            fd, tmp = tempfile.mkstemp(dir=job_dir, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(status, f, indent=2)
            os.replace(tmp, status_path)
        except Exception as exc:
            logger.error(f"Failed to write status for {job_id}: {exc}")

    with _job_file_logging(job_id, backend_root_str):
        logger = get_logger("worker")
        logger.info(f"Worker started for job {job_id} (PID {os.getpid()})")

        # ----- termination sentinel ----------------------------------------
        class _TerminationRequested(BaseException):
            """Raised to stop the job cooperatively; bypasses bare except Exception."""

        # ----- mark running -------------------------------------------------
        status = _read_status()
        status["status"] = "running"
        status["message"] = "Loading dataset..."
        _write_status(status)

        try:
            config_path = job_dir / "config.json"
            with open(config_path) as f:
                config = json.load(f)

            data_dir = str(Path(backend_root_str) / "data")
            logger.info(f"Loading dataset: {config['dataset_name']}")
            data_loader = DataLoader(cache_dir=data_dir)
            X_train, y_train = data_loader.load_dataset(
                config["dataset_name"],
                subset="train",
                max_samples=config.get("max_samples", 2000),
            )

            automl = HybridAutoML(
                X_train=X_train,
                y_train=y_train,
                population_size=config.get("population_size", 20),
                n_generations=config.get("n_generations", 10),
                bo_calls=config.get("bo_calls", 15),
                random_state=config.get("seed", 42),
                checkpoint_dir=str(job_dir / "checkpoints"),
                optimization_mode=config.get("optimization_mode", "multi_3d"),
                disable_bo=config.get("disable_bo", False),
            )

            def progress_callback(progress_info: dict) -> None:
                if stop_file.exists():
                    logger.info(f"Stop signal detected for job {job_id}")
                    raise _TerminationRequested()

                current = _read_status()
                current.update(
                    {
                        "current_generation": progress_info.get("current_generation"),
                        "total_generations": progress_info.get("total_generations"),
                        "progress": progress_info.get("progress"),
                        "message": progress_info.get("message"),
                    }
                )

                # Enrich with live metrics from the result store.
                try:
                    store = automl.result_store
                    total_lookups = store.cache_hit_count + store.cache_miss_count
                    successful = [
                        v for v in store.eval_cache.values()
                        if v.get("status") == "success"
                    ]
                    best_f1 = max(
                        (v.get("f1_score", 0.0) for v in successful),
                        default=0.0,
                    )
                    best_latency_ms = min(
                        (v.get("latency", float("inf")) * 1000 for v in successful),
                        default=0.0,
                    )
                    best_interpretability = max(
                        (v.get("interpretability", 0.0) for v in successful),
                        default=0.0,
                    )
                    cache_hit_rate = (
                        round(store.cache_hit_count / total_lookups * 100, 1)
                        if total_lookups > 0
                        else 0.0
                    )
                    current["best_f1"] = round(best_f1, 4)
                    current["best_latency_ms"] = round(best_latency_ms, 2)
                    current["best_interpretability"] = round(best_interpretability, 4)
                    current["cache_hit_rate"] = cache_hit_rate
                    current["total_evaluated"] = len(store.eval_cache)
                except Exception:
                    logger.debug("Live metric collection failed", exc_info=True)

                _write_status(current)

            status["message"] = "Running optimization..."
            _write_status(status)

            start_time = time.time()
            results = automl.run(callback=progress_callback)
            runtime_seconds = time.time() - start_time

            analyzer = ParetoAnalyzer()
            raw_metrics = analyzer.compute_metrics(results.get("all_solutions", []))
            results["metrics"] = _format_metrics(raw_metrics)
            results["runtime_seconds"] = runtime_seconds

            with open(result_path, "w") as f:
                json.dump(to_python_type(results), f, indent=2, allow_nan=False)

            status = _read_status()
            status["status"] = "completed"
            status["progress"] = 100
            status["message"] = "Optimization completed successfully"
            status["result_path"] = str(result_path)
            _write_status(status)
            logger.info(f"Job {job_id} completed successfully")

        except _TerminationRequested:
            logger.info(f"Job {job_id} terminated by user request")
            status = _read_status()
            status["status"] = "terminated"
            status["message"] = "Job was manually terminated"
            _write_status(status)

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
            status = _read_status()
            status["status"] = "failed"
            status["error"] = str(e)
            status["message"] = f"Failed: {e}"
            _write_status(status)


# ---------------------------------------------------------------------------
# Ablation study entry point
# ---------------------------------------------------------------------------


def run_ablation(
    mode: str,
    parent_job_id: str,
    dataset: str,
    disable_bo: bool,
    max_samples: int = 2000,
    population_size: int = 20,
    n_generations: int = 10,
    bo_calls: int = 15,
    seed: int = 42,
    backend_root_str: str = "",
) -> None:
    """Run one ablation study and save the result to results/ablations/.

    Fire-and-forget: no status tracking, no progress callback, no cancellation.
    Call GET /api/ablations to check for new results after the run finishes.

    Args:
        mode:             DEAP optimization mode (single_f1 / multi_2d / multi_3d).
        parent_job_id:    ID of the parent job whose config this ablation inherits.
        dataset:          Dataset identifier (resolved from parent job config).
        disable_bo:       If True, skip Bayesian Optimization (GA-only ablation).
        max_samples:      Maximum training samples.
        population_size:  GA population size.
        n_generations:    Number of GA generations.
        bo_calls:         BO calls (ignored when disable_bo=True).
        seed:             Random seed for reproducibility.
        backend_root_str: Absolute path to the backend package root.
    """
    _ensure_sys_path(backend_root_str)

    import json
    import time
    import warnings
    from pathlib import Path

    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from automl import HybridAutoML
    from automl.search_engine import OPTIMIZATION_MODES
    from utils.evaluation import ParetoAnalyzer
    from utils import DataLoader, to_python_type
    from utils.logger import get_logger

    ablation_log_id = f"ablation_{mode}{'_nobo' if disable_bo else ''}_{parent_job_id}"

    with _job_file_logging(ablation_log_id, backend_root_str):
        logger = get_logger("worker")
        logger.info(
            f"Ablation start — mode={mode}, parent={parent_job_id}, "
            f"dataset={dataset}, disable_bo={disable_bo}"
        )

        # Deterministic output path — shared with server.py idempotency check.
        output_dir = Path(backend_root_str) / "results" / "ablations"
        output_dir.mkdir(parents=True, exist_ok=True)
        name_parts = ["ablation", mode]
        if disable_bo:
            name_parts.append("nobo")
        name_parts.append(parent_job_id)
        output_file = output_dir / f"{'_'.join(name_parts)}.json"

        try:
            weights = OPTIMIZATION_MODES.get(mode, [1.0, -1.0, 1.0])

            data_dir = str(Path(backend_root_str) / "data")
            data_loader = DataLoader(cache_dir=data_dir)
            X_train, y_train = data_loader.load_dataset(
                dataset, subset="train", max_samples=max_samples
            )
            logger.info(f"Loaded {len(X_train)} samples for {dataset}")

            automl = HybridAutoML(
                X_train=X_train,
                y_train=y_train,
                population_size=population_size,
                n_generations=n_generations,
                bo_calls=0 if disable_bo else bo_calls,
                random_state=seed,
                optimization_mode=mode,
                disable_bo=disable_bo,
            )

            start = time.time()
            results = automl.run()
            elapsed = time.time() - start

            analyzer = ParetoAnalyzer()
            raw = analyzer.compute_metrics(results.get("all_solutions", []))
            metrics = _format_metrics(raw)
            # Ablation results omit knee_point (not needed for batch study comparisons).
            metrics.pop("knee_point", None)

            payload = {
                "mode": mode,
                "weights": weights,
                "dataset": dataset,
                "parent_job_id": parent_job_id,
                "disable_bo": disable_bo,
                "status": "completed",
                "config": {
                    "pop_size": population_size,
                    "generations": n_generations,
                    "bo_calls": 0 if disable_bo else bo_calls,
                    "max_samples": max_samples,
                    "seed": seed,
                },
                "metrics": metrics,
                "runtime_seconds": elapsed,
                "results": to_python_type(results),
            }

            tmp_file = output_file.with_suffix(f"{output_file.suffix}.tmp")
            with open(tmp_file, "w") as f:
                json.dump(to_python_type(payload), f, indent=2, allow_nan=False)
            tmp_file.replace(output_file)

            logger.info(f"Ablation saved → {output_file}")

        except Exception as e:
            logger.error(f"Ablation failed: {e}", exc_info=True)
            error_payload = {
                "mode": mode,
                "dataset": dataset,
                "parent_job_id": parent_job_id,
                "disable_bo": disable_bo,
                "status": "failed",
                "error": str(e),
                "metrics": {},
                "runtime_seconds": None,
            }
            try:
                with open(output_file, "w") as f:
                    json.dump(error_payload, f, indent=2)
            except Exception:
                logger.error(f"Failed to write error artifact for {output_file}")
