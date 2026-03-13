"""
Worker function executed in a separate process via concurrent.futures.ProcessPoolExecutor.

This module is intentionally thin — it contains only a top-level function so that
ProcessPoolExecutor can pickle it and send it to a worker process.  All heavy ML
imports are deferred inside the function body so they are only loaded in the worker
process, not in the main FastAPI process.

Why this approach instead of subprocess.Popen:
  - Cross-platform: no OS-specific signal groups or start_new_session flags.
  - Python-managed: the executor handles process lifecycle.
  - Cleaner IPC: we still use file-based state (status.json / result.json) which
    is already proven and avoids the complexity of multiprocessing queues for this
    use case.
"""


def run_automl_job(job_id: str, jobs_dir_str: str, backend_root_str: str) -> None:
    """
    Entry point for a ProcessPoolExecutor worker.

    This must be a module-level function (not a lambda or nested function) so
    that Python's pickle mechanism can locate it in worker processes that use
    the 'spawn' start method (the macOS default since Python 3.12).

    Args:
        job_id:           Unique job identifier (e.g. 'job_20240214_153000').
        jobs_dir_str:     Absolute path to the jobs directory.
        backend_root_str: Absolute path to the backend package root — inserted
                          into sys.path so that local packages are importable.
    """
    # ------------------------------------------------------------------ setup
    import os
    import sys

    if backend_root_str not in sys.path:
        sys.path.insert(0, backend_root_str)

    # Deferred heavy imports — only executed inside the worker process.
    import json
    import signal
    import time
    import traceback
    from pathlib import Path

    from automl import HybridAutoML
    from experiments.evaluation import ParetoAnalyzer
    from utils import DataLoader, to_python_type
    from utils.job_manager import JobManager
    from utils.logger import get_logger

    logger = get_logger("worker", log_file=f"run_{job_id}.log")

    # Propagate all sub-component loggers (automl, evaluator, search_engine, …)
    # to the same job log file.  Each of those modules calls get_logger() without
    # a log_file, so they only get a console handler pointing to this spawned
    # process's stdout — which is invisible.  Attaching the same RotatingFileHandler
    # to the root logger ensures every logger in the worker process writes to the file.
    import logging
    from logging.handlers import RotatingFileHandler as _RFH

    _log_dir = Path(backend_root_str) / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _root_file_handler = _RFH(
        str(_log_dir / f"run_{job_id}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    _root_file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(_root_file_handler)
    logging.getLogger().setLevel(logging.INFO)

    job_manager = JobManager(jobs_dir=jobs_dir_str)
    jobs_dir = Path(jobs_dir_str)
    stop_file = jobs_dir / job_id / "stop.signal"

    # ------------------------------------------------- termination bookkeeping
    _terminate_requested = False

    def _handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal _terminate_requested
        _terminate_requested = True
        logger.info("Worker received SIGTERM — will stop after current step")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    class _TerminationRequested(BaseException):
        """Raised (as BaseException) to bypass bare `except Exception` blocks."""

    # Write PID early so APIJobManager can force-kill this process if needed.
    status = job_manager.get_status(job_id) or {"job_id": job_id}
    status["pid"] = os.getpid()
    status["status"] = "running"
    status["message"] = "Loading dataset..."
    job_manager.update_status(job_id, status)
    logger.info(f"Worker started for job {job_id} (PID {os.getpid()})")

    # -------------------------------------------------------------- main logic
    try:
        config_path = jobs_dir / job_id / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # DataLoader: resolve data dir relative to backend root so the path is
        # correct regardless of which directory uvicorn was started from.
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
            checkpoint_dir=str(jobs_dir / job_id / "checkpoints"),
            optimization_mode=config.get("optimization_mode", "multi_3d"),
            disable_bo=config.get("disable_bo", False),
        )

        def progress_callback(progress_info: dict) -> None:
            nonlocal _terminate_requested
            if _terminate_requested or stop_file.exists():
                logger.info(f"Stop requested for job {job_id}")
                raise _TerminationRequested()

            current_status = job_manager.get_status(job_id)
            if not current_status:
                return

            current_status.update(
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
                best_f1 = (
                    max(
                        (v.get("f1_score", 0.0) for v in store.eval_cache.values()),
                        default=0.0,
                    )
                    if store.eval_cache
                    else 0.0
                )
                cache_hit_rate = (
                    round(store.cache_hit_count / total_lookups * 100, 1)
                    if total_lookups > 0
                    else 0.0
                )
                current_status["best_f1"] = round(best_f1, 4)
                current_status["cache_hit_rate"] = cache_hit_rate
                current_status["total_evaluated"] = len(store.eval_cache)
            except Exception:
                logger.debug("Live metric collection failed", exc_info=True)

            job_manager.update_status(job_id, current_status)

        status["message"] = "Running optimization..."
        job_manager.update_status(job_id, status)

        start_time = time.time()
        results = automl.run(callback=progress_callback)
        runtime_seconds = time.time() - start_time

        # Compute summary metrics.
        analyzer = ParetoAnalyzer()
        metrics = analyzer.compute_metrics(results.get("all_solutions", []))
        results["metrics"] = (
            {
                "best_f1": metrics["f1_score"]["max"],
                "best_latency_ms": metrics["latency"]["min"] * 1000,
                "best_interpretability": metrics["interpretability"]["max"],
                "pareto_front_size": metrics["pareto_front_size"],
                "total_solutions": metrics["total_solutions"],
                "hypervolume": metrics.get("hypervolume", 0.0),
                "knee_point": metrics.get("knee_point"),
            }
            if metrics
            else {
                "best_f1": 0.0,
                "best_latency_ms": 0.0,
                "best_interpretability": 0.0,
                "pareto_front_size": 0,
                "total_solutions": 0,
                "hypervolume": 0.0,
                "knee_point": None,
            }
        )
        results["runtime_seconds"] = runtime_seconds

        result_path = jobs_dir / job_id / "result.json"
        with open(result_path, "w") as f:
            json.dump(to_python_type(results), f, indent=2, allow_nan=False)

        status = job_manager.get_status(job_id)
        status["status"] = "completed"
        status["progress"] = 100
        status["message"] = "Optimization completed successfully"
        status["result_path"] = str(result_path)
        job_manager.update_status(job_id, status)
        logger.info("Job completed successfully")

    except _TerminationRequested:
        logger.info(f"Job {job_id} terminated by user request")
        status = job_manager.get_status(job_id) or {}
        status["status"] = "terminated"
        status["message"] = "Job was manually terminated"
        job_manager.update_status(job_id, status)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
        status = job_manager.get_status(job_id) or {}
        status["status"] = "failed"
        status["error"] = str(e)
        status["message"] = f"Failed: {e}"
        job_manager.update_status(job_id, status)
