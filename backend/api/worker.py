"""
THE KITCHEN LINE — worker.py
==============================
This module contains the functions that actually *run* the AutoML engine. They are
executed in separate OS processes (via ProcessPoolExecutor) so that the heavy ML
computation never blocks the FastAPI async event loop.

Think of the server (server.py) as the front-of-house that takes orders. This module
is the kitchen in the back: it receives the order ticket, does all the heavy cooking
(data loading, GA search, BO tuning), and writes the result to MongoDB when done.

Two important constraints shape this module's design:

1. Module-level top-level functions only: Python's 'spawn' start method (the default
   on macOS and Windows) launches worker processes by pickling the function and its
   arguments and re-importing the module from scratch. Only module-level functions
   can be pickled this way — lambdas and nested functions cannot.

2. Deferred ML imports: The main FastAPI process should start up fast and stay lean.
   Heavy packages (scikit-learn, numpy, HuggingFace datasets, etc.) are only imported
   inside the worker function bodies, so the server itself never loads them.
"""

from contextlib import contextmanager


def _suppress_sklearn_warnings() -> None:
    """Suppress noisy sklearn warnings that are expected during hyperparameter search.

    Called at the start of each worker entry point after deferred imports are available.
    Extracted here to avoid duplicating the same three filterwarnings() calls in both
    run_automl_job() and run_ablation(), keeping them in sync if new categories arise.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _ensure_sys_path(backend_root_str: str) -> None:
    """Insert backend root into sys.path so local packages are importable in spawned workers."""
    import sys

    if backend_root_str not in sys.path:
        sys.path.insert(0, backend_root_str)


@contextmanager
def _job_file_logging(job_id: str, backend_root_str: str):
    """Attach a per-job RotatingFileHandler to the root logger for the duration of a worker run.

    Guarantees handler cleanup on exit to prevent file-descriptor leaks across reused workers.
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
    """Normalise ParetoAnalyzer output into the flat shape the UI expects.

    All numeric values are cast to native Python types to avoid
    ``bson.errors.InvalidDocument`` when pymongo encounters numpy scalars.
    """
    if raw:
        from utils import to_python_type

        return {
            "best_f1": float(raw["f1_score"]["max"]),
            "best_latency_ms": float(raw["latency"]["min"]) * 1000,
            "best_interpretability": float(raw["interpretability"]["max"]),
            "pareto_front_size": int(raw["pareto_front_size"]),
            "total_solutions": int(raw["total_solutions"]),
            "hypervolume": float(raw.get("hypervolume", 0.0)),
            "knee_point": to_python_type(raw.get("knee_point")),
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


# Tracked job entry point


def run_automl_job(
    job_id: str, backend_root_str: str, mongo_uri: str, db_name: str
) -> None:
    """Run a tracked AutoML job, streaming progress to MongoDB on every generation callback."""
    _ensure_sys_path(backend_root_str)

    import os
    import time
    import traceback
    from pathlib import Path

    import certifi
    from pymongo import MongoClient

    _suppress_sklearn_warnings()

    from automl import HybridAutoML
    from utils import DataLoader, to_python_type
    from utils.evaluation import ParetoAnalyzer
    from utils.logger import get_logger

    # Each spawned worker process must create its own MongoClient. MongoDB connections
    # use sockets that cannot be safely shared across process boundaries — attempting
    # to do so causes unpredictable connection errors and data corruption.
    client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = client[db_name]
    jobs = db.jobs

    def _update_status(fields: dict) -> None:
        """Write status fields to MongoDB."""
        fields["last_updated"] = time.time()
        jobs.update_one({"_id": job_id}, {"$set": fields})

    def _is_stop_requested() -> bool:
        """Return True if the job's stop_requested flag is set in MongoDB."""
        doc = jobs.find_one({"_id": job_id}, {"stop_requested": 1})
        return bool(doc and doc.get("stop_requested"))

    with _job_file_logging(job_id, backend_root_str):
        logger = get_logger("worker")
        logger.info(f"Worker started for job {job_id} (PID {os.getpid()})")

        # _TerminationRequested inherits from BaseException (not Exception) deliberately.
        # A bare `except Exception` clause — common in sklearn internals — would silently
        # swallow a standard Exception, making termination invisible. BaseException
        # bypasses all bare except blocks and propagates cleanly up to the outer
        # `except _TerminationRequested` handler that logs and marks the job as terminated.
        class _TerminationRequested(BaseException):
            """Raised cooperatively when stop_requested is set; bypasses bare except Exception."""

        _update_status({"status": "running", "message": "Loading dataset..."})

        try:
            doc = jobs.find_one({"_id": job_id}, {"config": 1})
            config = doc["config"]

            data_dir = str(Path(backend_root_str) / "data")
            logger.info(f"Loading dataset: {config['dataset_name']}")
            data_loader = DataLoader(cache_dir=data_dir)
            X_train, y_train = data_loader.load_dataset(
                config["dataset_name"],
                subset="train",
                max_samples=config.get("max_samples", 2000),
                seed=config.get("seed", 42),
            )

            automl = HybridAutoML(
                X_train=X_train,
                y_train=y_train,
                population_size=config.get("population_size", 20),
                n_generations=config.get("n_generations", 10),
                bo_calls=config.get("bo_calls", 15),
                random_state=config.get("seed", 42),
                optimization_mode=config.get("optimization_mode", "multi_3d"),
                disable_bo=config.get("disable_bo", False),
            )

            # The progress callback serves a dual purpose:
            # 1. Live metrics push: after each GA generation, it writes the current
            #    best F1, latency, cache hit rate, and generation number to MongoDB,
            #    which the SSE stream then forwards to the frontend in real time.
            # 2. Cooperative termination: it checks MongoDB for a stop_requested flag
            #    before each generation. If the user has clicked "Stop", it raises
            #    _TerminationRequested to cleanly unwind the search.
            def progress_callback(progress_info: dict) -> None:
                if _is_stop_requested():
                    logger.info(f"Stop signal detected for job {job_id}")
                    raise _TerminationRequested()

                fields = {
                    "current_generation": progress_info.get("current_generation"),
                    "total_generations": progress_info.get("total_generations"),
                    "progress": progress_info.get("progress"),
                    "message": progress_info.get("message"),
                }

                try:
                    # get_live_metrics() encapsulates cache access — no direct
                    # coupling to ResultStore's internal dict schema here.
                    fields.update(automl.result_store.get_live_metrics())
                except Exception:
                    logger.debug("Live metric collection failed", exc_info=True)

                _update_status(fields)

            _update_status({"message": "Running optimization..."})

            start_time = time.time()
            results = automl.run(callback=progress_callback)
            runtime_seconds = time.time() - start_time

            analyzer = ParetoAnalyzer()
            raw_metrics = analyzer.compute_metrics(results.get("pipelines", []))
            results["metrics"] = _format_metrics(raw_metrics)
            results["runtime_seconds"] = runtime_seconds

            jobs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "result": to_python_type(results),
                        "status": "completed",
                        "progress": 100,
                        "message": "Optimization completed successfully",
                        "last_updated": time.time(),
                    }
                },
            )
            logger.info(f"Job {job_id} completed successfully")

        except _TerminationRequested:
            logger.info(f"Job {job_id} terminated by user request")
            _update_status(
                {
                    "status": "terminated",
                    "message": "Job was manually terminated",
                }
            )

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
            _update_status(
                {
                    "status": "failed",
                    "error": str(e),
                    "message": f"Failed: {e}",
                }
            )

        finally:
            client.close()



# --- Ablation Study Entry Point ---


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
    mongo_uri: str = "",
    db_name: str = "",
) -> None:
    """Run one ablation study and write the result to the parent job's ablations field in MongoDB."""
    _ensure_sys_path(backend_root_str)

    import time

    import certifi
    from pymongo import MongoClient

    _suppress_sklearn_warnings()

    from automl import HybridAutoML
    from automl.search_engine import OPTIMIZATION_MODES
    from utils import DataLoader, to_python_type
    from utils.evaluation import ParetoAnalyzer
    from utils.logger import get_logger

    client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = client[db_name]
    jobs = db.jobs

    eff_mode = (
        "random_search"
        if mode == "random_search"
        else ("ga_only" if disable_bo else mode)
    )

    ablation_log_id = f"ablation_{mode}{'_nobo' if disable_bo else ''}_{parent_job_id}"

    with _job_file_logging(ablation_log_id, backend_root_str):
        logger = get_logger("worker")
        logger.info(
            f"Ablation start — mode={mode}, parent={parent_job_id}, "
            f"dataset={dataset}, disable_bo={disable_bo}"
        )

        try:
            weights = OPTIMIZATION_MODES.get(mode, [1.0, -1.0, 1.0])

            # Random search is always a GA-only run (no BO tuning). This is already
            # implied by the mode name, but we set disable_bo explicitly to ensure
            # HybridAutoML doesn't accidentally run the Tuner in baseline mode.
            if mode == "random_search":
                disable_bo = True

            from pathlib import Path

            data_dir = str(Path(backend_root_str) / "data")
            data_loader = DataLoader(cache_dir=data_dir)
            X_train, y_train = data_loader.load_dataset(
                dataset, subset="train", max_samples=max_samples, seed=seed
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
            raw = analyzer.compute_metrics(results.get("pipelines", []))
            metrics = _format_metrics(raw)
            # knee_point is a single recommended solution — only meaningful when a human
            # is making a deployment decision from one result. In the ablation comparison
            # table we compare aggregate metrics (hypervolume, best F1, etc.) across
            # conditions, so the knee point is irrelevant and stripped to keep payloads lean.
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

            jobs.update_one(
                {"_id": parent_job_id},
                {"$set": {f"ablations.{eff_mode}": to_python_type(payload)}},
            )
            logger.info(f"Ablation saved → ablations.{eff_mode} on {parent_job_id}")

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
                jobs.update_one(
                    {"_id": parent_job_id},
                    {"$set": {f"ablations.{eff_mode}": error_payload}},
                )
            except Exception:
                logger.error(f"Failed to write error artifact for ablation {eff_mode}")

        finally:
            client.close()
