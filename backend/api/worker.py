"""
Worker entry points executed in separate processes via ProcessPoolExecutor.

Module-level functions are required for pickling under the 'spawn' start method
(macOS/Python 3.12+ default). Heavy ML imports are deferred to worker bodies so
the main FastAPI process never loads them.
"""

from contextlib import contextmanager


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
    import warnings
    from pathlib import Path

    import certifi
    from pymongo import MongoClient
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from automl import HybridAutoML
    from utils import DataLoader, to_python_type
    from utils.evaluation import ParetoAnalyzer
    from utils.logger import get_logger

    # Workers must create their own MongoClient; connections cannot be shared across fork boundaries.
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
                    store = automl.result_store
                    total_lookups = store.cache_hit_count + store.cache_miss_count
                    successful = [
                        v
                        for v in store.eval_cache.values()
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
                    fields["best_f1"] = round(float(best_f1), 4)
                    fields["best_latency_ms"] = round(float(best_latency_ms), 4)
                    fields["best_interpretability"] = round(
                        float(best_interpretability), 4
                    )
                    fields["cache_hit_rate"] = cache_hit_rate
                    fields["total_evaluated"] = len(store.eval_cache)
                except Exception:
                    logger.debug("Live metric collection failed", exc_info=True)

                _update_status(fields)

            _update_status({"message": "Running optimization..."})

            start_time = time.time()
            results = automl.run(callback=progress_callback)
            runtime_seconds = time.time() - start_time

            analyzer = ParetoAnalyzer()
            raw_metrics = analyzer.compute_metrics(results.get("all_solutions", []))
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



# Ablation study entry point


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
    import warnings

    import certifi
    from pymongo import MongoClient
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

            if mode == "random_search":
                disable_bo = True

            from pathlib import Path

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
            metrics.pop("knee_point", None)  # Not needed for batch comparison studies.

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
