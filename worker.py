import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automl import HybridAutoML
from utils import DataLoader, to_python_type
from utils.job_manager import JobManager
from utils.logger import get_logger

# Configure logger
logger = get_logger("worker")


def main():
    parser = argparse.ArgumentParser(description="AutoML Worker Process")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--config", required=True, help="Path to configuration JSON")
    parser.add_argument("--jobs-dir", default="jobs", help="Directory for job data")
    args = parser.parse_args()

    job_id = args.job_id
    config_path = args.config
    jobs_dir = args.jobs_dir

    logger.info(f"Worker started for job {job_id}")

    job_manager = JobManager(jobs_dir=jobs_dir)

    try:
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # Update status
        status = job_manager.get_status(job_id)
        status["status"] = "running"
        status["message"] = "Loading dataset..."
        job_manager.update_status(job_id, status)

        # Load data
        logger.info(f"Loading dataset: {config['dataset_name']}")
        data_loader = DataLoader(cache_dir="./data")
        X_train, y_train = data_loader.load_dataset(
            config["dataset_name"],
            subset="train",
            max_samples=config.get("max_samples", 2000),
        )

        # Define progress callback with enriched metrics
        def progress_callback(progress_info):
            current_status = job_manager.get_status(job_id)
            if current_status:
                # Core progress info
                current_status.update(
                    {
                        "current_generation": progress_info.get("current_generation"),
                        "total_generations": progress_info.get("total_generations"),
                        "progress": progress_info.get("progress"),
                        "message": progress_info.get("message"),
                    }
                )

                # Enriched live metrics from the result store
                try:
                    store = automl.result_store
                    total_cached = len(store.eval_cache)
                    total_history = len(store.search_history)

                    # Best F1 from evaluated solutions
                    best_f1 = 0.0
                    if store.eval_cache:
                        best_f1 = max(
                            (v.get("f1_score", 0.0) for v in store.eval_cache.values()),
                            default=0.0,
                        )

                    # Cache hit rate: history entries that were already in cache
                    # Each unique config is in eval_cache; repeated evals hit cache
                    cache_hit_rate = 0.0
                    if total_history > 0:
                        cache_hits = max(0, total_history - total_cached)
                        cache_hit_rate = round(cache_hits / total_history * 100, 1)

                    current_status["best_f1"] = round(best_f1, 4)
                    current_status["cache_hit_rate"] = cache_hit_rate
                    current_status["total_evaluated"] = total_cached
                # except Exception:
                #     pass  # Don't let metrics collection crash the callback
                except Exception:
                    logger.debug(
                        "Live metric collection failed for job %s",
                        job_id,
                        exc_info=True,
                    )

                job_manager.update_status(job_id, current_status)

        # Initialize AutoML
        automl = HybridAutoML(
            X_train=X_train,
            y_train=y_train,
            population_size=config.get("population_size", 20),
            n_generations=config.get("n_generations", 10),
            bo_calls=config.get("bo_calls", 15),
            random_state=42,
            checkpoint_dir=os.path.join(jobs_dir, job_id, "checkpoints"),
            optimization_mode=config.get("optimization_mode", "multi_3d"),
            disable_bo=config.get("disable_bo", False),
        )

        # Run AutoML
        start_time = time.time()
        status["message"] = "Running optimization..."
        job_manager.update_status(job_id, status)

        results = automl.run(callback=progress_callback)
        runtime_seconds = time.time() - start_time

        # Compute and persist key metrics
        from experiments.evaluation import ParetoAnalyzer

        analyzer = ParetoAnalyzer()
        metrics = analyzer.compute_metrics(results.get("all_solutions", []))
        if not metrics:
            results["metrics"] = {
                "best_f1": 0.0,
                "best_latency_ms": 0.0,
                "best_interpretability": 0.0,
                "pareto_front_size": 0,
                "total_solutions": 0,
                "hypervolume": 0.0,
            }
        else:
            results["metrics"] = {
                "best_f1": metrics["f1_score"]["max"],
                "best_latency_ms": metrics["latency"]["min"] * 1000,
                "best_interpretability": metrics["interpretability"]["max"],
                "pareto_front_size": metrics["pareto_front_size"],
                "total_solutions": metrics["total_solutions"],
                "hypervolume": metrics.get("hypervolume", 0.0),
            }
        results["runtime_seconds"] = runtime_seconds

        # Save results
        logger.info("Saving results...")
        result_path = os.path.join(jobs_dir, job_id, "result.json")
        with open(result_path, "w") as f:
            json.dump(to_python_type(results), f, indent=2)

        # Update status to completed
        status = job_manager.get_status(job_id)
        status["status"] = "completed"
        status["progress"] = 100
        status["message"] = "Optimization completed successfully"
        status["result_path"] = result_path
        job_manager.update_status(job_id, status)

        logger.info("Job completed successfully")

    except Exception as e:
        logger.error(f"Job failed: {e}")
        logger.error(traceback.format_exc())

        # Update status to failed
        status = job_manager.get_status(job_id) or {}
        status["status"] = "failed"
        status["error"] = str(e)
        status["message"] = f"Failed: {str(e)}"
        job_manager.update_status(job_id, status)

        sys.exit(1)


if __name__ == "__main__":
    main()
