"""
Ablation study worker for ProcessPoolExecutor.

Same pattern as worker_fn.py — top-level module-level function so Python's
pickle mechanism can locate it under the 'spawn' start method (macOS default).
"""


def run_ablation(
    mode: str,
    dataset: str,
    disable_bo: bool,
    max_samples: int = 2000,
    population_size: int = 20,
    n_generations: int = 10,
    bo_calls: int = 15,
    seed: int = 42,
    backend_root_str: str = "",
) -> None:
    """
    Run one ablation study and save the result to results/ablations/.

    Args:
        mode:             DEAP optimization mode (single_f1 / multi_2d / multi_3d).
        dataset:          Dataset identifier (ag_news / imdb / …).
        disable_bo:       If True, skip Bayesian Optimisation (GA-only ablation).
        max_samples:      Maximum training samples.
        population_size:  GA population size.
        n_generations:    Number of GA generations.
        bo_calls:         Bayesian optimisation calls (ignored when disable_bo=True).
        seed:             Random seed for reproducibility.
        backend_root_str: Absolute path to the backend package root.
    """
    import sys

    if backend_root_str not in sys.path:
        sys.path.insert(0, backend_root_str)

    # Heavy imports deferred so they only load inside the worker process.
    import json
    import time
    from pathlib import Path

    from automl import HybridAutoML
    from automl.search_engine import OPTIMIZATION_MODES
    from experiments.evaluation import ParetoAnalyzer
    from utils import DataLoader, to_python_type
    from utils.logger import get_logger

    logger = get_logger("ablation_worker")
    logger.info(
        f"Ablation start — mode={mode}, dataset={dataset}, disable_bo={disable_bo}"
    )

    weights = OPTIMIZATION_MODES.get(mode, [1.0, -1.0, 1.0])

    # ------------------------------------------------------------------ data
    data_dir = str(Path(backend_root_str) / "data")
    data_loader = DataLoader(cache_dir=data_dir)
    X_train, y_train = data_loader.load_dataset(
        dataset, subset="train", max_samples=max_samples
    )
    logger.info(f"Loaded {len(X_train)} samples for {dataset}")

    # ------------------------------------------------------------------ run
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

    # --------------------------------------------------------------- metrics
    analyzer = ParetoAnalyzer()
    raw = analyzer.compute_metrics(results.get("all_solutions", []))
    if raw:
        metrics = {
            "total_solutions": raw["total_solutions"],
            "pareto_front_size": raw["pareto_front_size"],
            "best_f1": raw["f1_score"]["max"],
            "best_latency_ms": raw["latency"]["min"] * 1000,
            "best_interpretability": raw["interpretability"]["max"],
            "hypervolume": raw.get("hypervolume", 0.0),
        }
    else:
        metrics = {
            "total_solutions": 0,
            "pareto_front_size": 0,
            "best_f1": 0.0,
            "best_latency_ms": 0.0,
            "best_interpretability": 0.0,
            "hypervolume": 0.0,
        }

    # ----------------------------------------------------------------- save
    output_dir = Path(backend_root_str) / "results" / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # GA-only gets a distinct filename so it doesn't overwrite the GA+BO run.
    name_parts = ["ablation", mode]
    if disable_bo:
        name_parts.append("nobo")
    name_parts.append(dataset)
    output_file = output_dir / f"{'_'.join(name_parts)}.json"

    payload = {
        "mode": mode,
        "weights": weights,
        "dataset": dataset,
        "disable_bo": disable_bo,
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
