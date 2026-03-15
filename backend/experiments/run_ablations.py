#!/usr/bin/env python
"""
Ablation study runner for T-AutoNLP.

Runs the AutoML pipeline under different optimization modes to evaluate the
contribution of each objective.  Each mode changes the DEAP fitness weights
while leaving everything else constant.

Usage examples
--------------
# Default 3-objective run:
    python experiments/run_ablations.py

# Single-objective (F1 only):
    python experiments/run_ablations.py --mode single_f1

# Two-objective (F1 + Latency):
    python experiments/run_ablations.py --mode multi_2d

# GA-only (no Bayesian Optimization):
    python experiments/run_ablations.py --disable-bo

# Custom parameters:
    python experiments/run_ablations.py --mode multi_3d \\
        --dataset imdb --max-samples 3000 --pop-size 30 \\
        --generations 15 --bo-calls 20 --output-dir results/ablations
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from automl import HybridAutoML
from automl.search_engine import OPTIMIZATION_MODES
from utils.evaluation import ParetoAnalyzer
from utils import DataLoader, to_python_type
from utils.logger import get_logger

logger = get_logger("ablation_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run T-AutoNLP ablation studies with different optimization modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=list(OPTIMIZATION_MODES.keys()),
        default="multi_3d",
        help="Optimization mode (default: multi_3d)",
    )
    parser.add_argument(
        "--dataset",
        default="20newsgroups",
        choices=["20newsgroups", "imdb", "ag_news", "banking77"],
        help="Dataset to use (default: 20newsgroups)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max training samples (default: 2000)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=20,
        help="GA population size (default: 20)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of GA generations (default: 10)",
    )
    parser.add_argument(
        "--bo-calls",
        type=int,
        default=15,
        help="Bayesian Optimization iterations (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablations",
        help="Directory for output JSON (default: results/ablations)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--disable-bo",
        action="store_true",
        default=False,
        help="Disable Bayesian Optimization (GA-only ablation: random hyperparams)",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Job ID of the master run this ablation belongs to (used in output filename)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = OPTIMIZATION_MODES[args.mode]
    logger.info(f"=== Ablation Study ===")
    logger.info(f"Mode          : {args.mode}")
    logger.info(f"Weights       : {weights}")
    logger.info(f"Dataset       : {args.dataset}")
    logger.info(f"Max samples   : {args.max_samples}")
    logger.info(f"Population    : {args.pop_size}")
    logger.info(f"Generations   : {args.generations}")
    logger.info(f"BO calls      : {args.bo_calls}")
    logger.info(f"Disable BO    : {args.disable_bo}")

    # --- Load data ---
    data_loader = DataLoader(cache_dir="./data")
    X_train, y_train = data_loader.load_dataset(
        args.dataset,
        subset="train",
        max_samples=args.max_samples,
    )
    logger.info(f"Loaded {len(X_train)} training samples")

    # --- Run AutoML ---
    automl = HybridAutoML(
        X_train=X_train,
        y_train=y_train,
        population_size=args.pop_size,
        n_generations=args.generations,
        bo_calls=args.bo_calls,
        random_state=args.seed,
        optimization_mode=args.mode,
        disable_bo=args.disable_bo,
    )

    start = time.time()
    results = automl.run()
    elapsed = time.time() - start

    # --- Compute metrics ---
    analyzer = ParetoAnalyzer()
    # metrics = analyzer.compute_metrics(results["all_solutions"])]
    metrics = analyzer.compute_metrics(results.get("all_solutions", []))
    if not metrics:
        logger.warning("No valid solutions were produced; writing empty metrics.")
        metrics = {
            "total_solutions": 0,
            "pareto_front_size": 0,
            "f1_score": {"max": 0.0},
            "latency": {"min": 0.0},
            "interpretability": {"max": 0.0},
            "hypervolume": 0.0,
        }

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f"  Ablation Results  —  mode = {args.mode}")
    print("=" * 60)
    print(f"  Weights (F1, Lat, Interp) : {weights}")
    print(f"  Total evaluations         : {metrics['total_solutions']}")
    print(f"  Pareto front size         : {metrics['pareto_front_size']}")
    print(f"  Best F1                   : {metrics['f1_score']['max']:.4f}")
    print(f"  Best Latency              : {metrics['latency']['min'] * 1000:.2f} ms")
    print(f"  Best Interpretability     : {metrics['interpretability']['max']:.4f}")
    print(f"  Hypervolume               : {metrics.get('hypervolume', 0.0):.4f}")
    print(f"  BO Disabled               : {args.disable_bo}")
    print(f"  Runtime                   : {elapsed:.1f}s")
    print("=" * 60 + "\n")

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Include job_id in filename so multiple runs on the same dataset don't collide
    job_suffix = f"_{args.job_id}" if args.job_id else ""
    output_file = output_dir / f"ablation_{args.mode}_{args.dataset}{job_suffix}.json"
    payload = {
        "mode": args.mode,
        "weights": weights,
        "dataset": args.dataset,
        "config": {
            "pop_size": args.pop_size,
            "generations": args.generations,
            "bo_calls": args.bo_calls,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "disable_bo": args.disable_bo,
        },
        "metrics": {
            "total_solutions": metrics["total_solutions"],
            "pareto_front_size": metrics["pareto_front_size"],
            "best_f1": metrics["f1_score"]["max"],
            "best_latency_ms": metrics["latency"]["min"] * 1000,
            "best_interpretability": metrics["interpretability"]["max"],
            "hypervolume": metrics.get("hypervolume", 0.0),
        },
        "runtime_seconds": elapsed,
        "results": to_python_type(results),
    }

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
