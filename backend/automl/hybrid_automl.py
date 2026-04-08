"""
Entry point for the Hybrid AutoML engine, orchestrating the NSGA-II
architecture search and Bayesian Optimization hyperparameter tuning.
"""

import time
from typing import Any, Dict

import numpy as np

from utils.logger import get_logger

from .bayesian_optimization import BayesianOptimizer
from .evaluator import PipelineEvaluator
from .pareto import get_pareto_front
from .persistence import ResultStore
from .search_engine import EvolutionarySearch

logger = get_logger("automl")


class HybridAutoML:
    """
    Hybrid AutoML system combining Genetic Algorithm for architecture search
    and Bayesian Optimization for hyperparameter tuning.
    """

    def __init__(
        self,
        X_train: list,
        y_train: np.ndarray,
        population_size: int = 20,
        n_generations: int = 10,
        bo_calls: int = 15,
        random_state: int = 42,
        optimization_mode: str = "multi_3d",
        disable_bo: bool = False,
        cv_folds: int = 3,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.5,
        stagnation_threshold: int = 3,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.optimization_mode = optimization_mode

        # Disable BO automatically if running random search baseline
        if optimization_mode == "random_search":
            disable_bo = True

        # Initialize in-memory cache for pipeline evaluations
        self.result_store = ResultStore()

        # Initialize Bayesian Optimizer for continuous hyperparameter tuning
        self.bo_optimizer = BayesianOptimizer(
            n_calls=bo_calls,
            cv=cv_folds,
            random_state=random_state,
            disable_bo=disable_bo,
        )

        # Initialize evaluator to compute F1, latency, and interpretability scores
        self.evaluator = PipelineEvaluator(
            X_train, y_train, self.bo_optimizer, self.result_store
        )

        # Initialize NSGA-II evolutionary search engine
        self.search_engine = EvolutionarySearch(
            population_size=population_size,
            n_generations=n_generations,
            random_state=random_state,
            result_store=self.result_store,
            evaluate_fn=self.evaluator.evaluate,
            optimization_mode=optimization_mode,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            stagnation_threshold=stagnation_threshold,
        )

    def run(self, callback=None) -> Dict[str, Any]:
        # Run the hybrid AutoML optimization and return the evaluated pipelines
        logger.info("Starting Hybrid AutoML...")
        start_time = time.time()

        # Execute the evolutionary search loop
        self.search_engine.run(callback)

        total_time = time.time() - start_time
        logger.info(f"Total runtime: {total_time:.2f}s")

        # Build temporal index to track when each unique architecture was first evaluated
        history_index: Dict[str, Dict] = {}
        for entry in self.result_store.search_history:
            if entry.get("status") != "success":
                continue
            arch_key = self.result_store.get_individual_key(
                [
                    entry["scaler"],
                    entry["dim_reduction"],
                    entry["vectorizer"],
                    entry["model"],
                    entry["ngram_range"],
                    entry["max_features"],
                ]
            )
            if arch_key not in history_index:
                history_index[arch_key] = {
                    "generation": entry.get("generation", 0),
                    "timestamp": entry.get("timestamp", 0.0),
                }

        # Retrieve deduplicated results from the evaluation cache and append temporal metadata
        valid_solutions: list = []
        for result in self.result_store.eval_cache.values():
            if result.get("status") != "success":
                continue
            arch_key = self.result_store.get_individual_key(
                [
                    result["scaler"],
                    result["dim_reduction"],
                    result["vectorizer"],
                    result["model"],
                    result["ngram_range"],
                    result["max_features"],
                ]
            )
            temporal = history_index.get(arch_key, {"generation": 0, "timestamp": 0.0})
            valid_solutions.append({**result, **temporal})

        # Compute the Pareto front from the final deduplicated pool of valid solutions
        pareto_solutions = get_pareto_front(valid_solutions)

        # Create an MD5-keyed set for O(1) Pareto membership lookups
        pareto_keys = {
            self.result_store.get_individual_key(
                [
                    s["scaler"],
                    s["dim_reduction"],
                    s["vectorizer"],
                    s["model"],
                    s["ngram_range"],
                    s["max_features"],
                ]
            )
            for s in pareto_solutions
        }

        # Tag each pipeline with its Pareto-optimality status for frontend visualization
        pipelines = [
            {
                **sol,
                "is_pareto_optimal": self.result_store.get_individual_key(
                    [
                        sol["scaler"],
                        sol["dim_reduction"],
                        sol["vectorizer"],
                        sol["model"],
                        sol["ngram_range"],
                        sol["max_features"],
                    ]
                )
                in pareto_keys,
            }
            for sol in valid_solutions
        ]

        penalty_counts = self.search_engine.get_penalty_history()
        total_penalties = sum(penalty_counts)

        logger.info(f"Search completed. Total constraint violations: {total_penalties}")
        logger.info(f"Constraint learning curve (Penalties per Gen): {penalty_counts}")

        return {
            "pipelines": pipelines,
            "stats": {
                "total_evaluations": len(self.result_store.eval_cache),
                "pareto_size": len(pareto_solutions),
                "objective_ranges": self.evaluator.objective_ranges,
                "total_penalties": total_penalties,
                "constraint_learning_curve": penalty_counts,
                "time_stats": {
                    "total_runtime": total_time,
                    "total_optimization_time": self.result_store.total_optimization_time,
                    "generation_times": self.result_store.generation_times,
                    "total_generations_run": len(self.result_store.generation_times),
                },
            },
        }
