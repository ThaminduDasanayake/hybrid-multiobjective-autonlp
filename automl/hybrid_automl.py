import time
import json
import random
import numpy as np
from typing import Dict, Any, List
from utils.logger import get_logger
from .bayesian_optimization import BayesianOptimizer
from .persistence import ResultStore
from .evaluator import PipelineEvaluator
from .search_engine import EvolutionarySearch
from experiments.evaluation import ParetoAnalyzer

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
        checkpoint_dir: str = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state

        # Initialize components
        self.result_store = ResultStore(checkpoint_dir)
        self.result_store.load_checkpoint()

        self.bo_optimizer = BayesianOptimizer(
            n_calls=bo_calls, cv=3, random_state=random_state
        )

        self.evaluator = PipelineEvaluator(
            X_train, y_train, self.bo_optimizer, self.result_store
        )

        self.search_engine = EvolutionarySearch(
            population_size=population_size,
            n_generations=n_generations,
            random_state=random_state,
            result_store=self.result_store,
            evaluate_fn=self.evaluator.evaluate,
        )

    def run(self, callback=None) -> Dict[str, Any]:
        """
        Run the hybrid AutoML optimization.
        """
        logger.info("Starting Hybrid AutoML...")
        start_time = time.time()

        self.search_engine.run(callback)

        total_time = time.time() - start_time
        logger.info(f"Total runtime: {total_time:.2f}s")

        # Prepare results
        all_solutions = list(self.result_store.eval_cache.values())

        # Use ParetoAnalyzer to get the true global Pareto front from all evaluated solutions
        # This ensures consistency with the UI metrics and catches any solutions missed by HOF
        pareto_solutions = ParetoAnalyzer.get_pareto_front(all_solutions)

        # Calculate Search Summary & Learning Curve
        # Retrieve penalty history directly from the search engine (Clean Separation)
        penalty_counts = self.search_engine.get_penalty_history()
        total_penalties = sum(penalty_counts)

        logger.info(f"Search completed. Total constraint violations: {total_penalties}")
        logger.info(f"Constraint learning curve (Penalties per Gen): {penalty_counts}")

        return {
            "pareto_front": pareto_solutions,
            "all_solutions": all_solutions,
            "search_history": self.result_store.search_history,
            "stats": {
                "total_evaluations": len(all_solutions),
                "unique_configurations": len(all_solutions),
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
