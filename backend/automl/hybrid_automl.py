"""
THE MANAGER — hybrid_automl.py
===============================
This is the single entry point for the entire AutoML engine. It wires together
all four subsystems (Memory, Architect, Tuner, Judge) and drives the search from
start to finish. Think of it as the project manager who assembles the team,
kicks off the work, and then collects and packages the final report.

The flow in a single run:
  1. __init__: Assemble the team — create Memory, build Tuner and Judge, hand them
               to the Architect (search engine).
  2. run():    Tell the Architect to run the GA. When it finishes, post-process
               the raw cache into a clean result: deduplicate, run Pareto analysis,
               flag each solution as Pareto-optimal or dominated.
"""

import time
import numpy as np
from typing import Dict, Any
from utils.logger import get_logger
from .bayesian_optimization import BayesianOptimizer
from .persistence import ResultStore
from .evaluator import PipelineEvaluator
from .search_engine import EvolutionarySearch
from .pareto import get_pareto_front

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

        # In random_search mode, the GA's evolutionary operators are bypassed entirely
        # (flat uniform sampling). There is no point running Bayesian Optimization on top
        # of random architecture selection, so BO is automatically disabled here.
        if optimization_mode == "random_search":
            disable_bo = True

        # The Memory is created first because every other subsystem depends on it
        # for caching, history recording, and hit/miss tracking.
        self.result_store = ResultStore()

        # The Tuner (BayesianOptimizer) is configured once and reused for every
        # individual the GA evaluates. It receives a fixed architecture (e.g.,
        # "tfidf + logistic") and searches for the best hyperparameters within it.
        self.bo_optimizer = BayesianOptimizer(
            n_calls=bo_calls,
            cv=cv_folds,
            random_state=random_state,
            disable_bo=disable_bo,
        )

        # The Judge wraps the Tuner and adds the fitness tuple (F1, latency,
        # interpretability) that DEAP needs. It also enforces compatibility rules
        # (e.g., NaiveBayes cannot follow a scaler that produces negative values).
        self.evaluator = PipelineEvaluator(
            X_train, y_train, self.bo_optimizer, self.result_store
        )

        # The Architect (EvolutionarySearch) runs the NSGA-II loop. It receives a
        # reference to the Judge's evaluate() method so it can score each individual
        # without knowing anything about BO or interpretability internally.
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
        """
        Run the hybrid AutoML optimization and return the full result payload.

        Returns a dict with two top-level keys:
          - 'pipelines': one entry per successfully evaluated architecture, each flagged
                         with is_pareto_optimal=True/False.
          - 'stats':     timing, Pareto front size, constraint violation counts, and
                         objective range metadata for the frontend.
        """
        logger.info("Starting Hybrid AutoML...")
        start_time = time.time()

        # Phase 1 — Run the evolutionary search. This call blocks until the GA finishes
        # all generations (or hits early stopping). The callback fires after each
        # generation to push live progress to the frontend via MongoDB.
        self.search_engine.run(callback)

        total_time = time.time() - start_time
        logger.info(f"Total runtime: {total_time:.2f}s")

        # Phase 2 — Build a temporal index from search_history.
        # search_history records every evaluation call in order, including cache re-hits.
        # We only want the *first time* each unique architecture was genuinely evaluated
        # (not a cache replay), so we take first-occurrence only. This gives each pipeline
        # the correct "born in generation X" metadata for the convergence chart.
        history_index: Dict[str, Dict] = {}
        for entry in self.result_store.search_history:
            if entry.get("status") != "success":
                continue
            arch_key = self.result_store.get_individual_key([
                entry["scaler"], entry["dim_reduction"], entry["vectorizer"],
                entry["model"], entry["ngram_range"], entry["max_features"],
            ])
            if arch_key not in history_index:
                history_index[arch_key] = {
                    "generation": entry.get("generation", 0),
                    "timestamp": entry.get("timestamp", 0.0),
                }

        # Phase 3 — Pull the deduplicated result set from eval_cache (not from history).
        # eval_cache stores exactly one entry per unique architecture (keyed by MD5),
        # so iterating it gives us a clean, deduplicated list of all evaluated pipelines.
        # We enrich each entry with its temporal metadata from the history_index above.
        valid_solutions: list = []
        for result in self.result_store.eval_cache.values():
            if result.get("status") != "success":
                continue
            arch_key = self.result_store.get_individual_key([
                result["scaler"], result["dim_reduction"], result["vectorizer"],
                result["model"], result["ngram_range"], result["max_features"],
            ])
            temporal = history_index.get(arch_key, {"generation": 0, "timestamp": 0.0})
            valid_solutions.append({**result, **temporal})

        # Phase 4 — Pareto dominance analysis.
        # A solution is Pareto-optimal if no other solution beats it on all three
        # objectives simultaneously (F1 ↑, latency ↓, interpretability ↑).
        # We recompute the full front here from the final deduplicated pool, rather
        # than using DEAP's internal HOF, to ensure correctness after caching.
        pareto_solutions = get_pareto_front(valid_solutions)

        # Build an MD5-keyed membership set for O(1) Pareto membership lookups.
        # We use MD5 keys rather than float-tuple fingerprints because floating-point
        # equality is unreliable under serialisation/deserialisation round-trips.
        pareto_keys = {
            self.result_store.get_individual_key([
                s["scaler"], s["dim_reduction"], s["vectorizer"],
                s["model"], s["ngram_range"], s["max_features"],
            ])
            for s in pareto_solutions
        }

        # Attach the is_pareto_optimal flag to every solution so the frontend can
        # highlight Pareto-front members in the visualisations without a separate lookup.
        pipelines = [
            {
                **sol,
                "is_pareto_optimal": self.result_store.get_individual_key([
                    sol["scaler"], sol["dim_reduction"], sol["vectorizer"],
                    sol["model"], sol["ngram_range"], sol["max_features"],
                ]) in pareto_keys,
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
