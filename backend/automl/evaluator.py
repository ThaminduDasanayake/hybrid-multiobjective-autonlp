"""
THE JUDGE — evaluator.py
=========================
This module sits between the Genetic Algorithm and the Bayesian Optimizer.
When the GA produces a new pipeline blueprint (chromosome), it hands it to the Judge.
The Judge first checks if this exact pipeline has been evaluated before (cache hit).
If not, it enforces compatibility rules, calls the Tuner (BO) to find the best
hyperparameters, and then computes a three-objective fitness score:
  (F1 score ↑, inference latency ↓, interpretability ↑)

This fitness tuple is what DEAP uses to rank individuals with NSGA-II.
"""

import time
from typing import Dict, Tuple

import numpy as np

from utils.logger import get_logger

from .bayesian_optimization import BayesianOptimizer
from .interpretability import interpretability_score
from .persistence import ResultStore

logger = get_logger("evaluator")

# Sentinel fitness values returned for pipelines that cannot be evaluated.
# The values are directionally "worst possible" for DEAP's weight vector (1.0, -1.0, 1.0):
#   F1=0 (worst), latency=huge (worst), interpretability=0 (worst).
# Two distinct sentinels are used:
#   PENALTY_FITNESS — for structurally invalid pipelines (e.g., NaiveBayes after
#                     a scaler that produces negative values). These are deterministic
#                     and are cached so the GA never wastes time re-evaluating them.
#   ERROR_FITNESS   — for transient runtime failures (unexpected exceptions). These
#                     are NOT cached, allowing the GA to retry the individual later.
PENALTY_FITNESS = (0.0, 1e6, 0.0)
ERROR_FITNESS = (0.0, 1e7, 0.0)


class PipelineEvaluator:
    """Evaluates GA pipeline individuals and returns (F1, latency, interpretability) fitness tuples."""

    def __init__(
        self,
        X_train: list,
        y_train: np.ndarray,
        bo_optimizer: BayesianOptimizer,
        result_store: ResultStore,
        objective_ranges: Dict[str, Dict[str, float]] = None,
    ):

        self.X_train = X_train
        self.y_train = y_train
        self.bo_optimizer = bo_optimizer
        self.result_store = result_store

        self.objective_ranges = objective_ranges or {
            "f1_score": {"min": float("inf"), "max": float("-inf")},
            "latency": {"min": float("inf"), "max": float("-inf")},
            "interpretability": {"min": float("inf"), "max": float("-inf")},
        }

    def evaluate(
        self, individual: list, generation: int = 0
    ) -> Tuple[float, float, float]:
        """
        Evaluate an individual (GA chromosome).

        Returns:
            Tuple of (f1_score, latency, interpretability)
            DEAP handles direction via weights (1.0, -1.0, 1.0).
        """
        cache_key = self.result_store.get_individual_key(individual)

        # Cache gate: if this exact architecture has been evaluated before (including
        # deterministic penalty results), return the stored score immediately.
        # Error results (transient failures) are the only exception — they are not
        # reused so the GA gets a chance to retry a pipeline that may have failed
        # due to a temporary resource issue.
        cached = self.result_store.get_cached_evaluation(cache_key)
        if cached and cached.get("status") != "error":
            return cached["f1_score"], cached["latency"], cached["interpretability"]

        # Unpack the 6-gene chromosome:
        # [scaler, dim_reduction, vectorizer, model, ngram_range, max_features]
        scaler_type = individual[0]
        dim_reduction_type = individual[1]
        vectorizer_type = individual[2]
        model_type = individual[3]
        ngram_range = individual[4]
        max_features = individual[5]

        # Compatibility check before spending any compute on BO.
        # Some gene combinations are mathematically illegal (e.g., MultinomialNB
        # requires non-negative inputs, so it cannot follow scalers or dim reduction
        # methods that produce negative values). Invalid combos receive PENALTY_FITNESS
        # and are cached so this check only runs once per unique architecture.
        if not self._validate_structure(scaler_type, dim_reduction_type, model_type):
            logger.warning(f"Invalid structure: {individual}. applying penalty.")
            self.result_store.cache_evaluation(
                cache_key,
                {
                    "status": "invalid_structure",
                    "scaler": scaler_type,
                    "dim_reduction": dim_reduction_type,
                    "vectorizer": vectorizer_type,
                    "model": model_type,
                    "ngram_range": ngram_range,
                    "max_features": max_features,
                    "params": {},
                    "f1_score": PENALTY_FITNESS[0],
                    "latency": PENALTY_FITNESS[1],
                    "interpretability": PENALTY_FITNESS[2],
                    "variance": 0.0,
                },
            )
            return PENALTY_FITNESS

        try:
            # Run Bayesian Optimization
            bo_result = self.bo_optimizer.optimize(
                scaler_type,
                dim_reduction_type,
                vectorizer_type,
                model_type,
                ngram_range,
                max_features,
                self.X_train,
                self.y_train,
            )

            # Track time
            if "optimization_time" in bo_result:
                self.result_store.add_time_stats(
                    optimization_time=bo_result["optimization_time"]
                )

            f1_score = bo_result["best_score"]
            latency = bo_result["inference_time"]

            # Compute interpretability via shared scoring function
            interpretability = interpretability_score(
                scaler_type,
                dim_reduction_type,
                vectorizer_type,
                model_type,
                ngram_range,
                max_features,
                bo_result["best_params"],
            )

            # Update observed objective ranges (for reporting only)
            self._update_objective_ranges(f1_score, latency, interpretability)

            # Results go to two separate stores with different purposes:
            # - eval_cache (via cache_evaluation): the deduplication store. Keyed by
            #   MD5 of the architecture. Only one entry per unique pipeline ever exists
            #   here. hybrid_automl.py reads this to build the final result payload.
            # - search_history (via add_to_history): the chronological log. Records
            #   every *new* successful evaluation in order, with generation and timestamp,
            #   so the convergence chart can show how the Pareto front evolved over time.
            result = {
                "status": "success",
                "scaler": scaler_type,
                "dim_reduction": dim_reduction_type,
                "vectorizer": vectorizer_type,
                "model": model_type,
                "ngram_range": ngram_range,
                "max_features": max_features,
                "params": bo_result["best_params"],
                "f1_score": f1_score,
                "latency": latency,
                "interpretability": interpretability,
                "variance": bo_result["variance"],
            }
            self.result_store.cache_evaluation(cache_key, result)

            # Add to history
            self.result_store.add_to_history(
                {
                    "status": "success",
                    "scaler": scaler_type,
                    "dim_reduction": dim_reduction_type,
                    "vectorizer": vectorizer_type,
                    "model": model_type,
                    "ngram_range": ngram_range,
                    "max_features": max_features,
                    "f1_score": f1_score,
                    "latency": latency,
                    "interpretability": interpretability,
                    "timestamp": time.time(),
                },
                generation=generation,
            )

            # Return raw values — DEAP handles direction via weights (1.0, -1.0, 1.0)
            return f1_score, latency, interpretability

        except Exception as e:
            logger.error(f"Error evaluating individual {individual}: {e}")
            # Don't cache transient runtime failures as final results.
            # Return sentinel for current generation; allow future retry.
            return ERROR_FITNESS

    def _update_objective_ranges(
        self, f1: float, latency: float, interpretability: float
    ):
        """Track observed min/max for each objective (for reporting only)."""
        for obj_name, value in (
            ("f1_score", f1),
            ("latency", latency),
            ("interpretability", interpretability),
        ):
            if value < self.objective_ranges[obj_name]["min"]:
                self.objective_ranges[obj_name]["min"] = value
            if value > self.objective_ranges[obj_name]["max"]:
                self.objective_ranges[obj_name]["max"] = value

    @staticmethod
    def _validate_structure(scaler: str, dim_reduction: str, model: str) -> bool:
        """Return False if the pipeline violates component compatibility constraints.

        MultinomialNB (Naive Bayes) works by computing word count probabilities.
        It mathematically requires all input values to be non-negative (you cannot
        have a negative word count). Two pipeline steps can violate this:
          - StandardScaler / RobustScaler: centre the data around zero, producing
            negative values for below-average features.
          - TruncatedSVD (our "pca" mode): decomposes the feature matrix into
            latent dimensions, which can also produce negative projections.
        Any pipeline that pairs NaiveBayes with these components is flagged as
        invalid before BO even runs, saving the evaluation budget.
        """
        if model == "naive_bayes":
            if scaler in ("standard", "robust"):
                return False
            if dim_reduction == "pca":
                return False

        return True
