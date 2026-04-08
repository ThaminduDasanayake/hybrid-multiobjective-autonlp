"""
Evaluates pipeline chromosomes, enforcing structural constraints and returning fitness tuples.
"""

import time
from typing import Dict, Tuple

import numpy as np

from utils.logger import get_logger

from .bayesian_optimization import BayesianOptimizer
from .interpretability import interpretability_score
from .persistence import ResultStore

logger = get_logger("evaluator")

# Sentinel fitness values for structural invalidity (cached) and runtime errors (uncached)
PENALTY_FITNESS = (0.0, 1e6, 0.0)
ERROR_FITNESS = (0.0, 1e7, 0.0)


class PipelineEvaluator:
    """Computes multi-objective fitness scores for GA individuals."""

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
        """Evaluate an individual and return its (f1_score, latency, interpretability) tuple."""
        cache_key = self.result_store.get_individual_key(individual)

        # Retrieve from cache if previously evaluated (excluding transient errors)
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

        # Verify architectural compatibility before assigning compute resources
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

            # Calculate structural interpretability score
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

            # Store result in the deduplication cache
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

            return f1_score, latency, interpretability

        except Exception as e:
            logger.error(f"Error evaluating individual {individual}: {e}")
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
        """Validate pipeline structural integrity (e.g., preventing negative inputs to Naive Bayes)."""
        if model == "naive_bayes":
            if scaler in ("standard", "robust"):
                return False
            if dim_reduction == "pca":
                return False

        return True
