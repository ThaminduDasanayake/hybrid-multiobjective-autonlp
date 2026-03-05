from typing import Dict, Any, Tuple
import numpy as np
import time
from utils.logger import get_logger
from .bayesian_optimization import BayesianOptimizer
from .persistence import ResultStore
from .interpretability import interpretability_score

logger = get_logger("evaluator")

# Sentinel fitness values for invalid/error individuals.
# Directionally correct for DEAP weights (1.0, -1.0, 1.0):
#   F1=0 (worst), latency=huge (worst), interpretability=0 (worst)
PENALTY_FITNESS = (0.0, 1e6, 0.0)
ERROR_FITNESS = (0.0, 1e7, 0.0)


class PipelineEvaluator:
    """
    Evaluates pipeline individuals using Bayesian Optimization.
    Calculates multi-objective fitness scores:
    1. F1 Score (Maximize)
    2. Latency (Minimize)
    3. Interpretability (Maximize)
    """

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

        # Check cache
        cached = self.result_store.get_cached_evaluation(cache_key)
        if cached:
            return cached["f1_score"], cached["latency"], cached["interpretability"]

        # Unpack 6 genes
        scaler_type = individual[0]
        dim_reduction_type = individual[1]
        vectorizer_type = individual[2]
        model_type = individual[3]
        ngram_range = individual[4]
        max_features = individual[5]

        # Safe-Pairing Logic
        if not self._validate_structure(scaler_type, dim_reduction_type, model_type):
            logger.warning(f"Invalid structure: {individual}. applying penalty.")
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
            interp = interpretability_score(
                scaler_type,
                dim_reduction_type,
                vectorizer_type,
                model_type,
                ngram_range,
                max_features,
                bo_result["best_params"],
            )

            # Update observed objective ranges (for reporting only)
            self._update_objective_ranges(f1_score, latency, interp)

            # Store result
            result = {
                "scaler": scaler_type,
                "dim_reduction": dim_reduction_type,
                "vectorizer": vectorizer_type,
                "model": model_type,
                "ngram_range": ngram_range,
                "max_features": max_features,
                "params": bo_result["best_params"],
                "f1_score": f1_score,
                "latency": latency,
                "interpretability": interp,
                "variance": bo_result["variance"],
            }
            self.result_store.cache_evaluation(cache_key, result)

            # Add to history
            self.result_store.add_to_history(
                {
                    "scaler": scaler_type,
                    "dim_reduction": dim_reduction_type,
                    "vectorizer": vectorizer_type,
                    "model": model_type,
                    "ngram_range": ngram_range,
                    "max_features": max_features,
                    "f1_score": f1_score,
                    "latency": latency,
                    "interpretability": interp,
                    "timestamp": time.time(),
                },
                generation=generation,
            )

            # Return raw values — DEAP handles direction via weights (1.0, -1.0, 1.0)
            return f1_score, latency, interp

        except Exception as e:
            logger.error(f"Error evaluating individual {individual}: {e}")
            return ERROR_FITNESS

    def _update_objective_ranges(
        self, f1: float, latency: float, interpretability: float
    ):
        """Track observed min/max for each objective (for reporting only)."""
        for obj_name, value in [
            ("f1_score", f1),
            ("latency", latency),
            ("interpretability", interpretability),
        ]:
            if value < self.objective_ranges[obj_name]["min"]:
                self.objective_ranges[obj_name]["min"] = value
            if value > self.objective_ranges[obj_name]["max"]:
                self.objective_ranges[obj_name]["max"] = value

    def _validate_structure(self, scaler: str, dim_reduction: str, model: str) -> bool:
        """
        Validate structure for compatibility.

        Rules:
        1. MultinomialNB requires non-negative input.
           - Incompatible with StandardScaler (produces negative values via centering).
           - Incompatible with RobustScaler (uses median/IQR, can produce negatives).
           - Incompatible with PCA (TruncatedSVD produces negative values).
        """
        if model == "naive_bayes":
            # Rule 1: Scalers that can produce negative values
            if scaler in ("standard", "robust"):
                return False

            # Rule 2: Dim Reduction
            # PCA (TruncatedSVD) produces negative values.
            if dim_reduction == "pca":
                return False

        return True


