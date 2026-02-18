from typing import Dict, Any, Tuple
import numpy as np
import time
from utils.logger import get_logger
from .bayesian_optimization import BayesianOptimizer
from .persistence import ResultStore

logger = get_logger("evaluator")

class PipelineEvaluator:
    """
    Evaluates pipeline individuals using Bayesian Optimization.
    Calculates multi-objective fitness scores:
    1. F1 Score (Maximize)
    2. Latency (Minimize)
    3. Interpretability (Maximize)
    """
    def __init__(self, 
                 X_train: list, 
                 y_train: np.ndarray, 
                 bo_optimizer: BayesianOptimizer,
                 result_store: ResultStore,
                 objective_ranges: Dict[str, Dict[str, float]] = None):
        
        self.X_train = X_train
        self.y_train = y_train
        self.bo_optimizer = bo_optimizer
        self.result_store = result_store
        
        self.objective_ranges = objective_ranges or {
            "f1_score": {"min": float("inf"), "max": float("-inf")},
            "latency": {"min": float("inf"), "max": float("-inf")},
            "interpretability": {"min": float("inf"), "max": float("-inf")},
        }

    def evaluate(self, individual: list, generation: int = 0) -> Tuple[float, float, float]:
        """
        Evaluate an individual (GA chromosome).
        
        Returns:
            Tuple of (neg_f1, latency, neg_interpretability)
            DEAP minimizes by default, so we negate maximized objectives.
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
            return -1.0, 2.0, -1.0
        
        try:
            # Run Bayesian Optimization
            bo_result = self.bo_optimizer.optimize(
                scaler_type, dim_reduction_type, vectorizer_type, model_type, ngram_range, max_features, self.X_train, self.y_train
            )
            
            # Track time
            if "optimization_time" in bo_result:
                self.result_store.add_time_stats(optimization_time=bo_result["optimization_time"])
                
            f1_score = bo_result["best_score"]
            latency = bo_result["inference_time"]
            
            # Compute interpretability
            interpretability = self._intrinsic_interpretability_score(
                scaler_type, dim_reduction_type, vectorizer_type, model_type, ngram_range, max_features, bo_result["best_params"]
            )
            
            # Normalize objectives
            norm_f1 = self._normalize_objective(f1_score, "f1_score")
            norm_latency = self._normalize_objective(latency, "latency")
            norm_interp = self._normalize_objective(
                interpretability, "interpretability"
            )
            
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
                "interpretability": interpretability,
                "variance": bo_result["variance"]
            }
            self.result_store.cache_evaluation(cache_key, result)
            
            # Add to history
            self.result_store.add_to_history({
                "scaler": scaler_type,
                "dim_reduction": dim_reduction_type,
                "vectorizer": vectorizer_type,
                "model": model_type,
                "ngram_range": ngram_range,
                "max_features": max_features,
                "f1_score": f1_score,
                "latency": latency,
                "interpretability": interpretability,
                "timestamp": time.time()
            }, generation=generation)
            
            # Return raw normalized values (DEAP handles maximization/minimization via weights)
            # Weights are (1.0, -1.0, 1.0) -> Max F1, Min Latency, Max Interp
            return norm_f1, norm_latency, norm_interp
            
        except Exception as e:
            logger.error(f"Error evaluating individual {individual}: {e}")
            return -1.0, 2.0, -1.0

    def _normalize_objective(self, value: float, obj_name: str) -> float:
        """Normalize objective value to [0, 1] range. Direction is handled by DEAP weights."""
        # Update ranges
        if value < self.objective_ranges[obj_name]["min"]:
            self.objective_ranges[obj_name]["min"] = value
        if value > self.objective_ranges[obj_name]["max"]:
            self.objective_ranges[obj_name]["max"] = value

        min_val = self.objective_ranges[obj_name]["min"]
        max_val = self.objective_ranges[obj_name]["max"]

        if abs(max_val - min_val) < 1e-10:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)

        # We want strict [0, 1] mapping where 0=min_val, 1=max_val
        # Direction is handled by DEAP weights
        return normalized

    def _validate_structure(self, scaler: str, dim_reduction: str, model: str) -> bool:
        """
        Validate structure for compatibility.
        
        Rules:
        1. MultinomialNB requires non-negative input.
           - Incompatible with StandardScaler (unless with_mean=False and data is non-neg? But user reported failure).
           - Incompatible with PCA (TruncatedSVD produces negative values).
        """
        if model == "naive_bayes":
            # Rule 1: Scaler
            # MaxAbsScaler is safe. MinMaxScaler is safe. None is safe.
            # StandardScaler is unsafe if it produces negative values.
            if scaler == "standard":
                return False
                
            # Rule 2: Dim Reduction
            # PCA (TruncatedSVD) produces negative values.
            if dim_reduction == "pca":
                return False
                
        return True

    def _intrinsic_interpretability_score(self, scaler: str, dim_reduction: str, vectorizer: str, model: str, ngram_range: str, max_features: Any, params: Dict[str, Any]) -> float:
        """Compute intrinsic interpretability score (0.0 to 1.0)."""
        score = 0.0
        
        # Model complexity (30%)
        model_scores = {
            "logistic": 1.0, 
            "naive_bayes": 0.9, 
            "svm": 0.7, 
            "random_forest": 0.4,
            "lightgbm": 0.3,
            "sgd": 0.9
        }
        score += 0.3 * model_scores.get(model, 0.5)
        
        # Feature transparency (Vectorization) (20%)
        vectorizer_scores = {
            "count": 1.0, 
            "tfidf": 0.8
        }
        score += 0.2 * vectorizer_scores.get(vectorizer, 0.5)
        
        # Preprocessing Complexity (Scaler + Dim Reduction) (20%)
        scaler_scores = {
            None: 1.0,
            "minmax": 0.9,
            "standard": 0.9,
            "robust": 0.85
        }
        dim_red_scores = {
            None: 1.0,
            "select_k_best": 0.8, # Selects features, keeps meaning
            "pca": 0.4            # Projects features, loses meaning
        }
        
        preprocessing_score = (scaler_scores.get(scaler, 0.5) + dim_red_scores.get(dim_reduction, 0.5)) / 2
        score += 0.2 * preprocessing_score

        # Hyperparameter simplicity (30%)
        simplicity = 0.0
        
        # 1. N-gram penalty
        # "1-1" (unigrams) -> simplest -> 1.0
        # "1-2" (bigrams) -> medium -> 0.7
        # "1-3" (trigrams) -> complex -> 0.4
        ngram_str = str(ngram_range)
        if ngram_str == "1-1":
            simplicity += 0.4 * 1.0
        elif ngram_str == "1-2":
            simplicity += 0.4 * 0.7
        else: # "1-3"
            simplicity += 0.4 * 0.4
        
        # 2. Max Features penalty
        # 5000 -> 1.0
        # 10000 -> 0.8
        # 20000 -> 0.6
        # "None" -> 0.2 (unlimited vocabulary is very hard to interpret)
        max_feat_str = str(max_features)
        if max_feat_str == "5000":
            feat_score = 1.0
        elif max_feat_str == "10000":
            feat_score = 0.8
        elif max_feat_str == "20000":
            feat_score = 0.6
        else: # "None"
            feat_score = 0.2
            
        simplicity += 0.3 * feat_score
        
        if model == "logistic":
            C = params.get("C", 1.0)
            simplicity += 0.3 * (1.0 / (1.0 + C))
        elif model == "random_forest":
            max_depth = params.get("max_depth", 10)
            simplicity += 0.3 * (1.0 / (1.0 + max_depth / 10))
        else:
            simplicity += 0.3 * 0.5
            
        score += 0.3 * simplicity
        
        return max(0.0, min(1.0, score))
