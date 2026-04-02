import time
from typing import Dict, Any, Optional

import numpy as np
from sklearn.model_selection import cross_validate
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from utils.logger import get_logger
from .pipeline_builder import build_pipeline

logger = get_logger("bayesian_optimization")



class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning of NLP pipelines.

    This class takes a pipeline configuration (vectorizer + model) and
    optimizes its hyperparameters using Gaussian Process-based BO.
    """

    def __init__(
        self,
        n_calls: int = 20,
        cv: int = 2,
        random_state: int = 42,
        disable_bo: bool = False,
    ):
        """
        Initialize the Bayesian Optimizer.

        Args:
            n_calls: Number of BO iterations
            cv: Number of cross-validation folds
            random_state: Random seed for reproducibility
            disable_bo: If True, skip gp_minimize and evaluate one random
                        hyperparameter sample instead (GA-only ablation)
        """
        self.n_calls = n_calls
        self.cv = cv
        self.random_state = random_state
        self.disable_bo = disable_bo
        self._rng = np.random.RandomState(random_state)

    @staticmethod
    def dataset_profile(X) -> str:
        """Return dataset size profile ('small', 'medium', or 'large') for adaptive configuration."""
        n = len(X)
        if n < 2_000:
            return "small"
        elif n < 50_000:
            return "medium"
        else:
            return "large"

    def optimize(
        self,
        scaler_type: Optional[str],
        dim_reduction_type: Optional[str],
        vectorizer_type: str,
        model_type: str,
        ngram_range: str,
        max_features: Any,
        X_train: list,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """Run Gaussian Process BO over the hyperparameter search space and return the best result dict."""
        # Measure optimization time
        optimization_start = time.time()

        # Profile dataset
        profile = self.dataset_profile(X_train)

        # Define search space based on pipeline configuration
        n_samples = len(X_train)
        space = self._get_search_space(
            dim_reduction_type,
            vectorizer_type,
            model_type,
            profile,
            max_features,
            n_samples,
        )

        # GA-only ablation: skip BO, sample one random config
        if self.disable_bo:
            return self._random_sample_evaluate(
                space,
                scaler_type,
                dim_reduction_type,
                vectorizer_type,
                model_type,
                ngram_range,
                max_features,
                X_train,
                y_train,
                optimization_start,
            )

        # Track evaluation metrics
        scores = []
        inference_times = []

        @use_named_args(space)
        def objective(**params):
            # Objective function for BO
            try:
                pipeline = build_pipeline(
                    scaler_type,
                    dim_reduction_type,
                    vectorizer_type,
                    model_type,
                    ngram_range,
                    max_features,
                    params,
                    random_state=self.random_state,
                )

                # return_estimator=True reuses the last fold's fitted model for latency measurement.
                cv_result = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=self.cv,
                    scoring="f1_weighted",
                    n_jobs=-1,
                    error_score=0.0,
                    return_estimator=True,
                )
                score = cv_result["test_score"].mean()
                scores.append(score)

                fitted = cv_result["estimator"][-1]
                n_test = min(100, len(X_train))
                inference_start = time.time()
                _ = fitted.predict(X_train[:n_test])
                inference_time = (time.time() - inference_start) / n_test
                inference_times.append(inference_time)

                # BO minimizes, so negate F1.
                if np.isnan(score):
                    return 0.0

                return -score

            except Exception:
                return 0.0  # Return poor score on failure

        # Run Bayesian Optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            n_initial_points=min(10, self.n_calls),
            random_state=self.random_state,
            n_jobs=1,  # Sequential for stability
            verbose=False,
        )

        # Extract best parameters
        best_params = {}
        for param_name, param_value in zip([s.name for s in space], result.x):
            best_params[param_name] = param_value

        best_score = -result.fun  # un-negate to recover F1

        # Compute variance from cross-validation
        variance = np.var(scores) if len(scores) > 0 else 0.0

        # Average inference time
        avg_inference_time = (
            np.mean(inference_times) if len(inference_times) > 0 else 0.001
        )

        optimization_end = time.time()
        optimization_time = optimization_end - optimization_start

        return {
            "best_params": best_params,
            "best_score": best_score,
            "variance": variance,
            "inference_time": avg_inference_time,
            "profile": profile,
            "optimization_time": optimization_time,
        }

    def _random_sample_evaluate(
        self,
        space,
        scaler_type,
        dim_reduction_type,
        vectorizer_type,
        model_type,
        ngram_range,
        max_features,
        X_train,
        y_train,
        optimization_start,
    ) -> Dict[str, Any]:
        """Sample one random hyperparameter set and evaluate it; used for the GA-only ablation."""
        random_params = {}
        for dim in space:
            sample = dim.rvs(n_samples=1, random_state=self._rng)
            # Guard against Categorical.rvs() returning a bare scalar string.
            if isinstance(sample, str):
                val = sample
            elif isinstance(sample, (list, np.ndarray)):
                val = sample[0]
            else:
                val = sample
            # sklearn validators require native Python types, not numpy scalars.
            if hasattr(val, "item"):
                val = val.item()
            random_params[dim.name] = val

        profile = self.dataset_profile(X_train)
        score = 0.0
        inference_time = 0.001
        variance = 0.0

        try:
            pipeline = build_pipeline(
                scaler_type,
                dim_reduction_type,
                vectorizer_type,
                model_type,
                ngram_range,
                max_features,
                random_params,
                random_state=self.random_state,
            )

            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=self.cv,
                scoring="f1_weighted",
                n_jobs=-1,
                error_score=0.0,
                return_estimator=True,
            )
            score = cv_result["test_score"].mean()
            variance = np.var(cv_result["test_score"])

            if np.isnan(score):
                score = 0.0

            fitted = cv_result["estimator"][-1]
            n_test = min(100, len(X_train))
            inf_start = time.time()
            _ = fitted.predict(X_train[:n_test])
            inference_time = (time.time() - inf_start) / n_test

        except Exception as e:
            logger.warning(
                f"Random pipeline failed (BO disabled), returning F1=0.0: {e}"
            )
            score = 0.0

        optimization_time = time.time() - optimization_start

        return {
            "best_params": random_params,
            "best_score": score,
            "variance": variance,
            "inference_time": inference_time,
            "profile": profile,
            "optimization_time": optimization_time,
        }

    @staticmethod
    def _get_search_space(
        dim_reduction_type: str,
        vectorizer_type: str,
        model_type: str,
        profile: str,
        max_features_val: Any,
        n_samples: int,
    ) -> list:
        """Build the skopt search space for the given pipeline configuration and dataset profile."""
        space = []

        # Scale max_iter to dataset size to avoid wasting BO budget on non-converging configs.
        if profile == "small":
            max_iter_logistic = 1000
            max_iter_svm = 1500
        elif profile == "medium":
            max_iter_logistic = 2000
            max_iter_svm = 3000
        else:  # large
            max_iter_logistic = 3000
            max_iter_svm = 5000

        if dim_reduction_type == "pca":
            # n_components must be < n_samples; TruncatedSVD is used for sparse text data.
            max_components = max(10, min(n_samples - 1, 100))
            space.extend([Integer(10, max_components, name="pca_n_components")])
        elif dim_reduction_type == "select_k_best":
            # Cap k at max_features to avoid SelectKBest requesting more features than exist.
            if str(max_features_val) == "None":
                limit_k = 2000
            else:
                try:
                    limit_k = min(2000, int(max_features_val))
                except (ValueError, TypeError):
                    limit_k = 2000
            space.extend([Integer(100, limit_k, name="k_best_k")])

        if vectorizer_type in ["tfidf", "count"]:
            space.extend(
                [
                    Integer(1, 10, name="min_df"),
                    Real(0.5, 1.0, name="max_df"),
                ]
            )

        # Model hyperparameters
        if model_type == "logistic":
            space.extend(
                [
                    Real(0.01, 10.0, prior="log-uniform", name="C"),
                    Integer(500, max_iter_logistic, name="max_iter"),
                ]
            )
        elif model_type == "naive_bayes":
            space.extend([Real(0.1, 10.0, name="alpha")])
        elif model_type == "svm":
            space.extend(
                [
                    Real(0.01, 10.0, prior="log-uniform", name="C"),
                    Categorical(["l1", "l2"], name="penalty"),
                    Integer(500, max_iter_svm, name="max_iter"),
                ]
            )

        return space
