"""
THE TUNER — bayesian_optimization.py
======================================
Once the Genetic Algorithm (the Architect) has decided *what kind* of pipeline to build
(e.g., "TF-IDF + no scaler + Logistic Regression"), this module finds the *best settings*
for that pipeline — its hyperparameters.

It uses Bayesian Optimization (BO) with a Gaussian Process (GP) surrogate model.
Think of it like a smart knob-turner: instead of randomly trying every combination,
it builds a probability model of the performance landscape and focuses its attempts
on regions that are likely to be better, based on what it has already tried.

The key trade-off: the first 10 calls are purely exploratory (random sampling to
initialize the GP). After that, the GP guides the remaining calls toward promising
regions. This is why n_calls must be meaningfully above 10 to get the benefit of BO.
"""

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

# Maps dataset size profile to per-solver max_iter caps.
# Defined at module level so _get_search_space() (called per BO evaluation)
# does a single O(1) dict lookup instead of an if/elif branch.
_MAX_ITER: dict = {
    "small":  {"logistic": 1000, "svm": 1500},
    "medium": {"logistic": 2000, "svm": 3000},
    "large":  {"logistic": 3000, "svm": 5000},
}



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

                # Cross-validate the pipeline with the current hyperparameter config.
                # return_estimator=True keeps the fitted model from the last fold so we
                # can measure real inference latency on it below.
                # n_jobs=1: this objective runs inside a ProcessPoolExecutor worker; spawning
                # additional loky subprocesses from within a spawned process causes nested
                # parallelism that silently degrades to sequential on macOS (spawn start method)
                # and can deadlock on some sklearn/joblib versions.
                cv_result = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=self.cv,
                    scoring="f1_weighted",
                    n_jobs=1,
                    error_score=0.0,
                    return_estimator=True,
                )
                score = cv_result["test_score"].mean()
                scores.append(score)

                inference_time = self._measure_inference_time(cv_result, X_train)
                inference_times.append(inference_time)

                # gp_minimize *minimizes* its objective, but we want to *maximize* F1.
                # Negating F1 converts our maximization problem into a minimization one.
                # A failed pipeline (NaN) is treated as F1=0 (worst possible score).
                if np.isnan(score):
                    return 0.0

                return -score

            except Exception as e:
                logger.warning(
                    f"BO objective failed for [{vectorizer_type}/{model_type}]: "
                    f"{type(e).__name__}: {e}"
                )
                return 0.0  # Treat failed pipeline as worst score (F1=0) for the minimizer

        # Run the Bayesian Optimization loop. The first n_initial_points calls are
        # random (to build an initial picture of the landscape). After that, the
        # Gaussian Process model guides subsequent calls toward promising regions.
        # n_jobs=1: sequential evaluation within the loop; see objective() comment above.
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            n_initial_points=min(10, self.n_calls),
            random_state=self.random_state,
            n_jobs=1,  # Sequential for stability
            verbose=False,
        )

        # result.x contains the best hyperparameter *values* found; result.fun is
        # the best (negated) F1 score seen. We map values back to named parameters
        # and un-negate to recover the true F1 score.
        best_params = {}
        for param_name, param_value in zip([s.name for s in space], result.x):
            best_params[param_name] = param_value

        best_score = -result.fun  # un-negate to recover F1

        # Variance across all BO objective evaluations — a proxy for how sensitive
        # this pipeline's performance is to hyperparameter choices.
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
        """Sample one random hyperparameter set and evaluate it; used for the GA-only ablation.

        When disable_bo=True, this replaces the full BO loop. Instead of 15+ guided
        calls, we draw a single random hyperparameter configuration and evaluate it once.
        This lets the ablation study answer: "how much does BO tuning actually help
        over just picking random hyperparameters and running the GA alone?"
        """
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
                n_jobs=1,  # must be 1 inside a worker process — see objective() comment above
                error_score=0.0,
                return_estimator=True,
            )
            score = cv_result["test_score"].mean()
            variance = np.var(cv_result["test_score"])

            if np.isnan(score):
                score = 0.0

            inference_time = self._measure_inference_time(cv_result, X_train)

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
    def _measure_inference_time(cv_result, X_train) -> float:
        """Measure per-sample inference latency using the last CV fold's fitted estimator."""
        fitted = cv_result["estimator"][-1]
        n_test = min(100, len(X_train))
        fitted.predict(X_train[:n_test])  # warmup: absorbs sklearn validation + first-call overhead
        start = time.perf_counter()
        fitted.predict(X_train[:n_test])
        return (time.perf_counter() - start) / n_test

    @staticmethod
    def _get_search_space(
        dim_reduction_type: str,
        vectorizer_type: str,
        model_type: str,
        profile: str,
        max_features_val: Any,
        n_samples: int,
    ) -> list:
        """Build the skopt search space for the given pipeline configuration and dataset profile.

        The search space is *conditional*: only axes relevant to the current pipeline
        are included. For example, a pipeline with no dim_reduction step has no
        pca_n_components axis; a NaiveBayes pipeline has no C or max_iter axis.
        This keeps the BO search space tight and makes each call meaningful.
        """
        space = []

        # max_iter (how long solvers run before giving up) is scaled to dataset size.
        # A small dataset converges quickly; a large dataset needs more iterations.
        # Capping max_iter prevents the BO budget being wasted on configs that simply
        # haven't converged yet rather than genuinely underperforming.
        _iters = _MAX_ITER[profile]
        max_iter_logistic = _iters["logistic"]
        max_iter_svm = _iters["svm"]

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
