import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from .pipeline_builder import build_pipeline
from utils.logger import get_logger

logger = get_logger("bayesian_optimization")

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning of NLP pipelines.

    This class takes a pipeline configuration (vectorizer + model) and
    optimizes its hyperparameters using Gaussian Process-based BO.
    """

    def __init__(self, n_calls: int = 20, cv: int = 2, random_state: int = 42, disable_bo: bool = False):
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

        # Set numpy random seed
        np.random.seed(random_state)

    @staticmethod
    def dataset_profile(X) -> str:
        """
        Profile dataset size for adaptive configuration.

        Args:
            X: Input data

        Returns:
            Profile string: 'small',
        """
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
        """
        Optimize hyperparameters for a given pipeline configuration.

        Args:
            scaler_type: Type of scaler ('standard', 'maxabs', 'robust', None)
            dim_reduction_type: Type of dim reduction ('pca', 'select_k_best', None)
            vectorizer_type: Type of vectorizer ('tfidf' or 'count')
            model_type: Type of model ('logistic', 'naive_bayes', 'svm', 'random_forest')
            X_train: Training texts
            y_train: Training labels

        Returns:
            Dictionary containing:
                - best_params: Best hyperparameters found
                - best_score: Best cross-validation score
                - variance: Variance of the best configuration
                - inference_time: Average inference time per sample
        """
        # Measure optimization time
        optimization_start = time.time()

        # Profile dataset
        profile = self.dataset_profile(X_train)

        # Define search space based on pipeline configuration
        n_samples = len(X_train)
        space = self._get_search_space(
            scaler_type,
            dim_reduction_type,
            vectorizer_type,
            model_type,
            profile,
            max_features,
            n_samples,
        )

        # ── GA-only ablation: skip BO, sample one random config ──────────
        if self.disable_bo:
            return self._random_sample_evaluate(
                space, scaler_type, dim_reduction_type,
                vectorizer_type, model_type, ngram_range,
                max_features, X_train, y_train, optimization_start,
            )

        # Track evaluation metrics
        scores = []
        inference_times = []

        @use_named_args(space)
        def objective(**params):
            # Objective function for BO
            try:
                # Build pipeline
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

                # Cross-validation score first (fail-fast: avoids wasted fit if CV fails)
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=self.cv,
                    scoring="f1_weighted",
                    n_jobs=-1,
                    error_score=0.0,
                )
                score = cv_scores.mean()
                scores.append(score)

                # Measure inference time (requires a full fit on all training data)
                pipeline.fit(X_train, y_train)
                inference_start = time.time()
                _ = pipeline.predict(X_train[:100])  # Sample for speed
                inference_time = (time.time() - inference_start) / 100
                inference_times.append(inference_time)

                # BO minimizes, so negate the score
                # Handle NaN if it slips through
                if np.isnan(score):
                    return 0.0

                return -score

            except Exception as e:
                # logger.debug(f"Error in BO iteration: {e}")
                return 0.0  # Return poor score on failure

        # Run Bayesian Optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            n_jobs=1,  # Sequential for stability
            verbose=False,
        )

        # Extract best parameters
        best_params = {}
        for param_name, param_value in zip([s.name for s in space], result.x):
            best_params[param_name] = param_value

        best_score = -result.fun  # Negate back to get actual score

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
        self, space, scaler_type, dim_reduction_type,
        vectorizer_type, model_type, ngram_range,
        max_features, X_train, y_train, optimization_start,
    ) -> Dict[str, Any]:
        """
        GA-only fallback: sample one random hyperparameter set from *space*,
        evaluate it with cross-validation, and return the result.

        On any failure the F1 score is set to 0.0 so the GA can naturally
        eliminate the bad individual without crashing the run.
        """
        # Sample one random point from the search space
        random_params = {}
        for dim in space:
            random_params[dim.name] = dim.rvs(random_state=self.random_state)[0]

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

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=self.cv,
                scoring="f1_weighted",
                n_jobs=-1,
                error_score=0.0,
            )
            score = cv_scores.mean()
            variance = np.var(cv_scores)

            if np.isnan(score):
                score = 0.0

            # Measure inference time
            pipeline.fit(X_train, y_train)
            inf_start = time.time()
            _ = pipeline.predict(X_train[:100])
            inference_time = (time.time() - inf_start) / 100

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

    def _get_search_space(
        self,
        scaler_type: str,
        dim_reduction_type: str,
        vectorizer_type: str,
        model_type: str,
        profile: str,
        max_features_val: Any,
        n_samples: int,
    ) -> list:
        """
        Define the hyperparameter search space.

        Args:
            scaler_type: Type of scaler
            dim_reduction_type: Type of dimensionality reduction
            vectorizer_type: Type of vectorizer
            model_type: Type of model
            profile: Dataset profile ('small', 'medium', 'large')

        Returns:
            List of skopt Space objects
        """
        space = []

        # Dataset-aware limits
        # max_iter is also scaled: small datasets converge in far fewer iterations,
        # so capping it avoids wasting BO budget on non-converging configs.
        if profile == "small":
            max_features_upper = 5000
            max_ngram = 2
            max_iter_logistic = 1000
            max_iter_svm      = 1500
        elif profile == "medium":
            max_features_upper = 10000
            max_ngram = 3
            max_iter_logistic = 2000
            max_iter_svm      = 3000
        else:  # large
            max_iter_logistic = 3000
            max_iter_svm      = 5000
            max_features_upper = 15000
            max_ngram = 3

        # Dim Reduction hyperparameters
        if dim_reduction_type == "pca":
            # Using TruncatedSVD for sparse text data
            # n_components must be < n_features and < n_samples
            # We'll tune it as an integer
            # Max components constrained by n_samples
            max_components = min(n_samples - 1, 100)
            if max_components < 10:
                max_components = 10

            space.extend([Integer(10, max_components, name="pca_n_components")])
        elif dim_reduction_type == "select_k_best":
            # Dynamic limit to avoid warnings
            # If max_features is None, we don't know exact count, but stick to profile default
            if str(max_features_val) == "None":
                limit_k = 2000
            else:
                try:
                    limit_k = min(2000, int(max_features_val))
                except (ValueError, TypeError):
                    limit_k = 2000

            space.extend([Integer(100, limit_k, name="k_best_k")])

        # Vectorizer hyperparameters
        if vectorizer_type in ["tfidf", "count"]:
            space.extend(
                [
                    # ngram_range and max_features are now fixed structure genes
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
        elif model_type == "random_forest":
            # Limit estimators for small datasets
            max_estimators = 100 if profile == "small" else 200
            space.extend(
                [
                    Integer(10, max_estimators, name="n_estimators"),
                    Integer(2, 20, name="max_depth"),
                    Integer(2, 10, name="min_samples_split"),
                ]
            )
        elif model_type == "lightgbm":
            space.extend(
                [
                    Integer(10, 100, name="n_estimators"),
                    Real(0.01, 0.3, name="learning_rate"),
                    Integer(2, 30, name="num_leaves"),
                ]
            )
        elif model_type == "sgd":
            space.extend(
                [
                    Categorical(["hinge", "log_loss", "modified_huber"], name="loss"),
                    Categorical(["l2", "l1", "elasticnet"], name="penalty"),
                    Real(1e-5, 1e-2, prior="log-uniform", name="alpha"),
                ]
            )

        return space

