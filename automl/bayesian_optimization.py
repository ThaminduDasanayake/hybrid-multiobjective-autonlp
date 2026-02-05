import time
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning of NLP pipelines.

    This class takes a pipeline configuration (vectorizer + model) and
    optimizes its hyperparameters using Gaussian Process-based BO.
    """

    def __init__(self, n_calls: int = 20, cv: int = 3, random_state: int = 42):
        """
        Initialize the Bayesian Optimizer.

        Args:
            n_calls: Number of BO iterations
            cv: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_calls = n_calls
        self.cv = cv
        self.random_state = random_state

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

    def optimize(self, vectorizer_type: str, model_type: str,
                 X_train: list, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a given pipeline configuration.

        Args:
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
        # Profile dataset
        profile = self.dataset_profile(X_train)

        # Define search space based on pipeline configuration
        space = self._get_search_space(vectorizer_type, model_type, profile)

        # Track evaluation metrics
        scores = []
        inference_times = []

        @use_named_args(space)
        def objective(**params):
            # Objective function for BO
            try:
                # Build pipeline
                pipeline = self._build_pipeline(vectorizer_type, model_type, params, profile)

                # Measure training time
                start_time = time.time()
                pipeline.fit(X_train, y_train)

                # Measure inference time
                inference_start = time.time()
                _ = pipeline.predict(X_train[:100])  # Sample for speed
                inference_time = (time.time() - inference_start) / 100
                inference_times.append(inference_time)

                # Cross-validation score
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train,
                    cv=self.cv, scoring='f1_weighted', n_jobs=-1
                )
                score = cv_scores.mean()
                scores.append(score)

                # BO minimizes, so negate the score
                return -score

            except Exception as e:
                print(f"Error in BO iteration: {e}")
                return 0.0  # Return poor score on failure

        # Run Bayesian Optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            n_jobs=1,  # Sequential for stability
            verbose=False
        )

        # Extract best parameters
        best_params = {}
        for param_name, param_value in zip([s.name for s in space], result.x):
            best_params[param_name] = param_value

        best_score = -result.fun  # Negate back to get actual score

        # Compute variance from cross-validation
        variance = np.var(scores) if len(scores) > 0 else 0.0

        # Average inference time
        avg_inference_time = np.mean(inference_times) if len(inference_times) > 0 else 0.001

        return {
            "best_params": best_params,
            "best_score": best_score,
            "variance": variance,
            "inference_time": avg_inference_time,
            "profile": profile,
        }

    def _get_search_space(self, vectorizer_type: str, model_type: str, profile: str) -> list:
        """
        Define the hyperparameter search space.

        Args:
            vectorizer_type: Type of vectorizer
            model_type: Type of model
            profile: Dataset profile ('small', 'medium', 'large')

        Returns:
            List of skopt Space objects
        """
        space = []

        # Dataset-aware limits
        if profile == "small":
            max_features_upper = 5000
            max_ngram = 2
        elif profile == "medium":
            max_features_upper = 10000
            max_ngram = 3
        else:  # large
            max_features_upper = 15000
            max_ngram = 3

        # Vectorizer hyperparameters
        if vectorizer_type in ["tfidf", "count"]:
            space.extend([
                Integer(1, max_ngram, name="ngram_range_max"),
                Integer(1, 10, name="min_df"),
                Real(0.5, 1.0, name="max_df"),
                Integer(1000, max_features_upper, name="max_features")
            ])

        # Model hyperparameters
        if model_type == "logistic":
            space.extend([
                Real(0.01, 10.0, prior="log-uniform", name="C"),
            ])
        elif model_type == "naive_bayes":
            space.extend([
                Real(0.1, 10.0, name="alpha")
            ])
        elif model_type == "svm":
            space.extend([
                Real(0.01, 10.0, prior="log-uniform", name="C"),
                Categorical(["l1", "l2"], name="penalty")
            ])
        elif model_type == "random_forest":
            # Limit estimators for small datasets
            max_estimators = 100 if profile == "small" else 200
            space.extend([
                Integer(10, max_estimators, name="n_estimators"),
                Integer(2, 20, name="max_depth"),
                Integer(2, 10, name="min_samples_split")
            ])

        return space

    def _build_pipeline(self, vectorizer_type: str, model_type: str,
                        params: Dict[str, Any], profile: str) -> Pipeline:
        """
        Build a sklearn pipeline from configuration and parameters.

        Args:
            vectorizer_type: Type of vectorizer
            model_type: Type of model
            params: Hyperparameters
            profile: Dataset profile

        Returns:
            Configured sklearn Pipeline
        """
        # Build vectorizer
        ngram_max = params.get("ngram_range_max", 1)

        if vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=(1, ngram_max),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
                max_features=params.get("max_features", 5000)
            )
        elif vectorizer_type == "count":
            vectorizer = CountVectorizer(
                ngram_range=(1, ngram_max),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
                max_features=params.get("max_features", 5000)
            )
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")

        # Build model
        if model_type == "logistic":
            model = LogisticRegression(
                C=params.get("C", 1.0),
                solver="saga",
                penalty="l2",
                max_iter=3000,
                n_jobs=-1,
                random_state=self.random_state
            )
        elif model_type == "naive_bayes":
            model = MultinomialNB(
                alpha=params.get("alpha", 1.0)
            )
        elif model_type == "svm":
            model = LinearSVC(
                C=params.get("C", 1.0),
                penalty=params.get("penalty", "l2"),
                dual="auto",
                max_iter=1000,
                random_state=self.random_state
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create pipeline
        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", model)
        ])

        return pipeline
