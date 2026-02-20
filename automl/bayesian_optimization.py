import time
from typing import Dict, Tuple, Any, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

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

        # Track evaluation metrics
        scores = []
        inference_times = []

        @use_named_args(space)
        def objective(**params):
            # Objective function for BO
            try:
                # Build pipeline
                pipeline = self._build_pipeline(
                    scaler_type,
                    dim_reduction_type,
                    vectorizer_type,
                    model_type,
                    ngram_range,
                    max_features,
                    params,
                    profile,
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
        if profile == "small":
            max_features_upper = 5000
            max_ngram = 2
        elif profile == "medium":
            max_features_upper = 10000
            max_ngram = 3
        else:  # large
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
                    Integer(1000, 3000, name="max_iter"),
                ]
            )
        elif model_type == "naive_bayes":
            space.extend([Real(0.1, 10.0, name="alpha")])
        elif model_type == "svm":
            space.extend(
                [
                    Real(0.01, 10.0, prior="log-uniform", name="C"),
                    Categorical(["l1", "l2"], name="penalty"),
                    Integer(1000, 5000, name="max_iter"),
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

    def _build_pipeline(
        self,
        scaler_type: str,
        dim_reduction_type: str,
        vectorizer_type: str,
        model_type: str,
        ngram_range: str,
        max_features: Any,
        params: Dict[str, Any],
        profile: str,
    ) -> Pipeline:
        """
        Build a sklearn pipeline from configuration and parameters.

        Args:
            scaler_type: Type of scaler
            dim_reduction_type: Type of dimensionality reduction
            vectorizer_type: Type of vectorizer
            model_type: Type of model
            params: Hyperparameters
            profile: Dataset profile

        Returns:
            Configured sklearn Pipeline
        """
        steps = []

        # Build vectorizer
        # Sanitize ngram_range: Handle numpy string and ensure tuple of ints
        if isinstance(ngram_range, (str, np.str_)):
            min_n, max_n = map(int, str(ngram_range).split("-"))
        else:
            # Fallback or assume it's already a tuple/list (though type hint says str)
            min_n, max_n = map(int, ngram_range)

        # Sanitize max_features: Handle numpy string/int and "None" string
        if str(max_features) == "None":
            max_feat = None
        else:
            max_feat = int(max_features)

        if vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=(min_n, max_n),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
                max_features=max_feat,
            )
        elif vectorizer_type == "count":
            vectorizer = CountVectorizer(
                ngram_range=(min_n, max_n),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
                max_features=max_feat,
            )
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
        steps.append(("vectorizer", vectorizer))

        # 2. Scaler
        if scaler_type == "standard":
            steps.append(("scaler", StandardScaler(with_mean=False)))
        elif scaler_type == "maxabs":
            # MaxAbsScaler is the correct choice for sparse text data (TF-IDF/Count).
            # It scales by maximum absolute value without shifting center, preserving sparsity.
            steps.append(("scaler", MaxAbsScaler()))
        elif scaler_type == "robust":
            steps.append(("scaler", RobustScaler(with_centering=False)))

        # 3. Dim Reduction
        if dim_reduction_type == "pca":
            # Using TruncatedSVD for interactions
            n_components = params.get("pca_n_components", 50)
            steps.append(("dim_reduction", TruncatedSVD(n_components=n_components)))
        elif dim_reduction_type == "select_k_best":
            k = int(params.get("k_best_k", 100))
            steps.append(("dim_reduction", SelectKBest(f_classif, k=k)))

        # 4. Model
        if model_type == "logistic":
            # Use saga universally — supports multiclass natively, sparse data, and L2 penalty.
            # liblinear is deprecated for multiclass in sklearn >=1.8.
            model = LogisticRegression(
                C=params.get("C", 1.0),
                solver="saga",
                penalty="l2",
                max_iter=params.get("max_iter", 3000),
                n_jobs=-1,
                random_state=self.random_state,
            )
        elif model_type == "naive_bayes":
            model = MultinomialNB(alpha=params.get("alpha", 1.0))
        elif model_type == "svm":
            # Dual selection: Prefer dual=True when n_samples < n_features or sparse data
            dual = "auto"
            if scaler_type != "standard":
                dual = True

            model = LinearSVC(
                C=params.get("C", 1.0),
                penalty=params.get("penalty", "l2"),
                dual=dual,
                max_iter=params.get("max_iter", 5000),
                random_state=self.random_state,
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif model_type == "lightgbm":
            if LGBMClassifier is None:
                # Log warning and raise exception to be caught by evaluator (or return dummy)
                # Evaluator catches Exception and returns penalty.
                # That fits "log a warning and skip".
                raise ImportError("LightGBM is not installed.")

            model = LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                num_leaves=params.get("num_leaves", 31),
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif model_type == "sgd":
            model = SGDClassifier(
                loss=params.get("loss", "hinge"),
                penalty=params.get("penalty", "l2"),
                alpha=params.get("alpha", 1e-4),
                max_iter=params.get("max_iter", 2000),
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        steps.append(("classifier", model))

        # Create pipeline
        pipeline = Pipeline(steps)

        return pipeline
