import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class RandomSearchBaseline:
    """
    Random search baseline for comparison with HybridAutoML.

    Randomly samples pipeline configurations and hyperparameters,
    providing a lower bound on performance.
    """

    def __init__(self, n_iterations: int = 50, cv: int = 3, random_state: int = 42):
        """
        Initialize random search baseline.

        Args:
            n_iterations: Number of random configurations to try
            cv: Number of cross-validation folds
            random_state: Random seed
        """
        self.n_iterations = n_iterations
        self.cv = cv
        self.random_state = random_state
        np.random.seed(random_state)

        self.results = []

    def run(self, X_train: list, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Run random search.

        Args:
            X_train: Training texts
            y_train: Training labels

        Returns:
            Dictionary containing all evaluated configurations
        """
        print(f"Running random search baseline with {self.n_iterations} iterations...")

        vectorizer_types = ["tfidf", "count"]
        model_types = ["logistic", "naive_bayes", "svm", "random_forest"]

        for i in range(self.n_iterations):
            # Random configuration
            vectorizer_type = np.random.choice(vectorizer_types)
            model_type = np.random.choice(model_types)

            # Random hyperparameters
            params = self._sample_random_params(vectorizer_type, model_type)

            try:
                # Build and evaluate pipeline
                pipeline = self._build_pipeline(vectorizer_type, model_type, params)

                # Measure inference time
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                inference_start = time.time()
                _ = pipeline.predict(X_train[:100])
                inference_time = (time.time() - inference_start) / 100

                # Cross-validation score
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train,
                    cv=self.cv, scoring='f1_weighted', n_jobs=-1
                )
                f1_score = cv_scores.mean()

                # Interpretability (use same scoring as HybridAutoML)
                interpretability = self._compute_interpretability(
                    vectorizer_type, model_type, params
                )

                self.results.append({
                    "vectorizer": vectorizer_type,
                    "model": model_type,
                    "params": params,
                    "f1_score": f1_score,
                    "latency": inference_time,
                    "interpretability": interpretability,
                    "iteration": i
                })

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{self.n_iterations} iterations")

            except Exception as e:
                print(f"  Error in iteration {i}: {e}")
                continue

        return {
            "all_solutions": self.results,
            "stats": {
                "total_evaluations": len(self.results),
                "best_f1": max([r["f1_score"] for r in self.results]),
                "best_latency": min([r["latency"] for r in self.results]),
                "best_interpretability": max([r["interpretability"] for r in self.results])
            }
        }

    def _sample_random_params(self, vectorizer_type: str, model_type: str) -> Dict[str, Any]:
        """Sample random hyperparameters."""
        params = {}

        # Vectorizer params
        params["ngram_range_max"] = np.random.randint(1, 4)
        params["min_df"] = np.random.randint(1, 11)
        params["max_df"] = np.random.uniform(0.5, 1.0)
        params["max_features"] = np.random.randint(1000, 10001)

        # Model params
        if model_type == "logistic":
            params["C"] = np.random.uniform(0.01, 10.0)
            params["penalty"] = np.random.choice(["l1", "l2"])
        elif model_type == "naive_bayes":
            params["alpha"] = np.random.uniform(0.1, 10.0)
        elif model_type == "svm":
            params["C"] = np.random.uniform(0.01, 10.0)
            params["penalty"] = np.random.choice(["l1", "l2"])
        elif model_type == "random_forest":
            params["n_estimators"] = np.random.randint(10, 201)
            params["max_depth"] = np.random.randint(2, 21)
            params["min_samples_split"] = np.random.randint(2, 11)

        return params

    def _build_pipeline(self, vectorizer_type: str, model_type: str,
                        params: Dict[str, Any]) -> Pipeline:
        """Build sklearn pipeline."""
        ngram_max = params.get("ngram_range_max", 1)

        if vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=(1, ngram_max),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
                max_features=params.get("max_features", 5000)
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=(1, ngram_max),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
                max_features=params.get("max_features", 5000)
            )

        if model_type == "logistic":
            model = LogisticRegression(
                C=params.get("C", 1.0),
                penalty=params.get("penalty", "l2"),
                solver="saga",
                max_iter=1000,
                random_state=self.random_state
            )
        elif model_type == "naive_bayes":
            model = MultinomialNB(alpha=params.get("alpha", 1.0))
        elif model_type == "svm":
            model = LinearSVC(
                C=params.get("C", 1.0),
                penalty=params.get("penalty", "l2"),
                dual="auto",
                max_iter=1000,
                random_state=self.random_state
            )
        else:  # random_forest
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=self.random_state,
                n_jobs=-1
            )

        return Pipeline([("vectorizer", vectorizer), ("classifier", model)])

    def _compute_interpretability(self, vectorizer_type: str,
                                  model_type: str,
                                  params: Dict[str, Any]) -> float:
        """Compute interpretability score (same as HybridAutoML)."""
        score = 0.0

        model_scores = {
            "logistic": 1.0,
            "naive_bayes": 0.9,
            "svm": 0.7,
            "random_forest": 0.4
        }
        score += 0.4 * model_scores.get(model_type, 0.5)

        vectorizer_scores = {"count": 1.0, "tfidf": 0.8}
        score += 0.3 * vectorizer_scores.get(vectorizer_type, 0.5)

        simplicity = 0.0
        ngram_max = params.get("ngram_range_max", 1)
        simplicity += 0.4 * (1.0 / ngram_max)

        max_features = params.get("max_features", 5000)
        simplicity += 0.3 * (1.0 - min(max_features / 10000, 1.0))

        if model_type == "logistic":
            C = params.get("C", 1.0)
            simplicity += 0.3 * (1.0 / (1.0 + C))
        elif model_type == "random_forest":
            max_depth = params.get("max_depth", 10)
            simplicity += 0.3 * (1.0 / (1.0 + max_depth / 10))
        else:
            simplicity += 0.3 * 0.5

        score += 0.3 * simplicity

        return max(0.0, min(1.0, score))


class GridSearchBaseline:
    """
    Simple grid search baseline (exhaustive but limited).

    Tries a predefined grid of configurations.
    """

    def __init__(self, cv: int = 3, random_state: int = 42):
        """
        Initialize grid search baseline.

        Args:
            cv: Number of cross-validation folds
            random_state: Random seed
        """
        self.cv = cv
        self.random_state = random_state
        self.results = []

    def run(self, X_train: list, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Run grid search over a small grid.

        Args:
            X_train: Training texts
            y_train: Training labels

        Returns:
            Dictionary containing all evaluated configurations
        """
        print("Running grid search baseline...")

        # Define small grid
        grid = {
            "vectorizer": ["tfidf", "count"],
            "model": ["logistic", "naive_bayes", "svm"],
            "max_features": [1000, 5000],
            "ngram_max": [1, 2]
        }

        total = (len(grid["vectorizer"]) * len(grid["model"]) *
                 len(grid["max_features"]) * len(grid["ngram_max"]))

        count = 0
        for vec in grid["vectorizer"]:
            for model in grid["model"]:
                for max_feat in grid["max_features"]:
                    for ngram in grid["ngram_max"]:
                        params = {
                            "max_features": max_feat,
                            "ngram_range_max": ngram,
                            "min_df": 1,
                            "max_df": 1.0,
                            "C": 1.0,
                            "alpha": 1.0,
                            "penalty": "l2"
                        }

                        try:
                            baseline = RandomSearchBaseline(n_iterations=1,
                                                            cv=self.cv,
                                                            random_state=self.random_state)
                            pipeline = baseline._build_pipeline(vec, model, params)

                            start_time = time.time()
                            pipeline.fit(X_train, y_train)
                            inference_start = time.time()
                            _ = pipeline.predict(X_train[:100])
                            inference_time = (time.time() - inference_start) / 100

                            cv_scores = cross_val_score(
                                pipeline, X_train, y_train,
                                cv=self.cv, scoring='f1_weighted', n_jobs=-1
                            )
                            f1_score = cv_scores.mean()

                            interpretability = baseline._compute_interpretability(
                                vec, model, params
                            )

                            self.results.append({
                                "vectorizer": vec,
                                "model": model,
                                "params": params,
                                "f1_score": f1_score,
                                "latency": inference_time,
                                "interpretability": interpretability
                            })

                            count += 1
                            print(f"  Completed {count}/{total} configurations")

                        except Exception as e:
                            print(f"  Error: {e}")
                            continue

        return {
            "all_solutions": self.results,
            "stats": {
                "total_evaluations": len(self.results),
                "best_f1": max([r["f1_score"] for r in self.results]) if self.results else 0,
                "best_latency": min([r["latency"] for r in self.results]) if self.results else 1,
                "best_interpretability": max([r["interpretability"] for r in self.results]) if self.results else 0
            }
        }