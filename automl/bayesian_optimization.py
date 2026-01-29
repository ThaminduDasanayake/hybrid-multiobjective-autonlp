import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


# -----------------------------------------------------------------------------
# 1. DEFINE THE HYPERPARAMETER SEARCH SPACE
# -----------------------------------------------------------------------------
# This dictionary maps each algorithm to its tunable hyperparameters.
# We use scikit-optimize's 'space' dimensions for this.
PARAM_SPACE = {
    "TfidfVectorizer": {
        "ngram_range": Categorical(["1,1", "1,2"], name="vectorizer__ngram_range"),
        "max_df": Real(0.5, 1.0, name="vectorizer__max_df"),
        "min_df": Integer(1, 5, name="vectorizer__min_df"),
    },
    "CountVectorizer": {
        "ngram_range": Categorical(["1,1", "1,2"], name="vectorizer__ngram_range"),
        "max_df": Real(0.5, 1.0, name="vectorizer__max_df"),
        "min_df": Integer(1, 5, name="vectorizer__min_df"),
    },
    "LogisticRegression": {
        "C": Real(1e-2, 1e2, prior="log-uniform", name="classifier__C"),
        "penalty": Categorical(
            ["l2"], name="classifier__penalty"
        ),  # l1 requires different solver
    },
    "SVC": {
        "C": Real(1e-2, 1e2, prior="log-uniform", name="classifier__C"),
        "gamma": Real(1e-3, 1e2, prior="log-uniform", name="classifier__gamma"),
    },
    "DecisionTreeClassifier": {
        "max_depth": Integer(3, 15, name="classifier__max_depth"),
        "min_samples_split": Integer(2, 10, name="classifier__min_samples_split"),
    },
}


class BayesianOptimizer:
    def __init__(self, X, y, pipeline_steps, param_space):
        self.X = X
        self.y = y
        self.pipeline = Pipeline(
            [
                ("vectorizer", pipeline_steps[0][1]()),
                ("classifier", pipeline_steps[1][1]()),
            ]
        )
        self.search_space = []

        if hasattr(self.pipeline.named_steps["vectorizer"], "ngram_range"):
            self.search_space.append(
                Categorical(["1,1", "1,2"], name="vectorizer__ngram_range")
            )

        # Unpack the pipeline steps to get the classes
        vec_class = pipeline_steps[0][1]
        clf_class = pipeline_steps[1][1]

        # Instantiate the classifier differently based on its type
        if clf_class == SVC:
            # If it's an SVC, enable probability estimates
            classifier_instance = clf_class(probability=True, max_iter=5000)
        else:
            # For all other classifiers, instantiate normally
            classifier_instance = clf_class()

        # Instantiate the classes to build the pipeline
        self.pipeline = Pipeline(
            [
                ("vectorizer", vec_class()),
                ("classifier", classifier_instance),
            ]
        )

        # Build search space: vectorizer params + classifier params
        vec_name = vec_class.__name__
        clf_name = clf_class.__name__

        # Add vectorizer parameters
        if vec_name in param_space:
            for param_name, param_space_obj in param_space[vec_name].items():
                self.search_space.append(param_space_obj)

        # Add classifier parameters
        if clf_name in param_space:
            for param_name, param_space_obj in param_space[clf_name].items():
                self.search_space.append(param_space_obj)

        # Convert to tuple for skopt
        self.search_space = tuple(self.search_space)

        # Combine the parameter spaces for the chosen algorithms
        # if clf_class.__name__ not in param_space:
        #     self.search_space = (
        #         param_space[vec_class.__name__]["ngram_range"],
        #         param_space[vec_class.__name__]["max_df"],
        #         param_space[vec_class.__name__]["min_df"],
        #     )
        #     return
        # self.search_space = (
        #     param_space[vec_class.__name__]["ngram_range"],
        #     param_space[vec_class.__name__]["max_df"],
        #     param_space[vec_class.__name__]["min_df"],
        #     param_space[clf_class.__name__]["C"],
        # )

        # Add gamma only if the classifier is SVC
        # if clf_class == SVC:
        #     self.search_space += (param_space[SVC.__name__]["gamma"],)

    def _objective(self, **params):
        # Convert ngram_range string to tuple if present
        if "vectorizer__ngram_range" in params:
            ngram_str = params["vectorizer__ngram_range"]
            params["vectorizer__ngram_range"] = tuple(map(int, ngram_str.split(",")))

        # ngram_str = params["vectorizer__ngram_range"]
        # params["vectorizer__ngram_range"] = tuple(map(int, ngram_str.split(",")))

        self.pipeline.set_params(**params)

        score = np.mean(
            cross_val_score(
                self.pipeline,
                self.X,
                self.y,
                cv=2,
                n_jobs=1,
                scoring="accuracy",
            )
        )

        # skopt minimizes functions, so we return 1.0 - accuracy
        return 1.0 - score

    def run(self, n_calls=15):
        if len(self.search_space) == 0:
            # No hyperparameters to tune
            score = np.mean(
                cross_val_score(
                    self.pipeline,
                    self.X,
                    self.y,
                    cv=2,
                    scoring="accuracy",
                    n_jobs=1,
                )
            )
            return {}, score, 0.0

        print(
            f"🚀 Starting Bayesian Optimization for pipeline: {self.pipeline.steps[0][1].__class__.__name__} -> {self.pipeline.steps[1][1].__class__.__name__}"
        )
        print(
            f"   Searching over {len(self.search_space)} hyperparameters for {n_calls} iterations..."
        )

        if not self.search_space:
            # No tunable params, just fit once and get accuracy
            print("   No hyperparameters to tune, evaluating with defaults...")
            self.pipeline.fit(self.X, self.y)
            acc = np.mean(
                cross_val_score(
                    self.pipeline, self.X, self.y, cv=2, scoring="accuracy", n_jobs=-1
                )
            )
            print("✅ Optimization finished!")
            return {}, acc, 0.0

        # We need a decorated version of the objective function
        # that can handle named arguments from the search space.
        @use_named_args(self.search_space)
        def wrapped_objective(**params):
            return self._objective(**params)

        result = gp_minimize(
            func=wrapped_objective,
            dimensions=self.search_space,
            n_calls=n_calls,
            random_state=42,
            verbose=False,
            n_jobs=1,
        )

        print("✅ Optimization finished!")

        best_params = {dim.name: val for dim, val in zip(result.space, result.x)}
        best_score = 1.0 - result.fun
        score_variance = np.var(result.func_vals)

        return best_params, best_score, score_variance
