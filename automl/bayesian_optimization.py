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
}


class BayesianOptimizer:
    def __init__(self, X, y, pipeline_steps, param_space):
        self.X = X
        self.y = y

        # Unpack the pipeline steps to get the classes
        vec_class = pipeline_steps[0][1]
        clf_class = pipeline_steps[1][1]

        # Instantiate the classifier differently based on its type
        if clf_class == SVC:
            # If it's an SVC, enable probability estimates
            classifier_instance = clf_class(probability=True)
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

        # Combine the parameter spaces for the chosen algorithms
        self.search_space = (
            param_space[vec_class.__name__]["ngram_range"],
            param_space[vec_class.__name__]["max_df"],
            param_space[vec_class.__name__]["min_df"],
            param_space[clf_class.__name__]["C"],
        )

        # Add gamma only if the classifier is SVC
        if clf_class == SVC:
            self.search_space += (param_space[SVC.__name__]["gamma"],)

    def _objective(self, **params):
        ngram_str = params["vectorizer__ngram_range"]
        params["vectorizer__ngram_range"] = tuple(map(int, ngram_str.split(",")))

        self.pipeline.set_params(**params)

        score = np.mean(
            cross_val_score(
                self.pipeline,
                self.X,
                self.y,
                cv=3,
                n_jobs=-1,  # Use all available CPU cores
                scoring="accuracy",
            )
        )

        # skopt minimizes functions, so we return 1.0 - accuracy
        return 1.0 - score

    def run(self, n_calls=20):
        print(
            f"🚀 Starting Bayesian Optimization for pipeline: {self.pipeline.steps[0][1].__class__.__name__} -> {self.pipeline.steps[1][1].__class__.__name__}"
        )
        print(
            f"   Searching over {len(self.search_space)} hyperparameters for {n_calls} iterations..."
        )

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
        )

        print("✅ Optimization finished!")

        best_params = {dim.name: val for dim, val in zip(result.space, result.x)}
        best_score = 1.0 - result.fun
        score_variance = np.var(result.func_vals)

        return best_params, best_score, score_variance


# -----------------------------------------------------------------------------
# 3. PUTTING IT ALL TOGETHER: EXAMPLE USAGE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Load sample data
    categories = ["sci.med", "sci.space"]
    newsgroups_train = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42
    )
    X_train = newsgroups_train.data
    y_train = newsgroups_train.target

    # This is the "winning" pipeline from our GA run.
    # We are providing the *classes* themselves, not instances.
    best_pipeline_from_ga = [("vectorizer", TfidfVectorizer), ("classifier", SVC)]

    # 1. Initialize the optimizer with the data and the pipeline structure
    bo_optimizer = BayesianOptimizer(
        X=X_train,
        y=y_train,
        pipeline_steps=best_pipeline_from_ga,
        param_space=PARAM_SPACE,
    )

    # 2. Run the optimization
    best_hyperparams, best_accuracy = bo_optimizer.run(
        n_calls=25
    )  # More calls = better, but slower

    # 3. Print the final results
    print("\n" + "=" * 50)
    print("🏆 Best Hyperparameters Found 🏆")
    for param, value in best_hyperparams.items():
        print(f"  - {param}: {value}")
    print(f"\n  - Best Cross-Validated Accuracy: {best_accuracy:.4f}")
    print("=" * 50)
