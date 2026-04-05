"""
THE ASSEMBLY LINE — pipeline_builder.py
=========================================
This module physically constructs sklearn Pipeline objects from the abstract gene
values that the GA and BO produce. It is the single source of truth for how a
chromosome like ["maxabs", "select_k_best", "tfidf", "logistic", "1-2", 5000]
gets translated into a runnable machine learning object.

Every pipeline follows the same four-step assembly order:
  1. Vectorizer  — convert raw text strings into a numeric feature matrix
  2. Scaler      — optionally normalise those numbers (not all pipelines have this)
  3. Dim Reduction — optionally compress or filter the feature space
  4. Classifier  — make the final class prediction

This fixed order mirrors a standard NLP preprocessing convention and ensures
that sklearn's Pipeline sequencing is always valid.
"""

import numpy as np
from typing import Any, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def build_pipeline(
    scaler_type: Optional[str],
    dim_reduction_type: Optional[str],
    vectorizer_type: str,
    model_type: str,
    ngram_range: str,
    max_features: Any,
    params: Dict[str, Any],
    random_state: int = 42,
) -> Pipeline:
    """Build and return a configured sklearn Pipeline from component names and hyperparameters."""
    steps = []

    # Step 1 — Vectorizer: turn raw text into a sparse numeric matrix.
    # The GA genes may arrive as numpy string types (np.str_) rather than plain Python
    # strings depending on how np.random.choice sampled them. Both cases are handled
    # here defensively to prevent sklearn type errors downstream.
    #
    # TF-IDF vs Count: TF-IDF down-weights common words (e.g., "the"), producing a
    # more nuanced feature representation. CountVectorizer is simpler and more
    # interpretable — you can directly read "this word appeared N times". The
    # interpretability scorer reflects this: Count gets a slightly higher score.
    if isinstance(ngram_range, (str, np.str_)):
        min_n, max_n = map(int, str(ngram_range).split("-"))
    else:
        min_n, max_n = map(int, ngram_range)

    # "None" arrives as a string from the gene pool ("None" != None in Python).
    # We explicitly convert it here so sklearn receives a proper None value.
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

    # Step 2 — Scaler (optional): normalise the feature values.
    # with_mean=False is required for StandardScaler and RobustScaler on sparse
    # matrices. Centering a sparse matrix (subtracting the mean) makes it dense,
    # which can exhaust memory on large vocabularies. MaxAbsScaler does not centre
    # so it is naturally sparse-safe.
    if scaler_type == "standard":
        steps.append(("scaler", StandardScaler(with_mean=False)))
    elif scaler_type == "maxabs":
        steps.append(("scaler", MaxAbsScaler()))
    elif scaler_type == "robust":
        steps.append(("scaler", RobustScaler(with_centering=False)))

    # Step 3 — Dimensionality Reduction (optional): compress or filter features.
    # We use TruncatedSVD (labeled "pca" in the gene pool) rather than standard PCA
    # because standard PCA requires a dense matrix (it computes the full covariance).
    # TruncatedSVD works directly on sparse TF-IDF/Count matrices, which is essential
    # for memory efficiency on large vocabularies.
    # SelectKBest keeps only the K features most correlated with the class labels
    # (using the F-statistic), which is more interpretable than SVD decomposition.
    if dim_reduction_type == "pca":
        n_components = params.get("pca_n_components", 50)
        steps.append(("dim_reduction", TruncatedSVD(n_components=n_components)))
    elif dim_reduction_type == "select_k_best":
        k = int(params.get("k_best_k", 100))
        steps.append(("dim_reduction", SelectKBest(f_classif, k=k)))

    # Step 4 — Classifier: make the final prediction.
    # Three interpretable linear models are available, each with a different trade-off:
    #   LogisticRegression — highest average F1, well-calibrated probabilities, slowest
    #   MultinomialNB      — fastest inference, requires non-negative input, simplest
    #   LinearSVC          — strong on high-dimensional sparse text, no probabilities
    if model_type == "logistic":
        model = LogisticRegression(
            C=params.get("C", 1.0),
            solver="saga",
            penalty="l2",
            max_iter=params.get("max_iter", 1000),
            n_jobs=1,
            random_state=random_state,
        )
    elif model_type == "naive_bayes":
        model = MultinomialNB(alpha=params.get("alpha", 1.0))
    elif model_type == "svm":
        penalty = params.get("penalty", "l2")
        # LinearSVC's `dual` parameter selects which formulation of the optimisation
        # problem to solve. The l1 penalty is only implemented in the primal formulation
        # (dual=False). For l2 we let sklearn choose "auto" when StandardScaler is in
        # use (dense data), and default to dual=True otherwise (better for sparse text).
        dual = False if penalty == "l1" else ("auto" if scaler_type == "standard" else True)
        model = LinearSVC(
            C=params.get("C", 1.0),
            penalty=penalty,
            dual=dual,
            max_iter=params.get("max_iter", 1500),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    steps.append(("classifier", model))
    return Pipeline(steps)
