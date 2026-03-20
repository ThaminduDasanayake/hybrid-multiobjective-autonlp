"""
Shared pipeline construction logic for NLP pipelines.

Provides a single source of truth for building sklearn Pipelines from
component configurations, used by both the Bayesian Optimizer and
the hold-out inference script.
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
    """
    Build a sklearn Pipeline from component configuration and hyperparameters.

    This is the single source of truth for pipeline construction, shared by
    the Bayesian Optimizer (during search) and run_final_inference.py (for
    hold-out evaluation).

    Args:
        scaler_type: Type of scaler ('standard', 'maxabs', 'robust', or None)
        dim_reduction_type: Dim reduction ('pca', 'select_k_best', or None)
        vectorizer_type: Vectorizer ('tfidf' or 'count')
        model_type: Classifier ('logistic', 'naive_bayes', 'svm')
        ngram_range: N-gram range string, e.g. '1-2'
        max_features: Max vocabulary size (int, str of int, or 'None')
        params: Hyperparameters dict (C, alpha, max_iter, etc.)
        random_state: Random seed for reproducibility

    Returns:
        Configured sklearn Pipeline
    """
    steps = []

    # 1. Vectorizer
    # Sanitize ngram_range: handle numpy strings and ensure tuple of ints
    if isinstance(ngram_range, (str, np.str_)):
        min_n, max_n = map(int, str(ngram_range).split("-"))
    else:
        min_n, max_n = map(int, ngram_range)

    # Sanitize max_features: handle numpy string/int and "None" string
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
        steps.append(("scaler", MaxAbsScaler()))
    elif scaler_type == "robust":
        steps.append(("scaler", RobustScaler(with_centering=False)))

    # 3. Dimensionality Reduction
    if dim_reduction_type == "pca":
        n_components = params.get("pca_n_components", 50)
        steps.append(("dim_reduction", TruncatedSVD(n_components=n_components)))
    elif dim_reduction_type == "select_k_best":
        k = int(params.get("k_best_k", 100))
        steps.append(("dim_reduction", SelectKBest(f_classif, k=k)))

    # 4. Classifier
    if model_type == "logistic":
        model = LogisticRegression(
            C=params.get("C", 1.0),
            solver="saga",
            penalty="l2",
            max_iter=params.get("max_iter", 1000),
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_type == "naive_bayes":
        model = MultinomialNB(alpha=params.get("alpha", 1.0))
    elif model_type == "svm":
        penalty = params.get("penalty", "l2")
        # l1 penalty requires dual=False in LinearSVC
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
