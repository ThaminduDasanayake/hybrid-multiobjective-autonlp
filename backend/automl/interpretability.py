"""
Computes a structural interpretability score [0, 1] for evaluated pipelines
"""

from typing import Any, Dict

# Pre-defined heuristic scores for pipeline components
_MODEL_SCORES: dict = {"logistic": 1.0, "naive_bayes": 0.9, "svm": 0.7}
_VECTORIZER_SCORES: dict = {"count": 1.0, "tfidf": 0.85}
_FEATURE_COUNT_SCORES: dict = {"5000": 1.0, "10000": 0.8, "None": 0.35}
_SCALER_SCORES: dict = {None: 1.0, "maxabs": 0.9, "standard": 0.9, "robust": 0.85}
_DIM_RED_SCORES: dict = {None: 1.0, "select_k_best": 0.8, "pca": 0.4}
_NGRAM_SIMPLICITY: dict = {"1-1": 1.0, "1-2": 0.7}


def interpretability_score(
    scaler: str,
    dim_reduction: str,
    vectorizer: str,
    model: str,
    ngram_range: str,
    max_features: Any,
    params: Dict[str, Any],
) -> float:
    """Calculate the overall interpretability score based on structural and parameter complexity."""
    score = 0.0

    # Component 1: Model complexity (30% of total score).
    score += 0.3 * _MODEL_SCORES.get(model, 0.5)

    # Component 2: Feature space transparency (20% of total score).
    feature_transparency = _VECTORIZER_SCORES.get(
        vectorizer, 0.5
    ) * _FEATURE_COUNT_SCORES.get(str(max_features), 0.5)
    score += 0.2 * feature_transparency

    # Component 3: Preprocessing complexity (20% of total score).
    preprocessing_score = (
        _SCALER_SCORES.get(scaler, 0.5) + _DIM_RED_SCORES.get(dim_reduction, 0.5)
    ) / 2
    score += 0.2 * preprocessing_score

    # Component 4: Hyperparameter simplicity (30% of total score).
    simplicity = 0.0

    # N-gram simplicity accounts for 40% of the hyperparameter weight
    simplicity += 0.4 * _NGRAM_SIMPLICITY.get(str(ngram_range), 0.4)

    # Regularization strength accounts for 60% of the hyperparameter weight
    if model == "logistic":
        C = params.get("C", 1.0)
        simplicity += 0.6 * (
            1.0 / (1.0 + C)
        )  # Higher regularization (lower C) increases interpretability
    else:
        simplicity += 0.6 * 0.5

    score += 0.3 * simplicity

    return max(0.0, min(1.0, score))
