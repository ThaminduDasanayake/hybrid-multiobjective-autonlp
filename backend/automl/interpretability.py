"""Structural interpretability scoring function, shared by PipelineEvaluator and ablation baselines."""

from typing import Any, Dict


def interpretability_score(
    scaler: str,
    dim_reduction: str,
    vectorizer: str,
    model: str,
    ngram_range: str,
    max_features: Any,
    params: Dict[str, Any],
) -> float:
    """Return a structural interpretability score in [0, 1] (Lipton, 2016; Rudin, 2019).

    Weighted sum of four structural components: model complexity (30%),
    feature space transparency (20%), preprocessing complexity (20%),
    and hyperparameter simplicity (30%).
    """
    score = 0.0

    # Model complexity (30%)
    model_scores = {
        "logistic": 1.0,
        "naive_bayes": 0.9,
        "svm": 0.7,
    }
    score += 0.3 * model_scores.get(model, 0.5)

    # Feature space transparency (20%)
    vectorizer_scores = {
        "count": 1.0,
        "tfidf": 0.85,
    }
    feature_count_scores = {
        "5000": 1.0,
        "10000": 0.8,
        "None": 0.35,
    }
    feature_transparency = vectorizer_scores.get(
        vectorizer, 0.5
    ) * feature_count_scores.get(str(max_features), 0.5)
    score += 0.2 * feature_transparency

    # Preprocessing complexity (Scaler + Dim Reduction) (20%)
    scaler_scores = {None: 1.0, "maxabs": 0.9, "standard": 0.9, "robust": 0.85}
    dim_red_scores = {
        None: 1.0,
        "select_k_best": 0.8,
        "pca": 0.4,
    }
    preprocessing_score = (
        scaler_scores.get(scaler, 0.5) + dim_red_scores.get(dim_reduction, 0.5)
    ) / 2
    score += 0.2 * preprocessing_score

    # Hyperparameter simplicity (30%)
    simplicity = 0.0

    # N-gram range (40% of simplicity)
    ngram_str = str(ngram_range)
    if ngram_str == "1-1":
        simplicity += 0.4 * 1.0
    elif ngram_str == "1-2":
        simplicity += 0.4 * 0.7
    else:  # "1-3"
        simplicity += 0.4 * 0.4

    # Model regularisation (60% of simplicity)
    if model == "logistic":
        C = params.get("C", 1.0)
        simplicity += 0.6 * (1.0 / (1.0 + C))
    else:
        simplicity += 0.6 * 0.5

    score += 0.3 * simplicity

    return max(0.0, min(1.0, score))
