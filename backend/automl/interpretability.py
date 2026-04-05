"""
THE TRANSLATOR — interpretability.py
======================================
This module answers the question: "How easy is this pipeline for a human to understand?"

The score is rule-based and structural — it does not depend on the data or the model's
predictions. Instead, it evaluates the *architecture* of the pipeline using four
components grounded in interpretability research (Lipton, 2016; Rudin, 2019):

  1. Model complexity (30%)     — simpler models are more interpretable
  2. Feature transparency (20%) — fewer, clearer features are easier to reason about
  3. Preprocessing complexity (20%) — fewer transformations preserve the link to raw text
  4. Hyperparameter simplicity (30%) — regularized, constrained models generalize better
                                       and their decisions are easier to explain

The result is a float in [0, 1], where 1.0 means "maximally interpretable". This score
becomes the third objective in the NSGA-II fitness tuple alongside F1 and latency.
"""

from typing import Any, Dict

# --- Module-level scoring constants ---
# Defined here so they are allocated once at import time and reused across every
# call to interpretability_score() (which runs for every pipeline the GA evaluates).

_MODEL_SCORES: dict = {"logistic": 1.0, "naive_bayes": 0.9, "svm": 0.7}
_VECTORIZER_SCORES: dict = {"count": 1.0, "tfidf": 0.85}
_FEATURE_COUNT_SCORES: dict = {"5000": 1.0, "10000": 0.8, "None": 0.35}
_SCALER_SCORES: dict = {None: 1.0, "maxabs": 0.9, "standard": 0.9, "robust": 0.85}
_DIM_RED_SCORES: dict = {None: 1.0, "select_k_best": 0.8, "pca": 0.4}
# N-gram simplicity weights: unigrams (1-1) are most interpretable; anything else
# defaults to 0.4 (covers the theoretical "1-3" range, not in the current gene pool).
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
    """Return a structural interpretability score in [0, 1] (Lipton, 2016; Rudin, 2019).

    Weighted sum of four structural components: model complexity (30%),
    feature space transparency (20%), preprocessing complexity (20%),
    and hyperparameter simplicity (30%).
    """
    score = 0.0

    # Component 1 — Model complexity (30% of total score).
    # Logistic Regression is the gold standard for interpretable NLP: its weights
    # directly map to "how much does this word push the prediction toward each class".
    # NaiveBayes is similarly transparent. SVM's decision boundary is harder to
    # explain to a non-technical stakeholder, hence the lower score.
    score += 0.3 * _MODEL_SCORES.get(model, 0.5)

    # Component 2 — Feature space transparency (20% of total score).
    # Count Vectorizer produces raw word counts — directly human-readable.
    # TF-IDF applies a logarithmic re-weighting that makes the features less
    # immediately intuitive, hence a slightly lower score.
    # Fewer max_features means a tighter, more focused vocabulary — easier to
    # inspect. An unbounded vocabulary (None) is harder to audit.
    feature_transparency = _VECTORIZER_SCORES.get(vectorizer, 0.5) * _FEATURE_COUNT_SCORES.get(str(max_features), 0.5)
    score += 0.2 * feature_transparency

    # Component 3 — Preprocessing complexity (20% of total score).
    # Each transformation step between raw text and the model is a layer of abstraction
    # that makes it harder to trace *why* the model made a decision. No scaler and no
    # dim reduction is the most transparent. TruncatedSVD (pca) is penalised most heavily
    # because it projects into latent dimensions with no direct semantic meaning.
    preprocessing_score = (
        _SCALER_SCORES.get(scaler, 0.5) + _DIM_RED_SCORES.get(dim_reduction, 0.5)
    ) / 2
    score += 0.2 * preprocessing_score

    # Component 4 — Hyperparameter simplicity (30% of total score).
    # This component captures how the *settings* of the chosen model affect interpretability.
    simplicity = 0.0

    # N-gram range (40% of hyperparameter simplicity):
    # Unigrams (1-1) are the most interpretable — each feature is a single word with
    # an obvious meaning. Bigrams (1-2) add "word pairs" which are harder to reason
    # about at scale.
    simplicity += 0.4 * _NGRAM_SIMPLICITY.get(str(ngram_range), 0.4)

    # Model regularisation (60% of hyperparameter simplicity):
    # For Logistic Regression, the C parameter controls regularisation strength.
    # A smaller C means stronger regularisation, which pushes most feature weights
    # toward zero — producing a simpler, more interpretable model. We use a
    # decreasing function 1/(1+C) so that low C (strong regularisation) scores
    # near 1.0 and high C (weak regularisation) scores near 0.0.
    if model == "logistic":
        C = params.get("C", 1.0)
        simplicity += 0.6 * (1.0 / (1.0 + C))
    else:
        simplicity += 0.6 * 0.5

    score += 0.3 * simplicity

    return max(0.0, min(1.0, score))
