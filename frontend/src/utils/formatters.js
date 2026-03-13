/**
 * Display-name helpers for pipeline gene values.
 *
 * Centralised here so ParetoFront3D hover text and SolutionsTable cells
 * always use the same labels.
 */

export const MODEL_LABELS = {
  logistic: "Logistic Regression",
  naive_bayes: "Naive Bayes",
  svm: "SVM",
  lightgbm: "LightGBM",
  sgd: "SGD Classifier",
};

export const VECTORIZER_LABELS = {
  tfidf: "TF-IDF",
  count: "Count Vec.",
};

export const SCALER_LABELS = {
  maxabs: "MaxAbs",
  robust: "Robust",
  standard: "Standard",
  minmax: "MinMax",
};

export const DIM_LABELS = {
  select_k_best: "SelectKBest",
  pca: "PCA",
};

/** Format a raw gene value into a human-readable label. */
export const fmt = {
  model: (v) => MODEL_LABELS[v] ?? v ?? "—",
  vectorizer: (v) => VECTORIZER_LABELS[v] ?? v ?? "—",
  scaler: (v) => (v ? (SCALER_LABELS[v] ?? v) : "None"),
  dim: (v) => (v ? (DIM_LABELS[v] ?? v) : "None"),
  ngram: (v) => v ?? "—",

  /** F1 Score rounded to 4 decimal places. */
  f1: (v) => Number(v ?? 0).toFixed(4),

  /** Latency in ms, 4 decimal places (raw value is in seconds). */
  latency_ms: (v) => (Number(v ?? 0) * 1000).toFixed(4),

  /** Interpretability score, 3 decimal places. */
  interp: (v) => Number(v ?? 0).toFixed(3),

  /** Format a Unix timestamp as a localised date+time string. */
  date: (ts) =>
    ts
      ? new Date(ts * 1000).toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        })
      : "—",
};
