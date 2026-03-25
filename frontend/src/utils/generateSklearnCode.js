/**
 * generateSklearnCode
 *
 * Accepts a Pareto-optimal solution config object and returns formatted
 * Python code that reconstructs the sklearn Pipeline.
 *
 * Mirrors backend/automl/pipeline_builder.py exactly.
 */

export function generateSklearnCode(config, datasetName = "ag_news") {
  const {
    vectorizer,
    scaler,
    dim_reduction,
    model,
    ngram_range,
    max_features,
    params = {},
  } = config;

  // ── Parse ngram_range ("1-2" → (1, 2)) ────────────────────────────
  let minN = 1,
    maxN = 1;
  if (ngram_range) {
    const parts = String(ngram_range).split("-");
    if (parts.length === 2) {
      minN = parseInt(parts[0], 10);
      maxN = parseInt(parts[1], 10);
    }
  }

  // ── Parse max_features ─────────────────────────────────────────────
  const maxFeat =
    !max_features || String(max_features) === "None" ? "None" : String(parseInt(max_features, 10));

  // ── Collect imports ────────────────────────────────────────────────
  const imports = ["from sklearn.pipeline import Pipeline"];

  // Vectorizer
  if (vectorizer === "tfidf") {
    imports.push("from sklearn.feature_extraction.text import TfidfVectorizer");
  } else if (vectorizer === "count") {
    imports.push("from sklearn.feature_extraction.text import CountVectorizer");
  }

  // Scaler
  if (scaler === "standard") {
    imports.push("from sklearn.preprocessing import StandardScaler");
  } else if (scaler === "maxabs") {
    imports.push("from sklearn.preprocessing import MaxAbsScaler");
  } else if (scaler === "robust") {
    imports.push("from sklearn.preprocessing import RobustScaler");
  }

  // Dimensionality reduction
  if (dim_reduction === "pca") {
    imports.push("from sklearn.decomposition import TruncatedSVD");
  } else if (dim_reduction === "select_k_best") {
    imports.push("from sklearn.feature_selection import SelectKBest, f_classif");
  }

  // Classifier
  if (model === "logistic") {
    imports.push("from sklearn.linear_model import LogisticRegression");
  } else if (model === "naive_bayes") {
    imports.push("from sklearn.naive_bayes import MultinomialNB");
  } else if (model === "svm") {
    imports.push("from sklearn.svm import LinearSVC");
  }

  // ── Build pipeline steps ───────────────────────────────────────────
  const steps = [];

  // 1. Vectorizer
  const minDf = params.min_df ?? 1;
  const maxDf = params.max_df ?? 1.0;
  if (vectorizer === "tfidf") {
    steps.push(
      `    ("vectorizer", TfidfVectorizer(\n` +
        `        ngram_range=(${minN}, ${maxN}),\n` +
        `        min_df=${minDf},\n` +
        `        max_df=${fmt(maxDf)},\n` +
        `        max_features=${maxFeat},\n` +
        `    )),`,
    );
  } else if (vectorizer === "count") {
    steps.push(
      `    ("vectorizer", CountVectorizer(\n` +
        `        ngram_range=(${minN}, ${maxN}),\n` +
        `        min_df=${minDf},\n` +
        `        max_df=${fmt(maxDf)},\n` +
        `        max_features=${maxFeat},\n` +
        `    )),`,
    );
  }

  // 2. Scaler
  if (scaler === "standard") {
    steps.push(`    ("scaler", StandardScaler(with_mean=False)),`);
  } else if (scaler === "maxabs") {
    steps.push(`    ("scaler", MaxAbsScaler()),`);
  } else if (scaler === "robust") {
    steps.push(`    ("scaler", RobustScaler(with_centering=False)),`);
  }

  // 3. Dimensionality reduction
  if (dim_reduction === "pca") {
    const nComp = params.pca_n_components ?? 50;
    steps.push(`    ("dim_reduction", TruncatedSVD(n_components=${nComp})),`);
  } else if (dim_reduction === "select_k_best") {
    const k = params.k_best_k ?? 100;
    steps.push(`    ("dim_reduction", SelectKBest(f_classif, k=${parseInt(k, 10)})),`);
  }

  // 4. Classifier
  if (model === "logistic") {
    const C = params.C ?? 1.0;
    const maxIter = params.max_iter ?? 1000;
    steps.push(
      `    ("classifier", LogisticRegression(\n` +
        `        C=${fmt(C)},\n` +
        `        solver="saga",\n` +
        `        penalty="l2",\n` +
        `        max_iter=${maxIter},\n` +
        `        n_jobs=-1,\n` +
        `        random_state=42,\n` +
        `    )),`,
    );
  } else if (model === "naive_bayes") {
    const alpha = params.alpha ?? 1.0;
    steps.push(`    ("classifier", MultinomialNB(alpha=${fmt(alpha)})),`);
  } else if (model === "svm") {
    const C = params.C ?? 1.0;
    const penalty = params.penalty ?? "l2";
    const dual = penalty === "l1" ? "False" : scaler === "standard" ? '"auto"' : "True";
    const maxIter = params.max_iter ?? 1500;
    steps.push(
      `    ("classifier", LinearSVC(\n` +
        `        C=${fmt(C)},\n` +
        `        penalty="${penalty}",\n` +
        `        dual=${dual},\n` +
        `        max_iter=${maxIter},\n` +
        `        random_state=42,\n` +
        `    )),`,
    );
  }

  // ── Assemble output ────────────────────────────────────────────────
  const pipelineCode = [...imports, "", "", "pipeline = Pipeline([", ...steps, "])"].join("\n");

  // Handle the banking77 mirror edge case safely
  const hfDataset = datasetName === "banking77" ? "mteb/banking77" : datasetName;

  const usageText = `
# ==========================================
# Full Execution & Evaluation Script
# ==========================================
if __name__ == "__main__":
    from datasets import load_dataset
    from sklearn.metrics import classification_report
    import time

    print(f"\\n[1/3] Loading ${hfDataset} dataset...")
    dataset = load_dataset("${hfDataset}")
    
    # Extract text and labels (assuming standard Hugging Face format)
    # Update the column names ('text', 'label') if your dataset uses different keys
    X_train, y_train = dataset['train']['text'], dataset['train']['label']
    X_test, y_test = dataset['test']['text'], dataset['test']['label']

    print(f"[2/3] Training optimal pipeline on {len(X_train)} samples...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"[3/3] Evaluating on {len(X_test)} unseen samples...")
    predictions = pipeline.predict(X_test)
    
    print("\\n" + "="*50)
    print("FINAL MODEL REPORT")
    print("="*50)
    print(f"Training Time: {train_time:.2f} seconds")
    print(classification_report(y_test, predictions, digits=4))
    
    # Optional: Save the model
    # import joblib
    # joblib.dump(pipeline, "optimal_nlp_model.pkl")
    # print("Model saved to optimal_nlp_model.pkl")
`;

  return pipelineCode + "\n" + usageText;
}

/**
 * Format a number for Python: ensure floats always show a decimal point.
 * e.g. 1 → "1.0", 0.5 → "0.5"
 */
function fmt(value) {
  const n = Number(value);
  return Number.isInteger(n) ? `${n}.0` : String(n);
}
