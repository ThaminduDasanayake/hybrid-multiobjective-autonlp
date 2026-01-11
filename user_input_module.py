import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# -----------------------------------------------------------------------------
# AVAILABLE OPTIONS
# -----------------------------------------------------------------------------
AVAILABLE_COMPONENTS = {
    "vectorizer": {
        "TfidfVectorizer": TfidfVectorizer,
        "CountVectorizer": CountVectorizer,
    },
    "classifier": {
        "LogisticRegression": LogisticRegression,
        "SVC": SVC,
    },
}

DEFAULT_OBJECTIVES = ["accuracy", "interpretability"]


# -----------------------------------------------------------------------------
# DATA UPLOAD
# -----------------------------------------------------------------------------
def upload_dataset():
    print("\n=== Dataset Upload ===")
    print("Provide path to CSV file (must contain 'text' and 'label' columns).")
    print("Leave blank to use default sample dataset (20 Newsgroups).")
    file_path = input("Enter file path: ").strip()

    if file_path == "":
        print("Using default dataset: 20 Newsgroups (sci.med, sci.space).")
        from sklearn.datasets import fetch_20newsgroups

        data = fetch_20newsgroups(
            subset="train",
            categories=["sci.med", "sci.space"],
            shuffle=True,
            random_state=42,
        )
        return data.data, data.target
    else:
        if not os.path.exists(file_path):
            print("File not found. Please check path.")
            return upload_dataset()
        df = pd.read_csv(file_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns.")
        return df["text"].tolist(), df["label"].tolist()


# -----------------------------------------------------------------------------
# PIPELINE DEFINITION
# -----------------------------------------------------------------------------
def select_pipeline_components():
    print("\n=== Pipeline Definition ===")
    print("Available vectorizers:")
    for i, name in enumerate(AVAILABLE_COMPONENTS["vectorizer"].keys()):
        print(f"  {i+1}. {name}")
    v_choice = input(
        "Select vectorizer [1 or Enter for default TfidfVectorizer]: "
    ).strip()
    v_choice = (
        "TfidfVectorizer"
        if v_choice == ""
        else list(AVAILABLE_COMPONENTS["vectorizer"].keys())[int(v_choice) - 1]
    )

    print("\nAvailable classifiers:")
    for i, name in enumerate(AVAILABLE_COMPONENTS["classifier"].keys()):
        print(f"  {i+1}. {name}")
    c_choice = input(
        "Select classifier [1 or Enter for default LogisticRegression]: "
    ).strip()
    c_choice = (
        "LogisticRegression"
        if c_choice == ""
        else list(AVAILABLE_COMPONENTS["classifier"].keys())[int(c_choice) - 1]
    )

    return (
        AVAILABLE_COMPONENTS["vectorizer"][v_choice],
        AVAILABLE_COMPONENTS["classifier"][c_choice],
    )


# -----------------------------------------------------------------------------
# OBJECTIVE SELECTION
# -----------------------------------------------------------------------------
def select_objectives():
    print("\n=== Objective Configuration ===")
    print("Available objectives: accuracy, interpretability, efficiency")
    print(
        "Enter comma-separated list (or press Enter for defaults: accuracy, interpretability)"
    )
    user_input = input("Objectives: ").strip()
    if user_input == "":
        return DEFAULT_OBJECTIVES
    else:
        chosen = [x.strip().lower() for x in user_input.split(",")]
        return [
            o for o in chosen if o in ["accuracy", "interpretability", "efficiency"]
        ]


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
def collect_user_inputs():
    X, y = upload_dataset()
    vectorizer_class, classifier_class = select_pipeline_components()
    objectives = select_objectives()
    print("\nConfiguration Summary:")
    print(f"  - Dataset size: {len(X)} samples")
    print(f"  - Vectorizer: {vectorizer_class.__name__}")
    print(f"  - Classifier: {classifier_class.__name__}")
    print(f"  - Objectives: {objectives}")
    return X, y, vectorizer_class, classifier_class, objectives
