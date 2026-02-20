import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path to import utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_loader import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def main():
    print("Loading data...")
    # Seed for reproducibility
    np.random.seed(42)

    data_loader = DataLoader(cache_dir="./data")
    # Load 5000 samples
    X, y = data_loader.load_dataset("20newsgroups", subset="train", max_samples=5000)
    print(f"Data loaded: {len(X)} samples")

    # Define Baselines
    pipelines = [
        (
            "Baseline A (Simple)",
            Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())]),
        ),
        (
            "Baseline B (Standard)",
            Pipeline(
                [
                    ("vect", TfidfVectorizer(max_features=10000)),
                    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
                ]
            ),
        ),
        (
            "Baseline C (Complex)",
            Pipeline(
                [
                    ("vect", TfidfVectorizer()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
                ]
            ),
        ),
    ]

    print("\nRunning Baselines...")
    print("-" * 80)
    print(f"{'Model Name':<25} | {'F1 Score':<10} | {'Time (s)':<10}")
    print("-" * 80)

    results = []

    automl_f1 = 0.686
    automl_time_seconds = 19569
    automl_hours = automl_time_seconds / 3600

    for name, pipeline in pipelines:
        start_time = time.time()

        # Evaluate
        # n_jobs=-1 uses all processors
        scores = cross_val_score(pipeline, X, y, cv=3, scoring="f1_weighted", n_jobs=-1)
        mean_f1 = scores.mean()

        end_time = time.time()
        duration = end_time - start_time

        print(f"{name:<25} | {mean_f1:.4f}     | {duration:.2f}")
        results.append({"name": name, "f1": mean_f1, "time": duration})

    print("-" * 80)

    # AutoML Comparison
    print("\nAutoML Comparison (Provided Results)")
    print("-" * 30)

    print(f"AutoML Best F1 : {automl_f1}")
    print(f"AutoML Time    : {automl_time_seconds}s ({automl_hours:.2f} hours)")
    print("-" * 80)
    print(f"{'Comparison vs AutoML':<30} | {'F1 Gain':<10} | {'Gain/Hour':<10}")
    print("-" * 80)

    for res in results:
        baseline_f1 = res["f1"]
        baseline_time_hours = res["time"] / 3600

        f1_gain = automl_f1 - baseline_f1

        # Time cost difference (AutoML took X more hours than baseline)
        time_cost_hours = automl_hours - baseline_time_hours

        if time_cost_hours > 0:
            gain_per_hour = f1_gain / time_cost_hours
        else:
            gain_per_hour = 0.0  # Should not happen given the vast difference

        print(f"vs {res['name']:<27} | {f1_gain:+.4f}     | {gain_per_hour:+.4f}")

    print("-" * 80)
    print("Gain/Hour = (AutoML F1 - Baseline F1) / (AutoML Hours - Baseline Hours)")


if __name__ == "__main__":
    main()
