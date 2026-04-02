#!/usr/bin/env python3
"""TPOT baseline benchmark for T-AutoNLP comparison.

Runs TPOTClassifier as a single-objective (F1 weighted) AutoML baseline on the
same text classification datasets used by T-AutoNLP.  Results can be compared
against the multi-objective GA+BO system to demonstrate the value of
Pareto-optimal trade-offs.

Usage:
    python scripts/tpot_baseline.py --dataset ag_news --max-samples 2000 --time-mins 22
    python scripts/tpot_baseline.py --dataset imdb --generations 5 --population-size 20 --output results/tpot_imdb.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add backend to sys.path so we can reuse the DataLoader
_BACKEND_ROOT = str((Path(__file__).parent.parent / "backend").resolve())
sys.path.insert(0, _BACKEND_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TPOT baseline on a text classification dataset",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. ag_news, imdb, banking77, 20newsgroups)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max training samples (default: 2000)",
    )
    parser.add_argument(
        "--time-mins",
        type=int,
        default=None,
        help="TPOT max time budget in minutes (takes precedence over --generations)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="TPOT generations when --time-mins is not set (default: 5)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="TPOT population size (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (optional)",
    )
    args = parser.parse_args()

    # Lazy imports so --help is fast
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from tpot import TPOTClassifier

    from utils.data_loader import DataLoader

    # ── Load dataset ──────────────────────────────────────────────────────
    data_dir = str(Path(_BACKEND_ROOT) / "data")
    data_loader = DataLoader(cache_dir=data_dir)
    X_texts, y = data_loader.load_dataset(
        args.dataset, subset="train", max_samples=args.max_samples,
    )
    print(f"Loaded {len(X_texts)} samples for '{args.dataset}'")

    # ── Vectorize text (TPOT works on numeric features) ───────────────────
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_texts)

    # ── Train/test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # ── Configure & run TPOT ──────────────────────────────────────────────
    tpot_params: dict = {
        "scoring": "f1_weighted",
        "random_state": args.seed,
        "verbosity": 2,
        "n_jobs": -1,
        "population_size": args.population_size,
    }
    if args.time_mins is not None:
        tpot_params["max_time_mins"] = args.time_mins
        print(f"TPOT budget: {args.time_mins} minutes")
    else:
        tpot_params["generations"] = args.generations
        print(f"TPOT budget: {args.generations} generations x {args.population_size} pop")

    tpot = TPOTClassifier(**tpot_params)

    start = time.time()
    tpot.fit(X_train, y_train)
    elapsed = time.time() - start

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred = tpot.predict(X_test)
    test_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # ── Inference latency (100-sample probe) ──────────────────────────────
    # Use exactly 100 samples so the per-sample figure is directly comparable
    # to T-AutoNLP's latency measurement.
    latency_samples = X_test[:100]
    latency_start = time.perf_counter()
    tpot.predict(latency_samples)
    latency_end = time.perf_counter()
    latency_ms_per_sample = ((latency_end - latency_start) / 100) * 1000

    # ── Pipeline step breakdown ───────────────────────────────────────────
    pipeline_steps = [
        {"step": name, "estimator": type(estimator).__name__}
        for name, estimator in tpot.fitted_pipeline_.steps
    ]

    print(f"\n{'=' * 50}")
    print(f"TPOT Results for '{args.dataset}':")
    print(f"  Best F1 (weighted):            {test_f1:.4f}")
    print(f"  Runtime:                       {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"  Per-sample inference latency:  {latency_ms_per_sample:.4f} ms")
    print(f"  Pipeline steps ({len(pipeline_steps)}):")
    for s in pipeline_steps:
        print(f"    [{s['step']}] {s['estimator']}")
    print(f"{'=' * 50}")

    # ── Export JSON results ───────────────────────────────────────────────
    if args.output:
        result = {
            "baseline": "tpot",
            "dataset": args.dataset,
            "best_f1_weighted": test_f1,
            "inference_latency_ms_per_sample": latency_ms_per_sample,
            "runtime_seconds": elapsed,
            "pipeline_str": str(tpot.fitted_pipeline_),
            "pipeline_steps": pipeline_steps,
            "config": {
                "max_samples": args.max_samples,
                "population_size": args.population_size,
                "generations": args.generations if args.time_mins is None else None,
                "time_mins": args.time_mins,
                "seed": args.seed,
                "tfidf_max_features": 10000,
                "tfidf_ngram_range": [1, 2],
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # ── Export best pipeline as Python code ────────────────────────────────
    pipeline_file = f"tpot_best_pipeline_{args.dataset}.py"
    tpot.export(pipeline_file)
    print(f"Pipeline exported to {pipeline_file}")


if __name__ == "__main__":
    main()
