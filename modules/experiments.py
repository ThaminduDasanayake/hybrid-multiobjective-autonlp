import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from automl.hybrid_automl import HybridAutoML
from automl.genetic_algorithm import GENE_POOL
from automl.bayesian_optimization import PARAM_SPACE

EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)


@st.cache_resource
def run_automl_process(X, y, ngen, population_size):
    """Runs the computationally expensive AutoML process."""
    automl = HybridAutoML(
        X=X,
        y=y,
        gene_pool=GENE_POOL,
        param_space=PARAM_SPACE,
    )
    return automl.run(ngen=ngen, population_size=population_size)


def save_run(results, dataset_name):
    """Persists run results to CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{ts}"

    rows = []
    for r in results:
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset_name,
                "vectorizer": r["vectorizer"],
                "classifier": r["classifier"],
                "accuracy": r["accuracy"],
                "efficiency": r["efficiency"],
                "interpretability": r["interpretability"],
                "timestamp": ts,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(EXPERIMENT_DIR / f"{run_id}.csv", index=False)
    return run_id


def load_history():
    """Loads all historical runs from CSVs."""
    files = sorted(EXPERIMENT_DIR.glob("run_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def get_run_ids():
    """Fast scan of available run IDs (filenames) without loading data."""
    if not EXPERIMENT_DIR.exists():
        return []
    # Return filenames without extension, sorted newest first
    return sorted([f.stem for f in EXPERIMENT_DIR.glob("run_*.csv")], reverse=True)


def load_single_run(run_id):
    """Loads a specific run CSV."""
    file_path = EXPERIMENT_DIR / f"{run_id}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        return df.to_dict("records")
    return None
