"""
Thesis Defense tab: comparison tables for ablation study results.

Scans ``results/ablations/`` for JSON files produced by
``experiments/run_ablations.py`` and builds two summary tables:

* **Table 1** - Single-Objective vs. Multi-Objective
* **Table 2** - Ablation Studies (GA+BO, GA-only, 2-Objective)

Provides a single job selector to pick the master run (like the History &
Analysis tab) and one-click buttons to launch ablation runs.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Default directory where run_ablations.py saves results
ABLATION_DIR = Path("results/ablations")

# Locate project root and ablation script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ABLATION_SCRIPT = _PROJECT_ROOT / "experiments" / "run_ablations.py"

# ── Session-state keys ────────────────────────────────────────────────
_SS_PROC = "ablation_bg_proc"
_SS_LABEL = "ablation_bg_label"


# ══════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════════════════════


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning *None* on any failure."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _metric_or_pending(
    data: Optional[Dict[str, Any]], key: str, fmt: str = ".4f"
) -> str:
    """Extract a metric value as a formatted string, or '⏳ Run pending'."""
    if data is None:
        return "⏳ Run pending"
    value = data.get("metrics", {}).get(key)
    if value is None:
        return "—"
    return f"{value:{fmt}}"


def _runtime_or_pending(data: Optional[Dict[str, Any]]) -> str:
    """Extract ``runtime_seconds`` (root-level key) as a formatted string."""
    if data is None:
        return "⏳ Run pending"
    value = data.get("runtime_seconds")
    if value is None:
        return "—"
    return f"{value:.1f} s"


# ── Job discovery (mirrors History & Analysis tab pattern) ────────────

def _discover_completed_jobs() -> Dict[str, Dict[str, Any]]:
    """
    Return ``{job_id: status_dict}`` for all completed jobs, sorted by
    start_time descending.  Each status dict also carries the job config.
    """
    jobs: Dict[str, Dict[str, Any]] = {}
    jobs_dir = Path("jobs")
    if not jobs_dir.exists():
        return jobs

    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
        status = _load_json(job_dir / "status.json")
        if not status or status.get("status") != "completed":
            continue
        if not (job_dir / "result.json").exists():
            continue
        # Attach config for display
        cfg = _load_json(job_dir / "config.json") or {}
        status["_config"] = cfg
        status["_job_dir"] = str(job_dir)
        jobs[job_dir.name] = status

    return dict(
        sorted(jobs.items(), key=lambda x: x[1].get("start_time", 0), reverse=True)
    )


def _load_job_as_master(job_id: str, job_status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load a completed job's result, using pre-stored metrics when available."""
    job_dir = Path(job_status["_job_dir"])
    raw = _load_json(job_dir / "result.json")
    if not raw:
        return None

    cfg = job_status.get("_config", {})
    dataset = cfg.get("dataset_name", "20newsgroups")

    # Prefer pre-stored metrics (written by worker.py since the update)
    stored_metrics = raw.get("metrics")
    if stored_metrics:
        return {
            "mode": "multi_3d",
            "dataset": dataset,
            "runtime_seconds": raw.get("runtime_seconds"),
            "config": {
                "disable_bo": False,
                "pop_size": cfg.get("population_size", 20),
                "generations": cfg.get("n_generations", 10),
                "bo_calls": cfg.get("bo_calls", 15),
                "max_samples": cfg.get("max_samples", 2000),
                "seed": 42,
            },
            "metrics": stored_metrics,
        }

    # Fallback: compute on-the-fly for older jobs without stored metrics
    solutions = raw.get("all_solutions", [])
    if not solutions:
        return None

    try:
        from experiments.evaluation import ParetoAnalyzer
        metrics = ParetoAnalyzer.compute_metrics(solutions)
    except Exception:
        return None

    return {
        "mode": "multi_3d",
        "dataset": dataset,
        "runtime_seconds": raw.get("runtime_seconds"),
        "config": {
            "disable_bo": False,
            "pop_size": cfg.get("population_size", 20),
            "generations": cfg.get("n_generations", 10),
            "bo_calls": cfg.get("bo_calls", 15),
            "max_samples": cfg.get("max_samples", 2000),
            "seed": 42,
        },
        "metrics": {
            "best_f1": metrics["f1_score"]["max"],
            "pareto_front_size": metrics["pareto_front_size"],
            "hypervolume": metrics.get("hypervolume", 0.0),
            "best_latency_ms": metrics["latency"]["min"] * 1000,
            "best_interpretability": metrics["interpretability"]["max"],
            "total_solutions": metrics["total_solutions"],
        },
    }


def _load_ablations_for_dataset(
    dataset: str,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    """
    Load ablation results for *dataset*.
    Returns ``(single_f1_data, multi_2d_data, ga_only_data)``.
    """
    single_f1 = _load_json(ABLATION_DIR / f"ablation_single_f1_{dataset}.json")
    multi_2d = _load_json(ABLATION_DIR / f"ablation_multi_2d_{dataset}.json")

    # GA-only = multi_3d with disable_bo=True
    ga_only = None
    candidate = _load_json(ABLATION_DIR / f"ablation_multi_3d_{dataset}.json")
    if candidate and candidate.get("config", {}).get("disable_bo", False):
        ga_only = candidate

    return single_f1, multi_2d, ga_only


# ══════════════════════════════════════════════════════════════════════
#  Config extraction + subprocess helpers
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG = {
    "dataset": "20newsgroups",
    "max_samples": 2000,
    "pop_size": 20,
    "generations": 10,
    "bo_calls": 15,
    "seed": 42,
}


def _get_ablation_config(
    main_run_data: Optional[Dict[str, Any]], dataset: str
) -> Dict[str, Any]:
    """Extract config from the master run for apples-to-apples ablation."""
    if main_run_data is None:
        return {**_DEFAULT_CONFIG, "dataset": dataset}

    cfg = main_run_data.get("config", {})
    return {
        "dataset": main_run_data.get("dataset", cfg.get("dataset", dataset)),
        "max_samples": cfg.get("max_samples", _DEFAULT_CONFIG["max_samples"]),
        "pop_size": cfg.get("pop_size", _DEFAULT_CONFIG["pop_size"]),
        "generations": cfg.get("generations", _DEFAULT_CONFIG["generations"]),
        "bo_calls": cfg.get("bo_calls", _DEFAULT_CONFIG["bo_calls"]),
        "seed": cfg.get("seed", _DEFAULT_CONFIG["seed"]),
    }


def _build_cmd(mode: str, cfg: Dict[str, Any], disable_bo: bool = False) -> list:
    cmd = [
        sys.executable, str(_ABLATION_SCRIPT),
        "--mode", mode,
        "--dataset", str(cfg["dataset"]),
        "--max-samples", str(cfg["max_samples"]),
        "--pop-size", str(cfg["pop_size"]),
        "--generations", str(cfg["generations"]),
        "--bo-calls", str(cfg["bo_calls"]),
        "--seed", str(cfg["seed"]),
    ]
    if disable_bo:
        cmd.append("--disable-bo")
    return cmd


def _launch_ablation(label: str, cmd: list) -> None:
    proc = subprocess.Popen(
        cmd, cwd=str(_PROJECT_ROOT), start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    st.session_state[_SS_PROC] = proc
    st.session_state[_SS_LABEL] = label


def _is_ablation_running() -> bool:
    proc = st.session_state.get(_SS_PROC)
    if proc is None:
        return False
    if proc.poll() is None:
        return True
    st.session_state[_SS_PROC] = None
    st.session_state[_SS_LABEL] = None
    return False


# ══════════════════════════════════════════════════════════════════════
#  Main render function
# ══════════════════════════════════════════════════════════════════════


def render_thesis_defense():
    """Render the Thesis Defense comparison tables tab."""
    st.header("🎓 Thesis Defense: Baseline & Ablation Comparisons")

    st.markdown(
        """
    These tables summarise the results of your ablation experiments, proving
    the contribution of each component (multi-objective optimisation, Bayesian
    optimisation) to the final system quality.
    """
    )

    # ── Job Selector (single dropdown, like History & Analysis) ───────
    completed_jobs = _discover_completed_jobs()

    main_run_data: Optional[Dict[str, Any]] = None
    dataset = "20newsgroups"  # fallback

    if not completed_jobs:
        st.info("No completed runs found. Run a job first, then come back.")
    else:
        selected_job_id = st.selectbox(
            "Select a previous run:",
            options=list(completed_jobs.keys()),
            format_func=lambda x: (
                f"{completed_jobs[x].get('job_id', x)} "
                f"({completed_jobs[x].get('status')}) - "
                f"{time.ctime(completed_jobs[x].get('start_time', 0))}"
            ),
            key="td_job_selector",
        )

        if selected_job_id:
            job_status = completed_jobs[selected_job_id]
            main_run_data = _load_job_as_master(selected_job_id, job_status)
            if main_run_data:
                dataset = main_run_data.get("dataset", "20newsgroups")
                st.success(f"Loaded results for job **{selected_job_id}**")
            else:
                st.error(f"Could not load results for job {selected_job_id}.")

    # Display config of selected run
    if main_run_data:
        st.subheader("⚙️ Configuration")
        cfg = main_run_data.get("config", {})
        cfg_cols = st.columns(6)
        labels_cfg = ["Dataset", "Samples", "Pop Size", "Generations", "BO Calls", "Seed"]
        values_cfg = [
            main_run_data.get("dataset", "—"),
            cfg.get("max_samples", "—"),
            cfg.get("pop_size", "—"),
            cfg.get("generations", "—"),
            cfg.get("bo_calls", "—"),
            cfg.get("seed", "—"),
        ]
        for col, lbl, val in zip(cfg_cols, labels_cfg, values_cfg):
            col.metric(lbl, val)

    # Load ablation results for the selected dataset
    single_f1_data, multi_2d_data, ga_only_data = _load_ablations_for_dataset(dataset)

    # ═══════════════════════════════════════════════════════════════════
    #  Execution Buttons
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🚀 Run Ablation Experiments")

    currently_running = _is_ablation_running()
    ablation_cfg = _get_ablation_config(main_run_data, dataset)

    # Status banner
    if currently_running:
        label = st.session_state.get(_SS_LABEL, "ablation")
        st.info(
            f"⏳ **{label}** is running in the background. "
            "Buttons are disabled until it completes."
        )

    # Buttons in a 2×2 grid
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🎯 Run Single-Objective Baseline",
            disabled=currently_running,
            width='stretch',
            key="btn_single_f1",
        ):
            cmd = _build_cmd("single_f1", ablation_cfg)
            _launch_ablation("Single-Objective (F1 Only)", cmd)
            st.rerun()

        if st.button(
            "🔬 Run GA-Only (No BO) Ablation",
            disabled=currently_running,
            width='stretch',
            key="btn_ga_only",
        ):
            cmd = _build_cmd("multi_3d", ablation_cfg, disable_bo=True)
            _launch_ablation("GA-Only (No BO)", cmd)
            st.rerun()

    with col2:
        if st.button(
            "📐 Run 2-Objective (No Interp) Ablation",
            disabled=currently_running,
            width='stretch',
            key="btn_multi_2d",
        ):
            cmd = _build_cmd("multi_2d", ablation_cfg)
            _launch_ablation("2-Objective (No Interp)", cmd)
            st.rerun()

        if st.button(
            "🏆 Run Full 3-Objective Master",
            disabled=currently_running,
            width='stretch',
            key="btn_multi_3d",
        ):
            cmd = _build_cmd("multi_3d", ablation_cfg)
            _launch_ablation("Full 3-Objective Master", cmd)
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════
    #  Table 1: Single-Objective vs. Multi-Objective
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 Table 1: Single-Objective vs. Multi-Objective")
    st.caption(
        "Compares optimising F1 alone against the full 3-objective formulation."
    )

    table1 = pd.DataFrame(
        {
            "Method": [
                "Single-Objective (F1 Only)",
                "Multi-Objective (F1 + Latency + Interp)",
            ],
            "Best F1": [
                _metric_or_pending(single_f1_data, "best_f1"),
                _metric_or_pending(main_run_data, "best_f1"),
            ],
            "Pareto Size": [
                _metric_or_pending(single_f1_data, "pareto_front_size", "d"),
                _metric_or_pending(main_run_data, "pareto_front_size", "d"),
            ],
            "Hypervolume": [
                _metric_or_pending(single_f1_data, "hypervolume"),
                _metric_or_pending(main_run_data, "hypervolume"),
            ],
            "Runtime": [
                _runtime_or_pending(single_f1_data),
                _runtime_or_pending(main_run_data),
            ],
        }
    )
    table1.index = range(1, len(table1) + 1)
    st.dataframe(table1, width='stretch', hide_index=True)

    missing_t1 = []
    if single_f1_data is None:
        missing_t1.append(
            "`python experiments/run_ablations.py --mode single_f1`"
        )
    if main_run_data is None:
        missing_t1.append(
            "`python experiments/run_ablations.py --mode multi_3d`"
        )
    if missing_t1:
        st.info(
            "**Missing data?** Run the following to populate:\n\n"
            + "\n\n".join(f"- {cmd}" for cmd in missing_t1)
        )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    #  Table 2: Ablation Studies
    # ═══════════════════════════════════════════════════════════════════
    st.subheader("🔬 Table 2: Ablation Studies")
    st.caption(
        "Proves the contribution of BO and interpretability to overall quality."
    )

    table2 = pd.DataFrame(
        {
            "Configuration": [
                "Full GA + BO (3-Objective)",
                "GA-Only — Random Hyperparams",
                "2-Objective (No Interpretability)",
            ],
            "Best F1": [
                _metric_or_pending(main_run_data, "best_f1"),
                _metric_or_pending(ga_only_data, "best_f1"),
                _metric_or_pending(multi_2d_data, "best_f1"),
            ],
            "Hypervolume": [
                _metric_or_pending(main_run_data, "hypervolume"),
                _metric_or_pending(ga_only_data, "hypervolume"),
                _metric_or_pending(multi_2d_data, "hypervolume"),
            ],
            "Runtime": [
                _runtime_or_pending(main_run_data),
                _runtime_or_pending(ga_only_data),
                _runtime_or_pending(multi_2d_data),
            ],
        }
    )
    table2.index = range(1, len(table2) + 1)
    st.dataframe(table2, width='stretch', hide_index=True)

    missing_t2 = []
    if main_run_data is None:
        missing_t2.append(
            "`python experiments/run_ablations.py --mode multi_3d`"
        )
    if ga_only_data is None:
        missing_t2.append(
            "`python experiments/run_ablations.py --mode multi_3d --disable-bo`"
        )
    if multi_2d_data is None:
        missing_t2.append(
            "`python experiments/run_ablations.py --mode multi_2d`"
        )
    if missing_t2:
        st.info(
            "**Missing data?** Run the following to populate:\n\n"
            + "\n\n".join(f"- {cmd}" for cmd in missing_t2)
        )

    # ── Auto-refresh while a background ablation is running ───────────
    if currently_running:
        time.sleep(5)
        st.rerun()
