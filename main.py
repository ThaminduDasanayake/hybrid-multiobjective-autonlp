"""
Main Streamlit Application
"""

import sys
import time
from pathlib import Path

import streamlit as st
from sklearn.exceptions import ConvergenceWarning

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.job_manager import JobManager
from utils.logger import get_logger
from ui import render_header, render_analysis_view, run_automl, render_thesis_defense

import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Page configuration
st.set_page_config(
    page_title="Multi-Objective AutoML for NLP",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = get_logger("streamlit_app")


def main():
    """Main application entry point."""

    # Render header
    render_header()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📖 Project Guide",
            "🚀 Run AutoML",
            "📂 History & Analysis",
            "🎓 Thesis Defense",
        ]
    )

    # --- Tab 1: Project Guide ---
    with tab1:
        st.markdown("## 👋 Welcome to T-AutoNLP")
        st.markdown(
            """
        This system is designed to help you **explore and optimize** NLP pipelines for your specific dataset.
        
        ### 🧬 Search Space (6 Genes)
        We optimize six key components of the NLP pipeline:
        1. **Scaler**: Data normalization (e.g., Standard, MaxAbs, Robust)
        2. **Dimensionality Reduction**: Feature reduction (e.g., PCA, SelectKBest)
        3. **Vectorizer**: Text representation (TF-IDF, Count)
        4. **Model**: The classifier (e.g., Logistic Regression, SVM, LightGBM, SGD)
        5. **N-gram Range**: Unigrams, bigrams, or trigrams (1-1, 1-2, 1-3)
        6. **Max Features**: Vocabulary size limit (5K, 10K, 20K, unlimited)

        ### 🎯 Objectives (3 Comparisons)
        The system finds the best trade-offs between:
        - **F1 Score**: Predictive performance (Higher is better)
        - **Latency**: Inference speed (Lower is better)
        - **Interpretability**: Model explainability (Higher is better)
        """
        )

        st.info("👈 Go to the **'Run AutoML'** tab to start a new experiment.")

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "baseline_results" not in st.session_state:
        st.session_state.baseline_results = None
    if "active_job_id" not in st.session_state:
        st.session_state.active_job_id = None
    # Post-processing tab state
    if "pp_job_id" not in st.session_state:
        st.session_state.pp_job_id = None
    if "pp_inference_proc" not in st.session_state:
        st.session_state.pp_inference_proc = None
    if "pp_baseline_proc" not in st.session_state:
        st.session_state.pp_baseline_proc = None

    job_manager = JobManager()

    # --- Tab 2: Run AutoML ---
    with tab2:
        run_automl(job_manager)

    # --- Tab 3: History & Analysis ---
    with tab3:
        st.header("📂 History & Analysis")

        # Job Selector
        all_jobs = job_manager.list_jobs()

        # Filter out active runs
        viewable_jobs = {
            job_id: job_data
            for job_id, job_data in all_jobs.items()
            if job_data.get("status") in ["completed", "failed"]
        }

        if not viewable_jobs:
            if all_jobs:
                st.info(
                    "Job(s) currently running. No completed previous runs available yet."
                )
            else:
                st.info("No past jobs found.")
        else:
            selected_job_id = st.selectbox(
                "Select a previous run:",
                options=list(viewable_jobs.keys()),
                format_func=lambda x: f"{viewable_jobs[x].get('job_id', x)} ({viewable_jobs[x].get('status')}) - {time.ctime(viewable_jobs[x].get('start_time', 0))}",
            )

            if selected_job_id:
                # Load button (optional, can just load on select)
                # But loading might be heavy so maybe button?
                # "Re-Visualization" task said "When a job is selected, load..." implying auto.

                # Check status
                job_status = all_jobs[selected_job_id]
                if job_status["status"] != "completed":
                    if job_status.get("status") == "failed":
                        st.error(
                            f"❌ Job failed: {job_status.get('error', 'Unknown error')}"
                        )
                    else:
                        st.warning(
                            f"This job is {job_status['status']}. Results may not be available."
                        )

                    # Offer resume if a checkpoint exists (population was saved)
                    checkpoint_path = (
                        project_root
                        / "jobs"
                        / selected_job_id
                        / "checkpoints"
                        / "checkpoint.pkl"
                    )
                    if checkpoint_path.exists():
                        completed_gen = job_status.get("current_generation", "?")
                        total_gen = job_status.get("total_generations", "?")
                        st.info(
                            f"💾 Checkpoint found (generation {completed_gen}/{total_gen} saved). "
                            "You can resume this job without losing progress."
                        )
                        if st.button(
                            "▶ Resume Job from Checkpoint",
                            key=f"resume_{selected_job_id}",
                            type="primary",
                        ):
                            if job_manager.resume_job(selected_job_id):
                                st.session_state.active_job_id = selected_job_id
                                st.success(
                                    "Job resumed — switch to the 'Run AutoML' tab to monitor progress."
                                )
                                st.rerun()
                            else:
                                st.error(
                                    "Failed to resume job. Check that config.json exists in the job directory."
                                )

                # Load results
                loaded_results = job_manager.get_result(selected_job_id)

                if loaded_results:
                    st.success(f"Loaded results for job {selected_job_id}")

                    # Display job configuration
                    config_path = project_root / "jobs" / selected_job_id / "config.json"
                    if config_path.exists():
                        import json as _json
                        with open(config_path) as _f:
                            job_cfg = _json.load(_f)
                        st.subheader("⚙️ Configuration")
                        cfg_cols = st.columns(5)
                        cfg_labels = ["Dataset", "Samples", "Pop Size", "Generations", "BO Calls"]
                        cfg_values = [
                            job_cfg.get("dataset_name", "—"),
                            job_cfg.get("max_samples", "—"),
                            job_cfg.get("population_size", "—"),
                            job_cfg.get("n_generations", "—"),
                            job_cfg.get("bo_calls", "—"),
                        ]
                        for col, lbl, val in zip(cfg_cols, cfg_labels, cfg_values):
                            col.metric(lbl, val)

                    render_analysis_view(
                        loaded_results, key_prefix=f"history_{selected_job_id}"
                    )
                else:
                    if job_status["status"] == "completed":
                        st.error("Results file not found for this job.")

    # --- Tab 4: Thesis Defense ---
    with tab4:
        render_thesis_defense()


if __name__ == "__main__":
    main()
