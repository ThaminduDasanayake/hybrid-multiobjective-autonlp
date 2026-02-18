"""
Main Streamlit Application
"""

import uuid

import streamlit as st
import sys
import time
import os
import json
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import DataLoader, to_python_type, to_json_safe
from utils.job_manager import JobManager
from utils.logger import get_logger
from automl import HybridAutoML
from experiments import ParetoAnalyzer, RandomSearchBaseline
from ui import (
    render_header,
    # render_config, # Imported locally in main()
    render_footer,
    render_results_summary,
    # render_knee_point_info,
    render_decision_support_panel,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_search_history,
    show_solutions_table,
    # compare_with_baseline
)

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
    tab1, tab2, tab3 = st.tabs(
        ["📖 Project Guide", "🚀 Run AutoML", "📂 History & Analysis"]
    )

    # --- Tab 1: Project Guide ---
    with tab1:
        st.markdown("## 👋 Welcome to T-AutoNLP")
        st.markdown(
            """
        This system is designed to help you **explore and optimize** NLP pipelines for your specific dataset.
        
        ### 🧬 Search Space (4 Genes)
        We optimize four key components of the NLP pipeline:
        1. **Scaler**: Data normalization (e.g., Standard, MinMax)
        2. **Dimensionality Reduction**: Feature reduction (e.g., PCA, SelectKBest)
        3. **Vectorizer**: Text representation (e.g., TF-IDF, Count)
        4. **Model**: The classifier (e.g., Logistic Regression, SVM, LightGBM)

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

    job_manager = JobManager()

    # --- Tab 2: Run AutoML ---
    with tab2:
        # Render configuration (formerly sidebar)
        # Note: We import render_config instead of render_sidebar now
        from ui.layout import (
            render_config,
        )  # Import here to avoid circular dependency issues if any, or just use the renamed function

        config = render_config()

        # Quick Demo Mode
        st.markdown("---")
        st.subheader("⚡ Quick Demo Mode")
        quick_demo = st.checkbox(
            "Enable Quick Demo (3 minutes)",
            value=False,
            help="Fast configuration for quick testing: 1K samples, 10 pop, 5 gen, 10 BO",
        )

        if quick_demo:
            config["max_samples"] = 1000
            config["population_size"] = 10
            config["n_generations"] = 5
            config["bo_calls"] = 10
            st.success("✓ Quick demo mode enabled!")
            st.write(f"Est. runtime: ~3-5 minutes")

        # Check for active job status
        active_job = None
        if st.session_state.active_job_id:
            active_job = job_manager.get_status(st.session_state.active_job_id)
            if active_job and active_job["status"] in ["completed", "failed"]:
                # If job just finished, load results
                if (
                    active_job["status"] == "completed"
                    and st.session_state.results is None
                ):
                    st.success("🎉 Job completed successfully!")
                    results = job_manager.get_result(st.session_state.active_job_id)
                    if results:
                        st.session_state.results = results
                elif active_job["status"] == "failed":
                    st.error(
                        f"❌ Job failed: {active_job.get('error', 'Unknown error')}"
                    )

        # Run button
        col1, col2 = st.columns([1, 4])
        with col1:
            # Disable run button if job is running
            is_running = (active_job is not None) and (
                active_job.get("status") == "running"
            )
            run_button = st.button(
                "🚀 Run AutoML", type="primary", width="stretch", disabled=is_running
            )
        with col2:
            if is_running:
                st.info(f"🔄 Job {st.session_state.active_job_id} is running...")
            elif st.session_state.results is not None:
                st.success("✅ Results available - scroll down to view")

        if run_button:
            # Clear previous results
            st.session_state.results = None
            st.session_state.baseline_results = None
            st.session_state.active_job_id = None

            # Create new job
            with st.spinner("Initializing job..."):
                job_id = job_manager.create_job(config)
                st.session_state.active_job_id = job_id
                st.rerun()

        # Job Progress Monitoring
        if active_job and active_job["status"] == "created":
            st.info("⏳ Job created, waiting for worker...")
            time.sleep(1)
            st.rerun()

        elif active_job and active_job["status"] == "running":
            st.subheader("🔄 Optimization in Progress")

            progress = active_job.get("progress", 0)
            curr_gen = active_job.get("current_generation", 0)
            total_gen = active_job.get("total_generations", config["n_generations"])
            message = active_job.get("message", "")

            st.progress(progress / 100)
            st.text(f"Status: {message}")
            st.text(f"Generation: {curr_gen}/{total_gen}")

            if st.button("Stop Monitoring (Job continues in background)"):
                st.session_state.active_job_id = None
                st.rerun()

            # Auto-refresh
            time.sleep(2)
            st.rerun()

        # Log Viewer
        with st.expander("📝 View Logs"):
            if st.session_state.active_job_id:
                log_file = f"logs/run_{st.session_state.active_job_id}.log"
                if os.path.exists(log_file):
                    st.caption(f"Showing logs for job: {st.session_state.active_job_id}")
                    with open(log_file, "r") as f:
                        # Read last 50 lines
                        lines = f.readlines()[-50:]
                        st.code("".join(lines))
                else:
                    st.info("Waiting for logs...")
            else:
                st.info("Start a job to view logs.")

        # Display results if available
        if st.session_state.results is not None:
            results = st.session_state.results
            render_analysis_view(results, key_prefix="live_run")

    # --- Tab 3: History & Analysis ---
    with tab3:
        st.header("📂 History & Analysis")

        # Job Selector
        all_jobs = job_manager.list_jobs()

        if not all_jobs:
            st.info("No past jobs found.")
        else:
            job_options = {
                job_id: f"{job_data.get('start_time', 'Unknown')} - {job_data.get('status', 'Unknown')}"
                for job_id, job_data in all_jobs.items()
            }

            # Add formatted date? For now just raw time or job ID
            # Let's make it prettier if possible
            # But keys must be unique.

            selected_job_id = st.selectbox(
                "Select a previous run:",
                options=list(all_jobs.keys()),
                format_func=lambda x: f"{all_jobs[x].get('job_id', x)[:8]}... ({all_jobs[x].get('status')}) - {time.ctime(all_jobs[x].get('start_time', 0))}",
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

                # Load results
                loaded_results = job_manager.get_result(selected_job_id)

                if loaded_results:
                    st.success(f"Loaded results for job {selected_job_id[:8]}...")
                    render_analysis_view(
                        loaded_results, key_prefix=f"history_{selected_job_id}"
                    )
                else:
                    if job_status["status"] == "completed":
                        st.error("Results file not found for this job.")


def render_analysis_view(results, key_prefix="default"):
    """Result visualization helper to be reused in both tabs."""

    st.markdown("---")
    st.header("📈 Results Analysis")

    # Compute metrics and recompute Pareto front from all solutions for consistency
    analyzer = ParetoAnalyzer()
    metrics = analyzer.compute_metrics(results["all_solutions"])
    # Ensure pareto_front is present or recompute
    if "pareto_front" not in results:
        pareto_front = analyzer.get_pareto_front(results["all_solutions"])
    else:
        pareto_front = results["pareto_front"]

    results_for_ui = dict(results)
    results_for_ui["pareto_front"] = pareto_front

    # Results summary
    render_results_summary(results_for_ui, metrics)

    st.markdown("---")

    # Decision support panel
    render_decision_support_panel(results_for_ui, metrics)

    st.markdown("---")

    # Visualizations
    st.subheader("📊 Pareto Front Visualization")

    t1, t2, t3 = st.tabs(["3D View", "2D Projections", "Search History"])

    with t1:
        st.markdown("### 3D Pareto Front")
        plot_pareto_front_3d(
            results["all_solutions"],
            results_for_ui["pareto_front"],
            metrics["knee_point"],
            key=f"{key_prefix}_3d",
        )

    with t2:
        st.markdown("### 2D Projections")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### F1 Score vs Latency")
            plot_pareto_front_2d(
                results["all_solutions"],
                results_for_ui["pareto_front"],
                metrics["knee_point"],
                x_metric="f1_score",
                y_metric="latency",
                key=f"{key_prefix}_2d_f1_lat",
            )

        with c2:
            st.markdown("#### F1 Score vs Interpretability")
            plot_pareto_front_2d(
                results["all_solutions"],
                results_for_ui["pareto_front"],
                metrics["knee_point"],
                x_metric="f1_score",
                y_metric="interpretability",
                key=f"{key_prefix}_2d_f1_int",
            )

        with c3:
            st.markdown("#### Latency vs Interpretability")
            plot_pareto_front_2d(
                results["all_solutions"],
                results_for_ui["pareto_front"],
                metrics["knee_point"],
                x_metric="latency",
                y_metric="interpretability",
                key=f"{key_prefix}_2d_lat_int",
            )

    with t3:
        st.markdown("### Search Progress Over Generations")
        plot_search_history(results["search_history"], key=f"{key_prefix}_history")

    st.markdown("---")

    # Solutions table
    st.subheader("📋 All Solutions")

    show_solutions_table(
        results["all_solutions"], results_for_ui["pareto_front"], metrics["knee_point"]
    )

    # Download results
    st.markdown("---")
    st.subheader("💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        # Download all solutions
        results_json = json.dumps(to_python_type(results["all_solutions"]), indent=2)
        st.download_button(
            label="📥 Download All Solutions (JSON)",
            data=results_json,
            file_name="automl_solutions.json",
            mime="application/json",
            key=f"download_all_{key_prefix}",
        )

    with col2:
        # Download Pareto front
        pareto_json = json.dumps(to_json_safe(results_for_ui["pareto_front"]), indent=2)
        st.download_button(
            label="📥 Download Pareto Front (JSON)",
            data=pareto_json,
            file_name="pareto_front.json",
            mime="application/json",
            key=f"download_pareto_{key_prefix}",
        )

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
