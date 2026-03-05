import json

import numpy as np
import streamlit as st

from experiments import ParetoAnalyzer
from ui.visuals import (
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    show_solutions_table,
    plot_search_history,
)

from utils import clean_params
from utils import to_python_type, to_json_safe
from utils.formatting import format_time


def render_header():
    """Render the main header and system disclaimer."""
    st.title("🔬 Multi-Objective AutoML for NLP")

    st.markdown(
        """
    ### Human-Centered Pipeline Exploration System

    **Important:** This system explores and compares alternative NLP pipelines under multiple constraints. 
    It does **not** recommend a single best model.
    """
    )

    with st.expander("ℹ️ What is this system?"):
        st.markdown(
            """
        This is a **diagnostic, decision-support AutoML framework** that:

        ✅ **Explores** multiple pipeline configurations  
        ✅ **Optimizes** three objectives simultaneously:
        - Predictive Performance (F1 Score)
        - Computational Efficiency (Inference Latency)
        - Intrinsic Interpretability

        ✅ **Returns** a Pareto front of non-dominated solutions  
        ✅ **Supports** visual inspection and trade-off analysis  

        #### What This System Is NOT:

        ❌ A black-box AutoML recommender  
        ❌ A deployment tool  
        ❌ A deep learning NAS system  
        ❌ A system that picks "the best" model for you

        #### Research Alignment:

        This system aligns with recent research in:
        - 3D Inference Scaling
        - Latency-aware Neural Architecture Search
        - Multi-objective Bayesian Optimization
        - Human-centered AutoML
        """
        )

    with st.expander("🎯 How to use this system"):
        st.markdown(
            """
        1. **Select a dataset** from the sidebar
        2. **Configure search parameters** (population size, generations)
        3. **Run the exploration** and wait for results
        4. **Analyze the Pareto front** to understand trade-offs
        5. **Select a solution** based on your priorities:
           - Use the **knee point** for a balanced solution
           - Choose solutions that prioritize specific objectives
           - Compare different pipeline architectures

        #### Understanding the Results:

        - **Pareto Front**: Solutions where no other solution is better in all objectives
        - **Knee Point**: The most balanced solution (closest to ideal in all objectives)
        - **Trade-offs**: Improving one objective often degrades another
        """
        )


def render_config():
    """
    Render optimized configuration controls using a column-based layout.
    """
    st.header("⚙️ Configuration")

    # --- Dataset Section ---
    with st.container():
        st.subheader("Dataset Selection")

        # Use columns for dataset and sample size to shorten the slider length
        col_ds1, col_ds2 = st.columns([1, 1], gap="medium")

        with col_ds1:
            dataset_name = st.selectbox(
                "Choose a dataset:",
                ["20newsgroups", "imdb", "ag_news", "banking77"],
                index=0,
                help="Select the text classification dataset to use",
            )

            dataset_info = {
                "20newsgroups": "Multi-class (20 categories) news article classification",
                "imdb": "Binary sentiment analysis (positive/negative reviews)",
                "ag_news": "News categorization (4 categories)",
                "banking77": "Intent classification (77 banking intents)",
            }
            st.caption(f"ℹ️ {dataset_info[dataset_name]}")

        with col_ds2:
            max_samples = st.slider(
                "Max samples (Prototyping):",
                min_value=500,
                max_value=10000,
                value=2000,
                step=500,
                help="Limit dataset size for faster experimentation",
            )

    st.divider()

    # --- Search Configuration Section ---
    st.subheader("Search Strategy")

    # Use a 3-column layout to make sliders significantly shorter and more manageable
    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        population_size = st.select_slider(
            "Population Size:",
            options=list(range(10, 55, 5)),
            value=20,
            help="Number of individuals in each generation",
        )

    with col2:
        n_generations = st.select_slider(
            "Generations:",
            options=list(range(5, 35, 5)),
            value=10,
            help="Number of evolutionary generations",
        )

    with col3:
        bo_calls = st.select_slider(
            "BO Iterations:",
            options=list(range(10, 35, 5)),
            value=15,
            help="Bayesian optimization iterations for hyperparameter tuning",
        )

    # --- Summary Logic ---
    total_evals = population_size + (population_size * (n_generations - 1))
    st.info(
        f"🧬 **Search Budget:** Approximately **{total_evals}** unique pipelines will be explored."
    )

    return {
        "dataset_name": dataset_name,
        "max_samples": max_samples,
        "population_size": population_size,
        "n_generations": n_generations,
        "bo_calls": bo_calls,
    }


def render_footer():
    """Render footer with additional information."""
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666;'>
        <p>Multi-Objective AutoML for NLP | Built with Streamlit</p>
        <p>For research and educational purposes</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_results_summary(results: dict, metrics: dict):
    """
    Render summary of AutoML results.

    Args:
        results: Results from HybridAutoML
        metrics: Computed metrics from evaluation
    """
    st.subheader("📊 Results Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Evaluations", results["stats"]["total_evaluations"])

    with col2:
        st.metric("Pareto Front Size", metrics["pareto_front_size"])

    with col3:
        st.metric("Dominated Solutions", metrics["dominated_solutions"])

    with col4:
        best_f1 = metrics["f1_score"]["max"]
        st.metric("Best F1 Score", f"{best_f1:.4f}")

    with col5:
        hv = metrics.get("hypervolume", 0.0)
        st.metric(
            "Hypervolume",
            f"{hv:.4f}",
            help="Volume of objective space dominated by the Pareto front. Higher = better quality front.",
        )

    # Objective statistics
    st.markdown("### Objective Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**F1 Score**")
        st.write(f"Min: {metrics['f1_score']['min']:.4f}")
        st.write(f"Max: {metrics['f1_score']['max']:.4f}")
        st.write(f"Mean: {metrics['f1_score']['mean']:.4f}")
        st.write(f"Std: {metrics['f1_score']['std']:.4f}")

    with col2:
        st.markdown("**Latency (ms)**")
        st.write(f"Min: {metrics['latency']['min'] * 1000:.2f}")
        st.write(f"Max: {metrics['latency']['max'] * 1000:.2f}")
        st.write(f"Mean: {metrics['latency']['mean'] * 1000:.2f}")
        st.write(f"Std: {metrics['latency']['std'] * 1000:.2f}")

    with col3:
        st.markdown("**Interpretability**")
        st.write(f"Min: {metrics['interpretability']['min']:.4f}")
        st.write(f"Max: {metrics['interpretability']['max']:.4f}")
        st.write(f"Mean: {metrics['interpretability']['mean']:.4f}")
        st.write(f"Std: {metrics['interpretability']['std']:.4f}")

    # Time statistics
    if "time_stats" in results["stats"]:
        st.markdown("### ⏱️ Time Statistics")
        time_stats = results["stats"]["time_stats"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Runtime", format_time(time_stats["total_runtime"]))

        with col2:
            st.metric(
                "Optimization (BO) Time",
                format_time(time_stats["total_optimization_time"]),
            )

        with col3:
            total_ga_time = (
                sum(time_stats["generation_times"])
                if time_stats["generation_times"]
                else 0
            )
            avg_gen_time = (
                np.mean(time_stats["generation_times"])
                if time_stats["generation_times"]
                else 0
            )
            st.metric("Total GA Time", format_time(total_ga_time))
            st.caption(f"Avg: {avg_gen_time:.2f}s / gen")

        with col4:
            st.metric("Generations Run", f"{time_stats['total_generations_run']}")

        with st.expander("View Generation Details"):
            st.write("Time per generation:")
            for i, duration in enumerate(time_stats["generation_times"]):
                st.write(f"Generation {i+1}: {format_time(duration)}")


def render_decision_support_panel(results: dict, metrics: dict):
    st.subheader("🎯 Decision Support: Choose Your Solution")

    st.markdown(
        """
    Based on your **priorities**, select from these recommended solutions:
    """
    )

    pareto = results["pareto_front"]
    knee_point = metrics["knee_point"]

    if not pareto:
        st.warning("No solutions in Pareto front")
        return

    # Find specific solutions
    best_f1_sol = max(pareto, key=lambda x: x["f1_score"])
    best_latency_sol = min(pareto, key=lambda x: x["latency"])
    best_interp_sol = max(pareto, key=lambda x: x["interpretability"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 High Accuracy Solution")
        st.markdown("**Choose this if**: Accuracy is your top priority")
        st.markdown(
            f"**Pipeline**: {best_f1_sol['vectorizer']} + {best_f1_sol['model']}"
        )
        st.metric("F1 Score", f"{best_f1_sol['f1_score']:.4f}")
        st.metric("Latency", f"{best_f1_sol['latency'] * 1000:.2f} ms")
        st.metric("Interpretability", f"{best_f1_sol['interpretability']:.4f}")

        with st.expander("View Hyperparameters"):
            st.json(clean_params(best_f1_sol["params"]))
            # st.json(best_f1_sol['params'])

    with col2:
        st.markdown("### ⚡ High Speed Solution")
        st.markdown("**Choose this if**: Fast inference is critical")
        st.markdown(
            f"**Pipeline**: {best_latency_sol['vectorizer']} + {best_latency_sol['model']}"
        )
        st.metric("F1 Score", f"{best_latency_sol['f1_score']:.4f}")
        st.metric("Latency", f"{best_latency_sol['latency'] * 1000:.2f} ms")
        st.metric("Interpretability", f"{best_latency_sol['interpretability']:.4f}")

        with st.expander("View Hyperparameters"):
            st.json(clean_params(best_latency_sol["params"]))

    with col3:
        st.markdown("### 🔍 High Interpretability Solution")
        st.markdown("**Choose this if**: Explainability is required")
        st.markdown(
            f"**Pipeline**: {best_interp_sol['vectorizer']} + {best_interp_sol['model']}"
        )
        st.metric("F1 Score", f"{best_interp_sol['f1_score']:.4f}")
        st.metric("Latency", f"{best_interp_sol['latency'] * 1000:.2f} ms")
        st.metric("Interpretability", f"{best_interp_sol['interpretability']:.4f}")

        with st.expander("View Hyperparameters"):
            st.json(clean_params(best_interp_sol["params"]))

    # Knee point (balanced solution)
    if knee_point:
        st.markdown("---")
        st.markdown("### ⚖️ Balanced Solution (Knee Point)")
        st.info(
            "**Recommended if**: You value all three objectives equally and want a balanced compromise"
        )
        st.markdown(
            """
            The **knee point** represents the most balanced solution across all three objectives.
            It offers a good compromise when you don't want to prioritize any single objective.
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Pipeline Configuration**")
            st.write(f"Vectorizer: `{knee_point['vectorizer']}`")
            st.write(f"Model: `{knee_point['model']}`")

        with col2:
            st.markdown("**Performance Metrics**")
            st.write(f"F1 Score: **{knee_point['f1_score']:.4f}**")
            st.write(f"Latency: **{knee_point['latency'] * 1000:.2f} ms**")
            st.write(f"Interpretability: **{knee_point['interpretability']:.4f}**")

        with st.expander("View Hyperparameters"):
            st.json(clean_params(knee_point["params"]))


def render_analysis_view(results, key_prefix="default"):
    """Result visualization helper to be reused in both tabs."""

    st.markdown("---")

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
