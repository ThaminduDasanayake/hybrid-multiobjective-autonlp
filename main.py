"""
Main Streamlit Application
"""

import streamlit as st
import sys
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import DataLoader, to_python_type, to_json_safe
from automl import HybridAutoML
from experiments import ParetoAnalyzer, RandomSearchBaseline
from ui import (
    render_header,
    render_sidebar,
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
    initial_sidebar_state="expanded"
)


def main():
    """Main application entry point."""

    # Render header
    render_header()

    # Render sidebar and get configuration
    config = render_sidebar()

    # Quick Demo Mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Demo Mode")
    quick_demo = st.sidebar.checkbox(
        "Enable Quick Demo (3 minutes)",
        value=False,
        help="Fast configuration for quick testing: 1K samples, 10 pop, 5 gen, 10 BO"
    )

    if quick_demo:
        config['max_samples'] = 1000
        config['population_size'] = 10
        config['n_generations'] = 5
        config['bo_calls'] = 10
        st.sidebar.success("✓ Quick demo mode enabled!")
        st.sidebar.write(f"Est. runtime: ~3-5 minutes")

    # Runtime warning
    if not quick_demo:
        est_minutes = (config['population_size'] * config['n_generations'] *
                       config['bo_calls'] * config['max_samples']) / 50000
        # st.sidebar.warning(
        #     f"⚠️ Estimated runtime: ~{est_minutes:.0f}-{est_minutes * 2:.0f} minutes. "
        #     f"Consider enabling Quick Demo Mode for faster exploration."
        # )

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'baseline_results' not in st.session_state:
        st.session_state.baseline_results = None

    # Run button
    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("🚀 Run AutoML", type="primary", width="stretch")
    with col2:
        if st.session_state.results is not None:
            st.success("✅ Results available - scroll down to view")

    if run_button:
        # Clear previous results
        st.session_state.results = None
        st.session_state.baseline_results = None

        # Load data
        st.info(f"📥 Loading {config['dataset_name']} dataset...")
        data_loader = DataLoader(cache_dir="./data")

        try:
            X_train, y_train = data_loader.load_dataset(
                config['dataset_name'],
                subset='train',
                max_samples=config['max_samples']
            )

            dataset_info = data_loader.get_dataset_info(config['dataset_name'])
            st.success(f"✅ Loaded {len(X_train)} samples from {config['dataset_name']}")

            with st.expander("Dataset Information"):
                st.json(dataset_info)

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            return

        # Run HybridAutoML
        st.info("🔬 Running Hybrid AutoML (GA + BO)...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Initialize and run AutoML
            automl = HybridAutoML(
                X_train=X_train,
                y_train=y_train,
                population_size=config['population_size'],
                n_generations=config['n_generations'],
                bo_calls=config['bo_calls'],
                random_state=42,
                early_stopping=True
            )

            status_text.text("Running evolutionary optimization...")
            results = automl.run()

            progress_bar.progress(100)
            status_text.text("✅ AutoML completed!")

            # Store results
            st.session_state.results = results

            st.success(f"✅ AutoML completed! Evaluated {results['stats']['total_evaluations']} configurations.")

            # Show objective ranges
            st.info(
                f"📊 Objective ranges found:\n"
                f"- F1: {results['stats']['objective_ranges']['f1_score']['min']:.4f} - "
                f"{results['stats']['objective_ranges']['f1_score']['max']:.4f}\n"
                f"- Latency: {results['stats']['objective_ranges']['latency']['min']:.4f}s - "
                f"{results['stats']['objective_ranges']['latency']['max']:.4f}s\n"
                f"- Interpretability: {results['stats']['objective_ranges']['interpretability']['min']:.4f} - "
                f"{results['stats']['objective_ranges']['interpretability']['max']:.4f}"
            )

        except Exception as e:
            st.error(f"❌ Error running AutoML: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

        # Run baseline if requested
        # if config['run_baseline']:
        #     st.info("📊 Running Random Search Baseline...")
        #
        #     try:
        #         baseline = RandomSearchBaseline(
        #             n_iterations=config['baseline_iterations'],
        #             cv=3,
        #             random_state=42
        #         )
        #
        #         baseline_results = baseline.run(X_train, y_train)
        #         st.session_state.baseline_results = baseline_results
        #
        #         st.success("✅ Baseline completed!")
        #
        #     except Exception as e:
        #         st.error(f"❌ Error running baseline: {e}")

    # Display results if available
    if st.session_state.results is not None:
        results = st.session_state.results

        st.markdown("---")
        st.header("📈 Results Analysis")

        # Compute metrics and recompute Pareto front from all solutions for consistency
        analyzer = ParetoAnalyzer()
        metrics = analyzer.compute_metrics(results['all_solutions'])
        pareto_front = analyzer.get_pareto_front(results['all_solutions'])
        results_for_ui = dict(results)
        results_for_ui["pareto_front"] = pareto_front

        # Results summary
        render_results_summary(results_for_ui, metrics)

        st.markdown("---")

        # Decision support panel
        render_decision_support_panel(results_for_ui, metrics)

        st.markdown("---")

        # Knee point information
        # render_knee_point_info(metrics['knee_point'])

        st.markdown("---")

        # Visualizations
        st.subheader("📊 Pareto Front Visualization")

        tab1, tab2, tab3 = st.tabs(["3D View", "2D Projections", "Search History"])

        with tab1:
            st.markdown("### 3D Pareto Front")
            plot_pareto_front_3d(
                results['all_solutions'],
                results_for_ui['pareto_front'],
                metrics['knee_point']
            )

        with tab2:
            st.markdown("### 2D Projections")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### F1 Score vs Latency")
                plot_pareto_front_2d(
                    results['all_solutions'],
                    results_for_ui['pareto_front'],
                    metrics['knee_point'],
                    x_metric='f1_score',
                    y_metric='latency'
                )

            with col2:
                st.markdown("#### F1 Score vs Interpretability")
                plot_pareto_front_2d(
                    results['all_solutions'],
                    results_for_ui['pareto_front'],
                    metrics['knee_point'],
                    x_metric='f1_score',
                    y_metric='interpretability'
                )

            st.markdown("#### Latency vs Interpretability")
            plot_pareto_front_2d(
                results['all_solutions'],
                results_for_ui['pareto_front'],
                metrics['knee_point'],
                x_metric='latency',
                y_metric='interpretability'
            )

        with tab3:
            st.markdown("### Search Progress Over Generations")
            plot_search_history(results['search_history'])

        st.markdown("---")

        # Solutions table
        st.subheader("📋 All Solutions")

        show_solutions_table(
            results['all_solutions'],
            results_for_ui['pareto_front'],
            metrics['knee_point']
        )



        # Baseline comparison
        # if st.session_state.baseline_results is not None:
        #     st.markdown("---")
        #     compare_with_baseline(results_for_ui, st.session_state.baseline_results)

        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")

        import json

        col1, col2 = st.columns(2)

        with col1:
            # Download all solutions
            results_json = json.dumps(to_python_type(results["all_solutions"]), indent=2)
            st.download_button(
                label="📥 Download All Solutions (JSON)",
                data=results_json,
                file_name="automl_solutions.json",
                mime="application/json"
            )

        with col2:
            # Download Pareto front
            pareto_json = json.dumps(to_json_safe(results_for_ui['pareto_front']), indent=2)
            st.download_button(
                label="📥 Download Pareto Front (JSON)",
                data=pareto_json,
                file_name="pareto_front.json",
                mime="application/json"
            )

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
