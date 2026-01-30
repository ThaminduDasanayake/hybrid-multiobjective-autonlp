import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules import data, experiments, visuals

# page config
st.set_page_config(layout="wide", page_title="T-AutoNLP")
st.title("T-AutoNLP")
st.caption("Human-centered, multi-objective AutoML for NLP.")
st.info(
    "This system explores and compares alternative NLP pipelines under multiple constraints. "
    "It does not recommend a single best model."
)

# sidebar config
st.sidebar.header("AutoML Configuration")

# Tabs to separate New Run vs History
tab_new, tab_history = st.sidebar.tabs(["New Run", "Load History"])

# Initialize variables
run_btn = False
load_run_btn = False
show_full_history = False
dataset_name = "IMDb"  # Default
selected_run_id = None
dev_mode = False

with tab_new:
    dataset_name = st.selectbox("Dataset", ["IMDb", "AG News", "Banking77"])

    # Dev Mode Checkbox
    dev_mode = st.checkbox("Development Mode (Fast Run)", value=False, help="Reduces data size & generations for quick testing.")

    if dev_mode:
        ngen = st.slider("Generations", 2, 5, 2)
        pop_size = st.slider("Population size", 4, 10, 4)
        st.caption("⚠️ Dev Mode: Using 500 samples & 3 BO calls.")
    else:
        ngen = st.slider("Generations", 2, 10, 4)
        pop_size = st.slider("Population size", 4, 20, 6)

    run_btn = st.button("Run T-AutoNLP", type="primary")

with tab_history:
    st.write("### Past Experiments")
    all_runs = experiments.get_run_ids()

    if all_runs:
        selected_run_id = st.selectbox("Select Run ID", all_runs)
        load_run_btn = st.button("Load Selected Run")
    else:
        st.info("No past experiments found. Run a new experiment first.")

    # Optional: Checkbox to show the aggregate table of ALL runs
    show_full_history = st.checkbox("Show aggregate history table")

# --- DATA LOADING (for new runs) ---
# Only load dataset if we're running a new experiment or if no dataset is loaded
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = dataset_name

# Load current dataset for sidebar info and potential new runs
current_dataset = st.session_state.get("dataset_name", dataset_name)

# Determine sample size based on dev mode
# If dev_mode is checked, use 500 samples, otherwise None (full)
n_samples = 500 if dev_mode else None
n_bo_calls = 3 if dev_mode else 15

X_train, y_train = data.load_selected_dataset(current_dataset, n_samples)
data.display_dataset_info(current_dataset, X_train, y_train)

# Store in session state for use in visualizations
if "X_train" not in st.session_state or "y_train" not in st.session_state:
    st.session_state["X_train"] = X_train
    st.session_state["y_train"] = y_train

# --- EXECUTION LOGIC (NEW RUN) ---
if run_btn:
    st.session_state["dataset_name"] = dataset_name

    # Reload dataset for the selected option
    X_train, y_train = data.load_selected_dataset(dataset_name, n_samples)
    st.session_state["X_train"] = X_train
    st.session_state["y_train"] = y_train

    with st.spinner(f"Running optimization on {dataset_name}..."):
        try:
            results = experiments.run_automl_process(X_train, y_train, ngen, pop_size, n_bo_calls)
            run_id = experiments.save_run(results, dataset_name)

            # Update Session State
            st.session_state["results"] = results
            st.session_state["last_run_id"] = run_id

            st.success(f"✅ Finished! Run ID: {run_id}")

        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            st.exception(e)

# --- EXECUTION LOGIC (LOAD HISTORY) ---
if load_run_btn and selected_run_id:
    with st.spinner(f"Loading {selected_run_id}..."):
        loaded_results = experiments.load_single_run(selected_run_id)

        if loaded_results:
            st.session_state["results"] = loaded_results
            st.session_state["last_run_id"] = selected_run_id

            # Update dataset name based on what's in the file
            if "dataset" in loaded_results[0]:
                loaded_dataset = loaded_results[0]["dataset"]
                st.session_state["dataset_name"] = loaded_dataset

                # Reload the dataset for visualization purposes
                X_train, y_train = data.load_selected_dataset(loaded_dataset, n_samples=None)
                st.session_state["X_train"] = X_train
                st.session_state["y_train"] = y_train

            st.success(f"✅ Loaded run: {selected_run_id}")
            st.rerun()  # Refresh to update main view
        else:
            st.error(f"Failed to load {selected_run_id}")


# results visualization
if "results" in st.session_state:
    results = st.session_state["results"]

    # Check if results is not empty
    if not results or len(results) == 0:
        st.warning("⚠️ No results to display.")
        st.stop()

    df = pd.DataFrame(results)

    st.markdown(f"## Results for: {st.session_state['dataset_name']}")

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Solutions", len(results))
    col2.metric("SVC Pipelines", df[df["classifier"] == "SVC"].shape[0])
    col3.metric(
        "LogisticRegression", df[df["classifier"] == "LogisticRegression"].shape[0]
    )
    col4.metric(
        "DecisionTree", df[df["classifier"] == "DecisionTreeClassifier"].shape[0]
    )

    # 1. pareto charts
    visuals.plot_pareto_fronts(df)

    st.divider()

    # 2. Interactive inspection
    # Get X_train and y_train from session state
    X_train = st.session_state.get("X_train", [])
    y_train = st.session_state.get("y_train", [])

    # 2. interactive inspection
    visuals.inspect_solution(df, X_train, y_train)

else:
    st.info(
        "👈 Configure AutoML in the sidebar to start, or load a past experiment from history."
    )

# history section
if show_full_history:
    st.divider()
    st.subheader("Aggregate Search History")
    history_df = experiments.load_history()

    if not history_df.empty:
        st.dataframe(
            history_df.sort_values("accuracy", ascending=False),
            use_container_width=True,
        )

        # Diversity chart
        st.markdown("### Exploration Diversity")
        if "classifier" in history_df.columns and len(history_df) > 0:
            st.bar_chart(
                history_df["classifier"].value_counts(),
                height=250,
            )
    else:
        st.info("No previous history found.")
