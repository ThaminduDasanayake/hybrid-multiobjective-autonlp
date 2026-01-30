import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# import new modules
from modules import data, experiments, visuals

# page config
st.set_page_config(layout="wide", page_title="T-AutoNLP")
st.title("T-AutoNLP")
st.caption("Human-centered, multi-objective AutoML for NLP.")

# sidebar config
st.sidebar.header("AutoML Configuration")

# Tabs to separate New Run vs History
tab_new, tab_history = st.sidebar.tabs(["New Run", "Load History"])

with tab_new:
    dataset_name = st.selectbox("Dataset", ["IMDb", "AG News", "Banking77"])
    ngen = st.slider("Generations", 2, 10, 4)
    pop_size = st.slider("Population size", 4, 20, 6)
    run_btn = st.button("Run T-AutoNLP", type="primary")

with tab_history:
    st.write("### Past Experiments")
    all_runs = experiments.get_run_ids()
    selected_run_id = st.selectbox("Select Run ID", all_runs)
    load_run_btn = st.button("Load Selected Run")

    # Optional: Checkbox to show the aggregate table of ALL runs
    show_full_history = st.checkbox("Show aggregate history table")

# --- DATA LOADING (Global) ---
# Default to selection, but override if a historical run is loaded
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = dataset_name

# data loading
X_train, y_train = data.load_selected_dataset(dataset_name)
data.display_dataset_info(dataset_name, X_train, y_train)

# --- EXECUTION LOGIC (NEW RUN) ---
if run_btn:
    # ... (Same logic as before for running new experiments) ...
    # Ensure we update the session state dataset_name to the selected one
    st.session_state["dataset_name"] = dataset_name

    with st.spinner(f"Running optimization on {dataset_name}..."):
        # ... (Run optimization logic) ...
        results = experiments.run_automl_process(X_train, y_train, ngen, pop_size)
        run_id = experiments.save_run(results, dataset_name)
        st.session_state["results"] = results
        # ...

# --- EXECUTION LOGIC (LOAD HISTORY) ---
if load_run_btn and selected_run_id:
    with st.spinner(f"Loading {selected_run_id}..."):
        loaded_results = experiments.load_single_run(selected_run_id)
        if loaded_results:
            st.session_state["results"] = loaded_results
            st.session_state["last_run_id"] = selected_run_id

            # Update dataset name based on what's in the file
            # (Assuming the CSV has a 'dataset' column as defined in save_run)
            if "dataset" in loaded_results[0]:
                st.session_state["dataset_name"] = loaded_results[0]["dataset"]

            st.success(f"Loaded run: {selected_run_id}")
            st.rerun()  # Refresh to update main view


# execution logic
if run_btn:
    with st.spinner(f"Running optimization on {dataset_name}..."):
        try:
            results = experiments.run_automl_process(X_train, y_train, ngen, pop_size)
            run_id = experiments.save_run(results, dataset_name)

            # Update Session State
            st.session_state["results"] = results
            st.session_state["last_run_id"] = run_id
            st.session_state["dataset_name"] = dataset_name
            st.success(f"Finished! Run ID: {run_id}")

        except Exception as e:
            st.error(f"Optimization failed: {e}")

# results visualization
if "results" in st.session_state:
    results = st.session_state["results"]
    df = pd.DataFrame(results)

    st.markdown(f"## Results for: {st.session_state['dataset_name']}")

    # 1. pareto charts
    visuals.plot_pareto_fronts(df)

    st.divider()

    # 2. interactive inspection
    visuals.inspect_solution(df, X_train, y_train)

else:
    st.info("👈 Configure AutoML in the sidebar to start.")

# history section
if show_full_history:
    st.divider()
    st.subheader("Aggregate Search History")
    history_df = experiments.load_history()

    if not history_df.empty:
        st.dataframe(history_df.sort_values("accuracy", ascending=False), width=True)
    else:
        st.info("No previous history found.")
