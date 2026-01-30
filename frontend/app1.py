import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px

from automl.bayesian_optimization import PARAM_SPACE
from automl.genetic_algorithm import GENE_POOL
from automl.hybrid_automl import HybridAutoML

from utils.data_loader import load_imdb, load_ag_news, load_banking77


# Page setup
st.set_page_config(layout="wide")
st.title("T-AutoNLP")
st.caption(
    "Human-centered, multi-objective AutoML for NLP. "
    "Optimizing accuracy, efficiency, and interpretability."
)
st.info(
    "This system explores and compares alternative NLP pipelines under multiple constraints. "
    "It does not recommend a single best model."
)

# Sidebar
st.sidebar.header("AutoML Configuration")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["IMDb", "AG News", "Banking77"],
)

ngen = st.sidebar.slider("Generations", 2, 10, 4)
population_size = st.sidebar.slider("Population size", 4, 20, 6)

run_button = st.sidebar.button("Run T-AutoNLP")


@st.cache_data
def load_selected_dataset(name):
    if name == "IMDb":
        X_train, _, y_train, _ = load_imdb()
    elif name == "AG News":
        X_train, _, y_train, _ = load_ag_news()
    elif name == "Banking77":
        X_train, _, y_train, _ = load_banking77()
    else:
        raise ValueError("Unknown dataset")

    return X_train, y_train


X_train, y_train = load_selected_dataset(dataset_name)

# Display dataset info
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
st.sidebar.info(
    f"**{dataset_name}**\n\n"
    f"Samples: {len(X_train)}\n\n"
    f"Classes: {len(set(y_train))}"
)

# --------------------------------------------------
# Experiment persistence
# --------------------------------------------------
EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)


def save_run(results, dataset_name):
    """Persist evaluated pipelines to CSV"""
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


def load_all_runs():
    """Load all historical runs"""
    files = sorted(EXPERIMENT_DIR.glob("run_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


@st.cache_resource
def run_automl(X, y, ngen, population_size):
    automl = HybridAutoML(
        X=X,
        y=y,
        gene_pool=GENE_POOL,
        param_space=PARAM_SPACE,
    )
    return automl.run(ngen=ngen, population_size=population_size)


# --------------------------------------------------
# Run AutoML
# --------------------------------------------------
if run_button:
    with st.spinner(f"Running multi-objective AutoML on {dataset_name}..."):
        try:
            results = run_automl(X_train, y_train, ngen, population_size)

            run_id = save_run(results, dataset_name=dataset_name)

            st.session_state["results"] = results
            st.session_state["last_run_id"] = run_id

            st.session_state["dataset_name"] = dataset_name
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train

            st.success(
                f"AutoML finished on {dataset_name}. Found {len(results)} Pareto-optimal solutions. "
                f"Run ID: {run_id}"
            )
        except Exception as e:
            st.error(f"❌ AutoML optimization failed: {e}")
            st.exception(e)


# --------------------------------------------------
# Visualization and inspection
# --------------------------------------------------
if "results" in st.session_state:

    results = st.session_state["results"]
    dataset_name = st.session_state.get("dataset_name", "Unknown Dataset")
    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]

    if not results or len(results) == 0:
        st.warning(
            "⚠️ No Pareto-optimal solutions were found. Try running again with different parameters."
        )
        st.stop()

    df = pd.DataFrame(results)

    st.markdown(f"## Results for: {dataset_name}")

    # Display Pareto diversity metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Solutions", len(results))
    col2.metric("SVC Pipelines", df[df["classifier"] == "SVC"].shape[0])
    col3.metric(
        "LogisticRegression", df[df["classifier"] == "LogisticRegression"].shape[0]
    )
    col4.metric(
        "DecisionTree", df[df["classifier"] == "DecisionTreeClassifier"].shape[0]
    )

    st.subheader("2D Trade-off Projections")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Accuracy vs Interpretability")
        fig1 = px.scatter(
            df,
            x="accuracy",
            y="interpretability",
            color="classifier",
            hover_data=["vectorizer", "mode"],
            title="Accuracy-Interpretability Trade-off",
        )
        fig1.update_traces(marker=dict(size=10))
        st.plotly_chart(fig1, width="stretch")

    with col2:
        st.caption("Accuracy vs Efficiency")
        fig2 = px.scatter(
            df,
            x="accuracy",
            y="efficiency",
            color="classifier",
            hover_data=["vectorizer", "mode"],
            title="Accuracy-Efficiency Trade-off",
        )
        fig2.update_traces(marker=dict(size=10))
        st.plotly_chart(fig2, width="stretch")

    with col3:
        st.caption("Efficiency vs Interpretability")
        fig3 = px.scatter(
            df,
            x="efficiency",
            y="interpretability",
            color="classifier",
            hover_data=["vectorizer", "mode"],
            title="Efficiency-Interpretability Trade-off",
        )
        fig3.update_traces(marker=dict(size=10))
        st.plotly_chart(fig3, width="stretch")

    st.subheader("Pareto Front (3D Trade-off Space)")

    fig = px.scatter_3d(
        df,
        x="accuracy",
        y="efficiency",
        z="interpretability",
        color="classifier",
        symbol="vectorizer",
        size_max=12,
        hover_data={
            "accuracy": ":.3f",
            "efficiency": ":.3f",
            "interpretability": ":.3f",
            "vectorizer": True,
            "classifier": True,
            "mode": True,
        },
        title="3D Pareto Front: Multi-Objective Optimization Space",
    )

    fig.update_traces(marker=dict(size=8, opacity=0.9))
    fig.update_layout(
        scene=dict(
            xaxis_title="Accuracy",
            yaxis_title="Efficiency",
            zaxis_title="Interpretability",
        ),
        height=600,
    )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    # --------------------------------------------------
    # Solution inspection
    # --------------------------------------------------
    st.subheader("Inspect Pareto-optimal Solution")

    idx = st.selectbox(
        "Select solution index", df.index, format_func=lambda x: f"Solution #{x+1}"
    )
    selected = df.loc[idx]

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{selected.accuracy:.4f}")
    col2.metric("Efficiency", f"{selected.efficiency:.4f}")
    col3.metric("Interpretability", f"{selected.interpretability:.4f}")

    st.markdown("### Pipeline Configuration")
    st.json(
        {
            "Vectorizer": selected.vectorizer,
            "Classifier": selected.classifier,
        }
    )

    # --------------------------------------------------
    # Rebuild pipeline for explanation
    # --------------------------------------------------
    st.markdown("### Model Structure and Diagnostics")

    solution = selected.solution

    # Decode structure
    vec_class = GENE_POOL[0][solution[0]]
    clf_class = GENE_POOL[1][solution[1]]

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("vectorizer", vec_class()),
            ("classifier", clf_class()),
        ]
    )

    st.caption(
        "Note: Diagnostics are shown for a freshly trained pipeline with default "
        "hyperparameters. Exact BO-tuned parameters are not reconstructed here."
    )

    # Fit fresh pipeline for explanation
    with st.spinner("Training pipeline for diagnostics..."):
        pipeline.fit(X_train, y_train)

    # --------------------------------------------------
    # In-situ explanations
    # --------------------------------------------------
    if selected.classifier == "LogisticRegression":

        st.markdown("#### Top weighted features (Logistic Regression)")

        coef = pipeline.named_steps["classifier"].coef_[0]
        vocab = pipeline.named_steps["vectorizer"].get_feature_names_out()

        top_features = sorted(
            zip(vocab, coef),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:15]

        feature_df = pd.DataFrame(top_features, columns=["Feature", "Weight"])
        feature_df["Absolute Weight"] = feature_df["Weight"].abs()

        fig_features = px.bar(
            feature_df,
            x="Absolute Weight",
            y="Feature",
            orientation="h",
            title="Most Important Features (by coefficient magnitude)",
            color="Weight",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig_features, width="stretch")

    elif selected.classifier == "DecisionTreeClassifier":

        st.markdown("#### Decision Tree Properties")

        tree = pipeline.named_steps["classifier"]

        col1, col2 = st.columns(2)
        col1.metric("Tree Depth", tree.get_depth())
        col2.metric("Number of Leaves", tree.get_n_leaves())

        # Feature importances
        if hasattr(tree, "feature_importances_"):
            importances = tree.feature_importances_
            vocab = pipeline.named_steps["vectorizer"].get_feature_names_out()

            # Get top 15 features
            top_idx = importances.argsort()[-15:][::-1]
            top_features = [
                (vocab[i], importances[i]) for i in top_idx if importances[i] > 0
            ]

            if top_features:
                importance_df = pd.DataFrame(
                    top_features, columns=["Feature", "Importance"]
                )
                fig_imp = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Feature Importances",
                )
                st.plotly_chart(fig_imp, width="stretch")

    elif selected.classifier == "SVC":
        st.markdown("#### SVM Properties")
        svm = pipeline.named_steps["classifier"]

        col1, col2 = st.columns(2)
        col1.metric("Number of Support Vectors", svm.n_support_.sum())
        col2.metric("Kernel", svm.kernel)

        st.info(
            "ℹ️ SVC is a black-box model with low intrinsic interpretability. "
            "Consider LogisticRegression or DecisionTree for better explainability."
        )

else:
    st.info(
        "👈 Configure AutoML in the sidebar and click 'Run T-AutoNLP' to start optimization."
    )

    # Show example of what to expect
    with st.expander("ℹ️ What is T-AutoNLP?"):
        st.markdown(
            """
        **T-AutoNLP** is a multi-objective AutoML system that simultaneously optimizes:

        1. **Accuracy**: Predictive performance on text classification tasks
        2. **Efficiency**: Training time, inference speed, and memory footprint
        3. **Interpretability**: Model transparency and explainability

        Unlike traditional AutoML that focuses solely on accuracy, T-AutoNLP produces a 
        **Pareto front** of solutions, allowing you to choose the pipeline that best fits 
        your deployment constraints and interpretability requirements.

        ### How it works:
        - **Genetic Algorithm (GA)** explores different pipeline structures
        - **Bayesian Optimization (BO)** fine-tunes hyperparameters
        - **NSGA-II** maintains diversity in the multi-objective space

        ### Supported Datasets:
        - **20 Newsgroups**: Binary classification (sci.med vs sci.space)
        - **IMDB Reviews**: Sentiment analysis (positive vs negative)

        ### Tips for IMDB dataset:
        - First load may take 1-2 minutes (downloading ~80MB)
        - Use smaller population size (4-6) for faster results
        - Convergence warnings for SVC are normal for text data
        """
        )

st.divider()
st.subheader("Search History & Exploration Summary")

history_df = load_all_runs()

if history_df.empty:
    st.info("No previous search history found.")
else:
    total = len(history_df)

    # Approximate dominance: treat last-run Pareto as non-dominated
    pareto_df = pd.DataFrame(st.session_state.get("results", []))
    pareto_count = len(pareto_df)
    dominated = total - pareto_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total pipelines evaluated", total)
    col2.metric("Pareto-optimal (current run)", pareto_count)
    col3.metric("Dominated / discarded", dominated)

    st.markdown("### Evaluated Pipelines (All Runs)")
    st.dataframe(
        history_df.sort_values(["accuracy", "interpretability"], ascending=False),
        width=True,
    )

st.markdown("### Exploration Diversity")

if "classifier" in history_df.columns and len(history_df) > 0:
    st.bar_chart(
        history_df["classifier"].value_counts(),
        height=250,
    )
else:
    st.info("No classifier data available yet. Run an experiment first.")
