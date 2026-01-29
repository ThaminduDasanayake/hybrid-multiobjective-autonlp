import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px

from automl.bayesian_optimization import PARAM_SPACE
from automl.genetic_algorithm import GENE_POOL
from automl.hybrid_automl import HybridAutoML

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Page setup
st.set_page_config(layout="wide")
st.title("T-AutoNLP")
st.caption(
    "Human-centered, multi-objective AutoML for NLP. "
    "Optimizing accuracy, efficiency, and interpretability."
)

# Sidebar
st.sidebar.header("AutoML Configuration")

# Dataset selection
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["20 Newsgroups", "IMDB Reviews"],
    help="Choose the NLP dataset for AutoML optimization",
)

ngen = st.sidebar.slider("Generations", 2, 10, 4)
population_size = st.sidebar.slider("Population size", 4, 20, 6)

run_button = st.sidebar.button("Run T-AutoNLP")


# Load datasets
@st.cache_data
def load_20newsgroups():
    """Load 20 Newsgroups dataset"""
    categories = ["sci.med", "sci.space"]
    data = fetch_20newsgroups(
        subset="all", categories=categories, shuffle=True, random_state=42
    )
    X_train, _, y_train, _ = train_test_split(
        data.data,
        data.target,
        train_size=0.5,
        stratify=data.target,
        random_state=42,
    )
    return X_train, y_train, "20 Newsgroups (sci.med vs sci.space)"


@st.cache_data
def load_imdb():
    """Load IMDB Reviews dataset using sklearn's load_files"""
    try:
        from sklearn.datasets import load_files
        import tempfile
        import urllib.request
        import tarfile
        import os
        import warnings

        # Download and extract IMDB dataset
        temp_dir = tempfile.mkdtemp()
        url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        with st.spinner("Downloading IMDB dataset (this may take a moment)..."):
            tar_path = os.path.join(temp_dir, "imdb.tar.gz")
            urllib.request.urlretrieve(url, tar_path)

            with tarfile.open(tar_path, "r:gz") as tar:
                # Suppress the Python 3.14 deprecation warning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    tar.extractall(temp_dir)

        # Load train data
        train_data = load_files(
            os.path.join(temp_dir, "aclImdb", "train"),
            categories=["pos", "neg"],
            shuffle=True,
            random_state=42,
        )

        # Use subset for faster experimentation
        X_train, _, y_train, _ = train_test_split(
            train_data.data,
            train_data.target,
            train_size=0.3,  # Use 30% for speed
            stratify=train_data.target,
            random_state=42,
        )

        # Convert bytes to strings
        X_train = [doc.decode("utf-8", errors="ignore") for doc in X_train]

        return X_train, y_train, "IMDB Movie Reviews (positive vs negative)"

    except Exception as e:
        st.error(f"Failed to load IMDB dataset: {e}")
        st.info("Falling back to 20 Newsgroups dataset")
        return load_20newsgroups()


# Load selected dataset
if dataset_choice == "20 Newsgroups":
    X_train, y_train, dataset_name = load_20newsgroups()
else:
    X_train, y_train, dataset_name = load_imdb()

# Display dataset info
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
st.sidebar.info(
    f"**{dataset_name}**\n\n"
    f"Samples: {len(X_train)}\n\n"
    f"Classes: {len(set(y_train))}"
)


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
            # automl = HybridAutoML(
            #     X=X_train,
            #     y=y_train,
            #     gene_pool=GENE_POOL,
            #     param_space=PARAM_SPACE,
            # )

            # results = automl.run(
            #     ngen=ngen,
            #     population_size=population_size,
            # )

            results = run_automl(X_train, y_train, ngen, population_size)

            st.session_state["results"] = results
            st.session_state["dataset_name"] = dataset_name
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train

            st.success(
                f"✅ AutoML finished on {dataset_name}. Found {len(results)} Pareto-optimal solutions."
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
            "Interpretability mode": selected["mode"],
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
