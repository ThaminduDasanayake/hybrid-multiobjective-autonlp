import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from automl.genetic_algorithm import GENE_POOL


def plot_pareto_fronts(df):
    """Renders the 2D and 3D Pareto front charts."""
    st.subheader("2D Trade-off Projections")
    col1, col2, col3 = st.columns(3)

    metrics = [
        ("accuracy", "interpretability", col1),
        ("accuracy", "efficiency", col2),
        ("efficiency", "interpretability", col3),
    ]

    for x_ax, y_ax, col in metrics:
        with col:
            st.caption(f"{x_ax.title()} vs {y_ax.title()}")
            fig = px.scatter(
                df,
                x=x_ax,
                y=y_ax,
                color="classifier",
                hover_data=["vectorizer"],
                template="plotly_white",
            )
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, width=True)

    st.subheader("Pareto Front (3D Trade-off Space)")
    fig_3d = px.scatter_3d(
        df,
        x="accuracy",
        y="efficiency",
        z="interpretability",
        color="classifier",
        symbol="vectorizer",
        hover_data=["accuracy", "efficiency", "interpretability"],
        height=600,
    )
    fig_3d.update_traces(marker=dict(size=6, opacity=0.8))
    st.plotly_chart(fig_3d, width=True)


def inspect_solution(df, X_train, y_train):
    """UI for selecting and explaining a specific pipeline."""
    st.subheader("Inspect Pareto-optimal Solution")

    if "solution" not in df.columns:
        st.error(
            "Cannot inspect solutions - 'solution' data not available in loaded results."
        )
        st.info(
            "This feature only works for newly run experiments, not loaded history."
        )
        return

    # Selection Widget
    idx = st.selectbox(
        "Select solution index",
        df.index,
        format_func=lambda x: f"Solution #{x+1} ({df.loc[x, 'classifier']})",
    )
    selected = df.loc[idx]

    # Metrics Display
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{selected.accuracy:.4f}")
    c2.metric("Efficiency", f"{selected.efficiency:.4f}")
    c3.metric("Interpretability", f"{selected.interpretability:.4f}")

    # Rebuild Pipeline logic
    st.markdown("### Model Diagnostics")
    solution = selected.solution
    vec_class = GENE_POOL[0][solution[0]]
    clf_class = GENE_POOL[1][solution[1]]

    pipeline = Pipeline([("vectorizer", vec_class()), ("classifier", clf_class())])

    with st.spinner("Training pipeline for diagnostics..."):
        pipeline.fit(X_train, y_train)

    # Specific Model Visualizations
    _render_model_explanation(pipeline, selected.classifier)


def _render_model_explanation(pipeline, clf_name):
    """Internal helper to render specific model charts."""
    if clf_name == "LogisticRegression":
        st.caption("Top weighted features")
        coef = pipeline.named_steps["classifier"].coef_[0]
        vocab = pipeline.named_steps["vectorizer"].get_feature_names_out()

        # Sort and plot
        top_feats = sorted(zip(vocab, coef), key=lambda x: abs(x[1]), reverse=True)[:15]
        f_df = pd.DataFrame(top_feats, columns=["Feature", "Weight"])
        f_df["Abs"] = f_df["Weight"].abs()

        fig = px.bar(f_df, x="Abs", y="Feature", orientation="h", color="Weight")
        st.plotly_chart(fig, width=True)

    elif clf_name == "DecisionTreeClassifier":
        tree = pipeline.named_steps["classifier"]
        c1, c2 = st.columns(2)
        c1.metric("Depth", tree.get_depth())
        c2.metric("Leaves", tree.get_n_leaves())

        if hasattr(tree, "feature_importances_"):
            imps = tree.feature_importances_
            vocab = pipeline.named_steps["vectorizer"].get_feature_names_out()
            top_idx = imps.argsort()[-15:][::-1]
            top_f = [(vocab[i], imps[i]) for i in top_idx if imps[i] > 0]

            if top_f:
                df_imp = pd.DataFrame(top_f, columns=["Feature", "Importance"])
                fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h")
                st.plotly_chart(fig, width=True)

    elif clf_name == "SVC":
        st.info("SVC is a black-box model. Limited interpretability available.")
