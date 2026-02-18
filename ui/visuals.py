import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import List, Dict, Any


def plot_pareto_front_2d(solutions: List[Dict[str, Any]],
                         pareto_front: List[Dict[str, Any]],
                         knee_point: Dict[str, Any] = None,
                         x_metric: str = "f1_score",
                         y_metric: str = "latency",
                         key: str = None):
    """
    Create 2D scatter plot of solutions with Pareto front highlighted.

    Args:
        solutions: All evaluated solutions
        pareto_front: Pareto-optimal solutions
        knee_point: Knee point solution (optional)
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
    """
    # Create DataFrame
    df_all = pd.DataFrame(solutions)
    df_pareto = pd.DataFrame(pareto_front)

    # Metric labels
    metric_labels = {
        "f1_score": "F1 Score",
        "latency": "Latency (s)",
        "interpretability": "Interpretability"
    }

    # Create figure
    fig = go.Figure()

    # All solutions
    fig.add_trace(go.Scatter(
        x=df_all[x_metric],
        y=df_all[y_metric],
        mode='markers',
        name='All Solutions',
        marker=dict(
            size=8,
            color='lightblue',
            opacity=0.6,
            line=dict(width=1, color='white')
        ),
        text=[f"{row['vectorizer']}-{row['model']}" for _, row in df_all.iterrows()],
        hovertemplate='<b>%{text}</b><br>' +
                      f'{metric_labels[x_metric]}: %{{x:.4f}}<br>' +
                      f'{metric_labels[y_metric]}: %{{y:.4f}}<extra></extra>'
    ))

    # Pareto front
    if len(df_pareto) > 0:
        fig.add_trace(go.Scatter(
            x=df_pareto[x_metric],
            y=df_pareto[y_metric],
            mode='markers',
            name='Pareto Front',
            marker=dict(
                size=12,
                color='red',
                symbol='star',
                line=dict(width=2, color='darkred')
            ),
            text=[f"{row['vectorizer']}-{row['model']}" for _, row in df_pareto.iterrows()],
            hovertemplate='<b>%{text}</b><br>' +
                          f'{metric_labels[x_metric]}: %{{x:.4f}}<br>' +
                          f'{metric_labels[y_metric]}: %{{y:.4f}}<extra></extra>'
        ))

    # Knee point
    if knee_point is not None:
        fig.add_trace(go.Scatter(
            x=[knee_point[x_metric]],
            y=[knee_point[y_metric]],
            mode='markers',
            name='Knee Point',
            marker=dict(
                size=16,
                color='gold',
                symbol='diamond',
                line=dict(width=3, color='orange')
            ),
            text=[f"{knee_point['vectorizer']}-{knee_point['model']}"],
            hovertemplate='<b>Knee Point: %{text}</b><br>' +
                          f'{metric_labels[x_metric]}: %{{x:.4f}}<br>' +
                          f'{metric_labels[y_metric]}: %{{y:.4f}}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=f"{metric_labels[x_metric]} vs {metric_labels[y_metric]}",
        xaxis_title=metric_labels[x_metric],
        yaxis_title=metric_labels[y_metric],
        hovermode='closest',
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, width="stretch", key=key)


def plot_pareto_front_3d(solutions: List[Dict[str, Any]],
                         pareto_front: List[Dict[str, Any]],
                         knee_point: Dict[str, Any] = None,
                         key: str = None):
    """
    Create 3D scatter plot showing all three objectives.

    Args:
        solutions: All evaluated solutions
        pareto_front: Pareto-optimal solutions
        knee_point: Knee point solution (optional)
    """
    df_all = pd.DataFrame(solutions)
    df_pareto = pd.DataFrame(pareto_front)

    # Create figure
    fig = go.Figure()

    # All solutions
    fig.add_trace(go.Scatter3d(
        x=df_all['f1_score'],
        y=df_all['latency'],
        z=df_all['interpretability'],
        mode='markers',
        name='All Solutions',
        marker=dict(
            size=5,
            color='lightblue',
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        text=[f"{row['vectorizer']}-{row['model']}" for _, row in df_all.iterrows()],
        hovertemplate='<b>%{text}</b><br>' +
                      'F1: %{x:.4f}<br>' +
                      'Latency: %{y:.4f}s<br>' +
                      'Interpretability: %{z:.4f}<extra></extra>'
    ))

    # Pareto front
    if len(df_pareto) > 0:
        fig.add_trace(go.Scatter3d(
            x=df_pareto['f1_score'],
            y=df_pareto['latency'],
            z=df_pareto['interpretability'],
            mode='markers',
            name='Pareto Front',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                line=dict(width=1, color='darkred')
            ),
            text=[f"{row['vectorizer']}-{row['model']}" for _, row in df_pareto.iterrows()],
            hovertemplate='<b>%{text}</b><br>' +
                          'F1: %{x:.4f}<br>' +
                          'Latency: %{y:.4f}s<br>' +
                          'Interpretability: %{z:.4f}<extra></extra>'
        ))

    # Knee point
    if knee_point is not None:
        fig.add_trace(go.Scatter3d(
            x=[knee_point['f1_score']],
            y=[knee_point['latency']],
            z=[knee_point['interpretability']],
            mode='markers',
            name='Knee Point',
            marker=dict(
                size=12,
                color='gold',
                symbol='diamond',
                line=dict(width=2, color='orange')
            ),
            text=[f"{knee_point['vectorizer']}-{knee_point['model']}"],
            hovertemplate='<b>Knee Point: %{text}</b><br>' +
                          'F1: %{x:.4f}<br>' +
                          'Latency: %{y:.4f}s<br>' +
                          'Interpretability: %{z:.4f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title="3D Pareto Front: Performance × Efficiency × Interpretability",
        scene=dict(
            xaxis_title="F1 Score",
            yaxis_title="Latency (s)",
            zaxis_title="Interpretability",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig, width="stretch", key=key)


def plot_search_history(search_history: List[Dict[str, Any]], key: str = None):
    """
    Plot search history showing objective evolution over generations.

    Args:
        search_history: List of evaluation records
    """
    df = pd.DataFrame(search_history)

    if len(df) == 0:
        st.warning("No search history available")
        return

    # Group by generation
    gen_stats = df.groupby('generation').agg({
        'f1_score': ['mean', 'max'],
        'latency': ['mean', 'min'],
        'interpretability': ['mean', 'max']
    }).reset_index()

    # Create subplots
    fig = go.Figure()

    # F1 Score
    fig.add_trace(go.Scatter(
        x=gen_stats['generation'],
        y=gen_stats['f1_score']['max'],
        mode='lines+markers',
        name='Best F1 Score',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=gen_stats['generation'],
        y=gen_stats['f1_score']['mean'],
        mode='lines',
        name='Mean F1 Score',
        line=dict(color='lightblue', width=1, dash='dash')
    ))

    fig.update_layout(
        title="F1 Score Evolution Across Generations",
        xaxis_title="Generation",
        yaxis_title="F1 Score",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, width="stretch", key=f"{key}_f1" if key else None)

    # Latency and Interpretability
    col1, col2 = st.columns(2)

    with col1:
        fig_latency = go.Figure()
        fig_latency.add_trace(go.Scatter(
            x=gen_stats['generation'],
            y=gen_stats['latency']['min'],
            mode='lines+markers',
            name='Best Latency',
            line=dict(color='green', width=2)
        ))
        fig_latency.update_layout(
            title="Best Latency per Generation",
            xaxis_title="Generation",
            yaxis_title="Latency (s)",
            height=300
        )
        st.plotly_chart(fig_latency, width="stretch", key=f"{key}_latency" if key else None)

    with col2:
        fig_interp = go.Figure()
        fig_interp.add_trace(go.Scatter(
            x=gen_stats['generation'],
            y=gen_stats['interpretability']['max'],
            mode='lines+markers',
            name='Best Interpretability',
            line=dict(color='purple', width=2)
        ))
        fig_interp.update_layout(
            title="Best Interpretability per Generation",
            xaxis_title="Generation",
            yaxis_title="Interpretability",
            height=300
        )
        st.plotly_chart(fig_interp, width="stretch", key=f"{key}_interp" if key else None)


def show_solutions_table(solutions: List[Dict[str, Any]],
                         pareto_front: List[Dict[str, Any]],
                         knee_point: Dict[str, Any] = None):
    """
    Display solutions in an interactive table.

    Args:
        solutions: All evaluated solutions
        pareto_front: Pareto-optimal solutions
        knee_point: Knee point solution (optional)
    """
    df = pd.DataFrame(solutions)

    def solution_key(sol: Dict[str, Any]) -> str:
        params = sol.get("params", {})
        params_key = json.dumps(params, sort_keys=True, default=str)
        return "|".join([
            str(sol.get("vectorizer")),
            str(sol.get("model")),
            params_key,
            f"{sol.get('f1_score')}",
            f"{sol.get('latency')}",
            f"{sol.get('interpretability')}",
        ])

    pareto_keys = {solution_key(s) for s in pareto_front}
    
    # Calculate flags using original solution objects to avoid DataFrame type conversion issues (e.g. NaN for missing keys)
    df['is_pareto'] = [solution_key(s) in pareto_keys for s in solutions]

    if knee_point:
        knee_key = solution_key(knee_point)
        df['is_knee'] = [solution_key(s) == knee_key for s in solutions]
    else:
        df['is_knee'] = False

    # Reorder columns
    display_df = df[[
        'vectorizer', 'model', 'f1_score', 'latency',
        'interpretability', 'is_pareto', 'is_knee'
    ]].copy()

    # Format
    display_df['latency'] = display_df['latency'].apply(lambda x: f"{x * 1000:.2f} ms")
    display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}")
    display_df['interpretability'] = display_df['interpretability'].apply(lambda x: f"{x:.4f}")
    display_df['is_pareto'] = display_df['is_pareto'].apply(lambda x: "⭐" if x else "")
    display_df['is_knee'] = display_df['is_knee'].apply(lambda x: "💎" if x else "")

    # Rename columns
    display_df.columns = [
        'Vectorizer', 'Model', 'F1 Score', 'Latency',
        'Interpretability', 'Pareto', 'Knee'
    ]

    display_df.index.name = "No."

    # Start index from 1 instead of 0
    display_df.index = range(1, len(display_df) + 1)

    st.dataframe(
        display_df,
        width="stretch",
        height=400
    )

    st.markdown("**Legend:** ⭐ = Pareto Front, 💎 = Knee Point")


# def compare_with_baseline(automl_results: Dict[str, Any],
#                           baseline_results: Dict[str, Any]):
#     """
#     Compare AutoML results with baseline.
#
#     Args:
#         automl_results: Results from HybridAutoML
#         baseline_results: Results from baseline
#     """
#     st.subheader("📊 AutoML vs. Baseline Comparison")
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         st.markdown("**HybridAutoML**")
#         st.metric("Evaluations", automl_results["stats"]["total_evaluations"])
#         st.metric("Pareto Front Size", automl_results["stats"]["pareto_size"])
#
#         # Best objectives from Pareto front
#         pareto = automl_results["pareto_front"]
#         if pareto:
#             best_f1 = max([s["f1_score"] for s in pareto])
#             best_latency = min([s["latency"] for s in pareto])
#             best_interp = max([s["interpretability"] for s in pareto])
#
#             st.write(f"Best F1: {best_f1:.4f}")
#             st.write(f"Best Latency: {best_latency * 1000:.2f} ms")
#             st.write(f"Best Interpretability: {best_interp:.4f}")
#
#     with col2:
#         st.markdown("**Random Search Baseline**")
#         st.metric("Evaluations", baseline_results["stats"]["total_evaluations"])
#
#         # Best objectives
#         baseline_sols = baseline_results["all_solutions"]
#         if baseline_sols:
#             best_f1 = max([s["f1_score"] for s in baseline_sols])
#             best_latency = min([s["latency"] for s in baseline_sols])
#             best_interp = max([s["interpretability"] for s in baseline_sols])
#
#             st.write(f"Best F1: {best_f1:.4f}")
#             st.write(f"Best Latency: {best_latency * 1000:.2f} ms")
#             st.write(f"Best Interpretability: {best_interp:.4f}")
#
#     # Visualize comparison
#     st.markdown("### Objective Space Comparison")
#
#     # Combine data
#     automl_df = pd.DataFrame(automl_results["all_solutions"])
#     automl_df['source'] = 'HybridAutoML'
#
#     baseline_df = pd.DataFrame(baseline_results["all_solutions"])
#     baseline_df['source'] = 'Random Search'
#
#     combined_df = pd.concat([automl_df, baseline_df], ignore_index=True)
#
#     # F1 vs Latency
#     fig = px.scatter(
#         combined_df,
#         x='f1_score',
#         y='latency',
#         color='source',
#         symbol='source',
#         title='F1 Score vs Latency: AutoML vs Baseline',
#         labels={'f1_score': 'F1 Score', 'latency': 'Latency (s)'},
#         color_discrete_map={'HybridAutoML': 'red', 'Random Search': 'blue'}
#     )
#
#     st.plotly_chart(fig, width="stretch")
