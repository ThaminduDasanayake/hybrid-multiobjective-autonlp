"""UI modules for Streamlit interface."""

from .layout import (
    render_header,
    render_config,
    render_footer,
    render_results_summary,
    render_decision_support_panel,
    render_analysis_view,
)
from .visuals import (
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_search_history,
    show_solutions_table,
)

from .tabs import (
    run_automl,
    render_thesis_defense,
)

__all__ = [
    "render_header",
    "render_config",
    "render_footer",
    "render_results_summary",
    "render_decision_support_panel",
    "render_analysis_view",
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_search_history",
    "show_solutions_table",
    "run_automl",
    "render_thesis_defense",
]
