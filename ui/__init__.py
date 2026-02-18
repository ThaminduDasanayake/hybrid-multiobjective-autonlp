"""UI modules for Streamlit interface."""

from .layout import (
    render_header,
    render_config,
    render_footer,
    render_results_summary,
    # render_knee_point_info,
    render_decision_support_panel
)
from .visuals import (
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_search_history,
    show_solutions_table,
    # compare_with_baseline
)

__all__ = [
    "render_header",
    "render_config",
    "render_footer",
    "render_results_summary",
    # "render_knee_point_info",
    "render_decision_support_panel",
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_search_history",
    "show_solutions_table",
    # "compare_with_baseline"
]
