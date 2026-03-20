/**
 * Shared Plotly chart theming for dark-mode UI.
 *
 * All chart components import from here instead of hardcoding hex colors
 * and layout objects individually.
 */

// Color Palette
export const COLORS = {
  primary: "#f97316",
  primaryDark: "#ea580c",
  amber: "#f59e0b",
  blue: "#3b82f6",
  blueDark: "#2563eb",

  slate200: "#cbd5e1",
  slate400: "#94a3b8",
  slate500: "#64748b",
  slate600: "#475569",
  slate800: "#1e293b",
  slate950: "#0f172a",

  transparent: "rgba(0,0,0,0)",
  legendBg: "rgba(2, 6, 23, 0.8)",
  dominatedSolution: "rgba(148, 163, 184, 0.35)",
  dominatedSolutionMed: "rgba(148, 163, 184, 0.6)",
  orangeFill: "rgba(249, 115, 22, 0.1)",
};

// Reusable Axis / Layout / Legend Objects
export const AXIS_STYLE = {
  gridcolor: COLORS.slate800,
  zerolinecolor: COLORS.slate800,
  tickfont: { size: 10, color: COLORS.slate500 },
};

export const CHART_LAYOUT = {
  autosize: true,
  paper_bgcolor: COLORS.transparent,
  plot_bgcolor: COLORS.transparent,
  font: { color: COLORS.slate400 },
};

export const LEGEND_STYLE = {
  bgcolor: COLORS.legendBg,
  bordercolor: COLORS.slate800,
  borderwidth: 1,
  font: { size: 11, color: COLORS.slate200 },
};

// Plotly Config Presets
export const CHART_CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
  displaylogo: false,
};

export const CHART_CONFIG_MINIMAL = {
  responsive: true,
  displayModeBar: false,
};
