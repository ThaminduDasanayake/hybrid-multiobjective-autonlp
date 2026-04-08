// Shared Plotly theme for dark mode — import here instead of hardcoding colors per chart.

// Color Palette
export const COLORS = {
  primary: "#f97316",
  primaryDark: "#ea580c",
  amber: "#f59e0b",

  slate200: "#cbd5e1",
  slate500: "#a1a1aa",
  slate600: "#475569",
  slate800: "#27272a",
  slate950: "#0f172a",

  transparent: "rgba(0,0,0,0)",
  legendBg: "rgba(24, 24, 27, 0.8)",
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
  font: { color: COLORS.slate500 },
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
