# T-AutoNLP Frontend

React + Vite SPA for the T-AutoNLP system. Communicates with the FastAPI backend over REST and Server-Sent Events.

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| React | 19 | UI framework |
| Vite | 7 | Build tool + dev server |
| React Router | 7 | Client-side routing |
| Tailwind CSS | 4 | Utility-first styling |
| shadcn/ui (Radix) | — | Accessible component primitives |
| Plotly.js | 3 | Interactive charts (3D Pareto, 2D projections, convergence) |
| Zustand | 5 | Lightweight global state |
| Lucide React | — | Icons |

## Directory Structure

```
src/
├── pages/
│   ├── RunAutoML.jsx         # Job configuration and live progress tracking
│   ├── HistoryAnalysis.jsx   # Full result view: metrics, charts, solutions table
│   └── Experiments.jsx     # Comparative ablation results view
├── components/
│   ├── run/
│   │   ├── ConfigForm.jsx    # Dataset + hyperparameter form (Slider-based)
│   │   ├── LiveTracker.jsx   # SSE-driven progress bar and status
│   │   ├── SliderField.jsx   # Labelled slider with live value display
│   │   └── InputField.jsx    # Labelled text/number input
│   ├── shared/
│   │   ├── StatCard.jsx      # Single metric card (label + value + unit)
│   │   └── StatusBadge.jsx   # Job status badge (created/running/completed/failed)
│   ├── ui/                   # shadcn/ui primitives (auto-generated, do not edit)
│   │   └── button, card, badge, input, select, slider, progress, alert, ...
│   ├── JobConfigCard.jsx     # Run configuration parameter display
│   ├── DecisionSupport.jsx   # Best Accuracy / Speed / Interpretability recommendations
│   ├── ParetoFront3D.jsx     # 3D Plotly scatter (lazy-loaded)
│   ├── ParetoFront2D.jsx     # 2D Pareto projection (lazy-loaded, reusable)
│   ├── ConvergenceChart.jsx  # GA convergence line chart (lazy-loaded)
│   ├── SolutionsTable.jsx    # Sortable Pareto solutions table
│   ├── DropdownSelector.jsx  # shadcn Select wrapper
│   ├── MetricCard.jsx        # Compact metric display with optional accent
│   └── Layout.jsx            # App shell with sidebar navigation
├── api.js                    # All fetch / SSE calls to the FastAPI backend
├── store.js                  # Zustand store (active job ID, reset)
├── constants.js              # Datasets, defaults, nav items
├── main.jsx                  # App entry point
└── utils/
    ├── formatters.js         # Human-readable metric formatting
    └── knee.js               # Knee-point selection algorithm
```

## Setup

```bash
npm install
npm run dev       # Dev server on http://localhost:5173
npm run build     # Production build → dist/
npm run preview   # Preview production build locally
npm run lint      # ESLint
npm run format    # Prettier (write)
```

## Backend Connection

The API base URL is configured in [src/api.js](src/api.js). By default it points to `http://localhost:8000`. To change it for a deployment, set the `VITE_API_BASE` environment variable:

```bash
VITE_API_BASE=https://api.example.com npm run build
```

## Design System

All colors use semantic CSS variable tokens defined in `src/index.css` (`--primary`, `--muted`, `--chart-1` through `--chart-5`, etc.) and registered as Tailwind utilities via `@theme inline`. Hardcoded palette colors (e.g. `text-green-600`, `bg-amber-100`) are not used — use the semantic tokens instead.

## Key Pages

### Run AutoML (`/run`)
Configure a new job (dataset, population size, generations, BO calls, seed) and submit it. Switches to a live progress view (SSE) once the job starts.

### History Analysis (`/history`)
Select any completed job and view:
- Summary stat cards (best F1, latency, interpretability, solutions count)
- Run configuration parameters
- Decision support panel (recommended solutions)
- Interactive 3D Pareto front
- 2D Pareto projections (F1 vs Latency, F1 vs Interpretability)
- GA convergence chart
- Full sortable solutions table

### Experiments (`/experiments`)
Side-by-side ablation comparison across algorithm variants.
