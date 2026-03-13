# T-AutoNLP: Human-Centered Multi-Objective AutoML for NLP

A hybrid Genetic Algorithm + Bayesian Optimization framework for automated NLP pipeline discovery, optimizing simultaneously for **accuracy**, **inference speed**, and **intrinsic interpretability**.

## Architecture

```
code/
├── backend/        # FastAPI server, AutoML engine, experiments
├── frontend/       # React + Vite SPA (shadcn/ui, Tailwind CSS v4, Plotly)
└── Dockerfile      # Multi-stage Docker build (frontend → backend)
```

The system is fully decoupled: the React frontend communicates with the FastAPI backend over a REST + SSE API. They can be run independently for development or served together from a single container in production.

## Features

- **Multi-objective optimization** — simultaneous Pareto-optimal search over F1, latency, and interpretability
- **Hybrid GA + Bayesian Optimization** — DEAP-powered genetic algorithm for pipeline structure search; scikit-optimize BO for hyperparameter refinement of top candidates
- **Intrinsic interpretability** — optimizes for explainability-by-design using structural model properties (not post-hoc explanations)
- **Real-time job tracking** — Server-Sent Events stream live progress, generation counts, and best-F1 to the UI
- **Pareto front visualization** — interactive 3D scatter (F1 / Latency / Interpretability), 2D projections, and GA convergence chart
- **Decision support** — automatic recommendation of Best Accuracy, Best Speed, and Best Interpretability candidates
- **Ablation experiments** — scripted comparative runs for thesis evaluation

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.13 |
| Node.js | ≥ 20 |
| [uv](https://docs.astral.sh/uv/) | latest |

### 1. Backend

```bash
cd backend
uv sync
uv run uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI is available at `http://localhost:5173`.

### 3. Docker (production — single container)

```bash
# Build
docker build -t t-autonlp .

# Run
docker run -p 8000:8000 \
  -e CORS_ORIGINS=http://localhost:8000 \
  -v $(pwd)/jobs:/app/jobs \
  t-autonlp
```

The React SPA is served statically by FastAPI at `http://localhost:8000`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8000` | Comma-separated list of allowed origins |

## API Overview

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/jobs` | Start a new AutoML job |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/jobs/{id}/status` | Poll job status |
| `GET` | `/api/jobs/{id}/stream` | SSE stream of live progress |
| `GET` | `/api/jobs/{id}/result` | Fetch full Pareto results |
| `GET` | `/api/jobs/{id}/logs` | Fetch worker log output |
| `DELETE` | `/api/jobs/{id}` | Terminate a running job |
| `GET` | `/api/ablations` | List ablation study results |
| `POST` | `/api/ablations` | Trigger ablation run |

## Running Tests

```bash
cd backend
uv run pytest tests/ -v
```

165 tests across API routes, AutoML core, data loading, Pareto logic, persistence, and serialization.
