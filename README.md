# T-AutoNLP: Human-Centered Multi-Objective AutoML for NLP

A hybrid Genetic Algorithm + Bayesian Optimization framework for automated NLP pipeline discovery, optimizing simultaneously for **accuracy**, **inference speed**, and **intrinsic interpretability**.

## Architecture

```
code/
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── api/
│   │   ├── core/                  # Config, lifespan
│   │   ├── routes/                # HTTP route handlers (jobs, ablations, feedback, health)
│   │   ├── services/              # Business logic (job, ablation, analytics)
│   │   ├── models/                # Pydantic schemas (JobConfig, AblationConfig, FeedbackCreate)
│   │   ├── workers/               # ProcessPoolExecutor, job manager, worker entry points
│   │   └── db.py                  # MongoDB client
│   ├── automl/                    # GA engine, BO, evaluator, Pareto logic
│   └── utils/                     # Data loading, evaluation, interpretability, logging
├── frontend/                      # React + Vite SPA (shadcn/ui, Tailwind CSS v4, Plotly)
│   ├── src/pages/                 # RunAutoML, HistoryAnalysis, Experiments
│   ├── src/components/
│   │   ├── history-analysis/      # Charts & tables for run analysis
│   │   ├── experiments/           # Ablation comparison components
│   │   ├── run-automl/            # Config form & live job tracker
│   │   └── ui/                    # shadcn/ui primitives
│   └── src/store.js               # Zustand global state (survives navigation)
└── Dockerfile                     # Multi-stage Docker build (targeting Hugging Face Spaces)
```

The system is fully decoupled: the React frontend communicates with the FastAPI backend over a REST + SSE API. They can be run independently for development or served together from a single container in production.

## Features

- **Multi-objective optimization** — simultaneous Pareto-optimal search over F1, latency, and interpretability
- **Hybrid GA + Bayesian Optimization** — DEAP-powered genetic algorithm for pipeline structure search; scikit-optimize BO for hyperparameter refinement of top candidates
- **Intrinsic interpretability** — optimizes for explainability-by-design using structural model properties (not post-hoc explanations)
- **Real-time job tracking** — Server-Sent Events stream live progress, generation counts, and best-F1 to the UI
- **Comprehensive visualization** — interactive 3D Pareto front, 2D projections, parallel coordinates, convergence charts (F1 + hypervolume), model distribution, and objective box plots
- **Decision support** — automatic recommendation of Best Accuracy, Best Speed, Best Interpretability, and Knee-point candidates
- **Ablation studies** — side-by-side comparison tables and bar charts for Single-Objective, 2-Objective, GA-Only, and full GA+BO configurations
- **Navigation-safe state** — Zustand global store ensures running-job and queued-ablation states survive page navigation

## Quick Start

### Prerequisites

| Tool                             | Version |
| -------------------------------- | ------- |
| Python                           | >= 3.13 |
| Node.js                          | >= 20   |
| MongoDB                          | >= 6    |
| [uv](https://docs.astral.sh/uv/) | latest  |

### 1. Backend

```bash
cd backend
uv sync
uv run fastapi dev main.py
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
docker build -t t-autonlp .
docker run -p 7860:7860 \
  -e CORS_ORIGINS=http://localhost:7860 \
  -e MONGODB_URI=mongodb://host.docker.internal:27017 \
  t-autonlp
```

The React SPA is served statically by FastAPI at `http://localhost:7860`.

## Environment Variables

| Variable       | Default                                                              | Description                              |
| -------------- | -------------------------------------------------------------------- | ---------------------------------------- |
| `CORS_ORIGINS` | `http://localhost:5173,https://t-autonlp.vercel.app`                | Comma-separated list of allowed origins  |
| `MONGODB_URI`  | `mongodb://localhost:27017`                                          | MongoDB connection URI                   |
| `MONGODB_DB`   | `tautonlp`                                                           | Database name                            |

## Pages

| Page               | Route          | Description                                                                                                                                                   |
| ------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Run AutoML         | `/`            | Configure and launch a new multi-objective optimization run with live SSE tracking                                                                            |
| History & Analysis | `/history`     | Explore completed runs: 3D/2D Pareto fronts, parallel coordinates, convergence charts, model & objective distributions, decision support, and solution tables |
| Experiments        | `/experiments` | Ablation study comparison tables, grouped bar charts, and buttons to trigger missing ablation runs                                                            |

## API Overview

| Method   | Route                                | Description                                 |
| -------- | ------------------------------------ | ------------------------------------------- |
| `GET`    | `/api/health`                        | Health check                                |
| `POST`   | `/api/jobs`                          | Start a new AutoML job                      |
| `GET`    | `/api/jobs`                          | List all jobs                               |
| `GET`    | `/api/jobs/{id}/stream`              | SSE stream of live progress                 |
| `GET`    | `/api/jobs/{id}/result`              | Fetch full Pareto results                   |
| `GET`    | `/api/jobs/{id}/hypervolume-history` | Per-generation hypervolume convergence data |
| `DELETE` | `/api/jobs/{id}`                     | Terminate a running job                     |
| `DELETE` | `/api/jobs/{id}/data`                | Permanently delete a completed job          |
| `GET`    | `/api/ablations`                     | List ablation study results                 |
| `POST`   | `/api/ablations`                     | Trigger an ablation run                     |
| `POST`   | `/api/feedback`                      | Submit user feedback                        |

## Running Tests

```bash
cd backend
uv run pytest tests/ -v
```
