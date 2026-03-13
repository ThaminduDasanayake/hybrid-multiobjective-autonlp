# T-AutoNLP Backend

FastAPI backend for the T-AutoNLP system. Runs AutoML jobs in a `ProcessPoolExecutor`, persists results to disk, and exposes a REST + SSE API consumed by the React frontend.

## Directory Structure

```
backend/
├── api/
│   ├── server.py           # FastAPI app, all route handlers
│   ├── job_manager.py      # APIJobManager — ProcessPoolExecutor orchestration
│   ├── worker_fn.py        # Top-level worker function (runs inside executor)
│   └── ablation_worker.py  # Worker for ablation experiment runs
├── automl/
│   ├── hybrid_automl.py    # HybridAutoML — GA + BO outer loop
│   ├── search_engine.py    # DEAP-based genetic algorithm
│   ├── bayesian_optimization.py  # scikit-optimize BO refinement
│   ├── evaluator.py        # Pipeline evaluation (F1, latency, interpretability)
│   ├── pipeline_builder.py # Builds sklearn pipelines from genome representations
│   ├── interpretability.py # Interpretability scoring
│   └── persistence.py      # Checkpoint read/write (ResultStore)
├── experiments/
│   ├── evaluation.py       # Experiment runner and metrics aggregation
│   └── run_ablations.py    # CLI entry-point for ablation studies
├── utils/
│   ├── data_loader.py      # HuggingFace Datasets loading + preprocessing
│   ├── job_manager.py      # Base JobManager (file-based state helpers)
│   ├── serialization.py    # JSON-safe serialisation (handles inf/nan)
│   ├── formatting.py       # Human-readable metric formatting
│   └── logger.py           # Centralised logging configuration
└── tests/
    ├── test_api_routes.py
    ├── test_automl_core.py
    ├── test_data_loader.py
    ├── test_inference.py
    ├── test_pareto.py
    ├── test_persistence.py
    ├── test_serialization.py
    └── test_utils.py
```

## Setup

```bash
# Install dependencies
uv sync

# Run development server (hot-reload)
uv run uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# Run production server
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi[standard]` | Web framework + Uvicorn |
| `deap` | Genetic algorithm (GA search engine) |
| `scikit-optimize` | Bayesian optimization |
| `scikit-learn` | Pipeline building, model evaluation |
| `datasets` | HuggingFace dataset loading |
| `numpy` | Numerical operations |
| `pymoo` | Pareto-front utilities |

Dev dependencies: `pytest`, `httpx` (async FastAPI test client).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8000` | Comma-separated allowed origins |

## Running Tests

```bash
uv run pytest tests/ -v
# or directly via the venv:
.venv/bin/python -m pytest tests/ -v
```

## Job Lifecycle

```
POST /api/jobs  →  APIJobManager.create_job()
                    ├── writes jobs/{id}/config.json
                    ├── clears jobs/{id}/checkpoints/ (prevents stale cache)
                    ├── sets status → "created"
                    └── submits worker_fn.run_automl_job() to ProcessPoolExecutor

worker_fn  →  HybridAutoML.run()
               ├── GA generations (DEAP) — checkpointed each generation
               ├── BO refinement of top-k candidates
               └── writes jobs/{id}/result.json + status → "completed"

GET /api/jobs/{id}/stream   →  SSE: polls status file every 0.5 s
GET /api/jobs/{id}/result   →  returns result.json merged with config.json
```

## Ablation Experiments

```bash
uv run python -m experiments.run_ablations
```

Results are written to `results/ablations/` and exposed via `GET /api/ablations`.
