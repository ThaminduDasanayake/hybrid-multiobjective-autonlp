# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Build the React frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (layer-cached unless package-lock.json changes).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build.
COPY frontend/ ./
RUN npm run build
# Output lives at /app/frontend/dist


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — FastAPI backend + bundled React SPA
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS backend

# Install uv (Astral's fast Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching.
COPY backend/pyproject.toml backend/uv.lock ./

# Install Python dependencies from the lockfile (no dev extras).
RUN uv sync --no-dev --frozen

# Copy backend source.
COPY backend/ ./

# Create runtime directories that the app writes to at run time.
# Mount these as Docker volumes in production to persist data across restarts.
RUN mkdir -p jobs results/ablations data logs

# Copy the React build into static/ so FastAPI can serve the SPA.
COPY --from=frontend-build /app/frontend/dist ./static/

EXPOSE 8000

# uvicorn is executed through uv so the project virtual-env is on the path.
CMD ["uv", "run", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
