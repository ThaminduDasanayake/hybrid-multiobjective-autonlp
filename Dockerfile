FROM python:3.13-slim

# Install uv (Astral's fast Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching.
COPY backend/pyproject.toml backend/uv.lock ./

# Install Python dependencies from the lockfile (no dev extras).
RUN uv sync --no-dev --frozen

# Copy backend source.
COPY backend/ ./

# Create runtime directories the app writes to.
# Mount these as Docker volumes in production to persist data across restarts.
RUN mkdir -p jobs results/ablations data logs

EXPOSE 7860

# uvicorn is executed through uv so the project virtual-env is on the path.
# Port 7860 is required by Hugging Face Spaces.
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
