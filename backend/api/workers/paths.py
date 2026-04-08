from pathlib import Path

# Absolute path to the backend root (the directory containing automl/, utils/, api/, etc.).
# Resolved at import time so all modules that need to reference backend-relative paths
# or pass the root to spawned worker processes share a single source of truth.
#
# File location: backend/api/workers/paths.py
#   parents[0] → backend/api/workers/
#   parents[1] → backend/api/
#   parents[2] → backend/          ← BACKEND_ROOT
BACKEND_ROOT: Path = Path(__file__).resolve().parents[2]
