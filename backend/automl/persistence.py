"""
THE MEMORY — persistence.py
=============================
This module is the in-memory database for a single AutoML run. It has two jobs:

1. Deduplication cache (eval_cache): Stores the result of every unique pipeline
   that has been evaluated. When the GA produces a chromosome it has seen before
   (which happens often as good genes are re-selected), the cache returns the
   stored result instantly without re-running BO. This is one of the biggest
   runtime savers in the system.

2. Chronological history (search_history): A running log of every evaluation in
   arrival order. Used after the search finishes to reconstruct which generation
   each pipeline was first discovered in, for the convergence chart.

All state lives in memory and is scoped to one job. Nothing is written to disk here.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from utils.logger import get_logger

logger = get_logger("persistence")


class ResultStore:
    """
    In-memory cache and history store for a single AutoML job.

    eval_cache:     MD5-keyed dict, one entry per unique pipeline architecture.
    search_history: Ordered list of every evaluation event (for convergence tracking).
    """

    def __init__(self):
        self.eval_cache = {}
        self.search_history = []

        # Hit/miss counters are reported live to the frontend during the run
        # so the user can see how efficiently the cache is working.
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        # Timing accumulators: total BO time and per-generation wall-clock times.
        # Collected here so hybrid_automl.py can include them in the result payload.
        self.total_optimization_time = 0.0
        self.generation_times = []

    def get_individual_key(self, individual: list) -> str:
        """Convert a pipeline chromosome into a stable, collision-proof cache key.

        We serialise the 6 genes to a sorted JSON string and MD5-hash the result.
        Sorted keys ensure the hash is identical regardless of dict insertion order.
        String-casting the last two genes (ngram_range, max_features) guards against
        type inconsistencies introduced by numpy (e.g., np.str_ vs str). MD5 is
        chosen for speed, not cryptographic security — collision risk is negligible
        for the ~200 unique pipelines a typical run evaluates.
        """
        # Individual structure: [scaler, dim_reduction, vectorizer, model, ngram_range, max_features]
        config = {
            "scaler": individual[0],
            "dim_reduction": individual[1],
            "vectorizer": individual[2],
            "model": individual[3],
            "ngram_range": str(individual[4]),  # Cast to string for safety
            "max_features": str(individual[5]),  # Cast to string for safety
        }
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def get_cached_evaluation(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation from cache, incrementing hit/miss counters."""
        result = self.eval_cache.get(key)
        if result is not None:
            self.cache_hit_count += 1
        else:
            self.cache_miss_count += 1
        return result

    def peek(self, key: str) -> Optional[Dict[str, Any]]:
        """Check cache without affecting hit/miss counters (probe-only).

        Used by the search engine's stagnation detector to count genuinely new
        individuals without skewing the cache metrics reported to the frontend.
        If peek() incremented the miss counter, the live cache hit rate would
        be artificially depressed during stagnation checks.
        """
        return self.eval_cache.get(key)

    def cache_evaluation(self, key: str, result: Dict[str, Any]):
        """Store evaluation in cache."""
        self.eval_cache[key] = result

    def add_to_history(self, entry: Dict[str, Any], generation: Optional[int] = None):
        """Add entry to search history."""
        if generation is not None:
            entry["generation"] = generation
        self.search_history.append(entry)

    def add_time_stats(self, optimization_time: float = 0, generation_time: float = 0):
        """Update time statistics."""
        if optimization_time > 0:
            self.total_optimization_time += optimization_time

        if generation_time > 0:
            self.generation_times.append(generation_time)

    def get_live_metrics(self) -> Dict[str, Any]:
        """Return current best-of-run metrics for live progress reporting.

        Called after every GA generation to push a live dashboard snapshot to
        MongoDB, which the frontend reads via the SSE stream. Encapsulates cache
        access so the worker process doesn't need to know ResultStore's internal schema.
        """
        successful = [v for v in self.eval_cache.values() if v.get("status") == "success"]
        total_lookups = self.cache_hit_count + self.cache_miss_count
        return {
            "best_f1": round(float(max((v["f1_score"] for v in successful), default=0.0)), 4),
            "best_latency_ms": round(
                float(min((v["latency"] * 1000 for v in successful), default=0.0)), 4
            ),
            "best_interpretability": round(
                float(max((v["interpretability"] for v in successful), default=0.0)), 4
            ),
            "cache_hit_rate": (
                round(self.cache_hit_count / total_lookups * 100, 1) if total_lookups > 0 else 0.0
            ),
            "total_evaluated": len(self.eval_cache),
        }


