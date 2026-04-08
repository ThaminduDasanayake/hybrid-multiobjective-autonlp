"""
Manages in-memory caching of evaluated pipeline architectures and chronologically logs search history.
"""

import hashlib
import json
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger("persistence")


class ResultStore:
    """In-memory store for deduplicating pipeline evaluations and tracking optimization metrics."""

    def __init__(self):
        self.eval_cache = {}
        self.search_history = []

        # Counters for live cache efficiency tracking
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        # Timing accumulators for the final result payload
        self.total_optimization_time = 0.0
        self.generation_times = []

    @staticmethod
    def get_individual_key(individual: list) -> str:
        """Generates a stable MD5 hash key from a JSON-serialized pipeline chromosome."""

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
        """Retrieves an evaluation result from the cache and updates hit/miss counters."""
        result = self.eval_cache.get(key)
        if result is not None:
            self.cache_hit_count += 1
        else:
            self.cache_miss_count += 1
        return result

    def peek(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cache entry without modifying hit/miss counters, preventing skewed metrics during stagnation checks."""
        return self.eval_cache.get(key)

    def cache_evaluation(self, key: str, result: Dict[str, Any]):
        """Stores a pipeline evaluation result in the deduplication cache."""
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
        """Aggregates current best-of-run metrics for live progress reporting."""
        successful = [
            v for v in self.eval_cache.values() if v.get("status") == "success"
        ]
        total_lookups = self.cache_hit_count + self.cache_miss_count
        return {
            "best_f1": round(
                float(max((v["f1_score"] for v in successful), default=0.0)), 4
            ),
            "best_latency_ms": round(
                float(min((v["latency"] * 1000 for v in successful), default=0.0)), 4
            ),
            "best_interpretability": round(
                float(max((v["interpretability"] for v in successful), default=0.0)), 4
            ),
            "cache_hit_rate": (
                round(self.cache_hit_count / total_lookups * 100, 1)
                if total_lookups > 0
                else 0.0
            ),
            "total_evaluated": len(self.eval_cache),
        }
