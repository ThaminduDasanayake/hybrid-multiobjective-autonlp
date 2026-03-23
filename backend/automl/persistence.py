import json
import hashlib
from typing import Dict, List, Any, Optional
from utils.logger import get_logger

logger = get_logger("persistence")


class ResultStore:
    """
    Handles persistence of AutoML results including caching and search history.
    """

    def __init__(self):
        self.eval_cache = {}
        self.search_history = []

        # Cache hit/miss counters for live reporting
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        # Time statistics
        self.total_optimization_time = 0.0
        self.generation_times = []

    def get_individual_key(self, individual: list) -> str:
        """
        Convert individual to a hashable cache key.
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
        """Check cache without affecting hit/miss counters (probe-only)."""
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


