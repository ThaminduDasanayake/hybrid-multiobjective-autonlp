import os
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional
from utils.logger import get_logger

logger = get_logger("persistence")

class ResultStore:
    """
    Handles persistence of AutoML results, including caching and checkpointing.
    """
    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        self.eval_cache = {}
        self.search_history = []
        
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
            "ngram_range": str(individual[4]), # Cast to string for safety
            "max_features": str(individual[5]) # Cast to string for safety
        }
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def get_cached_evaluation(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation from cache."""
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
            
    def set_generation_time(self, gen_idx: int, duration: float):
        """Set processed generation time (idempotent for retries)."""
        if len(self.generation_times) <= gen_idx:
            self.generation_times.append(duration)
        else:
            self.generation_times[gen_idx] = duration

    def save_checkpoint(self, extra_state: Dict[str, Any] = None, filename: str = "checkpoint.pkl"):
        """Save current state to a checkpoint file."""
        if not self.checkpoint_dir:
            return

        path = os.path.join(self.checkpoint_dir, filename)
        try:
            state = {
                "eval_cache": self.eval_cache,
                "search_history": self.search_history,
                "total_optimization_time": self.total_optimization_time,
                "generation_times": self.generation_times,
            }
            
            if extra_state:
                state.update(extra_state)
                
            with open(path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filename: str = "checkpoint.pkl") -> Optional[Dict[str, Any]]:
        """Load state from a checkpoint file and return extra state."""
        if not self.checkpoint_dir:
            return None

        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self.eval_cache = state.get("eval_cache", {})
            self.search_history = state.get("search_history", [])
            self.total_optimization_time = state.get("total_optimization_time", 0.0)
            self.generation_times = state.get("generation_times", [])

            logger.info(f"Checkpoint loaded from {path}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
