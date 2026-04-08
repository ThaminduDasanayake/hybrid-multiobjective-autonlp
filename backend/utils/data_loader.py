"""
Handles downloading, caching, and loading of benchmark NLP datasets.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups

from utils.logger import get_logger

logger = get_logger("data_loader")

# Dataset metadata
_DATASET_INFO: dict = {
    "20newsgroups": {
        "description": "Multi-class document classification (20 categories)",
        "num_classes": 20,
        "task": "multi-class classification",
        "domain": "news articles",
    },
    "imdb": {
        "description": "Binary sentiment analysis (positive/negative)",
        "num_classes": 2,
        "task": "binary classification",
        "domain": "movie reviews",
    },
    "ag_news": {
        "description": "News categorization (4 categories)",
        "num_classes": 4,
        "task": "multi-class classification",
        "domain": "news headlines",
    },
    "banking77": {
        "description": "Intent classification (77 intents)",
        "num_classes": 77,
        "task": "multi-class classification",
        "domain": "banking queries",
    },
}


class DataLoader:
    """Manages the retrieval and persistent local caching of benchmark datasets."""

    def __init__(self, cache_dir: str = "./data"):
        """Initialize the DataLoader with a local cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def load_dataset(
        self,
        dataset_name: str,
        subset: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[List[str], np.ndarray]:
        """Returns the dataset split, utilizing local cache and deterministic subsampling if specified."""
        cache_file = self.cache_dir / f"{dataset_name}_{subset}.pkl"

        if cache_file.exists():
            try:
                logger.info(f"Loading {dataset_name} ({subset}) from cache...")
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                texts, labels = data["texts"], data["labels"]
            except Exception as e:
                # Corrupt or truncated cache file — delete and re-download.
                logger.warning(
                    f"Cache file {cache_file} is unreadable ({type(e).__name__}: {e}). "
                    f"Deleting and re-downloading."
                )
                cache_file.unlink(missing_ok=True)
                texts, labels = self._download_and_cache(
                    dataset_name, subset, cache_file
                )
        else:
            texts, labels = self._download_and_cache(dataset_name, subset, cache_file)

        # Apply deterministic subsampling using a dedicated RandomState to ensure ablation consistency
        if max_samples is not None and len(texts) > max_samples:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = labels[indices]

        return texts, labels

    def _download_and_cache(
        self, dataset_name: str, subset: str, cache_file: Path
    ) -> Tuple[List[str], np.ndarray]:
        """Downloads the dataset and persists it to disk as a pickle file."""
        logger.info(f"Downloading {dataset_name} ({subset})...")
        texts, labels = self._download_dataset(dataset_name, subset)
        with open(cache_file, "wb") as f:
            pickle.dump({"texts": texts, "labels": labels}, f)
        logger.info(f"Cached {len(texts)} samples to {cache_file}")
        return texts, labels

    def _download_dataset(
        self, dataset_name: str, subset: str
    ) -> Tuple[List[str], np.ndarray]:
        """Routes dataset requests to the appropriate Hugging Face or scikit-learn loader."""
        if dataset_name == "20newsgroups":
            return self._load_20newsgroups(subset)
        elif dataset_name == "imdb":
            return self._load_imdb(subset)
        elif dataset_name == "ag_news":
            return self._load_ag_news(subset)
        elif dataset_name == "banking77":
            return self._load_banking77(subset)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    @staticmethod
    def _load_20newsgroups(subset: str) -> Tuple[List[str], np.ndarray]:
        """Loads 20 Newsgroups, stripping metadata to prevent spurious feature correlation."""
        data = fetch_20newsgroups(
            subset=subset, remove=("headers", "footers", "quotes")
        )
        return data.data, np.array(data.target)

    @staticmethod
    def _load_imdb(subset: str) -> Tuple[List[str], np.ndarray]:
        """Load IMDb sentiment dataset."""
        dataset = load_dataset("imdb", split=subset)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    @staticmethod
    def _load_ag_news(subset: str) -> Tuple[List[str], np.ndarray]:
        """Load AG News dataset."""
        dataset = load_dataset("ag_news", split=subset)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    @staticmethod
    def _load_banking77(subset: str) -> Tuple[List[str], np.ndarray]:
        """Load Banking77 dataset."""
        dataset = load_dataset("banking77", split=subset)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    @staticmethod
    def get_dataset_info(dataset_name: str) -> dict:
        """Return metadata dict for a supported dataset."""
        return _DATASET_INFO.get(dataset_name, {"description": "Unknown dataset"})
