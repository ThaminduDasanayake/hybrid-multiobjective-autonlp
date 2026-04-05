"""
THE LIBRARIAN — utils/data_loader.py
======================================
This module handles dataset loading for all AutoML jobs and ablation studies.
Its two responsibilities are:

1. Fetching datasets from their sources (Hugging Face Hub or sklearn's built-in
   datasets) the first time they are requested.

2. Caching them as pickle files on disk so that subsequent runs — including every
   ablation study variant — load instantly without hitting the network again.

The cache is stored under backend/data/ and is persistent across server restarts.
Each dataset + split combination gets its own file (e.g., imdb_train.pkl).
If a cache file is corrupt or truncated, it is automatically deleted and re-downloaded.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups

from utils.logger import get_logger

logger = get_logger("data_loader")

# Dataset metadata dict defined once at module level rather than inside
# get_dataset_info() so it isn't reconstructed on every call.
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
    """Loads and locally caches benchmark NLP datasets.

    Supported datasets and their characteristics:
      - 20newsgroups: 20-class document classification, ~18,000 train samples
      - imdb:         Binary sentiment (positive/negative), 25,000 train samples
      - ag_news:      4-class news categorisation, 120,000 train samples
      - banking77:    77-class intent classification, ~10,000 train samples
    """

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
        """Return (texts, labels) for the requested dataset split, loading from cache if available.

        Args:
            dataset_name: One of "20newsgroups", "imdb", "ag_news", "banking77".
            subset: Dataset split — "train" or "test".
            max_samples: If set, subsample the dataset to at most this many examples.
            seed: Random seed used for the subsampling step (independent of GA/BO seed).
        """
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
                texts, labels = self._download_and_cache(dataset_name, subset, cache_file)
        else:
            texts, labels = self._download_and_cache(dataset_name, subset, cache_file)

        # Subsampling: if max_samples is set and the dataset is larger, draw a
        # reproducible random subset. Using a seeded RandomState (independent of the
        # GA/BO seed) means the same subset is used for both the main job and all its
        # ablation studies, ensuring fair apples-to-apples comparisons.
        if max_samples is not None and len(texts) > max_samples:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = labels[indices]

        return texts, labels

    def _download_and_cache(
        self, dataset_name: str, subset: str, cache_file: Path
    ) -> Tuple[List[str], np.ndarray]:
        """Download a dataset, persist it to cache_file, and return (texts, labels)."""
        logger.info(f"Downloading {dataset_name} ({subset})...")
        texts, labels = self._download_dataset(dataset_name, subset)
        with open(cache_file, "wb") as f:
            pickle.dump({"texts": texts, "labels": labels}, f)
        logger.info(f"Cached {len(texts)} samples to {cache_file}")
        return texts, labels

    def _download_dataset(
        self, dataset_name: str, subset: str
    ) -> Tuple[List[str], np.ndarray]:
        """Download a dataset from Hugging Face or sklearn and return (texts, labels)."""
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

    def _load_20newsgroups(self, subset: str) -> Tuple[List[str], np.ndarray]:
        """Load 20 Newsgroups dataset.

        headers, footers, and quotes are stripped to prevent the model from
        learning from metadata artefacts rather than the actual article content.
        This is the standard preprocessing for this benchmark in the literature.
        """
        data = fetch_20newsgroups(
            subset=subset, remove=("headers", "footers", "quotes")
        )
        return data.data, np.array(data.target)

    def _load_imdb(self, subset: str) -> Tuple[List[str], np.ndarray]:
        """Load IMDb sentiment dataset."""
        dataset = load_dataset("imdb", split=subset)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    def _load_ag_news(self, subset: str) -> Tuple[List[str], np.ndarray]:
        """Load AG News dataset."""
        dataset = load_dataset("ag_news", split=subset)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    def _load_banking77(self, subset: str) -> Tuple[List[str], np.ndarray]:
        """Load Banking77 dataset."""
        dataset = load_dataset("banking77", split=subset)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    def get_dataset_info(self, dataset_name: str) -> dict:
        """Return metadata dict for a supported dataset."""
        return _DATASET_INFO.get(dataset_name, {"description": "Unknown dataset"})
