import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups


class DataLoader:
    """Loads and locally caches benchmark datasets for the AutoML system."""

    def __init__(self, cache_dir: str = "./data"):
        """Initialize the DataLoader with a local cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def load_dataset(
        self, dataset_name: str, subset: str = "train", max_samples: int = None
    ) -> Tuple[List[str], np.ndarray]:
        """Return (texts, labels) for the requested dataset split, loading from cache if available."""
        cache_file = self.cache_dir / f"{dataset_name}_{subset}.pkl"

        if cache_file.exists():
            print(f"Loading {dataset_name} from cache...")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                texts, labels = data["texts"], data["labels"]
        else:
            print(f"Downloading {dataset_name}...")
            texts, labels = self._download_dataset(dataset_name, subset)

            with open(cache_file, "wb") as f:
                pickle.dump({"texts": texts, "labels": labels}, f)

        if max_samples is not None and len(texts) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = labels[indices]

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
        """Load 20 Newsgroups dataset."""
        data = fetch_20newsgroups(
            subset=subset, remove=("headers", "footers", "quotes")
        )
        return data.data, np.array(data.target)

    def _load_imdb(self, subset: str) -> Tuple[List[str], np.ndarray]:
        """Load IMDb sentiment dataset."""
        split = "train" if subset == "train" else "test"
        dataset = load_dataset("imdb", split=split)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    def _load_ag_news(self, subset: str) -> Tuple[List[str], np.ndarray]:
        """Load AG News dataset."""
        split = "train" if subset == "train" else "test"
        dataset = load_dataset("ag_news", split=split)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    def _load_banking77(self, subset: str) -> Tuple[List[str], np.ndarray]:
        """Load Banking77 dataset."""
        split = "train" if subset == "train" else "test"
        dataset = load_dataset("banking77", split=split)
        texts = dataset["text"]
        labels = np.array(dataset["label"])
        return texts, labels

    def get_dataset_info(self, dataset_name: str) -> dict:
        """Return metadata dict for a supported dataset."""
        info = {
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

        return info.get(dataset_name, {"description": "Unknown dataset"})
