from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)


def load_imdb():
    cache_dir = DATA_ROOT / "imdb"
    dataset = load_dataset("imdb", cache_dir=str(cache_dir))

    X = dataset["train"]["text"]
    y = dataset["train"]["label"]

    return train_test_split(X, y, train_size=0.5, stratify=y, random_state=42)


def load_ag_news():
    cache_dir = DATA_ROOT / "ag_news"
    dataset = load_dataset("ag_news", cache_dir=str(cache_dir))

    texts = [
        t + " " + d
        for t, d in zip(dataset["train"]["title"], dataset["train"]["description"])
    ]
    labels = dataset["train"]["label"]

    return train_test_split(
        texts, labels, train_size=0.5, stratify=labels, random_state=42
    )


def load_banking77():
    cache_dir = DATA_ROOT / "banking77"
    dataset = load_dataset("banking77", cache_dir=str(cache_dir))

    X = dataset["train"]["text"]
    y = dataset["train"]["label"]

    return train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
