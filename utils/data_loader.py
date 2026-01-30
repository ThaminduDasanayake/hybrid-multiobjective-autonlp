from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)


def load_imdb(n_samples=None):
    cache_dir = DATA_ROOT / "imdb"
    dataset = load_dataset("imdb", cache_dir=str(cache_dir))

    X = dataset["train"]["text"]
    y = dataset["train"]["label"]

    if n_samples:
        # If n_samples is requested, we can just take a smaller slice.
        # But for better distribution, we use train_test_split (stratified) to get a small subset.
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, stratify=y, random_state=42
        )
        # We don't need to split further because we return X, y directly as "train"
        # However, the original code returned a split. Let's respect the return signature.
        # The original code did: return train_test_split(..., train_size=0.5, ...)
        
        # If we want a SMALL training set for Dev Mode:
        return X, None, y, None # We just return the subsampled data as "train" part.
        
        # Wait, the original code returns train_test_split(X, y, train_size=0.5, ...)
        # which returns: X_train, X_test, y_train, y_test.
        # But load_selected_dataset only unpacks 4 values: X_train, _, y_train, _
        # So we need to return 4 values.
        
    return train_test_split(X, y, train_size=0.5, stratify=y, random_state=42)


def load_ag_news(n_samples=None):
    cache_dir = DATA_ROOT / "ag_news"
    dataset = load_dataset("ag_news", cache_dir=str(cache_dir))

    texts = [
        t + " " + d
        for t, d in zip(dataset["train"]["title"], dataset["train"]["description"])
    ]
    labels = dataset["train"]["label"]

    if n_samples:
        X, _, y, _ = train_test_split(
            texts, labels, train_size=n_samples, stratify=labels, random_state=42
        )
        return X, None, y, None

    return train_test_split(
        texts, labels, train_size=0.5, stratify=labels, random_state=42
    )


def load_banking77(n_samples=None):
    cache_dir = DATA_ROOT / "banking77"
    dataset = load_dataset("banking77", cache_dir=str(cache_dir))

    X = dataset["train"]["text"]
    y = dataset["train"]["label"]

    if n_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, stratify=y, random_state=42
        )
        return X, None, y, None

    return train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
