import streamlit as st
from utils.data_loader import load_imdb, load_ag_news, load_banking77


@st.cache_data
def load_selected_dataset(name, n_samples=None):
    """Loads dataset based on selection."""
    if name == "IMDb":
        X_train, _, y_train, _ = load_imdb(n_samples)
    elif name == "AG News":
        X_train, _, y_train, _ = load_ag_news(n_samples)
    elif name == "Banking77":
        X_train, _, y_train, _ = load_banking77(n_samples)
    else:
        raise ValueError("Unknown dataset")
    return X_train, y_train


def display_dataset_info(name, X, y):
    """Sidebar component to show dataset stats."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.info(
        f"**{name}**\n\n" f"Samples: {len(X)}\n\n" f"Classes: {len(set(y))}"
    )
