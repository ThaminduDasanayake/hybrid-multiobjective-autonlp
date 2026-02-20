"""Utilities for converting objects to Python-native or JSON-safe formats."""

import numpy as np


def to_python_type(val):
    """Convert a value to a native Python type (handles numpy types, dicts, lists)."""
    if isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, np.bool_):
        return bool(val)
    elif isinstance(val, np.str_):
        return str(val)
    elif isinstance(val, dict):
        return {k: to_python_type(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return type(val)(to_python_type(v) for v in val)
    return val


# Aliases for backward compatibility
to_json_safe = to_python_type
clean = to_python_type


def clean_params(params: dict) -> dict:
    """Clean parameter dict — convert numpy types and remove internal keys."""
    if not isinstance(params, dict):
        return params
    return {k: to_python_type(v) for k, v in params.items() if not k.startswith("_")}
