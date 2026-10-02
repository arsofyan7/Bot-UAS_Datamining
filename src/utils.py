"""
Utilities and Helper Functions Module
====================================
Helper functions for configuration loading, random seed management, dataset loading,
and model persistence.
"""

import os
import random
import yaml
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        # Check relative to repo root
        alt_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42) -> None:
    """Sets global random seeds across random, numpy, and environment variables."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_seed(seed: int = 42) -> None:
    """Alias for seed_everything."""
    seed_everything(seed)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset CSV into a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        # Try finding in root
        alt_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset tidak ditemukan di path: {filepath}")

    df = pd.read_csv(filepath)
    return df


def save_model(model_obj: Any, filepath: str) -> None:
    """Saves model/pipeline object to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_obj, filepath)


def load_model(filepath: str) -> Any:
    """Loads model/pipeline object from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file tidak ditemukan di: {filepath}")
    return joblib.load(filepath)
