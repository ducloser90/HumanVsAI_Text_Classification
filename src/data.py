"""
data.py
-------
Handles loading of local JSONL datasets.
"""

import logging
import os

import pandas as pd

from .config import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

TRAIN_FILE = "train_data.jsonl"
DEV_FILE   = "dev_data.jsonl"
TEST_FILE  = "test_data.jsonl"


def load_data(path: str) -> pd.DataFrame:
    """Load a JSONL file and return a DataFrame with 'text' and 'label' columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: '{path}'")

    logger.info("Loading dataset from '%s' …", path)

    try:
        df = pd.read_json(path, lines=True)
    except ValueError as e:
        raise ValueError(f"Failed to parse JSONL file: {path}") from e

    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)

    if missing:
        raise KeyError(f"Missing columns in '{path}': {missing}")

    return df[["text", "label"]].copy()


def load_train() -> pd.DataFrame:
    return load_data(os.path.join(DEFAULT_DATA_DIR, TRAIN_FILE))


def load_dev() -> pd.DataFrame:
    return load_data(os.path.join(DEFAULT_DATA_DIR, DEV_FILE))


def load_test() -> pd.DataFrame:
    return load_data(os.path.join(DEFAULT_DATA_DIR, TEST_FILE))
