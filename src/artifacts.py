"""
artifacts.py
------------
Saving, loading, and running inference with trained pipeline artifacts.
"""

import logging
import os
from typing import List, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion

from .config import DEFAULT_MODEL_DIR, MODEL_FILENAME, VECTORIZER_FILENAME

logger = logging.getLogger(__name__)


def save_artifacts(
    vectorizer: FeatureUnion,
    clf: RandomForestClassifier,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> None:
    """Persist the fitted vectorizer and classifier to disk using joblib."""
    os.makedirs(model_dir, exist_ok=True)

    vec_path   = os.path.join(model_dir, VECTORIZER_FILENAME)
    model_path = os.path.join(model_dir, MODEL_FILENAME)

    joblib.dump(vectorizer, vec_path)
    logger.info("Vectorizer saved → '%s'", vec_path)

    joblib.dump(clf, model_path)
    logger.info("Model saved      → '%s'", model_path)


def load_artifacts(
    model_dir: str = DEFAULT_MODEL_DIR,
) -> tuple[FeatureUnion, RandomForestClassifier]:
    """Load vectorizer and classifier from disk."""
    vec_path   = os.path.join(model_dir, VECTORIZER_FILENAME)
    model_path = os.path.join(model_dir, MODEL_FILENAME)

    for path in (vec_path, model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: '{path}'")

    logger.info("Loading vectorizer from '%s' …", vec_path)
    vectorizer: FeatureUnion = joblib.load(vec_path)

    logger.info("Loading model from '%s' …", model_path)
    clf: RandomForestClassifier = joblib.load(model_path)

    logger.info("Artifacts loaded successfully.")
    return vectorizer, clf


def predict(
    texts: Union[str, List[str]],
    vectorizer: FeatureUnion,
    clf: RandomForestClassifier,
) -> np.ndarray:
    """Run inference on one or more text strings."""
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        raise ValueError("'texts' must contain at least one string.")

    logger.info("Running inference on %d sample(s) …", len(texts))
    X = vectorizer.transform(texts)
    predictions: np.ndarray = clf.predict(X)
    return predictions
