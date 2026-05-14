"""
model.py
--------
Training and evaluation logic for the Random Forest text classifier.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from .config import RF_CONFIG

logger = logging.getLogger(__name__)


def train_model(
    X_train: spmatrix,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Fit a Random Forest classifier on the provided feature matrix."""
    logger.info("Training RandomForestClassifier with config: %s", RF_CONFIG)
    clf = RandomForestClassifier(**RF_CONFIG)
    clf.fit(X_train, y_train)
    logger.info("Training complete.")
    return clf


def evaluate_model(
    clf: RandomForestClassifier,
    X_test: spmatrix,
    y_test: pd.Series,
) -> Tuple[dict, str]:
    """Evaluate the classifier on the test set and log the results."""
    logger.info("Evaluating model on test set (%d samples) …", len(y_test))
    y_pred: np.ndarray = clf.predict(X_test)

    metrics: dict = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall":    recall_score(y_test, y_pred, average="binary"),
        "f1":        f1_score(y_test, y_pred, average="binary"),
    }
    report: str = classification_report(y_test, y_pred)

    logger.info("── Test Set Metrics ──────────────────────")
    for name, value in metrics.items():
        logger.info("  %-10s: %.4f", name.capitalize(), value)
    logger.info("── Classification Report ─────────────────\n%s", report)

    return metrics, report
