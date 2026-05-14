"""
train.py
--------
Main entry point for the text classification pipeline.
Run from the project root: python scripts/train.py
"""

import argparse
import logging
import sys

import pandas as pd

from src.artifacts import load_artifacts, predict, save_artifacts
from src.config import DEFAULT_MODEL_DIR
from src.data import load_dev, load_test, load_train
from src.features import build_vectorizer
from src.model import evaluate_model, train_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text classification pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory to save / load model artifacts.",
    )
    parser.add_argument(
        "--predict",
        metavar="TEXT",
        default=None,
        help="Skip training; load saved model and classify TEXT.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def run_training_pipeline(model_dir: str) -> None:
    train_df = load_train()
    dev_df   = load_dev()
    test_df  = load_test()

    train_dev_df = pd.concat([train_df, dev_df], ignore_index=True)

    logger.info("Train+Dev: %d rows | Test: %d rows", len(train_dev_df), len(test_df))

    vectorizer = build_vectorizer()

    logger.info("Fitting TF-IDF vectorizer …")
    X_train_dev = vectorizer.fit_transform(train_dev_df["text"])
    X_test      = vectorizer.transform(test_df["text"])

    y_train_dev = train_dev_df["label"]
    y_test      = test_df["label"]

    clf = train_model(X_train_dev, y_train_dev)
    evaluate_model(clf, X_test, y_test)
    save_artifacts(vectorizer, clf, model_dir=model_dir)

    logger.info("Training complete. Model saved to '%s'.", model_dir)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def run_prediction(text: str, model_dir: str) -> None:
    vectorizer, clf = load_artifacts(model_dir=model_dir)
    prediction = predict(text, vectorizer, clf)
    print(f"\nInput : {text}")
    print(f"Label : {prediction[0]}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.predict:
        run_prediction(args.predict, args.model_dir)
    else:
        run_training_pipeline(args.model_dir)


if __name__ == "__main__":
    main()
