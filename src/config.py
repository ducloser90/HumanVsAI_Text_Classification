"""
config.py
---------
Central configuration for the text classification pipeline.
All tunable parameters and path defaults live here.
"""

import os

# ---------------------------------------------------------------------------
# Google Drive source URLs
# ---------------------------------------------------------------------------

GDRIVE_URLS: dict[str, str] = {
    "train": "https://drive.google.com/uc?id=1HeCgnLuDoUHhP-2OsTSSC3FXRLVoI6OG",
    "dev":   "https://drive.google.com/uc?id=1e_G-9a66AryHxBOwGWhriePYCCa4_29e",
    "test":  "https://drive.google.com/uc?id=1-TN7sfSK1BuYHXlqxHHfwjEIE0JfarPk",
}

# ---------------------------------------------------------------------------
# Default file paths
# ---------------------------------------------------------------------------
# NOTE: Train/dev/test paths are managed inside data.py.
# Only the model artifact directory is needed here.

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_DIR:   str = os.path.join(BASE_DIR, "saved_model")
DEFAULT_DATA_DIR:    str = os.path.join(BASE_DIR, "data")

VECTORIZER_FILENAME: str = "vectorizer.pkl"
MODEL_FILENAME:      str = "rf_model.pkl"

# ---------------------------------------------------------------------------
# Vectorizer hyper-parameters
# ---------------------------------------------------------------------------

WORD_TFIDF_CONFIG: dict = {
    "lowercase":    False,
    "max_features": 20_000,
    "ngram_range":  (1, 2),
}

CHAR_TFIDF_CONFIG: dict = {
    "analyzer":     "char_wb",
    "ngram_range":  (3, 5),
    "max_features": 10_000,
    "lowercase":    False,
}

# ---------------------------------------------------------------------------
# Random Forest hyper-parameters
# ---------------------------------------------------------------------------

RF_CONFIG: dict = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs":       -1,
}
