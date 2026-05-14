"""
features.py
-----------
Builds the TF-IDF FeatureUnion vectorizer (word n-grams + char n-grams).
"""

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .config import CHAR_TFIDF_CONFIG, WORD_TFIDF_CONFIG

logger = logging.getLogger(__name__)


def build_vectorizer() -> FeatureUnion:
    """Create a FeatureUnion of word-level and character-level TF-IDF vectorizers."""
    logger.info(
        "Building FeatureUnion: word TF-IDF %s + char TF-IDF %s",
        WORD_TFIDF_CONFIG["ngram_range"],
        CHAR_TFIDF_CONFIG["ngram_range"],
    )

    vectorizer = FeatureUnion(
        transformer_list=[
            ("word_tfidf", TfidfVectorizer(**WORD_TFIDF_CONFIG)),
            ("char_tfidf", TfidfVectorizer(**CHAR_TFIDF_CONFIG)),
        ]
    )
    return vectorizer
