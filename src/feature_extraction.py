"""
Feature Extraction Module
=========================
Extracts numerical representations from textual utterances (TF-IDF, N-grams).
"""

import os
from typing import Tuple, Union, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureExtractor:
    """
    TF-IDF Feature Extractor wrapper with serialization support.
    """

    def __init__(
        self,
        vectorizer_type: str = "tfidf",
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True
    ):
        self.vectorizer_type = vectorizer_type.lower()
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        if self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                sublinear_tf=self.sublinear_tf
            )
        elif self.vectorizer_type == "count":
            self.vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df
            )
        else:
            raise ValueError(f"Tipe vectorizer '{vectorizer_type}' tidak didukung. Pilih 'tfidf' atau 'count'.")

    def fit(self, X_train: List[str]):
        """Fits vectorizer vocabulary on training texts."""
        self.vectorizer.fit(X_train)
        return self

    def transform(self, X: List[str]):
        """Transforms texts to numerical feature matrix."""
        return self.vectorizer.transform(X)

    def fit_transform(self, X_train: List[str]):
        """Fits and transforms texts in one step."""
        return self.vectorizer.fit_transform(X_train)

    def get_feature_names(self) -> List[str]:
        """Returns vocabulary feature names."""
        return self.vectorizer.get_feature_names_out().tolist()

    def save_vectorizer(self, filepath: str) -> None:
        """Saves fitted vectorizer object to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)

    @classmethod
    def load_vectorizer(cls, filepath: str) -> "FeatureExtractor":
        """Loads fitted vectorizer object from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File vectorizer tidak ditemukan di: {filepath}")
        loaded_vec = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = loaded_vec
        return instance


def create_vectorizer(
    method: str = "tfidf",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 5000,
    min_df: int = 1,
    max_df: float = 1.0
) -> Union[TfidfVectorizer, CountVectorizer]:
    """Factory function for backward-compatibility."""
    extractor = FeatureExtractor(
        vectorizer_type=method,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    return extractor.vectorizer
