"""
Machine Learning Models & Pipeline Module
=========================================
Factory and wrapper classes for Supervised Intent Classification models.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def get_model(model_type: str, random_state: int = 42, **kwargs) -> Any:
    """
    Factory function for classification models.

    Args:
        model_type: 'multinomial_nb', 'logistic_regression', or 'linear_svm'
        random_state: Seed for reproducibility
        **kwargs: Additional parameters passed to model
    """
    model_type = model_type.lower()

    if model_type in ["multinomial_nb", "naive_bayes", "mnb"]:
        alpha = kwargs.get("alpha", 1.0)
        return MultinomialNB(alpha=alpha)

    elif model_type in ["logistic_regression", "logreg", "lr"]:
        c = kwargs.get("C", 1.0)
        max_iter = kwargs.get("max_iter", 1000)
        return LogisticRegression(C=c, max_iter=max_iter, random_state=random_state)

    elif model_type in ["linear_svm", "svm", "svc"]:
        c = kwargs.get("C", 1.0)
        base_svm = LinearSVC(C=c, random_state=random_state, max_iter=3000)
        # CalibratedClassifierCV allows predict_proba for SVM
        return CalibratedClassifierCV(estimator=base_svm, cv=kwargs.get("cv", 3))

    else:
        raise ValueError(f"Model '{model_type}' tidak didukung. Pilih: multinomial_nb, logistic_regression, linear_svm.")


def get_classifier(model_name: str, random_state: int = 42, **kwargs) -> Any:
    """Alias for get_model for backward-compatibility."""
    return get_model(model_name, random_state=random_state, **kwargs)


def build_pipeline(vectorizer: Any, classifier: Any) -> Pipeline:
    """Helper to build standard scikit-learn Pipeline."""
    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])


class IntentClassifierPipeline:
    """
    End-to-end wrapper combining text vectorization and classification estimator
    with probability estimation, persistence, and confidence scoring.
    """

    def __init__(self, vectorizer: Any = None, classifier: Any = None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.classes_: Optional[np.ndarray] = None
        self.pipeline: Optional[Pipeline] = None

        if vectorizer is not None and classifier is not None:
            self._init_pipeline()

    def _init_pipeline(self):
        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", self.classifier)
        ])

    def fit(self, X_train: List[str], y_train: List[str]):
        """Fits vectorizer and classifier pipeline on text and intent labels."""
        if self.pipeline is None:
            self._init_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        # Save classes
        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "classes_"):
            self.classes_ = clf.classes_
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predicts intent labels for given text input."""
        if self.pipeline is None:
            raise ValueError("Pipeline belum diinisialisasi atau dilatih.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Computes class probability distribution for input text."""
        if self.pipeline is None:
            raise ValueError("Pipeline belum diinisialisasi atau dilatih.")

        clf = self.pipeline.named_steps["classifier"]
        vec = self.pipeline.named_steps["vectorizer"]

        X_vec = vec.transform(X)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X_vec)
        elif hasattr(clf, "decision_function"):
            # Softmax on decision function fallback
            df = clf.decision_function(X_vec)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            raise AttributeError("Classifier tidak mendukung estimasi probabilitas.")

    def predict_with_confidence(self, X: List[str], oos_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Predicts intent and confidence score, applying OOS threshold filtering.
        """
        probs = self.predict_proba(X)
        preds = self.predict(X)

        results = []
        for i, (pred, prob_dist) in enumerate(zip(preds, probs)):
            max_prob = float(np.max(prob_dist))
            is_oos = max_prob < oos_threshold

            results.append({
                "intent": "OOS_REJECTED" if is_oos else pred,
                "raw_intent": pred,
                "confidence": max_prob,
                "is_oos": is_oos,
                "probabilities": {cls: float(p) for cls, p in zip(self.classes_, prob_dist)} if self.classes_ is not None else {}
            })
        return results

    def save_model(self, filepath: str) -> None:
        """Saves entire pipeline to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "IntentClassifierPipeline":
        """Loads fitted pipeline from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File model tidak ditemukan di: {filepath}")
        loaded = joblib.load(filepath)
        return loaded
