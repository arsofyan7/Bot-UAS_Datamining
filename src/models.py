"""
================================================================================
MODUL: src/models.py
DESKRIPSI: Model Machine Learning & Pipeline Klasifikasi Intent Terkalibrasi
================================================================================

Tujuan Ilmiah / Konsep Teori:
-----------------------------
Modul ini mengelola inisialisasi algoritma Supervised Multi-Class Classification:

1. Multinomial Naive Bayes (MNB):
   - Model probabilistik generatif berbasis Teorema Bayes: P(C|X) ~ P(C) * Prod(P(x_i|C)).
   - Memiliki parameter Laplace Smoothing (alpha=1.0) untuk menangani kata yang belum
     pernah terlihat sebelumnya (zero probability problem).

2. Logistic Regression (LogReg):
   - Model diskriminatif linear yang menghitung log-odds dan menggunakan fungsi Softmax
     untuk menghasilkan distribusi probabilitas kelas: P(y=c|X) = exp(w^T X) / sum(exp(w_k^T X)).

3. Linear Support Vector Machine (Linear SVM) + CalibratedClassifierCV:
   - SVM mencari bidang hiper (hyperplane) dengan margin jarak terbesar yang memisahkan
     antar-kelas data (Maximum Margin Classifier).
   - Masalah: Linear SVM murni hanya menghasilkan jarak margin (decision_function), bukan
     probabilitas keyakinan (confidence).
   - Solusi Riset: Linear SVM dibungkus dengan metode Platt Scaling via CalibratedClassifierCV
     menggunakan 3-Fold Cross-Validation. Metode ini memetakan nilai margin ke fungsi Sigmoid
     terkalibrasi: P(y=1 | f(x)) = 1 / (1 + exp(A*f(x) + B)).
   - Keunggulan: Menghasilkan confidence score yang sangat tajam (>75-90% untuk kalimat yang cocok
     dan <20-30% untuk pertanyaan di luar domain), sehingga sangat ideal untuk deteksi Out-of-Scope!
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
    Factory function untuk membuat instans model Machine Learning.

    Args:
        model_type: Pilihan model ('multinomial_nb', 'logistic_regression', atau 'linear_svm').
        random_state: Bilangan acak untuk replikasi riset.
        **kwargs: Parameter tambahan untuk model.

    Returns:
        Instans estimator Scikit-Learn.
    """
    model_type = model_type.lower()

    if model_type in ["multinomial_nb", "naive_bayes", "mnb"]:
        # Multinomial Naive Bayes dengan smoothing alpha
        alpha = kwargs.get("alpha", 1.0)
        return MultinomialNB(alpha=alpha)

    elif model_type in ["logistic_regression", "logreg", "lr"]:
        # Logistic Regression dengan L2 Regularization (Ridge)
        c = kwargs.get("C", 1.0)
        max_iter = kwargs.get("max_iter", 1000)
        return LogisticRegression(C=c, max_iter=max_iter, random_state=random_state)

    elif model_type in ["linear_svm", "svm", "svc"]:
        # Linear SVM yang dikalibrasi Platt Scaling (CalibratedClassifierCV)
        c = kwargs.get("C", 1.0)
        base_svm = LinearSVC(C=c, random_state=random_state, max_iter=3000)
        return CalibratedClassifierCV(estimator=base_svm, cv=kwargs.get("cv", 3))

    else:
        raise ValueError(f"Model '{model_type}' tidak didukung. Pilih: multinomial_nb, logistic_regression, linear_svm.")


def get_classifier(model_name: str, random_state: int = 42, **kwargs) -> Any:
    """Alias untuk fungsi get_model (backward-compatibility)."""
    return get_model(model_name, random_state=random_state, **kwargs)


def build_pipeline(vectorizer: Any, classifier: Any) -> Pipeline:
    """
    Fungsi pembantu menyatukan vectorizer dan classifier ke dalam satu Scikit-Learn Pipeline.
    """
    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])


class IntentClassifierPipeline:
    """
    Kelas Wrapper Terpadu yang menyatukan ekstraksi fitur TF-IDF dan model klasifikasi.
    
    Menyediakan fungsionalitas inferensi end-to-end, estimasi probabilitas keyakinan
    (confidence score), deteksi Out-of-Scope (OOS), serta penyimpanan/pemuatan model via Joblib.
    """

    def __init__(self, vectorizer: Any = None, classifier: Any = None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.classes_: Optional[np.ndarray] = None
        self.pipeline: Optional[Pipeline] = None

        if vectorizer is not None and classifier is not None:
            self._init_pipeline()

    def _init_pipeline(self):
        """Membentuk objek Scikit-Learn Pipeline internal."""
        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", self.classifier)
        ])

    def fit(self, X_train: List[str], y_train: List[str]):
        """
        Melatih seluruh pipeline (fit TF-IDF pada teks dan fit model pada vektor fitur).
        """
        if self.pipeline is None:
            self._init_pipeline()
            
        self.pipeline.fit(X_train, y_train)
        
        # Simpan daftar label kelas unik yang dipelajari
        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "classes_"):
            self.classes_ = clf.classes_
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """
        Memprediksi label kelas intent yang memiliki probabilitas tertinggi.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline belum diinisialisasi atau dilatih.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Menghasilkan distribusi probabilitas (0.0 s/d 1.0) untuk setiap kelas intent.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline belum diinisialisasi atau dilatih.")

        clf = self.pipeline.named_steps["classifier"]
        vec = self.pipeline.named_steps["vectorizer"]

        X_vec = vec.transform(X)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X_vec)
        elif hasattr(clf, "decision_function"):
            # Fallback jika model menggunakan decision_function tanpa CalibratedClassifierCV
            df = clf.decision_function(X_vec)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            raise AttributeError("Classifier tidak mendukung estimasi probabilitas.")

    def predict_with_confidence(self, X: List[str], oos_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Logika Inferensi Cerdas & Penyaring Out-of-Scope (OOS Gate):
        
        1. Menghitung probabilitas untuk seluruh kelas intent.
        2. Mencari nilai probabilitas tertinggi: max_prob = max(probabilities).
        3. Jika max_prob < oos_threshold (misal < 50%), sistem menolak prediksi
           dan menandai intent sebagai 'OOS_REJECTED'.
        4. Jika max_prob >= oos_threshold, sistem menerima prediksi sebagai In-Scope.
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
        """
        Menyimpan keseluruhan objek pipeline (termasuk vectorizer dan bobot model) ke disk via Joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "IntentClassifierPipeline":
        """
        Memuat model tersimpan dari disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File model tidak ditemukan di: {filepath}")
        loaded = joblib.load(filepath)
        return loaded
