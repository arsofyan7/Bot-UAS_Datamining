"""
================================================================================
MODUL: src/feature_extraction.py
DESKRIPSI: Ekstraksi Fitur Teks Numerik (TF-IDF & N-Gram Feature Engineering)
================================================================================

Tujuan Ilmiah / Konsep Teori:
-----------------------------
Model Machine Learning (seperti Naive Bayes, Logistic Regression, dan SVM) hanya
dapat menerima masukan berupa matriks numerik, bukan teks langsung.

Modul ini bertugas mengubah teks string menjadi vektor berbobot statistik:
1. TF (Term Frequency): Mengukur seberapa sering suatu kata muncul di kalimat tertentu.
2. IDF (Inverse Document Frequency): Memberikan penalti bobot pada kata yang muncul
   di hampir semua dokumen (karena tidak unik), dan memberikan bobot tinggi pada kata
   kunci yang langka dan spesifik.
3. N-Gram (Unigram + Bigram): Menggabungkan kata tunggal ('biaya', 'pendaftaran')
   dan pasangan 2 kata ('biaya pendaftaran') untuk menangkap makna kontekstual frasa.
4. Sublinear TF: Menerapkan penskalaan logaritmik 1 + log(TF) agar frekuensi kata
   yang sangat sering muncul tidak mendominasi secara tidak proporsional.
"""

import os
from typing import Tuple, Union, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureExtractor:
    """
    Wrapper ekstraktor fitur berbasis Scikit-Learn dengan dukungan serialisasi model.
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
        """
        Inisialisasi konfigurasi vectorizer:
        
        Args:
            vectorizer_type: 'tfidf' (bobot statistik) atau 'count' (frekuensi murni).
            ngram_range: (1, 1) untuk unigram murni, (1, 2) untuk unigram + bigram.
            max_features: Batas maksimal kosakata teratas untuk mencegah 'curse of dimensionality'.
            min_df: Frekuensi dokumen minimal agar sebuah token masuk ke kosakata.
            max_df: Ambang batas frekuensi dokumen maksimal (mengabaikan kata yang terlalu sering muncul).
            sublinear_tf: Menggunakan rumus 1 + log(tf) untuk meredam skewness frekuensi.
        """
        self.vectorizer_type = vectorizer_type.lower()
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        # Inisialisasi Scikit-Learn Vectorizer yang sesuai
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
        """
        Mempelajari kosakata (vocabulary) dan menghitung nilai IDF dari kumpulan data latih (X_train).
        """
        self.vectorizer.fit(X_train)
        return self

    def transform(self, X: List[str]):
        """
        Mentransformasikan teks masukan menjadi matriks sparse TF-IDF berdasarkan kosakata yang telah dipelajari.
        """
        return self.vectorizer.transform(X)

    def fit_transform(self, X_train: List[str]):
        """
        Menggabungkan proses fit dan transform dalam satu langkah komputasi yang efisien.
        """
        return self.vectorizer.fit_transform(X_train)

    def get_feature_names(self) -> List[str]:
        """
        Mengembalikan daftar seluruh nama token kosakata yang telah dipelajari oleh vectorizer.
        """
        return self.vectorizer.get_feature_names_out().tolist()

    def save_vectorizer(self, filepath: str) -> None:
        """
        Menyimpan objek vectorizer yang telah di-fit ke media penyimpanan (disk) menggunakan joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)

    @classmethod
    def load_vectorizer(cls, filepath: str) -> "FeatureExtractor":
        """
        Memuat objek vectorizer dari berkas joblib di disk.
        """
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
    """
    Fungsi pembantu (factory function) untuk membuat instans Vectorizer secara ringkas.
    """
    extractor = FeatureExtractor(
        vectorizer_type=method,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    return extractor.vectorizer
