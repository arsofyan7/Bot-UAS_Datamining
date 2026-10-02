"""
================================================================================
MODUL: src/evaluation.py
DESKRIPSI: Evaluasi Metrik Riset (Macro F1, Confusion Matrix, & OOS Analysis)
================================================================================

Tujuan Ilmiah / Konsep Teori:
-----------------------------
1. Mengapa Macro F1 adalah Metrik Utama (Bukan Akurasi)?
   - Pada masalah klasifikasi teks percakapan, distribusi data sering mengalami
     ketimpangan kelas (class imbalance).
   - Akurasi hanya menghitung total prediksi benar dibagi total sampel, sehingga
     model yang hanya menebak kelas mayoritas akan tetap memperoleh skor tinggi
     walaupun gagal memprediksi kelas minoritas.
   - Macro F1 menghitung skor F1 untuk setiap kelas secara independen, lalu mengambil
     rata-rata aritmatikanya secara setara (equal weighting). Hal ini memaksa model
     untuk berkinerja baik di SEMUA kelas tanpa terkecuali.

2. Confusion Matrix Heatmap:
   - Memetakan label ground truth (sumbu Y) vs label prediksi (sumbu X).
   - Diagonal utama (kiri-atas ke kanan-bawah) menunjukkan True Positives (prediksi tepat).
   - Sel di luar diagonal menunjukkan False Positives & False Negatives (kesalahan klasifikasi).

3. Out-of-Scope (OOS) Rejection Evaluation:
   - Mengukur seberapa andal model menolak pertanyaan di luar topik dengan memeriksa
     apakah skor keyakinan berada di bawah ambang batas (Confidence Cutoff).
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Menghitung metrik performa riset standar:
    - Macro F1: Rata-rata F1 unweighted (Metrik Utama).
    - Macro Precision: Rata-rata ketepatan prediksi per kelas.
    - Macro Recall: Rata-rata cakupan deteksi per kelas.
    - Accuracy: Proporsi total prediksi yang benar.
    - Weighted F1: Rata-rata F1 yang diboboti oleh jumlah sampel per kelas.
    """
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    }


def evaluate_intent_model(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Menghasilkan laporan evaluasi lengkap termasuk classification report dan confusion matrix.
    """
    metrics = calculate_metrics(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics


def plot_and_save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    output_path: str = "reports/figures/confusion_matrix.png",
    title: str = "Confusion Matrix - Intent Classification"
) -> str:
    """
    Menggambar visualisasi heatmap Confusion Matrix menggunakan Seaborn dan Matplotlib,
    lalu menyimpannya ke direktori gambar (reports/figures/) dengan resolusi 300 DPI.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig_size = max(6, len(labels) * 1.2)
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label (Hasil Prediksi Model)", fontsize=11, labelpad=8)
    plt.ylabel("True Label (Label Sebenarnya)", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def evaluate_oos_threshold(
    model: Any,
    X: List[str],
    y_true: List[str],
    threshold: float = 0.5,
    oos_label: str = "OOS_REJECTED"
) -> Dict[str, Any]:
    """
    Mengevaluasi kemampuan sistem dalam menolak pertanyaan Out-of-Scope (OOS).
    
    Jika probabilitas tertinggi model pada sebuah sampel < threshold, prediksi diubah
    menjadi 'OOS_REJECTED'. Fungsi ini menghitung tingkat penolakan (Rejection Rate).
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        preds = model.predict(X)
    else:
        raise AttributeError("Model harus mendukung method predict_proba.")

    max_probs = np.max(probs, axis=1)
    adjusted_preds = []

    # Filter ambang batas OOS
    for pred, max_prob in zip(preds, max_probs):
        if max_prob < threshold:
            adjusted_preds.append(oos_label)
        else:
            adjusted_preds.append(pred)

    total_samples = len(y_true)
    rejected_count = sum(1 for p in adjusted_preds if p == oos_label)
    rejection_rate = rejected_count / total_samples if total_samples > 0 else 0.0

    metrics = calculate_metrics(y_true, adjusted_preds)
    metrics["rejection_rate"] = float(rejection_rate)
    metrics["rejected_count"] = rejected_count
    metrics["threshold"] = threshold

    return {
        "metrics": metrics,
        "adjusted_predictions": adjusted_preds,
        "max_probabilities": max_probs.tolist()
    }
