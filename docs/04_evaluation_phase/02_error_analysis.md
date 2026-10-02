# 🔍 02. Analisis Kesalahan (Error Analysis & Confusion Matrix)

Dokumen ini menjelaskan cara membaca dan menginterpretasikan visualisasi Confusion Matrix di [reports/figures/confusion_matrix.png](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/reports/figures/confusion_matrix.png) serta log metrik terstruktur di [reports/metrics/experiment_results.json](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/reports/metrics/experiment_results.json).

---

## 1. Membaca Heatmap Confusion Matrix

Matriks Konfusi (*Confusion Matrix*) memetakan frekuensi label sebenarnya (*Ground Truth / True Label*) pada sumbu vertikal (Y) terhadap label hasil prediksi model (*Predicted Label*) pada sumbu horizontal (X):

- **Diagonal Utama (Kiri-Atas ke Kanan-Bawah):** Menunjukkan jumlah prediksi yang benar (*True Positives*). Semakin gelap/biru pekat warna sel diagonal utama, semakin tinggi akurasi model pada kelas tersebut.
- **Sel di Luar Diagonal:** Menunjukkan kesalahan klasifikasi (*Misclassifications* / *False Positives* & *False Negatives*).

---

## 2. Pola Kesalahan Umum & Solusinya

| Gejala Kesalahan | Kemungkinan Penyebab | Tindakan Perbaikan |
| :--- | :--- | :--- |
| Dua intent saling tertukar (misal `syarat_masuk` diprediksi `biaya_pendaftaran`) | Adanya kata yang tumpang tindih (*overlap*) seperti kata "pendaftaran" | Tambahkan fitur N-Gram (Bigram) atau perkaya variasi frasa pembeda di data latih |
| Kalimat informal salah prediksi | Singkatan kata belum terdaftar di kamus slang | Tambahkan singkatan tersebut ke `DEFAULT_SLANG_DICT` di `src/preprocessing.py` |
| Pertanyaan in-scope ditolak OOS | Ambang batas (*OOS threshold*) diatur terlalu tinggi | Turunkan slider OOS threshold di Web UI (misal dari 0.70 ke 0.50) |
