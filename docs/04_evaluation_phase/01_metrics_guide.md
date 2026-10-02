# 📈 01. Panduan Metrik Evaluasi (Macro F1 vs Akurasi)

Dokumen ini menjelaskan alasan ilmiah pemilihan **Macro F1-Score** sebagai metrik evaluasi utama dalam penelitian ini di [src/evaluation.py](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/src/evaluation.py).

---

## 1. Masalah Ketimpangan Kelas (*Class Imbalance*)

Pada dataset teks percakapan dunia nyata, jumlah contoh kalimat per intent jarang sekali berimbang (misal: intent `biaya_pendaftaran` memiliki 50 kalimat, sedangkan `kontak_layanan` hanya memiliki 5 kalimat).

Jika kita hanya mengukur **Akurasi (Accuracy)**:
$$\text{Accuracy} = \frac{\text{Jumlah Prediksi Benar}}{\text{Total Sampel}}$$
Model yang hanya mahir memprediksi kelas mayoritas akan tetap memperoleh nilai akurasi tinggi (misal 90%), meskipun kelas minoritas gagal diprediksi sama sekali.

---

## 2. Definisi Matematis Metrik Evaluasi

### A. Precision & Recall Per Kelas ($c$)
$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

### B. F1-Score Per Kelas ($c$)
Rata-rata harmonik antara Precision dan Recall:
$$\text{F1}_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### C. Macro F1-Score (Metrik Utama Riset)
Menghitung rata-rata sederhana dari skor F1 setiap kelas tanpa memandang jumlah sampel:
$$\text{Macro F1} = \frac{1}{K} \sum_{c=1}^K \text{F1}_c$$

> [!IMPORTANT]
> **Mengapa Macro F1 Unggul?**
> Macro F1 memperlakukan setiap kelas intent dengan bobot kepentingan yang setara (*equal weight*). Jika model gagal pada kelas minoritas, nilai Macro F1 akan langsung turun drastis, menjadikannya metrik yang paling adil dan objektif untuk evaluasi klasifikasi multi-class.
