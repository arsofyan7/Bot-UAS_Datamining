# 🧠 01. Arsitektur Model Machine Learning

Dokumen ini menjelaskan teori dan implementasi tiga algoritma *Supervised Learning* yang dievaluasi pada proyek ini di [src/models.py](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/src/models.py).

---

## 1. Multinomial Naive Bayes (MNB)

- **Dasar Teori:** Menggunakan Teorema Bayes dengan asumsi independensi fitur (*naive*):
  $$P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{i=1}^n P(x_i \mid C_k)$$
- **Kelebihan:** Sangat cepat dilatih, efisien memori, bekerja baik pada teks sederhana.
- **Keterbatasan:** Probabilitas prediksi pada kalimat pendek sering terbagi rata pada banyak kelas karena pengaruh *Laplace smoothing* ($\alpha=1.0$), sehingga nilai *confidence score* puncak cenderung tidak tajam.

---

## 2. Logistic Regression (Multinomial / Softmax)

- **Dasar Teori:** Memetakan kombinasi linear bobot fitur ke dalam distribusi probabilitas kelas menggunakan fungsi Softmax:
  $$P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{k=1}^K e^{\mathbf{w}_k^T \mathbf{x} + b_k}}$$
- **Kelebihan:** Menghasilkan batas keputusan linear yang stabil dan probabilitas terdistribusi lebih proporsional.
- **Regularisasi:** Menggunakan penalti L2 (*Ridge*) untuk mencegah *overfitting* pada vektor berdimensi tinggi.

---

## 3. Linear Support Vector Machine (Linear SVM) + Kalibrasi Probabilitas

- **Dasar Teori:** Menemukan *hyperplane* pemisah dengan margin terbesar (*maximum margin classifier*):
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
- **Kalibrasi Platt (`CalibratedClassifierCV`):**
  Linear SVM murni hanya menghasilkan nilai jarak margin (*decision function*), bukan nilai probabilitas 0 s/d 1. Oleh karena itu, model dibungkus dengan metode *Platt Scaling* via 3-Fold Cross-Validation menggunakan fungsi sigmoid logistik:
- **Kelebihan Utama:** Memberikan estimasi *confidence score* yang sangat tegas (di atas 75–90% untuk kalimat in-scope yang cocok) dan rendah (di bawah 20–30% untuk pertanyaan acak), menjadikannya **model terbaik** untuk filtering Out-of-Scope (OOS).

---

## 4. Mekanisme Seleksi Model (Turnamen Model vs Ensemble Voting)

### Bagaimana Sistem Memilih Algoritma Terbaik?
Sistem tidak menjalankan 3 algoritma saat user sedang chat. Sebaliknya, proses pemilihan model dilakukan pada tahap eksperimen (**E2 Model Comparison** di `scripts/run_experiments.py`):

1. **Evaluasi Terstandarisasi:** Ketiga model (MultinomialNB, Logistic Regression, Linear SVM) dilatih pada data latih yang sama dan diuji pada data uji yang sama.
2. **Kriteria Penilaian:**
   - Skor **Macro F1** pada data evaluasi independen.
   - Ketajaman kalibrasi probabilitas keyakinan (*confidence score sharpness*) untuk membedakan pertanyaan in-scope vs out-of-scope.
3. **Persistensi Model Juara (*Champion Model*):** Model yang memenangkan turnamen (Linear SVM) disimpan ke berkas biner tunggal `models/best_model.joblib`.
4. **Eksekusi di Production (Streamlit):** Web UI hanya memuat 1 model juara tersebut ke dalam RAM, sehingga respon balasan chat berlangsung instan (**< 50 milidetik**) dan hemat memori server.

