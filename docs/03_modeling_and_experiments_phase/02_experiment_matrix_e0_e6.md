# 🔬 02. Matriks Eksperimen Riset (E0–E6)

Dokumen ini menjelaskan rancangan, metodologi, dan tujuan ilmiah dari masing-masing eksperimen yang dijalankan oleh skrip [scripts/run_experiments.py](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/scripts/run_experiments.py).

---

## 📑 Rincian Eksperimen E0 s/d E6

```text
E0 (Baseline)
    │  (MNB + TF-IDF Unigram + Minimal Preprocessing)
    ▼
E1 (Preprocessing Ablation)
    │  (MNB + TF-IDF Unigram + Full Preprocessing: Slang & Stopwords)
    ▼
E2 (Model Comparison)
    │  (MNB vs Logistic Regression vs Calibrated Linear SVM)
    ▼
E3 (N-Gram Feature Ablation)
    │  (Unigram (1,1) vs Unigram+Bigram (1,2))
    ▼
E4 (Cross-Validation Stability)
    │  (5-Fold Stratified CV pada kandidat model terbaik)
    ▼
E5 (Robustness Evaluation)
    │  (Pengujian variasi kalimat informal, slang berat, & typo)
    ▼
E6 (Out-of-Scope Thresholding)
       (Evaluasi penolakan pertanyaan di luar domain pada threshold 0.40 - 0.70)
```

---

### E0: Baseline Model
- **Tujuan:** Menentukan patokan performa awal (*lower bound*) sistem.
- **Konfigurasi:** Multinomial Naive Bayes + TF-IDF Unigram + Lowercase saja (tanpa penghapusan tanda baca kompleks / tanpa kamus slang).

### E1: Preprocessing Ablation
- **Tujuan:** Mengukur kontribusi nyata dari pipeline pra-pemrosesan teks Bahasa Indonesia (normalisasi slang + pembersihan tanda baca).
- **Pengukuran:** Menghitung selisih $\Delta \text{Macro F1} = \text{F1}_{E1} - \text{F1}_{E0}$.

### E2: Model Architecture Comparison
- **Tujuan:** Membandingkan 3 arsitektur pembelajaran terawasi (*MultinomialNB*, *LogisticRegression*, *LinearSVM*) pada kondisi fitur yang identik.
- **Kriteria Pemilihan:** Model dengan Macro F1 tertinggi dan kalibrasi probabilitas paling tajam dipilih untuk tahap selanjutnya.

### E3: N-Gram Feature Ablation
- **Tujuan:** Mengetahui apakah penambahan fitur dua kata (*Bigram*) membantu model membedakan intent yang memiliki kata tumpang tindih.

### E4: 5-Fold Stratified Cross-Validation
- **Tujuan:** Memvalidasi bahwa performa model bukan kebetulan (*overfitting* pada 1 pembagian data tertentu).
- **Output:** Rata-rata (*Mean*) dan standar deviasi (*Std*) skor Macro F1 di 5 lipatan data.

### E5: Robustness Evaluation (Uji Ketahanan)
- **Tujuan:** Menguji performa inferensi pada variasi penulisan pengguna di dunia nyata yang penuh singkatan ekstrem (*contoh: "brp bya dftr mhs baru?"*).

### E6: Out-of-Scope (OOS) Threshold Analysis
- **Tujuan:** Menemukan nilai batas ambang probabilitas optimal untuk menolak pertanyaan di luar topik (*Out-of-Scope rejection rate*).
