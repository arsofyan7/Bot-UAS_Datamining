# 🏛️ 01. Arsitektur Sistem & Metodologi Riset

## 1. Pendahuluan & Filosofi Desain

Proyek ini dibangun untuk menyelesaikan permasalahan **Intent Classification** (Klasifikasi Niat/Maksud Pengguna) pada teks percakapan Bahasa Indonesia. Sistem ini dirancang secara **Domain-Agnostic** (independen dari satu domain tertentu), sehingga seluruh arsitektur kode dapat dengan mudah ditransfer ke berbagai bidang (pendidikan, perbankan, kesehatan, e-commerce, pariwisata, dsb.) tanpa perlu merombak *core logic* program.

---

## 2. Metodologi Riset: Standar CRISP-DM

Pengembangan proyek ini mengikuti 6 tahapan standar industri **CRISP-DM (*Cross-Industry Standard Process for Data Mining*)**:

```text
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Business Understanding (Klasifikasi Intent Multi-Class)   │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. Data Understanding (Struktur 8 Kolom & Distribusi Split)  │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. Data Preparation (Indonesian NLP Preprocessing & TF-IDF) │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. Modeling (Eksperimen E0–E6: MNB, LogReg, Linear SVM)     │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 5. Evaluation (Prioritas Metrik Macro F1 & OOS Detection)   │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 6. Deployment (Antarmuka Interaktif Web UI Streamlit)       │
  └─────────────────────────────────────────────────────────────┘
```

### Tahap 1: Business Understanding
- **Tujuan Bisnis:** Membangun asisten virtual cerdas yang mampu mengidentifikasi kebutuhan spesifik pengguna dari kalimat masukan (*utterance*) secara akurat.
- **Tantangan Utama:** Variasi bahasa percakapan non-formal (*slang*, singkatan, *typo*) dan pertanyaan di luar domain (*Out-of-Scope*).

### Tahap 2: Data Understanding
- Data berupa pasangan kalimat percakapan (*utterance*) dan label intent (*target class*).
- Pembagian data dilakukan secara ketat (*Train, Validation, Test, dan Out-of-Scope Test*) untuk menguji generalisasi model.

### Tahap 3: Data Preparation
- Pembersihan teks Bahasa Indonesia secara bertahap (*Case Folding*, *Punctuation/Number Removal*, *Slang Normalization*, *Stopword Removal*, dan *Stemming*).
- Transformasi teks ke bentuk numerik menggunakan *Term Frequency-Inverse Document Frequency* (TF-IDF) dengan kombinasi fitur Unigram dan Bigram.

### Tahap 4: Modeling
- Pelatihan model pembelajaran terbimbing (*Supervised Learning*):
  - **Multinomial Naive Bayes (MNB):** Model probabilistik berbasis Teorema Bayes.
  - **Logistic Regression (LogReg):** Model linear berbasis fungsi logit/softmax.
  - **Linear Support Vector Machine (Linear SVM):** Model pencarian *hyperplane* optimal yang dikalibrasi (*CalibratedClassifierCV*) untuk menghasilkan estimasi probabilitas keyakinan.

### Tahap 5: Evaluation
- Penilaian performa menggunakan metrik **Macro F1-Score** sebagai metrik utama guna menghindari bias kelas minoritas.
- Evaluasi stabilitas dengan 5-Fold Stratified Cross-Validation dan analisis ambang batas penolakan Out-of-Scope (OOS).

### Tahap 6: Deployment
- Penyajian model dalam antarmuka Web UI Chatbot berbasis **Streamlit** dengan *real-time confidence badge* dan mekanisme penolakan OOS otomatis.

---

## 3. Diagram Alur Data End-to-End

```text
[ Input Pengguna: "brp bya dftr mhs baru?" ]
                     │
                     ▼
       ┌───────────────────────────┐
       │ 1. TextPreprocessor       │
       │    - Slang Normalization  │ -> "berapa biaya daftar mahasiswa baru"
       │    - Stopword & Cleaning  │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ 2. TF-IDF Feature Vector  │ -> Sparse Vector Matrix [1 x V]
       │    - Unigram + Bigram     │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ 3. Linear SVM (Calibrated)│
       │    - Predict Intent Class │ -> "biaya_pendaftaran"
       │    - Estimate Confidence  │ -> 77.17%
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ 4. OOS Decision Gate      │
       │    - Is Conf >= 50.0%?    │ -> YA (In-Scope)
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ 5. UI Response Generator  │
       │    - Render Bot Message   │ -> "💰 Biaya pendaftaran reguler Rp 350.000..."
       │    - Show Confidence Tag  │ -> 🎯 Intent: biaya_pendaftaran | Conf: 77.17%
       └───────────────────────────┘
```
