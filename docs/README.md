# 📚 Pusat Dokumentasi Proyek Intent Classification NLP

Selamat datang di repositori dokumentasi resmi untuk proyek penelitian **"Domain-Agnostic Intent Classification Chatbot Bahasa Indonesia"**. Dokumentasi ini disusun secara berjenjang (*hierarchical documentation*), mulai dari konsep arsitektur makro hingga detail teknis mikro setiap berkas kode.

---

## 🧭 Navigasi Dokumen

Klik pada tautan berkas `.md` di bawah untuk membaca dokumentasi modul terkait:

### 🏛️ 00. Overview & Arsitektur Makro
- 📄 [01_system_architecture.md](00_overview/01_system_architecture.md) — Arsitektur sistem makro, metodologi standar CRISP-DM, dan diagram alur data end-to-end.
- 📄 [02_project_structure.md](00_overview/02_project_structure.md) — Penjelasan fungsi dan tanggung jawab setiap folder serta berkas kode dalam repositori.
- 📄 [03_domain_adaptation_guide.md](00_overview/03_domain_adaptation_guide.md) — ⭐ **Panduan Lengkap 7 Langkah Adaptasi & Ganti Domain Baru** (tugas studi kasus UAS).

### 📊 01. Fase Dataset & Validasi
- 📄 [01_data_schema.md](01_data_phase/01_data_schema.md) — Spesifikasi 8 kolom standar dataset dan aturan pembagian data (*Train/Val/Test/OOS*).
- 📄 [02_data_validation.md](01_data_phase/02_data_validation.md) — Mekanisme kerja skrip validasi skema, null checks, dan pencegahan kebocoran data (*Data Leakage*).

### 🔤 02. Fase NLP Pipeline
- 📄 [01_preprocessing.md](02_nlp_pipeline_phase/01_preprocessing.md) — Tahapan pra-pemrosesan teks Bahasa Indonesia (Case Folding, Slang Normalization, Stopwords, Stemming).
- 📄 [02_feature_extraction.md](02_nlp_pipeline_phase/02_feature_extraction.md) — Konsep matematika ekstraksi fitur TF-IDF, N-Gram (Unigram vs Bigram), dan scaling sublinear TF.

### 🧠 03. Fase Modeling & Eksperimen
- 📄 [01_model_architectures.md](03_modeling_and_experiments_phase/01_model_architectures.md) — Teori dan implementasi MultinomialNB, Logistic Regression, dan Calibrated Linear SVM.
- 📄 [02_experiment_matrix_e0_e6.md](03_modeling_and_experiments_phase/02_experiment_matrix_e0_e6.md) — Desain riset dan metodologi matriks eksperimen E0 (Baseline) hingga E6 (OOS Analysis).
- 📄 [03_oos_threshold_theory.md](03_modeling_and_experiments_phase/03_oos_threshold_theory.md) — Teori dan mekanisme filter Out-of-Scope (OOS) berbasis *Confidence Thresholding*.

### 📈 04. Fase Evaluasi & Analisis
- 📄 [01_metrics_guide.md](04_evaluation_phase/01_metrics_guide.md) — Rasionalisasi pemilihan metrik utama Macro F1 vs Akurasi pada masalah *Class Imbalance*.
- 📄 [02_error_analysis.md](04_evaluation_phase/02_error_analysis.md) — Cara membaca visualisasi Confusion Matrix heatmap dan menginterpretasikan log metrik JSON.

### 🖥️ 05. Fase Deployment & Web UI
- 📄 [01_streamlit_flow.md](05_deployment_and_ui_phase/01_streamlit_flow.md) — Arsitektur runtime Web UI Streamlit, penanganan session state, dan optimasi resource cache.
- 📄 [02_how_to_run_and_troubleshoot.md](05_deployment_and_ui_phase/02_how_to_run_and_troubleshoot.md) — Daftar perintah eksekusi CLI lengkap dan panduan solusi kendala teknis (*troubleshooting*).

---

## 🚀 Alur Kerja Program Secara Ringkas

```text
[ Dataset CSV (8 Kolom) ]
          │
          ▼
[ Skrip Validasi & Anti-Leakage (scripts/validate_dataset.py) ]
          │
          ▼
[ NLP Preprocessing: Case Folding + Slang Normalization + Stopwords ]
          │
          ▼
[ Feature Extraction: TF-IDF (Unigram + Bigram) ]
          │
          ▼
[ Model Training & Eksperimen E0–E6 (scripts/run_experiments.py) ]
          │
          ├─► [ Evaluasi Riset (Macro F1 & Confusion Matrix) ]
          └─► [ Penyimpanan Model Terlatih (models/best_model.joblib) ]
                    │
                    ▼
          [ Web UI Chatbot Streamlit (app/main.py) ]
```
