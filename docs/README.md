# 📚 Pusat Dokumentasi Proyek Intent Classification NLP

Selamat datang di repositori dokumentasi resmi untuk proyek penelitian **"Domain-Agnostic Intent Classification Chatbot Bahasa Indonesia"**. Dokumentasi ini disusun secara berjenjang (*hierarchical documentation*), mulai dari konsep arsitektur makro hingga detail teknis mikro setiap berkas kode.

---

## 🧭 Navigasi Dokumen

```text
docs/
├── 00_overview/
│   ├── 01_system_architecture.md       # Arsitektur makro, metodologi CRISP-DM, & alur end-to-end
│   ├── 02_project_structure.md          # Penjelasan fungsi setiap folder dan berkas kode
│   └── 03_domain_adaptation_guide.md    # Panduan lengkap ganti domain tugas UAS (step-by-step)
│
├── 01_data_phase/
│   ├── 01_data_schema.md                # Spesifikasi 8 kolom dataset & pembagian split data
│   └── 02_data_validation.md            # Mekanisme validasi skema & pencegahan Data Leakage
│
├── 02_nlp_pipeline_phase/
│   ├── 01_preprocessing.md              # Pipeline pra-pemrosesan teks Bahasa Indonesia
│   └── 02_feature_extraction.md         # Ekstraksi fitur TF-IDF & analisis N-Gram
│
├── 03_modeling_and_experiments_phase/
│   ├── 01_model_architectures.md        # Teori algoritma MNB, Logistic Regression, & Linear SVM
│   ├── 02_experiment_matrix_e0_e6.md    # Desain & analisis matriks eksperimen riset E0–E6
│   └── 03_oos_threshold_theory.md       # Teori & implementasi filter Out-of-Scope (OOS)
│
├── 04_evaluation_phase/
│   ├── 01_metrics_guide.md              # Rasionalisasi metrik utama Macro F1 vs Akurasi
│   └── 02_error_analysis.md             # Analisis kesalahan (Confusion Matrix & JSON Logs)
│
└── 05_deployment_and_ui_phase/
    ├── 01_streamlit_flow.md             # Arsitektur runtime Web UI Chatbot (Streamlit)
    └── 02_how_to_run_and_troubleshoot.md # Panduan instalasi, eksekusi perintah, & solusi error
```

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
