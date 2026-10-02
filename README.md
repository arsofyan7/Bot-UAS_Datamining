# 🤖 Domain-Agnostic Intent Classification Chatbot (Bahasa Indonesia)

Proyek penelitian Data Mining / Natural Language Processing (NLP) berbasis pendekatan **CRISP-DM** untuk klasifikasi intent teks percakapan Bahasa Indonesia. Proyek ini mengimplementasikan algoritma *Supervised Multi-Class Classification* dengan metrik evaluasi utama **Macro F1** dan dilengkapi dengan antarmuka interaktif Web UI Chatbot berbasis **Streamlit**.

---

## 📌 Fitur Utama

- **Domain-Agnostic Architecture**: Fleksibel untuk diterapkan pada berbagai domain percakapan (pendidikan, layanan pelanggan, e-commerce, dsb.).
- **Indonesian NLP Preprocessing Pipeline**: Pembersihan teks, case folding, normalisasi kata gaul/slang, stopword removal, dan stemming.
- **Ekstraksi Fitur Fleksibel**: TF-IDF & N-Gram (Unigram, Bigram) dengan Scikit-Learn.
- **Model Machine Learning**: Multinomial Naive Bayes, Logistic Regression, dan Linear SVM (Calibrated).
- **Out-of-Scope (OOS) Detection**: Mendeteksi pertanyaan di luar domain menggunakan ambang batas confidence score (*OOS Threshold*).
- **Interactive Web UI (Streamlit)**: Simulasi percakapan langsung dengan chatbot dan visualisasi skor kepercayaan prediksi.

---

## 📂 Struktur Direktori

```text
intent-classification-nlp/
├── app/
│   ├── __init__.py
│   └── main.py                     # Entry point Web UI Chatbot (Streamlit)
├── data/
│   ├── raw/                        # Dataset mentah
│   ├── processed/                  # Dataset hasil pembersihan & splitting
│   └── templates/
│       └── dataset_template.csv    # Format standar skema dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Pipeline pra-pemrosesan teks Bahasa Indonesia
│   ├── feature_extraction.py       # Vectorizer (TF-IDF, Count, N-Grams)
│   ├── models.py                   # Definisi model & Scikit-Learn pipelines
│   ├── evaluation.py               # Perhitungan Macro F1 & metrik OOS
│   └── utils.py                    # Konfigurasi, seed, dan helper persistence
├── notebooks/
│   └── 01_dummy_experiment.ipynb   # Notebook eksplorasi & eksperimen
├── models/                         # Folder penyimpanan model terlatih (.joblib / .pkl)
├── reports/
│   ├── figures/                    # Grafik evaluasi & confusion matrix
│   └── metrics/                    # Hasil log evaluasi JSON/CSV
├── scripts/
│   └── validate_dataset.py         # Skrip validasi integritas dataset & leakage
├── config.yaml                     # Konfigurasi parameter & eksperimen global
├── requirements.txt                # Daftar pustaka dependensi Python
├── .gitignore                      # Konfigurasi git ignore
└── README.md                       # Dokumentasi utama proyek
```

---

## 🚀 Panduan Memulai

### 1. Setup Virtual Environment

Disarankan menggunakan Python 3.10+:

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate
```

### 2. Instalasi Dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Validasi Dataset

Sebelum melatih model, pastikan file dataset Anda mematuhi skema standar dan bebas dari *Data Leakage*:

```bash
# Validasi file template default
python scripts/validate_dataset.py

# Validasi file dataset kustom
python scripts/validate_dataset.py data/raw/dataset_kamu.csv
```

---

## 💬 Menjalankan Web UI Chatbot

Jalankan antarmuka interaktif Streamlit lokal:

```bash
python -m streamlit run app/main.py
# atau jika streamlit sudah ada di PATH:
# streamlit run app/main.py
```

Setelah perintah dijalankan, antarmuka browser akan terbuka secara otomatis di `http://localhost:8501`.

---

## 📊 Metrik Evaluasi

Proyek ini mengutamakan metrik **Macro F1-Score** untuk menjamin performa klasifikasi yang adil pada setiap kelas intent tanpa terbiaskan oleh ketimpangan distribusi data (*class imbalance*).
