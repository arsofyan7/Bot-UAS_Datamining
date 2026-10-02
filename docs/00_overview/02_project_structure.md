# 📂 02. Struktur Proyek & Penjelasan Berkas

Dokumen ini menjelaskan fungsi dan tanggung jawab setiap direktori dan berkas dalam repositori [Bot-UAS_Datamining](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining).

---

## 🌳 Pohon Direktori

```text
intent-classification-nlp/
├── app/
│   ├── __init__.py                 # Marker package aplikasi Web UI
│   └── main.py                     # Entry point antarmuka Web UI Chatbot (Streamlit)
│
├── data/
│   ├── raw/                        # Tempat penyimpanan dataset mentah (.csv)
│   ├── processed/                  # Tempat dataset hasil pembersihan / split
│   └── templates/
│       └── dataset_template.csv    # Template referensi 8 kolom standar
│
├── src/
│   ├── __init__.py                 # Inisialisasi package module core
│   ├── preprocessing.py           # Kelas TextPreprocessor (pembersihan, slang, stopwords, stem)
│   ├── feature_extraction.py       # Kelas FeatureExtractor (TF-IDF, N-Grams, persistence)
│   ├── models.py                   # Model factory & IntentClassifierPipeline
│   ├── evaluation.py               # Perhitungan Macro F1, plot confusion matrix, & OOS
│   └── utils.py                    # Loader konfigurasi YAML, setting seed, & helper I/O
│
├── notebooks/
│   └── 01_dummy_experiment.ipynb   # Jupyter Notebook untuk eksplorasi interaktif
│
├── models/
│   └── best_model.joblib           # Model terbaik hasil serialisasi pipeline ML
│
├── reports/
│   ├── figures/
│   │   └── confusion_matrix.png    # Gambar plot Confusion Matrix resolusi tinggi (300 DPI)
│   └── metrics/
│       └── experiment_results.json # Hasil ringkasan evaluasi seluruh matriks eksperimen E0–E6
│
├── scripts/
│   ├── validate_dataset.py         # Skrip CLI validasi format dataset & anti-data leakage
│   └── run_experiments.py         # Skrip CLI otomatisasi eksperimen matriks E0–E6
│
├── config.yaml                     # Konfigurasi terpusat parameter eksperimen, path, & UI
├── requirements.txt                # Daftar pustaka dependensi Python
├── .gitignore                      # Berkas pengecualian Git
└── README.md                       # Ringkasan panduan cepat proyek
```

---

## 📑 Penjelasan Detail Per Berkas

### 1. Folder `src/` (Core Logic Library)
- **`src/preprocessing.py`**: Berisi kelas `TextPreprocessor`. Bertugas mengubah kalimat mentah menjadi bentuk baku. Menangani *case folding*, penghapusan karakter khusus, normalisasi kata tidak baku (*slang* seperti `dmn` $\rightarrow$ `dimana`), penghapusan kata umum (*stopwords*), dan pemenggalan kata dasar (*stemming*).
- **`src/feature_extraction.py`**: Berisi kelas `FeatureExtractor`. Bertugas mengubah teks yang sudah bersih menjadi matriks vektor numerik menggunakan algoritma TF-IDF (*Term Frequency-Inverse Document Frequency*) dengan dukungan fitur N-Gram (Unigram dan Bigram).
- **`src/models.py`**: Berisi fungsi `get_model()` untuk menginisialisasi algoritma (MultinomialNB, LogisticRegression, LinearSVM) dan kelas `IntentClassifierPipeline` yang menyatukan vectorizer dan classifier ke dalam satu kesatuan model yang dapat disimpan/dimuat via `joblib`.
- **`src/evaluation.py`**: Berisi fungsi `calculate_metrics()`, `plot_and_save_confusion_matrix()`, dan `evaluate_oos_threshold()`. Bertugas menghitung metrik performa riset dengan fokus utama pada **Macro F1**.
- **`src/utils.py`**: Menyediakan fungsi pembantu seperti `load_config()`, `seed_everything()` untuk menjamin replikasi hasil riset, serta fungsi baca/tulis berkas.

### 2. Folder `scripts/` (Automated Runners)
- **`scripts/validate_dataset.py`**: Skrip mandiri yang memeriksa validitas dataset CSV sebelum dilakukan proses *training*. Memeriksa kelengkapan 8 kolom wajib, keabsahan nilai split (`train`, `val`, `test`, `oos-test`), serta mendeteksi kebocoran data (*Data Leakage*) jika ada kalimat pada data uji yang persis sama dengan data latih.
- **`scripts/run_experiments.py`**: Skrip otomasi riset yang menjalankan eksperimen E0 hingga E6 secara berurutan, menampilkan tabel evaluasi di terminal, menyimpan log ke file JSON, menggambar Confusion Matrix, dan mengekspor model terbaik ke `models/best_model.joblib`.

### 3. Folder `app/` (Web User Interface)
- **`app/main.py`**: Aplikasi web interaktif berbasis Streamlit. Memuat model terbaik yang telah dilatih, menyediakan antarmuka obrolan (*chat message*), memproses kalimat masukan secara *real-time*, menampilkan skor probabilitas keyakinan (*confidence score*), mendeteksi *Out-of-Scope*, dan membalas dengan respon yang sesuai.

### 4. Folder `data/`, `models/`, & `reports/` (Data & Artifacts)
- **`data/raw/`**: Tempat meletakkan berkas CSV dataset mentah Anda (misal `dataset.csv`).
- **`data/templates/dataset_template.csv`**: Contoh acuan format CSV dengan 8 kolom wajib.
- **`models/best_model.joblib`**: Berkas biner pipeline model terbaik yang dihasilkan dari `run_experiments.py`.
- **`reports/figures/`**: Folder keluaran gambar visualisasi analisis riset (contoh: `confusion_matrix.png`).
- **`reports/metrics/`**: Folder penyimpanan log hasil evaluasi dalam format terstruktur JSON (`experiment_results.json`).

### 5. File Konfigurasi
- **`config.yaml`**: Pusat kendali seluruh parameter sistem (random seed, path direktori, pilihan fitur, kandidat model, dan nilai ambang batas OOS).
- **`requirements.txt`**: Daftar pustaka Python yang wajib diinstal (`pandas`, `scikit-learn`, `sastrawi`, `streamlit`, dll.).
