# 🛠️ 02. Panduan Eksekusi & Solusi Kendala Teknis (Troubleshooting)

Dokumen ini berisi daftar perintah terminal yang sering digunakan serta solusi praktis jika Anda menemui kendala teknis saat menjalankan proyek ini.

---

## 1. Daftar Perintah Lengkap

| Perintah | Deskripsi |
| :--- | :--- |
| `python scripts/validate_dataset.py data/raw/dataset.csv` | Menjalankan validasi skema 8 kolom, null check, dan anti data leakage pada dataset. |
| `python scripts/run_experiments.py` | Menjalankan matriks eksperimen riset E0 s/d E6, membuat grafik confusion matrix, dan menyimpan model terbaik. |
| `python scripts/run_experiments.py --data <path_csv>` | Menjalankan eksperimen pada file dataset kustom/domain baru. |
| `python -m streamlit run app/main.py` | Menjalankan server lokal Web UI Chatbot di browser (`http://localhost:8501`). |

---

## 2. Solusi Kendala Teknis (Troubleshooting)

### A. `streamlit : The term 'streamlit' is not recognized...`
- **Penyebab:** Folder binary *Scripts* Python user belum masuk ke `PATH` environment variable Windows.
- **Solusi:** Jalankan melalui modul Python langsung:
  ```powershell
  python -m streamlit run app/main.py
  ```

### B. `ModuleNotFoundError: No module named '...'`
- **Penyebab:** Dependensi belum terpasang di environment Python Anda.
- **Solusi:** Jalankan instalasi dependensi:
  ```powershell
  pip install -r requirements.txt
  ```

### C. `UnicodeEncodeError: 'charmap' codec can't encode character...`
- **Penyebab:** Terminal Windows (PowerShell/CMD) menggunakan encoding default `cp1252` yang tidak dapat menampilkan karakter emoji atau simbol non-ASCII.
- **Solusi:** Seluruh script output di proyek ini telah disesuaikan menggunakan penanda ASCII aman seperti `[SUCCESS]`, `[PASS]`, dan `[FAIL]`.

### D. Model Terdeteksi `OOS_REJECTED` pada Pertanyaan yang Seharusnya Diterima
- **Penyebab:** Ambang batas *OOS threshold* diatur terlalu tinggi, atau model yang aktif belum diperbarui dengan Linear SVM yang terkalibrasi.
- **Solusi:** 
  1. Turunkan slider *Ambang Batas OOS* di sidebar Streamlit (misal ke `0.45` atau `0.50`).
  2. Latih ulang model dengan perintah `python scripts/run_experiments.py`.
