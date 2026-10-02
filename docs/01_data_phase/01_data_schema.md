# 📊 01. Skema Dataset & Pembagian Data (Data Splitting)

Dokumen ini menjelaskan spesifikasi struktur data masukan yang digunakan dalam proyek Intent Classification NLP.

---

## 1. Skema 8 Kolom Standar

Dataset wajib disimpan dalam format `.csv` dengan pemisah koma (`,`) dan pengodean UTF-8. Seluruh baris harus mematuhi 8 kolom wajib berikut:

| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
| :--- | :--- | :--- | :--- |
| **`id`** | String (Unik) | Pengenal unik untuk setiap baris data | `UTT-001`, `MED-012` |
| **`utterance`** | String | Kalimat asli/mentah yang diucapkan pengguna | `Berapa biaya pendaftaran kuliah?` |
| **`normalized_utterance`**| String | Kalimat setelah dilakukan normalisasi awal | `berapa biaya pendaftaran kuliah` |
| **`intent`** | String | Label kategori niat pengguna (*target class*) | `biaya_pendaftaran`, `jadwal_kuliah` |
| **`domain`** | String | Nama domain cakupan topik kalimat | `PENDIDIKAN`, `KESEHATAN` |
| **`source`** | String | Sumber perolehan data (*crowdsourced*, log chat, sintetik) | `synthetic`, `real_chat_log` |
| **`split`** | String | Penugasan peruntukan dataset | `train`, `val`, `test`, `oos-test` |
| **`boundary_case_notes`** | String (Opsional) | Catatan khusus jika kalimat ambigu / batas | `Pertanyaan multitafsir` |

---

## 2. Aturan Pembagian Split Data

Pembagian dataset (*data partitioning*) diatur secara eksplisit pada kolom `split` untuk menjamin replikasi riset yang konsisten:

```text
Dataset Keseluruhan (100%)
├── In-Scope Data
│   ├── train    (60% - 70%) : Data untuk fitting bobot model & TF-IDF
│   ├── val      (10% - 15%) : Data untuk penyesuaian parameter & pemilihan model
│   └── test     (15% - 20%) : Data untuk evaluasi final independen
└── Out-of-Scope Data
    └── oos-test (Data Uji)  : Kalimat di luar domain untuk menguji ketahanan filter OOS
```

### Panduan Penulisan Kalimat In-Scope vs Out-of-Scope:
- **In-Scope (`train`, `val`, `test`):** Kalimat yang masih memiliki relasi dengan salah satu kelas intent yang didefinisikan (misal seputar biaya, syarat, jadwal, kontak kampus).
- **Out-of-Scope (`oos-test`):** Kalimat yang sama sekali tidak relevan dengan sistem (misal resep masakan, ramalan cuaca, harga saham, tokoh politik, lelucon).
