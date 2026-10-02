# 🛡️ 02. Validasi Integritas Data & Anti Data Leakage

Dokumen ini menjelaskan mekanisme kerja skrip [scripts/validate_dataset.py](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/scripts/validate_dataset.py) untuk memastikan dataset bersih dan valid sebelum digunakan dalam proses pelatihan Machine Learning.

---

## 1. Mengapa Validasi Dataset Sangat Krusial?

Dalam riset Data Mining / NLP, kualitas data (*Garbage In, Garbage Out*) menentukan validitas hasil eksperimen. Kesalahan umum seperti kolom yang hilang, nilai kosong (*null*), label split salah ketik, atau kalimat uji yang bocor ke data latih (*Data Leakage*) dapat merusak validitas penelitian.

---

## 2. Empat Tahapan Pemeriksaan Validator

Skrip `validate_dataset.py` menjalankan 4 lapisan pengujian berurutan:

```text
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Skema Kolom Wajib                                        │
  │    Memastikan 8 nama kolom ada dan sesuai ejaan standar.     │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. Pengecekan Nilai Null & Duplikasi ID                    │
  │    Memastikan kolom esensial ('id', 'utterance', 'intent')  │
  │    tidak kosong dan setiap baris memiliki ID unik.          │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. Validasi Nilai Kolom Split                              │
  │    Nilai split hanya boleh: {'train', 'val', 'test', 'oos'} │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. Deteksi Kebocoran Data (Data Leakage Detection)          │
  │    Mencari apakah ada utterance pada val/test/oos yang      │
  │    persis sama dengan utterance pada train.                 │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Penjelasan Khusus: Deteksi Data Leakage

**Apa itu Data Leakage?**
Data Leakage terjadi apabila sebuah kalimat yang ada di kumpulan data evaluasi (`val` / `test`) ternyata duplikat persis dari kalimat di data `train`. Hal ini menyebabkan model tampak memiliki performa sempurna (Akurasi 100%), padahal model hanya "menghafal" kalimat tersebut.

Skrip validator mendeteksi ini dengan membandingkan himpunan teks kalimat:
$$\text{Leakage} = \{ x \in \text{Test/Val} \mid \text{lower}(x) \in \text{Train} \}$$

Jika ditemukan irisan teks, validator akan mengeluarkan pesan peringatan `[WARNING]` beserta nomor baris dan teks yang mengalami kebocoran.

---

## 4. Cara Menjalankan Validator

Jalankan perintah berikut di terminal:

```powershell
# Validasi dataset default (template)
python scripts/validate_dataset.py

# Validasi dataset kustom Anda
python scripts/validate_dataset.py data/raw/dataset.csv
```

**Arti Kode Warna Log Terminal:**
- `[INFO]`: Menampilkan progres pembacaan dan total baris data.
- `[SUCCESS]` (Hijau): Uji pemeriksaan berhasil lolos.
- `[WARNING]` (Kuning): Peringatan non-fatal (misal kolom tambahan atau potensi leakage).
- `[ERROR]` (Merah): Kesalahan fatal yang harus diperbaiki sebelum proses training.
