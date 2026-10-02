# 🔄 03. Panduan Lengkap Adaptasi Domain Baru (Domain Transfer Guide)

Dokumen ini adalah panduan langkah demi langkah (*step-by-step tutorial*) jika dosen Anda memberikan tugas dengan **domain/topik studi kasus yang berbeda** (misalnya: Perbankan, E-Commerce, Rumah Sakit/Kesehatan, Pariwisata, Layanan Publik, Restoran, dll.).

Karena arsitektur sistem ini dirancang secara **Domain-Agnostic**, Anda **TIDAK PERLU** merombak algoritma Machine Learning ataupun pipeline matematika di `src/`. Anda hanya perlu mengikuti 7 langkah terstruktur berikut:

---

## 📋 Ringkasan 7 Langkah Adaptasi Domain

```text
[ Langkah 1: Tentukan Taksonomi Intent Domain Baru ]
                        │
                        ▼
[ Langkah 2: Buat File Dataset CSV (8 Kolom Standar) ]
                        │
                        ▼
[ Langkah 3: Validasi Dataset dengan Skrip Validator ]
                        │
                        ▼
[ Langkah 4: (Opsional) Tambah Kata Slang Khusus Domain ]
                        │
                        ▼
[ Langkah 5: Sesuaikan config.yaml ]
                        │
                        ▼
[ Langkah 6: Update Jawaban Bot di app/main.py ]
                        │
                        ▼
[ Langkah 7: Eksekusi Ulang Training & Buka Web UI ]
```

---

## 🛠️ Panduan Detail Per Langkah

### Langkah 1: Rancang Taksonomi Intent Domain Baru
Tentukan niat (*intent*) apa saja yang ingin dikenali oleh chatbot Anda (biasanya 5–10 intent in-scope + 1 intent out-of-scope).

**Contoh Kasus: Domain Rumah Sakit / Layanan Kesehatan (KESEHATAN)**
- `jadwal_dokter`: Menanyakan jam praktek dokter spesialis.
- `biaya_rawat_inap`: Menanyakan tarif kamar rawat inap / kamar operasi.
- `cara_pendaftaran_bpjs`: Menanyakan prosedur berobat menggunakan BPJS Kesehatan.
- `fasilitas_igd`: Menanyakan ketersediaan ambulans dan IGD 24 jam.
- `lokasi_poliklinik`: Menanyakan denah ruangan atau gedung poliklinik.
- `kontak_darurat`: Menanyakan nomor hotline darurat rumah sakit.
- `OOS_DIFFERENT_DOMAIN`: Pertanyaan di luar topik medis (misal politik, olahraga).

---

### Langkah 2: Buat Berkas Dataset CSV Baru
Buat file CSV baru di folder `data/raw/` (contoh: `data/raw/dataset_kesehatan.csv`) dengan **8 kolom wajib**:

```csv
id,utterance,normalized_utterance,intent,domain,source,split,boundary_case_notes
MED-001,Kapan jadwal praktek dokter anak?,kapan jadwal praktek dokter anak,jadwal_dokter,KESEHATAN,synthetic,train,
MED-002,Berapa tarif rawat inap kamar VIP per malam?,berapa tarif rawat inap kamar vip per malam,biaya_rawat_inap,KESEHATAN,synthetic,train,
MED-003,Apakah bisa berobat menggunakan kartu BPJS?,apakah bisa berobat menggunakan kartu bpjs,cara_pendaftaran_bpjs,KESEHATAN,synthetic,train,
MED-004,Nomor ambulans gawat darurat berapa?,nomor ambulans gawat darurat berapa,kontak_darurat,KESEHATAN,synthetic,train,
MED-005,Jam berapa dokter spesialis kandungan ada?,jam berapa dokter spesialis kandungan ada,jadwal_dokter,KESEHATAN,synthetic,val,
MED-006,Rincian biaya kamar kelas satu berapa?,rincian biaya kamar kelas satu berapa,biaya_rawat_inap,KESEHATAN,synthetic,test,
MED-007,Siapa juara piala dunia sepakbola kemarin?,siapa juara piala dunia sepakbola kemarin,OOS_DIFFERENT_DOMAIN,OUT_OF_SCOPE,synthetic,oos-test,Pertanyaan olahraga di luar medis
```

> [!TIP]
> **Aturan Pembagian Split:**
> - `train`: 60% – 70% data in-scope.
> - `val`: 10% – 15% data in-scope.
> - `test`: 15% – 20% data in-scope.
> - `oos-test`: Kalimat-kalimat acak di luar domain untuk menguji ketahanan model.

---

### Langkah 3: Validasi Integritas Dataset
Pastikan dataset baru Anda lolos uji skema dan tidak mengalami kebocoran data (*Data Leakage*):

```powershell
python scripts/validate_dataset.py data/raw/dataset_kesehatan.csv
```

Jika terminal menampilkan `[SUCCESS] SELURUH VALIDASI BERHASIL!`, Anda dapat lanjut ke langkah berikutnya.

---

### Langkah 4: (Opsional) Tambahkan Slang / Istilah Khusus Domain
Jika domain baru Anda memiliki singkatan atau istilah populer khas (misal: di kesehatan ada `ugd`, `poli`, `obgyn`, `spesialis`), buka berkas `src/preprocessing.py` dan tambahkan pemetaannya pada `DEFAULT_SLANG_DICT`:

```python
# Tambahan di src/preprocessing.py
DEFAULT_SLANG_DICT.update({
    "ugd": "unit gawat darurat",
    "igd": "instalasi gawat darurat",
    "poli": "poliklinik",
    "obgyn": "spesialis kandungan",
    "dr": "dokter",
    "inap": "rawat inap"
})
```

---

### Langkah 5: Sesuaikan `config.yaml`
Buka file `config.yaml` dan sesuaikan nama proyek serta judul Web UI:

```yaml
project:
  name: "Chatbot Intent Classification - Domain Kesehatan"
  version: "1.0.0"
  random_seed: 42

paths:
  raw_data: "data/raw/dataset_kesehatan.csv"
  model_output: "models/"

ui:
  title: "Chatbot Rumah Sakit & Layanan Kesehatan"
  theme_color: "#2E7D32" # Warna hijau kesehatan
  show_confidence_score: true
  show_oos_warning: true
```

---

### Langkah 6: Perbarui Template Jawaban Bot di `app/main.py`
Buka berkas `app/main.py` dan sesuaikan dictionary `INTENT_RESPONSES` dengan respon informatif yang sesuai untuk setiap intent domain baru Anda:

```python
# app/main.py
INTENT_RESPONSES = {
    "jadwal_dokter": (
        "👨‍⚕️ **Jadwal Dokter Spesialis:**\n\n"
        "- Dokter Anak: Senin - Jumat (09.00 - 14.00 WIB)\n"
        "- Dokter Penyakit Dalam: Senin - Sabtu (10.00 - 16.00 WIB)\n"
        "- Dokter Kandungan: Selasa & Kamis (13.00 - 18.00 WIB)"
    ),
    "biaya_rawat_inap": (
        "🏥 **Tarif Kamar Rawat Inap (Per Hari):**\n\n"
        "- Kamar Kelas 3: Rp 200.000\n"
        "- Kamar Kelas 1: Rp 600.000\n"
        "- Kamar VIP: Rp 1.250.000 (termasuk konsumsi & fasilitas sofa penunggu)."
    ),
    "cara_pendaftaran_bpjs": (
        "💳 **Alur Pendaftaran Pasien BPJS:**\n\n"
        "1. Bawa kartu BPJS aktif & surat rujukan dari Faskes Tingkat 1 (Puskesmas/Klinik).\n"
        "2. Ambil antrean di loket admisi Gedung B lantai 1.\n"
        "3. Tunjukkan KTP dan kartu BPJS kepada petugas verifikasi."
    ),
    "kontak_darurat": (
        "🚨 **Kontak Layanan Gawat Darurat (24 Jam):**\n\n"
        "- **Hotline IGD:** (021) 555-1199\n"
        "- **Layanan Ambulans:** 119\n"
        "- **WhatsApp Informasi:** 0812-9988-7766"
    ),
    "OOS_REJECTED": (
        "⚠️ **Pertanyaan di Luar Domain Kesehatan:**\n\n"
        "Maaf, pertanyaan Anda berada di luar topik layanan rumah sakit atau tingkat keyakinan model rendah. Silakan tanyakan seputar jadwal dokter, rawat inap, BPJS, atau kontak darurat."
    )
}
```

---

### Langkah 7: Jalankan Ulang Training & Buka Web UI
Latih model baru pada dataset domain Anda secara otomatis:

```powershell
# 1. Jalankan eksperimen E0-E6 pada dataset domain baru
python scripts/run_experiments.py --data data/raw/dataset_kesehatan.csv

# 2. Jalankan Web UI Chatbot
python -m streamlit run app/main.py
```

🎉 **Selesai!** Chatbot Anda sekarang telah sepenuhnya bermutasi menjadi asisten virtual untuk domain baru dengan model Machine Learning dan antarmuka Web yang sepenuhnya terkalibrasi.
