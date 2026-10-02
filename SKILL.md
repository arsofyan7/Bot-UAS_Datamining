# 📜 AI Assistant Coding Guidelines & Rules (`SKILL.md`)

Panduan wajib dan pakem coding yang **HARUS** dibaca dan dipatuhi oleh setiap AI Assistant sebelum melakukan modifikasi kode pada repositori `Bot-UAS_Datamining`.

---

## 1. 🔄 Sinkronisasi Dokumentasi Wajib (Documentation Sync)

- **Aturan Emas:** Setiap kali melakukan penambahan fitur, perubahan alur, perbaikan bug, atau modifikasi parameter pada kode program (`src/`, `scripts/`, `app/`, `config.yaml`), AI **WAJIB** memperbarui dokumen terkait di dalam folder `docs/`.
- Dokumentasi tidak boleh usang (*stale*). Jika ada perubahan alur pra-pemrosesan, perbarui `docs/02_nlp_pipeline_phase/01_preprocessing.md` dan diagram di `docs/00_overview/01_system_architecture.md`.
- Jika ada penambahan perintah atau solusi troubleshooting baru, tambahkan ke `docs/05_deployment_and_ui_phase/02_how_to_run_and_troubleshoot.md`.

---

## 2. 💬 Kebijakan Komentar Kode (Code Comments Integrity)

- **Dilarang Menghapus Komentar Edukatif:** Komentar Bahasa Indonesia yang menjelaskan konsep data mining, teori model, dan metodologi riset di setiap file **TIDAK BOLEH** dihapus atau disederhanakan menjadi minim.
- **Kualitas Penjelasan:** Jika perlu mengubah atau menambah komentar, pastikan komentar tersebut ditulis dengan bahasa Indonesia yang jelas, bernilai akademis, dan mudah dipahami oleh mahasiswa saat sidang/ujian UAS di hadapan dosen.
- Berikan penjelasan **Mengapa (Why)** dan **Bagaimana (How)** pada baris logika yang krusial.

---

## 3. ⚡ Efisiensi Token & Kinerja Komputasi (Token Conservation)

- **Targeted Code Edits:** Gunakan alat edit file berbasis potongan blok spesifik (`replace_file_content` / `multi_replace_file_content`), hindari menimpa ulang seluruh isi file besar jika perubahannya hanya beberapa baris.
- **Ringkas & Padat:** Berikan penjelasan yang langsung ke inti masalah tanpa basa-basi berulang.
- **Hindari Output Terminal Masif:** Batasi pembacaan data/log yang terlalu panjang dengan memfilter baris penting atau membatasi head/tail output.

---

## 4. 📐 Standar Koding & Kaidah Data Mining (Engineering Best Practices)

- **Reproducibility:** Selalu sertakan `seed_everything(seed)` untuk setiap proses yang melibatkan pengacakan (shuffling, split, cross-validation, model training).
- **No Hardcoding:** Gunakan konfigurasi dari `config.yaml` untuk direktori path, ambang batas OOS, dan hyperparameter.
- **Defensive Preprocessing:**
  - URL harus dibersihkan sebelum tanda baca diubah menjadi spasi.
  - Whitelist kata kunci intent (*WH-words* seperti `dimana`, `kapan`, `berapa`, `syarat`, dll.) **TIDAK BOLEH** terhapus oleh stopword removal.
- **Model Probabilitas Terkalibrasi:** Linear SVM wajib dikalibrasi via `CalibratedClassifierCV` agar method `predict_proba` menghasilkan skor keyakinan yang tajam untuk filter OOS.
- **Evaluasi Adil:** Selalu gunakan **Macro F1-Score** sebagai metrik acuan utama untuk menghindari bias pada ketimpangan distribusi kelas (*Class Imbalance*).
- **Windows Terminal Encoding:** Hindari mencetak karakter emoji mentah pada skrip console CLI untuk mencegah `UnicodeEncodeError (charmap/cp1252)`. Gunakan penanda ASCII aman seperti `[INFO]`, `[SUCCESS]`, `[WARNING]`, `[PASS]`, dan `[FAIL]`.
