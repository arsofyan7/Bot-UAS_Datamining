# 🔤 01. Pipeline Pra-Pemrosesan Teks (NLP Preprocessing)

Dokumen ini menjelaskan implementasi teknis kelas `TextPreprocessor` di [src/preprocessing.py](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/src/preprocessing.py).

---

## 1. Tahapan Mikro Pembersihan Teks

Sebelum kalimat masukan diubah menjadi representasi vektor numerik, kalimat melewati 5 tahapan mikro berurutan:

```text
  [ Input Mentah: "Halo admin! Lokasi kmpusnya dmn y min? Tolong infonya... http://link.id" ]
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. Regex Cleaning & Case Folding                                            │
  │    - Hapus URL, mention @, tanda baca, simbol, angka, & extra spaces        │
  │    - Ubah huruf kapital menjadi huruf kecil                                 │
  │    Hasil: "halo admin lokasi kmpusnya dmn y min tolong infonya"             │
  └──────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 2. Slang & Abbreviation Normalization                                       │
  │    - Memetakan kata gaul/singkatan ke bentuk formal baku                    │
  │    - 'kmpusnya' -> 'kampus', 'dmn' -> 'dimana', 'y' -> 'ya', 'min' -> 'admin'│
  │    Hasil: "halo admin lokasi kampus dimana ya admin tolong informasi"       │
  └──────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 3. Indonesian Stopword Removal                                              │
  │    - Menghapus kata umum yang minim bobot diskriminatif ('halo', 'ya', dll.) │
  │    - Mempertahankan kata kunci penentu topik                                │
  │    Hasil: "lokasi kampus dimana informasi"                                  │
  └──────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 4. Stemming (Sastrawi Factory - Opsional)                                   │
  │    - Memotong imbuhan (awalan, sisipan, akhiran) ke kata dasar              │
  │    - 'informasi' -> 'informasi', 'pendaftaran' -> 'daftar'                  │
  │    Hasil: "lokasi kampus mana informasi"                                    │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Parameter Konfigurasi Preprocessor

Kelas `TextPreprocessor` memiliki parameter yang dapat dihidupkan atau dimatikan secara modular untuk kebutuhan eksperimen ablasi (E0 vs E1):

```python
preprocessor = TextPreprocessor(
    case_folding=True,        # Mengubah teks ke huruf kecil
    remove_punctuation=True,  # Menghapus seluruh tanda baca
    remove_numbers=True,      # Menghapus karakter angka
    normalization=True,       # Mengaktifkan kamus slang normalization
    stopword_removal=True,    # Menghapus stopword Bahasa Indonesia
    stemming=False            # Stemming Sastrawi (default: False untuk efisiensi kecepatan inferensi)
)
```

---

## 3. Kamus Slang Bahasa Indonesia (`DEFAULT_SLANG_DICT`)

Kamus internal memetakan kata-kata tidak baku ke bentuk formal:
- Kata tanya / petunjuk arah: `dmn` $\rightarrow$ `dimana`, `gmn`/`gimana` $\rightarrow$ `bagaimana`, `kpn` $\rightarrow$ `kapan`.
- Kata ganti / sapaan: `sy` $\rightarrow$ `saya`, `km` $\rightarrow$ `kamu`, `min` $\rightarrow$ `admin`, `kak` $\rightarrow$ `kakak`.
- Kata topik pendidikan / kampus: `bya` $\rightarrow$ `biaya`, `dftr` $\rightarrow$ `daftar`, `mhs` $\rightarrow$ `mahasiswa`, `kulyah` $\rightarrow$ `kuliah`, `prodi`/`jurusan` $\rightarrow$ `program studi`, `univ` $\rightarrow$ `universitas`.
