# 🎯 03. Teori & Mekanisme Out-of-Scope (OOS) Thresholding

Dokumen ini menjelaskan strategi penanganan pertanyaan di luar domain (*Out-of-Scope Detection*) menggunakan pendekatan *Confidence Thresholding*.

---

## 1. Mengapa Chatbot Butuh Deteksi OOS?

Sistem klasifikasi multi-class standar (*Closed-World Assumption*) selalu memaksakan kalimat masukan ke salah satu kelas yang ada, meskipun input tersebut sama sekali tidak relevan (misalnya menanyakan resep masakan kepada chatbot universitas).

Untuk mengubahnya menjadi sistem dunia nyata (*Open-World Setting*), sistem menerapkan gerbang penyaring probabilitas keyakinan (*Confidence Gate*).

---

## 2. Mekanisme Kerja

```text
[ Kalimat Input Pengguna ]
           │
           ▼
[ Pipeline Classifier (SVM Calibrated) ]
           │
           ▼
[ Distribusi Probabilitas Kelas: P(c1), P(c2), ..., P(cK) ]
           │
           ▼
[ Hitung Probabilitas Tertinggi: P_max = max(P) ]
           │
           ▼
   ┌──────────────────────┐
   │ Is P_max >= θ_OOS?   │  (Contoh: θ_OOS = 0.50 atau 50%)
   └──────────┬───────────┘
              │
      ┌───────┴───────┐
      │               │
     YA             TIDAK
      │               │
      ▼               ▼
[ In-Scope ]   [ OOS_REJECTED ]
Terima Intent  Tampilkan Pesan
& Jawab Info   Penolakan Sopan
```

---

## 3. Menentukan Nilai Ambang Batas ($\theta_{\text{OOS}}$)

Berdasarkan hasil eksperimen E6:
- **Threshold terlalu rendah ($\theta < 0.30$):** Pertanyaan OOS rentan lolos dan dijawab salah (*False Positive*).
- **Threshold terlalu tinggi ($\theta > 0.85$):** Pertanyaan in-scope yang valid tetapi agak informal rentan ditolak keliru (*False Negative*).
- **Threshold Optimal ($\theta = 0.50$ s/d $0.60$):** Memberikan keseimbangan terbaik di mana pertanyaan in-scope diterima ($\ge 75\%$) dan pertanyaan acak ditolak ($< 30\%$).
