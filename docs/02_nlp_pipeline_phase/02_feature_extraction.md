# 📐 02. Ekstraksi Fitur Teks (TF-IDF & N-Gram)

Dokumen ini menjelaskan implementasi teknis kelas `FeatureExtractor` di [src/feature_extraction.py](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/src/feature_extraction.py).

---

## 1. Konsep Matematika TF-IDF

Model Machine Learning tidak dapat memproses teks mentah secara langsung. Algoritma **TF-IDF (*Term Frequency-Inverse Document Frequency*)** mengubah kumpulan kata menjadi vektor berbobot statistik:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

1. **Term Frequency ($\text{TF}$):** Seberapa sering kata $t$ muncul dalam satu kalimat $d$.
   $$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
2. **Inverse Document Frequency ($\text{IDF}$):** Seberapa unik kata $t$ di seluruh kumpulan data $D$. Kata yang muncul di hampir semua dokumen diberi bobot lebih rendah.
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

---

## 2. Fitur N-Gram (Unigram vs Bigram)

Dalam klasifikasi intent, urutan kata sering kali mengubah arti kalimat:
- **Unigram (`ngram_range=(1, 1)`):** Menghitung kata per kata secara terpisah.
  *Contoh:* `"biaya"`, `"pendaftaran"`, `"kuliah"`.
- **Unigram + Bigram (`ngram_range=(1, 2)`):** Menggabungkan kata tunggal dan pasangan dua kata berurutan.
  *Contoh:* `"biaya"`, `"pendaftaran"`, `"kuliah"`, `"biaya pendaftaran"`, `"pendaftaran kuliah"`.

Fitur Bigram terbukti sangat efektif menangkap frasa kunci seperti `"biaya pendaftaran"` atau `"jadwal kuliah"` yang memiliki makna kuat terhadap target intent.

---

## 3. Parameter Utama `FeatureExtractor`

```python
extractor = FeatureExtractor(
    vectorizer_type="tfidf",
    ngram_range=(1, 2),    # Unigram dan Bigram
    max_features=5000,     # Batas maksimal ukuran kosakata teratas
    min_df=1,              # Frekuensi minimal kemunculan kata
    sublinear_tf=True      # Menggunakan scaling logaritmik 1 + log(tf) untuk meredam frekuensi ekstrem
)
```
