# 🖥️ 01. Arsitektur Runtime Web UI Chatbot (Streamlit)

Dokumen ini menjelaskan alur kerja aplikasi antarmuka pengguna di [app/main.py](file:///d:/Kuliah/1_Data%20Mining/Chatbot%20UAS/Bot-UAS_Datamining/app/main.py).

---

## 1. Siklus Hidup Eksekusi (*Execution Lifecycle*)

Aplikasi Streamlit berjalan secara reaktif. Setiap kali ada interaksi pengguna (mengirim pesan chat, menggeser slider, atau menekan tombol), Streamlit mengeksekusi skrip dari atas ke bawah.

```text
[ Browser Client Membuka http://localhost:8501 ]
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │ 1. get_model_and_config()            │
    │    - Menggunakan @st.cache_resource  │
    │    - Load best_model.joblib sekali   │
    │      saja ke dalam memory (RAM)      │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │ 2. Inisialisasi Session State        │
    │    - st.session_state.messages       │
    │    - Mempertahankan riwayat obrolan  │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │ 3. Render Sidebar Controls           │
    │    - Slider OOS Threshold            │
    │    - Status Model Aktif              │
    │    - Tombol Reset Chat               │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │ 4. Render Riwayat Pesan Chat         │
    │    - Menampilkan bubble User/Bot     │
    │    - Menampilkan Badge Confidence    │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │ 5. Event: Input Baru Pengguna        │
    │    - TextPreprocessor.transform()    │
    │    - IntentClassifierPipeline        │
    │    - Evaluasi OOS Gate               │
    │    - Append Respon ke Session State  │
    └──────────────────────────────────────┘
```

---

## 2. Optimasi Kinerja: `@st.cache_resource`

Pemuatan model Machine Learning (`joblib.load`) dan inisialisasi kamus pra-pemrosesan membutuhkan waktu komputasi. Dengan decorator `@st.cache_resource`, model hanya dibaca satu kali saat aplikasi pertama kali menyala, sehingga waktu respon saat membalas pesan obrolan pengguna berlangsung sangat cepat (**< 50 milidetik**).
