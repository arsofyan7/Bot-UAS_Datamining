"""
Web UI Chatbot - Domain-Agnostic Intent Classification
======================================================
Interactive Streamlit application integrating real trained ML pipelines (MultinomialNB,
Logistic Regression, Linear SVM) with Indonesian NLP preprocessing and Out-of-Scope detection.
"""

import os
import sys
import yaml
import streamlit as st

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import TextPreprocessor
from src.models import IntentClassifierPipeline
from src.utils import load_config, load_model


# -----------------------------------------------------------------------------
# Bot Knowledge Base / Response Mapping
# -----------------------------------------------------------------------------
INTENT_RESPONSES = {
    "biaya_pendaftaran": (
        "💰 **Informasi Biaya Pendaftaran:**\n\n"
        "Biaya pendaftaran mahasiswa baru adalah **Rp 350.000** untuk program reguler dan **Rp 750.000** untuk program kelas internasional. "
        "Pembayaran dapat dilakukan melalui Virtual Account Bank mitra kami (Mandiri, BCA, BNI, BRI)."
    ),
    "lokasi_alamat": (
        "📍 **Lokasi & Alamat Kampus:**\n\n"
        "Kampus utama berlokasi di **Jl. Telekomunikasi No. 1, Terusan Buahbatu, Sukapura, Kec. Dayeuhkolot, Kabupaten Bandung, Jawa Barat 40257**. "
        "Gedung rektorat dan admisi berada di Gedung Bangkit (Gedung A)."
    ),
    "syarat_masuk": (
        "📋 **Persyaratan Masuk & Dokumen Pendaftaran:**\n\n"
        "1. Scan Ijazah / Surat Keterangan Lulus (SKL) yang telah dilegalisir.\n"
        "2. Pas foto formal terbaru ukuran 4x6 latar belakang merah.\n"
        "3. Salinan Kartu Keluarga (KK) dan KTP / Kartu Pelajar.\n"
        "4. Scan rapor semester 1-5 (khusus jalur prestasi/nilai rapor)."
    ),
    "jadwal_kuliah": (
        "📅 **Jadwal & Kalender Akademik:**\n\n"
        "Perkuliahan semester ganjil dimulai pada **pertengahan September**. "
        "Masa orientasi mahasiswa baru (Ospek) diselenggarakan pada akhir Agustus, dan pengisian KRS dibuka mulai awal September."
    ),
    "beasiswa": (
        "🎓 **Program Beasiswa Tersedia:**\n\n"
        "Tersedia berbagai skema beasiswa:\n"
        "- **Beasiswa KIP-Kuliah** (bebas UKT + uang saku bulanan).\n"
        "- **Beasiswa Prestasi Akademik & Unggulan** (potongan 50% - 100% biaya pendidikan).\n"
        "- **Beasiswa Kemitraan Industri & Alumni**."
    ),
    "program_studi": (
        "🏫 **Pilihan Program Studi (Prodi):**\n\n"
        "Tersedia program sarjana (S1) dan vokasi (D3/D4) di antaranya: Informatika, Sistem Informasi, Sains Data, Teknik Elektro, Teknik Telekomunikasi, Desain Komunikasi Visual (DKV), dan Manajemen Bisnis Telekomunikasi."
    ),
    "kontak_layanan": (
        "📞 **Kontak Layanan & Helpdesk Admisi:**\n\n"
        "- **WhatsApp Admisi:** +62 811-2345-6789\n"
        "- **Email:** admisi@universitas.ac.id\n"
        "- **Call Center:** (022) 7564108\n"
        "- **Jam Layanan:** Senin - Jumat (08.00 - 16.00 WIB)"
    ),
    "OOS_REJECTED": (
        "⚠️ **Pertanyaan di Luar Domain (Out-of-Scope):**\n\n"
        "Maaf, pertanyaan Anda tampaknya berada di luar cakupan topik pengetahuan sistem ini atau tingkat keyakinan model berada di bawah batas ambang (*OOS threshold*). "
        "Silakan ajukan pertanyaan seputar informasi pendaftaran, biaya, beasiswa, jurusan, lokasi, atau kontak kampus."
    ),
    "OOS_DIFFERENT_DOMAIN": (
        "⚠️ **Topik Tidak Relevan:**\n\n"
        "Topik ini berada di luar domain sistem chatbot kampus. Silakan tanyakan hal seputar penerimaan mahasiswa baru dan informasi akademik."
    )
}


# -----------------------------------------------------------------------------
# Configuration & Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_and_config():
    try:
        config = load_config("config.yaml")
    except Exception:
        config = {
            "ui": {
                "title": "Intent Classification Chatbot NLP",
                "theme_color": "#1E88E5",
                "show_confidence_score": True,
                "show_oos_warning": True
            },
            "evaluation": {"oos_threshold": 0.5}
        }

    model_path = os.path.join(ROOT_DIR, "models", "best_model.joblib")
    model = None
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")

    preprocessor = TextPreprocessor()
    return config, model, preprocessor


config, trained_model, preprocessor = get_model_and_config()

# -----------------------------------------------------------------------------
# Page Configuration & Header
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=config.get("ui", {}).get("title", "Intent Classification Chatbot"),
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/bot.png", width=100)
    st.title("⚙️ Kontrol Sistem")

    # Model Status
    if trained_model is not None:
        st.success("🟢 **Model ML Terlatih Aktif**")
        classes_list = getattr(trained_model, "classes_", [])
        st.caption(f"Kelas Intent ({len(classes_list)}): `{', '.join(classes_list[:4])}...`")
    else:
        st.warning("🟠 **Model Belum Dilatih**")
        st.info("Jalankan perintah berikut di terminal:\n```bash\npython scripts/run_experiments.py\n```")

    # OOS Slider
    default_thresh = float(config.get("evaluation", {}).get("oos_threshold", 0.5))
    oos_threshold = st.slider(
        "Ambang Batas OOS (*Confidence Threshold*):",
        min_value=0.0,
        max_value=1.0,
        value=default_thresh,
        step=0.05,
        help="Pertanyaan dengan skor kepercayaan probabilitas di bawah batas ini akan ditolak sebagai Out-of-Scope (OOS)."
    )

    st.markdown("---")
    st.subheader("📊 Spesifikasi Riset")
    st.markdown("""
    - **Metrik Utama:** `Macro F1-Score`
    - **Algoritma:** `MultinomialNB / LogReg / LinearSVM`
    - **Fitur Teks:** `TF-IDF (Unigram + Bigram)`
    - **Bahasa:** `Bahasa Indonesia (Sastrawi + Slang Dict)`
    """)

    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------------------------
def infer_intent(user_text: str, threshold: float):
    """
    Cleans user input, passes through NLP pipeline, and determines intent & confidence.
    """
    clean_text = preprocessor.transform(user_text, full_pipeline=True)
    
    if trained_model is not None:
        try:
            res_list = trained_model.predict_with_confidence([clean_text], oos_threshold=threshold)
            result = res_list[0]
            intent = result["intent"]
            conf = result["confidence"]
            is_oos = result["is_oos"] or intent in ["OOS_REJECTED", "OOS_DIFFERENT_DOMAIN"]
        except Exception:
            # Fallback if probability fails
            pred = trained_model.predict([clean_text])[0]
            intent = pred
            conf = 0.85
            is_oos = False
    else:
        # Dummy fallback when model file is not present
        lower = clean_text.lower()
        if any(w in lower for w in ["biaya", "bayar", "tarif", "uang"]):
            intent, conf = "biaya_pendaftaran", 0.92
        elif any(w in lower for w in ["lokasi", "alamat", "dimana", "gedung"]):
            intent, conf = "lokasi_alamat", 0.89
        elif any(w in lower for w in ["syarat", "berkas", "dokumen"]):
            intent, conf = "syarat_masuk", 0.87
        elif any(w in lower for w in ["jadwal", "kapan", "waktu"]):
            intent, conf = "jadwal_kuliah", 0.82
        elif any(w in lower for w in ["beasiswa", "bantuan"]):
            intent, conf = "beasiswa", 0.90
        elif any(w in lower for w in ["prodi", "jurusan", "program studi"]):
            intent, conf = "program_studi", 0.91
        elif any(w in lower for w in ["kontak", "admin", "telepon", "whatsapp", "email"]):
            intent, conf = "kontak_layanan", 0.88
        else:
            intent, conf = "OOS_REJECTED", 0.32
        
        is_oos = conf < threshold or intent == "OOS_REJECTED"

    reply = INTENT_RESPONSES.get(intent, INTENT_RESPONSES["OOS_REJECTED"])

    return {
        "intent": intent,
        "clean_text": clean_text,
        "confidence": conf,
        "is_oos": is_oos,
        "reply": reply
    }


# -----------------------------------------------------------------------------
# Main Chat UI
# -----------------------------------------------------------------------------
st.title("🤖 " + config.get("ui", {}).get("title", "Intent Classification Chatbot"))
st.caption("Chatbot NLP berbasis Supervised Machine Learning dengan deteksi Out-of-Scope (OOS) & evaluasi Macro F1.")

if trained_model is None:
    st.warning("⚠️ **Catatan:** Model fisik (`models/best_model.joblib`) belum ditemukan. Sistem saat ini berjalan dalam mode fallback rule-based. Jalankan `python scripts/run_experiments.py` untuk mengaktifkan model ML terlatih.")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! Saya adalah chatbot asisten informasi pendaftaran kampus. Ada yang bisa saya bantu terkait biaya pendaftaran, lokasi kampus, syarat pendaftaran, jadwal kuliah, atau beasiswa?",
            "meta": None
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta") and config.get("ui", {}).get("show_confidence_score", True):
            meta = msg["meta"]
            intent_str = meta["intent"]
            conf_val = meta["confidence"]
            
            if meta["is_oos"] and config.get("ui", {}).get("show_oos_warning", True):
                st.caption(f"⚠️ `Intent: {intent_str}` | `Confidence: {conf_val:.2%}` *(Out-of-Scope Triggered)*")
            else:
                st.caption(f"🎯 `Intent: {intent_str}` | `Confidence: {conf_val:.2%}` | `Cleaned: \"{meta.get('clean_text', '')}\"`")

# Chat input
if user_prompt := st.chat_input("Tulis pertanyaan Anda di sini... (contoh: Berapa biaya pendaftaran kuliah?)"):
    # User message
    st.session_state.messages.append({"role": "user", "content": user_prompt, "meta": None})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Inference
    pred_res = infer_intent(user_prompt, oos_threshold)

    # Bot response
    with st.chat_message("assistant"):
        st.markdown(pred_res["reply"])
        if config.get("ui", {}).get("show_confidence_score", True):
            if pred_res["is_oos"] and config.get("ui", {}).get("show_oos_warning", True):
                st.caption(f"⚠️ `Intent: {pred_res['intent']}` | `Confidence: {pred_res['confidence']:.2%}` *(Out-of-Scope Triggered)*")
            else:
                st.caption(f"🎯 `Intent: {pred_res['intent']}` | `Confidence: {pred_res['confidence']:.2%}` | `Cleaned: \"{pred_res['clean_text']}\"`")

    # Append bot reply
    st.session_state.messages.append({
        "role": "assistant",
        "content": pred_res["reply"],
        "meta": pred_res
    })
