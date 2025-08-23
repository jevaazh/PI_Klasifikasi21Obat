import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tempfile
import base64
import time

# === OPTIONAL: suara lokal yang lebih manusiawi (jika tersedia) ===
try:
    import pyttsx3  # lokal, tidak butuh internet
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False

from gtts import gTTS
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ================================== KONFIGURASI ==================================
st.set_page_config(
    page_title="MediScan",
    layout="centered",
    page_icon="💊"
)

# ============================== DARK MODE (GLOBAL CSS) =============================
st.markdown("""
<style>
/* Global dark background & text */
html, body, [data-testid="stAppViewContainer"] {
  background: #0e1117 !important;
  color: #e6edf3 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: #0b0e14 !important;
  color: #e6edf3 !important;
  border-right: 1px solid #1f232b !important;
}

/* Cards/boxes */
.block-container { padding-top: 1.2rem; }
div[data-testid="stMetricValue"] { color: #e6edf3 !important; }
div[data-testid="stMetricDelta"] { color: #a6d189 !important; }

/* Buttons */
.stButton > button {
  background: #1f6feb !important;
  color: #fff !important;
  border: 1px solid #335bba !important;
  border-radius: 12px !important;
  padding: 0.6rem 1rem !important;
  font-weight: 600 !important;
  width: 100%;
  transition: transform .1s ease, background .2s ease;
}
.stButton > button:hover {
  background: #2a7df0 !important;
  transform: translateY(-1px);
}

/* Info/warning boxes */
div.stAlert {
  background: #0b1320 !important;
  border: 1px solid #18304f !important;
  color: #dbe7ff !important;
}

/* Expander */
.streamlit-expanderHeader {
  background: #0b1320 !important; 
  border-radius: 10px !important;
}

/* Inputs */
.css-1cpxqw2, .stTextInput, .stSelectbox, .stFileUploader, .stCameraInput {
  color: #e6edf3 !important;
}

/* Left menu items look like list rows */
.ms-list-row {
  display: flex; align-items: center; gap: .6rem;
  background: #0b1320; border: 1px solid #18304f;
  border-radius: 12px; padding: .8rem 1rem; cursor: pointer;
  user-select: none; min-height: 54px;
}
.ms-list-row:hover { background: #0e1a2b; }
.ms-list-row .chev { font-weight: 700; opacity: .9; }

/* Right detail panel */
.ms-detail {
  background: #0b1320; border: 1px solid #18304f;
  border-radius: 12px; padding: 1rem; min-height: 200px;
}

/* Subtle small text */
.small-muted { color: #9aa6b2; font-size: 0.92rem; margin-top: .25rem; }

/* Title area */
h1.ms-title { margin-bottom: .2rem; }
</style>
""", unsafe_allow_html=True)

# ============================== PARAMETER & UTILITAS ===============================
MIN_CONFIDENCE = 0.00  # ambang tolak gambar yang tidak jelas/ bukan obat

def center_crop_square(pil_img: Image.Image) -> Image.Image:
    """Crop tengah menjadi 1:1 (persegi)."""
    w, h = pil_img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return pil_img.crop((left, top, left + side, top + side))

# ============================== LOAD MODEL & DATASET ==============================
@st.cache_resource
def load_model_local():
    model_path = "model_obat_1.h5"
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            st.error(f"File model tidak ditemukan: {model_path}")
            return None
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

@st.cache_data
def load_dataset_local():
    dataset_path = "dataset_obat.csv"
    try:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            return df
        else:
            st.error(f"File dataset tidak ditemukan: {dataset_path}")
            return None
    except Exception as e:
        st.error(f"Gagal memuat dataset: {str(e)}")
        return None

# ============================== TTS (lebih manusiawi jika ada pyttsx3) ============
def speak_gtts(text, autoplay=True, lang='id'):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            gTTS(text=text, lang=lang, slow=False).save(tmp.name)
            audio_bytes = open(tmp.name, 'rb').read()
        os.unlink(tmp.name)
        b64 = base64.b64encode(audio_bytes).decode()
        if autoplay:
            st.markdown(f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <audio controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Gagal memutar audio: {e}")

def speak_pyttsx3(text):
    """
    Render audio secara lokal (lebih natural di beberapa mesin).
    Catatan: membutuhkan pyttsx3 & driver voice di host server.
    Kita generate file WAV dan embed.
    """
    try:
        engine = pyttsx3.init()
        # Atur kecepatan & volume agar terdengar lebih natural
        rate = engine.getProperty('rate')
        engine.setProperty('rate', int(rate*0.92))
        volume = engine.getProperty('volume')
        engine.setProperty('volume', min(1.0, volume))

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            outpath = tmp.name
        engine.save_to_file(text, outpath)
        engine.runAndWait()

        audio_bytes = open(outpath, 'rb').read()
        os.unlink(outpath)
        b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"TTS lokal gagal, fallback ke gTTS. Detail: {e}")
        speak_gtts(text, autoplay=True)

def speak(text, autoplay=True, prefer_local=True):
    """Abstraksi TTS: pakai pyttsx3 jika ada, else gTTS."""
    if prefer_local and HAS_PYTTSX3 and autoplay:
        speak_pyttsx3(text)
    else:
        speak_gtts(text, autoplay=autoplay)

def stop_all_audio():
    """Hentikan seluruh audio di halaman (pause & reset)."""
    st.markdown("""
    <script>
    const aud = document.querySelectorAll('audio');
    aud.forEach(a => { try { a.pause(); a.currentTime = 0; } catch(e){} });
    </script>
    """, unsafe_allow_html=True)

# ============================== APLIKASI ==========================================
def main():
    # =================== TITLE AREA ===================
    st.markdown(
        '<h1 class="ms-title">💊 <em>MediScan</em>: Klasifikasi 21 Citra Kemasan Obat dengan Integrasi <em>Text-to-Speech</em></h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="small-muted">Kirim foto kemasan obat—pastikan label/kemasan terlihat jelas agar hasil lebih akurat.</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ====== Load model & dataset ======
    model = load_model_local()
    obat_info_df = load_dataset_local()
    if model is None or obat_info_df is None:
        st.error("Gagal memuat model atau database. Silakan refresh halaman.")
        return

    class_names = sorted(obat_info_df['label'].unique())

    # ================== SIDEBAR INPUT ==================
    with st.sidebar:
        st.header("📷 Input Gambar")
        st.markdown("Pilih salah satu cara untuk memasukkan gambar obat:")

        img_file = st.file_uploader(
            "📁 Unggah Gambar Obat",
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG"
        )

        st.markdown("*ATAU*")

        enable_camera = st.toggle(
            "🔧 Aktifkan Kamera",
            value=False,
            help="Aktifkan toggle ini untuk menggunakan kamera"
        )

        camera_img = None
        if enable_camera:
            camera_img = st.camera_input(
                "📸 Ambil Gambar Realtime (otomatis 1:1)",
                help="Klik untuk mengambil foto obat secara langsung"
            )
        else:
            st.info("💡 Aktifkan toggle di atas untuk menggunakan kamera")

        # Pilih input
        img_input = img_file if img_file else camera_img

        if img_input:
            st.success("✅ Gambar berhasil dimuat!")

    # ================== MAIN CONTENT ==================
    if img_input:
        with st.spinner("🔍 Menganalisis gambar obat..."):
            try:
                img = Image.open(img_input).convert('RGB')

                # --- Otomatis 1:1 (center-crop), lalu resize untuk model ---
                img_square = center_crop_square(img)
                img_resized = img_square.resize((256, 256))

                # Preprocess
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediksi
                prediction = model.predict(img_array)[0]
                predicted_index = int(np.argmax(prediction))
                predicted_label = class_names[predicted_index]
                confidence = float(prediction[predicted_index])

                # Tolak jika confidence rendah (anggap bukan foto obat yang jelas)
                if confidence < MIN_CONFIDENCE:
                    st.error("❌ Gambar kurang jelas atau tidak menyerupai kemasan obat. "
                             "Silakan ambil ulang dengan pencahayaan baik dan fokus pada kemasan.")
                    st.stop()

                # Ambil info dari CSV
                info = obat_info_df[obat_info_df['label'] == predicted_label].iloc[0]

            except Exception as e:
                st.error(f"Gagal memproses gambar: {str(e)}")
                return

        # ================== HASIL PREDIKSI ==================
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img_square, caption=f"Hasil Deteksi: {predicted_label}", use_column_width=True)

        with col2:
            st.subheader(f"💊 {info['nama_obat']}")
            st.metric("🎯 Keyakinan Prediksi", f"{confidence*100:.2f}%")

            # Info dasar
            st.markdown(f"""
            *Golongan:* {info['golongan']}  
            *Jenis:* {info['jenis']}  
            """)

        # ================== DETAIL INFORMASI ==================
        st.markdown("---")
        st.subheader("📋 Detail Informasi")

        with st.expander("🔍 Lihat Detail", expanded=True):
            st.markdown(f"""
            *Manfaat:* {info['manfaat']}  
            *Aturan Minum:* {info['aturan_minum']}  
            *Catatan:* {info['catatan']}  
            """)

        # ================== PERINGATAN ==================
        st.warning("""
        ⚠ *PERINGATAN PENTING:* 
        Aturan minum dapat berbeda-beda pada setiap orang. Ikuti petunjuk dari dokter yang telah memeriksa kondisi Anda.
        """)

        # ================== TTS UTAMA (MANUAL) ==================
        # Teks utama yang lebih manusiawi
        main_text = (
            f"Nama obat ini adalah {info['nama_obat']}. "
            f"Golongan {info['golongan']}. "
            f"Merupakan {info['jenis']}, yang memiliki manfaat untuk {info['manfaat']}. "
            f"Dapat diminum {info['aturan_minum']}. "
            f"Dengan catatan, {info['catatan']}."
        )

        st.markdown("### 🔊 Dengarkan Ringkasan")
        if st.button("▶ Putar Ringkasan Obat"):
            # Manual (tidak auto) → tampilkan player kontrol
            speak_gtts(main_text, autoplay=False)

        # ================== INFORMASI LAINNYA (LIST KIRI → PANEL KANAN) ==================
        st.markdown("---")
        st.subheader("📂 Informasi Lainnya")

        # Map menu -> (judul tampil, field CSV, pembuka kalimat TTS)
        MENU_ITEMS = {
            "efek_samping": ("Efek Samping", "efek_samping", f"Efek samping dari {info['nama_obat']}: "),
            "pantangan": ("Pantangan Makanan", "pantangan_makanan", "Pantangan makanan: "),
            "interaksi": ("Interaksi Negatif", "interaksi_negatif", "Interaksi negatif: "),
            "lupa": ("Jika Lupa Minum?", "jika_lupa_minum", "Jika lupa minum: "),
            "simpan": ("Cara Penyimpanan", "penyimpanan", "Cara penyimpanan: "),
        }

        # siapkan state seleksi
        if "ms_selected" not in st.session_state:
            st.session_state.ms_selected = None

        left, right = st.columns([1, 2], vertical_alignment="top")

        with left:
            # Render list sebagai tombol vertikal seragam
            for key, (title, _, _) in MENU_ITEMS.items():
                # HTML baris dengan chevron ▶
                placeholder = st.container()
                with placeholder:
                    st.markdown(
                        f'<div class="ms-list-row">'
                        f'<span class="chev">▶</span> {title}</div>',
                        unsafe_allow_html=True
                    )
                    # invisible button overlay:
                    pressed = st.button(f" ", key=f"btn_{key}")
                # Klik → toggle select
                if pressed:
                    if st.session_state.ms_selected == key:
                        # klik lagi item yang sama → unselect + stop audio
                        st.session_state.ms_selected = None
                        with right:  # kosongkan panel kanan
                            stop_all_audio()
                            st.empty()
                    else:
                        st.session_state.ms_selected = key

        with right:
            panel = st.container()
            if st.session_state.ms_selected:
                k = st.session_state.ms_selected
                title, field, prefix = MENU_ITEMS[k]
                value = info.get(field, 'Informasi tidak tersedia')
                with panel:
                    st.markdown(f"### {title}")
                    st.markdown(f'<div class="ms-detail">{value}</div>', unsafe_allow_html=True)
                    # Auto-speak saat dipilih
                    speak(prefix + str(value), autoplay=True, prefer_local=True)
            else:
                # Panel kosong
                panel.empty()

    else:
        # ================== TAMPILAN AWAL ==================
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🔍 Selamat datang di <em>MediScan</em>!</h2>
            <p>Aplikasi deteksi kemasan obat berbasis AI untuk membantu Anda mengenali obat dan mendapatkan informasi yang tepat.</p>
            <p><strong>Unggah foto obat atau gunakan kamera dari sidebar.</strong><br>Pastikan kemasan, label, dan teks pada obat terlihat jelas.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📖 Panduan Penggunaan", expanded=True):
            st.markdown("""
            *Langkah-langkah:*
            1. *Upload Gambar* — Klik "Unggah Gambar Obat" dan pilih file gambar.
            2. *Atau Ambil Foto* — Aktifkan kamera dan ambil foto (otomatis 1:1).
            3. *Tunggu Analisis* — AI akan menganalisis gambar obat Anda.
            4. *Lihat Hasil* — Dapatkan *Detail Informasi* obat.
            5. *Dengarkan Audio* — Ringkasan *tidak otomatis*; tekan tombol untuk memutar.
            6. *Informasi Lainnya* — Klik item di daftar; panel kanan muncul & audio langsung memandu.

            *Tips:*
            - Pastikan gambar jelas, tajam, dan tidak blur.
            - Gunakan latar netral; hindari pantulan.
            - Foto kemasan asli untuk akurasi terbaik.
            """)

if __name__ == "__main__":
    main()