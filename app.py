import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tempfile
import base64
from gtts import gTTS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import io
import time

# ========== KONFIGURASI ========== #
st.set_page_config(
    page_title="MediScan",
    layout="centered",
    page_icon="💊"
)

# Sedikit CSS untuk konsistensi UI
st.markdown("""
<style>
/* Title styling */
.app-title{
  font-size: 1.8rem;
  font-weight: 700;
  margin: 0 0 .25rem 0;
}
.app-sub{
  font-size: .95rem;
  color: #6c757d;
  margin-bottom: .75rem;
}
hr{ border: none; border-top: 1px solid #e9ecef; margin: 1rem 0; }

/* Buttons */
.stButton > button {
    background-color: #0f172a !important;
    border: 2px solid #e9ecef !important;
    border-radius: 12px !important;
    padding: 0.55rem 1rem !important;
    margin: 0.25rem 0 !important;
    transition: all 0.2s ease !important;
    font-weight: 600 !important;
    width: 100% !important;
}
.stButton > button:hover {
    background-color: #e9ecef !important;
    color: #0f172a !important;
    border-color: #0f172a !important;
    transform: translateY(-1px);
}

/* Right panel info box fixed height */
.info-panel{
  border: 1px solid #e9ecef;
  border-radius: 12px;
  padding: 1rem;
  min-height: 220px; /* tinggi seragam */
  background: #4B4B4C;
}

/* Radio label as section label */
.block-label{
  font-weight: 600;
  margin-bottom: .25rem;
}

/* Hide default radio label since we render custom */
div[data-baseweb="radio"] > div:first-child{ display:none; }
</style>
""", unsafe_allow_html=True)

# ================== UTIL: AUDIO ================== #
def _render_audio_base64(audio_bytes: bytes, autoplay: bool):
    audio_base64 = base64.b64encode(audio_bytes).decode()
    auto = "autoplay" if autoplay else ""
    return f"""
    <audio {auto} controls style="width:100%;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """

def speak_text(text, lang='id', autoplay=True):
    """
    Konversi teks ke audio dan render <audio>. 
    Catatan: Untuk suara yang benar-benar natural (mirip manusia),
    sebaiknya integrasi layanan TTS seperti Azure/ElevenLabs.
    """
    try:
        # tambah sedikit tanda baca agar jedanya lebih natural
        refined = text.replace(". ", ".  ").replace(", ", ",  ")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts = gTTS(text=refined, lang=lang, slow=False)
            tts.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
        os.unlink(tmp_file.name)
        st.markdown(_render_audio_base64(audio_bytes, autoplay=autoplay), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Gagal memutar audio: {str(e)}")

def create_audio_player(text, lang='id'):
    """Buat audio player (tanpa autoplay) untuk TTS utama."""
    try:
        refined = text.replace(". ", ".  ").replace(", ", ",  ")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts = gTTS(text=refined, lang=lang, slow=False)
            tts.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
        os.unlink(tmp_file.name)
        return _render_audio_base64(audio_bytes, autoplay=False)
    except Exception as e:
        return f"<p>Error creating audio: {str(e)}</p>"

def stop_all_audio():
    """Hentikan audio dengan JS (pause semua <audio>)"""
    st.markdown("""
    <script>
    const audios = document.getElementsByTagName('audio');
    for (let a of audios){ a.pause(); a.currentTime = 0; }
    </script>
    """, unsafe_allow_html=True)

# ================== LOAD MODEL & DATA ================== #
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

# ================== IMAGE UTILS ================== #
def center_crop_to_square(pil_img: Image.Image) -> Image.Image:
    """Crop tengah menjadi persegi 1:1."""
    w, h = pil_img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    return pil_img.crop((left, top, right, bottom))

def preprocess_for_model(pil_img: Image.Image, size=(256, 256)):
    """Square-center-crop -> resize -> scale 0-1 -> expand dims."""
    sq = center_crop_to_square(pil_img)
    img_resized = sq.resize(size)
    arr = image.img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img_resized  # return resized for display

# ========== STATE KEYS ========== #
ACTIVE_INFO_KEY = "active_info"   # menyimpan menu info yang aktif

if ACTIVE_INFO_KEY not in st.session_state:
    st.session_state[ACTIVE_INFO_KEY] = None

# ================== MAIN APP ================== #
def main():
    # ======== HEADER ======== #
    st.markdown(
        '<div class="app-title"><i>MediScan</i>: Klasifikasi 21 Citra Kemasan Obat dengan Integrasi Text-to-Speech</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="app-sub">📷 Silakan kirim foto obat Anda, pastikan kemasan terlihat dengan jelas.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ======== LOAD MODEL & DATA ======== #
    model = load_model_local()
    obat_info_df = load_dataset_local()
    if model is None or obat_info_df is None or obat_info_df.empty:
        st.error("Gagal memuat model atau database. Silakan refresh halaman.")
        return

    # Dapatkan class names
    class_names = sorted(obat_info_df['label'].astype(str).unique())

    # ======== SIDEBAR INPUT ======== #
    with st.sidebar:
        st.header("📷 Input Gambar")
        st.markdown("Pilih salah satu cara untuk memasukkan gambar obat:")

        # Upload file
        img_file = st.file_uploader(
            "📁 Unggah Gambar Obat",
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG"
        )

        st.markdown("**ATAU**")

        # Toggle kamera
        enable_camera = st.toggle(
            "🔧 Aktifkan Kamera",
            value=False,
            help="Aktifkan untuk menggunakan kamera"
        )

        camera_img = None
        if enable_camera:
            # Catatan: st.camera_input preview square; kita tetap jaga 1:1 dengan center-crop saat preprocessing
            camera_img = st.camera_input(
                "📸 Ambil Gambar Realtime (1:1)",
                help="Pastikan kemasan obat terlihat penuh, tajam, dan tidak blur."
            )
        else:
            st.info("💡 Aktifkan toggle di atas untuk menggunakan kamera")

        # Pilih input yang akan digunakan
        img_input = img_file if img_file else camera_img
        if img_input:
            st.success("✅ Gambar berhasil dimuat!")

    # ======== MAIN CONTENT ======== #
    if img_input:
        with st.spinner("🔍 Menganalisis gambar obat..."):
            try:
                # Buka gambar
                img = Image.open(img_input).convert('RGB')
                # Preprocess: crop 1:1 & resize
                img_array, img_square_resized = preprocess_for_model(img, size=(256, 256))

                # Prediksi
                prediction = model.predict(img_array)[0]
                predicted_index = int(np.argmax(prediction))
                confidence = float(prediction[predicted_index])  # 0..1

                if confidence < 0.00:
                    st.error("❌ Gambar tidak terdeteksi sebagai obat. Silakan coba lagi dengan foto yang lebih jelas.")
                    stop_all_audio()
                    return

                predicted_label = class_names[predicted_index]
                # Ambil info dari CSV
                row = obat_info_df[obat_info_df['label'].astype(str) == str(predicted_label)]
                if row.empty:
                    st.error("Data obat tidak ditemukan pada database.")
                    stop_all_audio()
                    return
                info = row.iloc[0]

            except Exception as e:
                st.error(f"Gagal memproses gambar: {str(e)}")
                stop_all_audio()
                return

        # ======== HASIL PREDIKSI ======== #
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(
                img_square_resized,
                caption=f"Hasil Deteksi: {predicted_label}",
                use_column_width=True
            )

        with col2:
            st.subheader(f"💊 {info['nama_obat']}")
            st.metric("🎯 Akurasi Prediksi", f"{confidence*100:.2f}%")
            st.markdown(f"""
            **Golongan:** {info.get('golongan','-')}  
            **Jenis:** {info.get('jenis','-')}  
            """)

        # ======== DETAIL INFORMASI ======== #
        st.markdown("---")
        st.subheader("📋 Detail Informasi")
        with st.expander("🔍 Lihat Detail", expanded=True):
            st.markdown(f"""
            **Manfaat:** {info.get('manfaat','-')}  
            **Aturan Minum:** {info.get('aturan_minum','-')}  
            **Catatan:** {info.get('catatan','-')}  
            """)

        # ======== PERINGATAN ======== #
        st.warning("""
        ⚠️ **PERINGATAN PENTING:** 
        Aturan minum dapat berbeda pada setiap orang. Ikuti saran dokter yang sudah memeriksa kondisi pasien.
        """)

        # ======== NARASI TTS UTAMA (MANUAL PLAY) ======== #
        main_text = (
            f"Nama obat ini adalah {info.get('nama_obat','-')}. "
            f"Golongan {info.get('golongan','-')}. "
            f"Merupakan {info.get('jenis','-')}, yang memiliki manfaat untuk {info.get('manfaat','-')}. "
            f"Dapat diminum {info.get('aturan_minum','-')}. "
            f"Dengan catatan, {info.get('catatan','-')}."
        )

        if st.button("🔊 Dengarkan Info Obat"):
            audio_html = create_audio_player(main_text)
            st.markdown(audio_html, unsafe_allow_html=True)

        # ======== INFORMASI LAINNYA (KIRI: MENU VERTIKAL, KANAN: PANEL) ======== #
        st.markdown("---")
        st.subheader("📂 Informasi Lainnya")

        left, right = st.columns([1, 2])

        # Helper untuk toggle tombol info
        def toggle_info(label: str):
            if st.session_state[ACTIVE_INFO_KEY] == label:
                st.session_state[ACTIVE_INFO_KEY] = None
                # hentikan audio ketika menutup
                stop_all_audio()
            else:
                st.session_state[ACTIVE_INFO_KEY] = label
                # hentikan audio sebelumnya (jika ada)
                stop_all_audio()

        with left:
            st.markdown('<div class="block-label">Pilih informasi:</div>', unsafe_allow_html=True)
            if st.button("🔴 Efek Samping"):
                toggle_info("Efek Samping")
            if st.button("🚫 Pantangan Makanan"):
                toggle_info("Pantangan Makanan")
            if st.button("⚠️ Interaksi Negatif"):
                toggle_info("Interaksi Negatif")
            if st.button("🤔 Jika Lupa Minum?"):
                toggle_info("Jika Lupa Minum?")
            if st.button("📦 Cara Penyimpanan"):
                toggle_info("Cara Penyimpanan")

        with right:
            active = st.session_state[ACTIVE_INFO_KEY]
            if active:
                if active == "Efek Samping":
                    text = info.get('efek_samping', 'Informasi tidak tersedia')
                elif active == "Pantangan Makanan":
                    text = info.get('pantangan_makanan', 'Informasi tidak tersedia')
                elif active == "Interaksi Negatif":
                    text = info.get('interaksi_negatif', 'Informasi tidak tersedia')
                elif active == "Jika Lupa Minum?":
                    text = info.get('jika_lupa_minum', 'Informasi tidak tersedia')
                elif active == "Cara Penyimpanan":
                    text = info.get('penyimpanan', 'Informasi tidak tersedia')
                else:
                    text = "Informasi tidak tersedia"

                # Panel dengan tinggi seragam
                st.markdown(f"""
                <div class="info-panel">
                    <strong>{active} {info.get('nama_obat','')}:</strong><br/>{text}
                </div>
                """, unsafe_allow_html=True)

                # Otomatis TTS ketika menu ditekan
                speak_text(f"{active}: {text}", autoplay=True)
            else:
                # Kosongkan panel & hentikan audio
                st.markdown(f"""
                <div class="info-panel">
                    <em>Tidak ada informasi yang dipilih.</em>
                </div>
                """, unsafe_allow_html=True)
                stop_all_audio()

    else:
        # ======== TAMPILAN AWAL ======== #
        st.markdown("""
        <div style="text-align: center; padding: 1.25rem;">
            <h3>🔍 Selamat datang di <i>MediScan</i>!</h3>
            <p>Aplikasi deteksi obat menggunakan AI untuk membantu Anda mengenali obat dan mendapatkan informasi yang jelas.</p>
            <p><strong>Silakan unggah gambar obat atau ambil foto menggunakan kamera di sidebar.</strong></p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📖 Panduan Penggunaan", expanded=True):
            st.markdown("""
            **Langkah-langkah:**
            1. **Upload Gambar** - Klik "Unggah Gambar Obat" dan pilih file gambar.
            2. **Atau Ambil Foto** - Aktifkan kamera dan ambil foto (rasio 1:1).
            3. **Tunggu Analisis** - AI menganalisis gambar obat Anda.
            4. **Lihat Hasil** - Dapatkan informasi ringkas & detail obat.
            5. **Dengarkan Audio** - Tekan tombol untuk mendengarkan narasi.
            6. **Informasi Lainnya** - Pilih topik; info muncul di panel kanan dan otomatis dibacakan.

            **Tips:**
            - Pastikan gambar **jelas** dan **fokus**.
            - Gunakan pencahayaan baik, hindari bayangan.
            - Usahakan kemasan **asli** dan penuh dalam frame.
            """)

if __name__ == "__main__":
    main()
