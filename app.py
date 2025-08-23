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
# from streamlit_extras.stylable_container import stylable_container
import time

# ========== KONFIGURASI ==========
st.set_page_config(
    page_title="ObatVision", 
    layout="centered",
    page_icon="💊"
)

# ================== LOAD MODEL ================== #
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

# ================== LOAD CSV ================== #
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

# ========== FUNGSI TTS ==========
def speak_text(text, lang='id'):
    """Konversi teks ke audio menggunakan gTTS"""
    try:
        # Buat temporary file untuk audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(tmp_file.name)
            
            # Baca file audio dan encode ke base64
            with open(tmp_file.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Hapus temporary file
            os.unlink(tmp_file.name)
            
            # Encode ke base64 untuk HTML audio player
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # Tampilkan audio player
            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Gagal memutar audio: {str(e)}")

def create_audio_player(text, lang='id'):
    """Buat audio player untuk teks"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            
            with open(tmp_file.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            os.unlink(tmp_file.name)
            
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            return f"""
            <audio controls style="width: 100%;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
    except Exception as e:
        return f"<p>Error creating audio: {str(e)}</p>"

# ========== MAIN APP ==========
def main():
    # Header
    st.title("💊 ObatVision: Deteksi dan Info Obat")
    st.markdown("---")
    
    # Load model dan dataset
    model = load_model_local()
    obat_info_df = load_dataset_local()
    
    if model is None or obat_info_df is None:
        st.error("Gagal memuat model atau database. Silakan refresh halaman.")
        return
    
    # Dapatkan class names
    class_names = sorted(obat_info_df['label'].unique())
    
    # ========== SIDEBAR INPUT ==========
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
        
        # Toggle untuk mengaktifkan kamera
        enable_camera = st.toggle(
            "🔧 Aktifkan Kamera",
            value=False,
            help="Aktifkan toggle ini untuk menggunakan kamera"
        )
        
        # Camera input (hanya muncul jika toggle diaktifkan)
        camera_img = None
        if enable_camera:
            camera_img = st.camera_input(
                "📸 Ambil Gambar Realtime",
                help="Klik untuk mengambil foto obat secara langsung"
            )
        else:
            st.info("💡 Aktifkan toggle di atas untuk menggunakan kamera")
        
        # Pilih input yang akan digunakan
        img_input = img_file if img_file else camera_img
        
        if img_input:
            st.success("✅ Gambar berhasil dimuat!")
    
    # ========== MAIN CONTENT ==========
    if img_input:
        # Proses gambar
        with st.spinner("🔍 Menganalisis gambar obat..."):
            try:
                # Load dan resize gambar
                img = Image.open(img_input).convert('RGB')
                img_resized = img.resize((256, 256))
                
                # Preprocessing untuk model
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediksi
                prediction = model.predict(img_array)[0]
                predicted_index = np.argmax(prediction)
                predicted_label = class_names[predicted_index]
                confidence = prediction[predicted_index] * 100
                
                # Ambil info dari CSV
                info = obat_info_df[obat_info_df['label'] == predicted_label].iloc[0]
                
            except Exception as e:
                st.error(f"Gagal memproses gambar: {str(e)}")
                return
        
        # ========== HASIL PREDIKSI ==========
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(
                img, 
                caption=f"Hasil Deteksi: {predicted_label}", 
                use_column_width=True
            )
        
        with col2:
            st.subheader(f"💊 {info['nama_obat']}")
            st.metric("🎯 Akurasi Prediksi", f"{confidence:.2f}%")
            
            # Info dasar
            st.markdown(f"""
            **Golongan:** {info['golongan']}  
            **Jenis:** {info['jenis']}  
            """)
        
        # ========== INFO LENGKAP ==========
        st.markdown("---")
        st.subheader("📋 Informasi Lengkap Obat")
        
        with st.expander("🔍 Lihat Detail", expanded=True):
            st.markdown(f"""
            **Manfaat:** {info['manfaat']}  
            **Aturan Minum:** {info['aturan_minum']}  
            **Catatan:** {info['catatan']}  
            """)
        
        # ========== PERINGATAN ==========
        st.warning("""
        ⚠️ **PERINGATAN PENTING:** 
        🥧 Aturan minum dapat berbeda-beda pada setiap orang. Harus mengikuti saran dari dokter yang sudah cek kondisi pasien. 
        Kira-kira solusinya mungkin bisa menambahkan fitur untuk koreksi jadwal minum obat, jika memungkinkan.
        """)
        
        # Text untuk TTS
        main_text = f"""
        Obat yang terdeteksi adalah {info['nama_obat']}. 
        Aturan minum: {info['aturan_minum']}. 
        Perhatian: {info['catatan']}. 
        Peringatan: 🥧 Aturan minum dapat berbeda-beda pada setiap orang. Harap ikuti petunjuk dari dokter yang sudah memeriksa kondisi Anda.
        """
        
        # Auto play audio
        speak_text(main_text)
        
        # ========== MENU LANJUTAN ==========
        st.markdown("---")
        st.subheader("📂 Lihat lebih lanjut:")
        
        # Custom CSS for buttons
        st.markdown("""
        <style>
        .stButton > button {
            background-color: #417eba;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin: 0.3rem 0;
            transition: all 0.3s ease;
            font-weight: 500;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #e9ecef;
            border-color: #007bff;
            transform: translateY(-2px);
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔴 Efek Samping", key="efek_samping"):
                efek_samping = info.get('efek_samping', 'Informasi tidak tersedia')
                st.info(f"**Efek Samping {info['nama_obat']}:**\n{efek_samping}")
                audio_html = create_audio_player(f"Efek samping dari {info['nama_obat']}: {efek_samping}")
                st.markdown(audio_html, unsafe_allow_html=True)
            
            if st.button("🚫 Pantangan Makanan", key="pantangan"):
                pantangan = info.get('pantangan_makanan', 'Informasi tidak tersedia')
                st.info(f"**Pantangan Makanan:**\n{pantangan}")
                audio_html = create_audio_player(f"Pantangan makanan: {pantangan}")
                st.markdown(audio_html, unsafe_allow_html=True)
            
            if st.button("⚠️ Interaksi Negatif", key="interaksi"):
                interaksi = info.get('interaksi_negatif', 'Informasi tidak tersedia')
                st.info(f"**Interaksi Negatif:**\n{interaksi}")
                audio_html = create_audio_player(f"Interaksi negatif: {interaksi}")
                st.markdown(audio_html, unsafe_allow_html=True)
        
        with col2:
            if st.button("🤔 Jika Lupa Minum?", key="lupa_minum"):
                lupa_minum = info.get('jika_lupa_minum', 'Informasi tidak tersedia')
                st.info(f"**Jika Lupa Minum:**\n{lupa_minum}")
                audio_html = create_audio_player(f"Jika lupa minum: {lupa_minum}")
                st.markdown(audio_html, unsafe_allow_html=True)
            
            if st.button("📦 Cara Penyimpanan", key="penyimpanan"):
                penyimpanan = info.get('penyimpanan', 'Informasi tidak tersedia')
                st.info(f"**Cara Penyimpanan:**\n{penyimpanan}")
                audio_html = create_audio_player(f"Cara penyimpanan: {penyimpanan}")
                st.markdown(audio_html, unsafe_allow_html=True)
    
    else:
        # ========== TAMPILAN AWAL ==========
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🔍 Selamat datang di ObatVision!</h2>
            <p>Aplikasi deteksi obat menggunakan AI untuk membantu Anda mengenali obat dan mendapatkan informasi lengkap tentang penggunaan yang tepat.</p>
            <p><strong>Silakan unggah gambar obat atau ambil foto menggunakan kamera di sidebar.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Panduan penggunaan
        with st.expander("📖 Panduan Penggunaan", expanded=True):
            st.markdown("""
            **Langkah-langkah:**
            1. **Upload Gambar** - Klik "Unggah Gambar Obat" dan pilih file gambar
            2. **Atau Ambil Foto** - Klik "Ambil Gambar Realtime" untuk foto langsung
            3. **Tunggu Analisis** - AI akan menganalisis gambar obat Anda
            4. **Lihat Hasil** - Dapatkan informasi lengkap tentang obat
            5. **Dengarkan Audio** - Informasi akan dibacakan secara otomatis
            6. **Jelajahi Detail** - Klik menu untuk info lebih lanjut
            
            **Tips:**
            - Pastikan gambar obat jelas dan fokus
            - Pencahayaan yang baik akan meningkatkan akurasi
            - Obat sebaiknya dalam kemasan asli
            """)

if __name__ == "__main__":
    main()