import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import requests # Diperlukan untuk download gambar contoh

# --- Konfigurasi Halaman & Styling ---
st.set_page_config(layout="wide", page_title="Editor Citra Digital")

st.title("🖼️ Editor Citra Digital Interaktif")
st.write("Aplikasi ini memungkinkan Anda untuk melakukan analisis histogram, thresholding biner, dan equalisasi histogram pada sebuah gambar.")

# --- Fungsi Bantuan ---

@st.cache_data
def load_sample_image(url):
    """Fungsi untuk mengunduh dan memuat gambar contoh dari URL."""
    try:
        # Header untuk menyamar sebagai browser agar tidak diblokir (Error 403)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Cek jika ada error
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return np.array(img)
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal memuat gambar contoh: {e}")
        return None

def plot_histogram(image, title, ax, threshold_value=None):
    """Fungsi untuk menghitung dan memplot histogram gambar."""
    if len(image.shape) == 3: # Gambar RGB
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(histr, color=col)
        ax.set_title(title)
        ax.set_xlim([0, 256])
    else: # Gambar Grayscale
        histr = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(histr, color='gray')
        ax.set_title(title)
        ax.set_xlabel("Intensitas")
        ax.set_ylabel("Frekuensi")
        ax.set_xlim([0, 256])
        if threshold_value is not None:
            ax.axvline(x=threshold_value, color='r', linestyle='--', label=f'Threshold={int(threshold_value)}')
            ax.legend()

# --- Sidebar untuk Pengaturan ---
st.sidebar.title("⚙️ Panel Pengaturan")

# 1. Pengaturan Input Gambar
st.sidebar.header("1. Pengaturan Input")
use_sample = st.sidebar.checkbox("Gunakan gambar contoh (Lena)", value=True)
uploaded_file = st.sidebar.file_uploader("Atau upload gambar Anda", type=["jpg", "png", "jpeg"], disabled=use_sample)

image_to_process = None
if use_sample:
    # URL gambar Lena yang umum digunakan dalam pengolahan citra
    lena_url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    image_to_process = load_sample_image(lena_url)
elif uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB) # Konversi ke RGB

# --- Panel Utama ---
if image_to_process is not None:
    img_gray = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2GRAY)

    # 2. Opsi Thresholding di Sidebar
    st.sidebar.header("2. Opsi Thresholding")
    threshold_method = st.sidebar.radio(
        "Pilih Metode Thresholding",
        ["Manual", "Otsu Otomatis", "Mean Grayscale"]
    )

    info_message = ""
    if threshold_method == "Manual":
        threshold_value = st.sidebar.slider("Atur Threshold Manual", 0, 255, 127)
        info_message = f"Metode: Manual. Threshold diatur ke **{threshold_value}**."
    elif threshold_method == "Otsu Otomatis":
        threshold_value, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        info_message = f"Metode: Otsu Otomatis. Ditemukan threshold optimal di **{int(threshold_value)}**."
    elif threshold_method == "Mean Grayscale":
        threshold_value = int(np.mean(img_gray))
        info_message = f"Metode: Mean Grayscale. Threshold dari rata-rata intensitas adalah **{threshold_value}**."

    # Terapkan thresholding berdasarkan nilai yang didapat
    _, binary_image = cv2.threshold(img_gray, int(threshold_value), 255, cv2.THRESH_BINARY)
    
    # Menampilkan Citra Asli
    st.header("Citra Asli")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_to_process, caption="Citra Asli (RGB)", use_container_width=True)
    with col2:
        st.image(img_gray, caption="Citra Grayscale", use_container_width=True)
    st.divider()

    # Membuat Tabs untuk Hasil Analisis
    tab1, tab2, tab3 = st.tabs(["Analisis Histogram", "Thresholding Biner", "Equalisasi Histogram"])

    with tab1:
        st.header("Grafik Histogram")
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            fig_rgb, ax_rgb = plt.subplots()
            plot_histogram(image_to_process, "Histogram RGB", ax_rgb)
            st.pyplot(fig_rgb)
        with col_hist2:
            fig_gray, ax_gray = plt.subplots()
            plot_histogram(img_gray, "Histogram Grayscale", ax_gray)
            st.pyplot(fig_gray)

    with tab2:
        st.header("Segmentasi Citra Menjadi Hitam & Putih")
        st.info(info_message)
        col_bin1, col_bin2 = st.columns(2)
        with col_bin1:
            st.image(binary_image, caption=f"Hasil Thresholding", use_container_width=True)
            # Logika Tombol Download
            pil_img = Image.fromarray(binary_image)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Citra Biner (.png)",
                data=byte_im,
                file_name="citra_biner.png",
                mime="image/png"
            )
        with col_bin2:
            fig_bin, ax_bin = plt.subplots()
            plot_histogram(img_gray, "Histogram dengan Garis Threshold", ax_bin, threshold_value=threshold_value)
            st.pyplot(fig_bin)

    with tab3:
        st.header("Peningkatan Kontras Citra")
        equalized_image = cv2.equalizeHist(img_gray)
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.image(equalized_image, caption="Hasil Equalisasi Histogram", use_container_width=True)
        with col_eq2:
            fig_eq, ax_eq = plt.subplots()
            plot_histogram(equalized_image, "Histogram Setelah Equalisasi", ax_eq)
            st.pyplot(fig_eq)
        st.info("Equalisasi Histogram meratakan distribusi intensitas piksel untuk meningkatkan kontras global pada gambar.")

else:
    st.warning("Silakan upload gambar atau gunakan gambar contoh untuk memulai.")
