import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io # Diperlukan untuk tombol download

# --- Fungsi Bantuan ---
def plot_histogram(image, title, ax, threshold_value=None):
    """Fungsi untuk menghitung dan memplot histogram gambar."""
    if len(image.shape) == 3: # Gambar RGB
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(histr, color=col)
        ax.set_title(title)
        ax.set_xlim([0, 256])
    else: # Gambar Grayscale
        histr = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(histr, color='gray')
        ax.set_title(title)
        ax.set_xlabel("Intensitas") # Menambahkan label sumbu X
        ax.set_ylabel("Frekuensi") # Menambahkan label sumbu Y
        ax.set_xlim([0, 256])
        # -- PENAMBAHAN BARU: Menggambar garis threshold --
        if threshold_value is not None:
            ax.axvline(x=threshold_value, color='r', linestyle='--', label=f'Threshold={int(threshold_value)}')
            ax.legend()
    return histr.flatten().astype(int)

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Pengolahan Citra Digital")
st.title("Tugas: Thresholding & Equalization")

# --- Sidebar untuk Upload File ---
st.sidebar.header("Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload sebuah gambar (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.info("Silakan upload sebuah gambar melalui sidebar untuk memulai analisis.")
else:
    # --- Membaca dan Menampilkan Gambar Asli ---
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        st.header("1. Citra Asli")
        st.image(img_rgb, caption="Gambar yang Anda upload.", use_container_width=True)
        st.divider()

        # --- Tugas 1: Tampilkan Histogram RGB dan Greyscale ---
        st.header("2. Analisis Histogram")
        
        # --- Thresholding DULU untuk mendapatkan nilainya ---
        threshold_value, binary_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Histogram RGB")
            fig_rgb, ax_rgb = plt.subplots()
            plot_histogram(original_image, "Histogram RGB", ax_rgb)
            st.pyplot(fig_rgb)

        with col2:
            st.subheader("Histogram Grayscale dengan Garis Threshold")
            fig_gray, ax_gray = plt.subplots()
            # -- PENAMBAHAN BARU: Memasukkan threshold_value ke fungsi plot --
            plot_histogram(img_gray, "Histogram dengan Garis Threshold", ax_gray, threshold_value=threshold_value)
            st.pyplot(fig_gray)
            
        st.divider()

        # --- Tugas 2: Thresholding & Citra Biner ---
        st.header("3. Thresholding Menjadi Citra Biner")
        st.success(f"**Nilai Threshold yang ditemukan (Otsu's Method):** `{int(threshold_value)}`")
        st.image(binary_image, caption=f"Citra Biner (Threshold={int(threshold_value)})", use_container_width=True)

        # -- PENAMBAHAN BARU: Tombol Download ---
        # Konversi gambar biner (numpy array) ke format file PNG di memori
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

        st.divider()

        # --- Tugas 3: Histogram Equalization ---
        st.header("4. Histogram Equalization")
        equalized_image = cv2.equalizeHist(img_gray)
        
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.subheader("Citra Hasil Equalization")
            st.image(equalized_image, caption="Hasil setelah proses equalization.", use_container_width=True)
        with col_eq2:
            st.subheader("Histogram Hasil Equalization")
            fig_eq, ax_eq = plt.subplots()
            plot_histogram(equalized_image, "Histogram Setelah Equalization", ax_eq)
            st.pyplot(fig_eq)
            st.info("Perhatikan bagaimana histogram menjadi lebih merata setelah equalization.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar. Pastikan file gambar tidak rusak. Error: {e}")
