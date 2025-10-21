import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Fungsi Bantuan ---
def plot_histogram(image, title, ax):
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
        ax.set_xlim([0, 256])
    return histr.flatten().astype(int) # Kembalikan nilai angkanya

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
        # Mengubah file yang diupload menjadi format yang bisa dibaca OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # OpenCV membaca gambar sebagai BGR, kita konversi ke RGB untuk tampilan yang benar
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        st.header("1. Citra Asli")
        st.image(img_rgb, caption="Gambar yang Anda upload.", use_container_width=True)

        st.divider()

        # --- Tugas 1: Tampilkan Histogram RGB dan Greyscale ---
        st.header("2. Analisis Histogram")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Histogram RGB")
            fig_rgb, ax_rgb = plt.subplots()
            hist_rgb_data = plot_histogram(original_image, "Histogram RGB", ax_rgb)
            st.pyplot(fig_rgb)
            with st.expander("Lihat data angka Histogram RGB"):
                st.write("Data ini menunjukkan jumlah piksel untuk setiap tingkat intensitas (0-255) pada channel Biru, Hijau, dan Merah.")
                st.write(cv2.calcHist([original_image], [0], None, [256], [0, 256]).flatten().astype(int), " (Biru)")
                st.write(cv2.calcHist([original_image], [1], None, [256], [0, 256]).flatten().astype(int), " (Hijau)")
                st.write(cv2.calcHist([original_image], [2], None, [256], [0, 256]).flatten().astype(int), " (Merah)")


        with col2:
            st.subheader("Histogram Grayscale")
            fig_gray, ax_gray = plt.subplots()
            hist_gray_data = plot_histogram(img_gray, "Histogram Grayscale", ax_gray)
            st.pyplot(fig_gray)
            with st.expander("Lihat data angka Histogram Grayscale"):
                 st.write("Data ini menunjukkan jumlah piksel untuk setiap tingkat keabuan (0-255).")
                 st.write(hist_gray_data)

        st.divider()

        # --- Tugas 2: Thresholding & Citra Biner ---
        st.header("3. Thresholding Menjadi Citra Biner")
        # Menggunakan metode Otsu yang otomatis menemukan threshold optimal
        # (cocok untuk deskripsi "dua puncak histogram")
        threshold_value, binary_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        st.success(f"**Nilai Threshold yang ditemukan (Otsu's Method):** `{int(threshold_value)}`")
        st.image(binary_image, caption=f"Citra Biner dengan Threshold = {int(threshold_value)}", use_container_width=True)

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
            st.info("Perhatikan bagaimana histogram menjadi lebih merata (uniform) setelah equalization, yang biasanya meningkatkan kontras gambar.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar. Pastikan file gambar tidak rusak. Error: {e}")
