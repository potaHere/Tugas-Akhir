import streamlit as st
import pandas as pd

def show_sidebar():
    with st.sidebar:
        # Navigation styling
        st.markdown("""
        <style>
            div[role="radiogroup"] > label > div:first-child {
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
                transition: all 0.3s;
            }
            div[role="radiogroup"] > label > div:first-child:hover {
                background: #f0f2f6;
            }
            .sidebar .sidebar-content {
                padding: 4rem 1rem !important;
            }
        </style>
        """, unsafe_allow_html=True)

        with st.expander("📌 **Panduan Penggunaan**", expanded=True):
            st.markdown("""
            1. 🖼️ Pilih model analisis sentimen yang diinginkan
            2. ✍️ Masukkan teks yang ingin dianalisis
            3. ⏳ Klik tombol prediksi dan tunggu hasil analisis
            4. 📊 Hasil analisis akan ditampilkan di layar
            """)

        with st.expander("📊 **Statistik Model**"):
            st.markdown("""
            - **SVM:**
              - Akurasi: 90%
            - **Naive Bayes:**
              - Akurasi: 72%
            - **KNN:**
              - Akurasi: 65%
            """)

        with st.expander("ℹ️ **Informasi Teknis**"):
            st.markdown("""
            - **🧠 Model yang digunakan:**
              - SVM, Naive Bayes, KNN
            - **📁 Dataset:**
              - Dataset komentar Twitter tentang gaji dan kesehatan mental
            - **🔄 Teknik Preprocessing:**
              - Tokenization, Stopword Removal, TF-IDF Vectorization
            - **⚙️ Optimizer:**
              - SVM: Default
              - Naive Bayes: Default
              - KNN: Default
            """)

        st.markdown("---")
        st.warning("""
        **Disclaimer Medis:**  
        Hasil analisis ini bersifat informatif awal dan tidak menggantikan diagnosis medis profesional. 
        Selalu konsultasikan dengan dokter spesialis untuk pemeriksaan lengkap.
        """)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # Render navigation dengan styling improved
        st.sidebar.markdown("## 🧭 Navigasi Aplikasi")
        app = st.sidebar.radio(
            '',
            self.apps,
            format_func=lambda app: f"👉 {app['title']}",
            label_visibility="collapsed"
        )
        
        # Render sidebar content
        # show_sidebar()
        
        # Eksekusi app function
        app['function']()

if __name__ == "__main__":
    app()