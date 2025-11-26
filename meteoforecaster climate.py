##############################################
# DASHBOARD DSS IKLIM – SUMATERA SELATAN
# Tanpa Upload Data – Data Langsung Dibaca
##############################################

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------------------------------
# CONFIGURASI HALAMAN
# -------------------------------------------------
st.set_page_config(
    page_title="Dashboard DSS Iklim – Sumatera Selatan",
    page_icon="🌦️",
    layout="wide"
)

# -------------------------------------------------
# LOAD DATA OTOMATIS
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("SUMSEL.xlsx")
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    return df

df = load_data()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
# 🌦️ Dashboard Analisis & Prediksi Cuaca — **Sumatera Selatan**
Dashboard ini menyajikan analisis data iklim (suhu, kelembaban, angin, curah hujan)  
serta model *machine learning* untuk memprediksi kategori cuaca.
""")

# -------------------------------------------------
# DATA SUMMARY
# -------------------------------------------------
st.subheader("📊 Ringkasan Data Iklim")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rata-rata Suhu (°C)", f"{df['suhu'].mean():.2f}")
col2.metric("Rata-rata Kelembaban (%)", f"{df['kelembaban'].mean():.2f}")
col3.metric("Rata-rata Angin (m/s)", f"{df['angin'].mean():.2f}")
col4.metric("Total Curah Hujan", f"{df['curah_hujan'].sum():.2f} mm")

st.divider()

# -------------------------------------------------
# VISUALISASI DATA
# -------------------------------------------------
st.header("📈 Visualisasi Data Iklim")

tab1, tab2, tab3, tab4 = st.tabs(["Suhu", "Kelembaban", "Angin", "Curah Hujan"])

with tab1:
    fig = px.line(df, x='tanggal', y='suhu', title="Tren Suhu Harian")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.line(df, x='tanggal', y='kelembaban', title="Tren Kelembaban Harian")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.line(df, x='tanggal', y='angin', title="Tren Kecepatan Angin Harian")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = px.bar(df, x='tanggal', y='curah_hujan', title="Curah Hujan Harian")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# MACHINE LEARNING PREDIKSI CUACA
# -------------------------------------------------
st.header("🤖 Prediksi Cuaca Menggunakan Random Forest")

st.markdown("""
Model berikut dikembangkan untuk memprediksi cuaca (Hujan Lebat, Hujan Ringan, Tidak Hujan)  
berdasarkan variabel iklim.
""")

# Pastikan ada kolom label cuaca
if "cuaca" not in df.columns:
    st.error("Kolom 'cuaca' tidak ditemukan. Tambahkan kategori cuaca pada data.")
else:
    X = df[['suhu', 'kelembaban', 'angin', 'curah_hujan']]
    y = df['cuaca']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Tampilkan hasil evaluasi
    y_pred = model.predict(X_test)

    st.subheader("📘 Hasil Evaluasi Model")
    st.text(classification_report(y_test, y_pred))

    # Prediksi manual
    st.subheader("🧪 Coba Prediksi Cuaca")

    cs, ck, ca, ch = st.columns(4)
    suhu = cs.number_input("Suhu (°C)", value=28.0)
    kelembaban = ck.number_input("Kelembaban (%)", value=75.0)
    angin = ca.number_input("Angin (m/s)", value=3.0)
    curah = ch.number_input("Curah Hujan (mm)", value=0.0)

    if st.button("Prediksi Cuaca"):
        pred = model.predict([[suhu, kelembaban, angin, curah]])[0]
        st.success(f"🌤️ Prediksi Cuaca: **{pred}**")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.markdown("""
### © 2025 – Dashboard Iklim Sumatera Selatan  
Dikembangkan menggunakan **Streamlit & Machine Learning**
""")
