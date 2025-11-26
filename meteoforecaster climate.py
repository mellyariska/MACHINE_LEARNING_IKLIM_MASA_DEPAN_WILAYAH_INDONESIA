import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from io import BytesIO

# ============================================================
# 1. CONFIG DASBOR
# ============================================================
st.set_page_config(
    page_title="Dashboard Iklim Sumatera Selatan",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌦️ Dashboard Analisis & Prediksi Cuaca — Sumatera Selatan")

st.markdown("""
Dashboard ini menampilkan analisis data iklim (suhu, kelembaban, angin, curah hujan) serta model machine learning 
untuk memprediksi kondisi cuaca **Wilayah Sumatera Selatan**.
""")

# ============================================================
# 2. SIDEBAR
# ============================================================
st.sidebar.header("🔧 Pengaturan Data")

uploaded_file = st.sidebar.file_uploader("Unggah File Excel Iklim", type=["xlsx"])

st.sidebar.markdown("**Format wajib kolom data:**")
st.sidebar.code("""
tanggal
suhu
kelembaban
angin
curah_hujan
""")

# ============================================================
# 3. LOAD DATA
# ============================================================
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Pastikan kolom tanggal benar
    if "tanggal" in df.columns:
        df["tanggal"] = pd.to_datetime(df["tanggal"])

    st.success("File berhasil dimuat!")
else:
    st.warning("Silakan unggah file Excel terlebih dahulu.")
    st.stop()

# ============================================================
# 4. TAMPILKAN DATA
# ============================================================
st.subheader("📊 Data Iklim — Sumatera Selatan")

st.dataframe(df, height=300)

# ============================================================
# 5. STATISTIK RINGKAS
# ============================================================
st.subheader("📈 Statistik Deskriptif")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rata-rata Suhu", round(df["suhu"].mean(), 2))
col2.metric("Rata-rata Kelembaban", round(df["kelembaban"].mean(), 2))
col3.metric("Rata-rata Angin", round(df["angin"].mean(), 2))
col4.metric("Total Curah Hujan", round(df["curah_hujan"].sum(), 2))

# ============================================================
# 6. VISUALISASI
# ============================================================
st.subheader("📉 Visualisasi Tren Iklim")

# Line chart suhu
fig_temp = px.line(df, x="tanggal", y="suhu", title="Perubahan Suhu Harian")
st.plotly_chart(fig_temp, use_container_width=True)

# Line chart kelembaban
fig_hum = px.line(df, x="tanggal", y="kelembaban", title="Perubahan Kelembaban Harian")
st.plotly_chart(fig_hum, use_container_width=True)

# Line chart angin
fig_wind = px.line(df, x="tanggal", y="angin", title="Perubahan Kecepatan Angin")
st.plotly_chart(fig_wind, use_container_width=True)

# Bar chart curah hujan
fig_rain = px.bar(df, x="tanggal", y="curah_hujan", title="Curah Hujan Harian")
st.plotly_chart(fig_rain, use_container_width=True)

# ============================================================
# 7. MACHINE LEARNING PREDIKSI CUACA
# ============================================================
st.subheader("🤖 Prediksi Kondisi Cuaca dengan Random Forest")

# Label otomatis berdasarkan curah hujan
df["label"] = pd.cut(
    df["curah_hujan"],
    bins=[-1, 0, 20, 200],
    labels=["Tidak Hujan", "Hujan Ringan", "Hujan Lebat"],
)

features = df[["suhu", "kelembaban", "angin"]]
labels = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Tampilkan hasil model
st.text("Laporan Akurasi Model:")
st.code(classification_report(y_test, predictions))

# Prediksi seluruh dataset
df["prediksi_cuaca"] = model.predict(features)

st.subheader("📥 Hasil Prediksi Cuaca")
st.dataframe(df, height=300)

# ============================================================
# 8. DOWNLOAD
# ============================================================
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="XlsxWriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Prediksi")

st.download_button(
    label="📥 Download Hasil Prediksi (Excel)",
    data=buffer.getvalue(),
    file_name="hasil_prediksi_cuaca_sumatera_selatan.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
