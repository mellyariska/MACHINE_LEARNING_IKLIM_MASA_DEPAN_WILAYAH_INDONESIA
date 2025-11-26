import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="DSS Iklim Sumatera Selatan",
    layout="wide"
)

# ============================================================
# 1. DATA DIBUAT LANGSUNG DI DALAM KODE (TANPA FILE EXCEL)
# ============================================================

# membuat 30 hari data simulasi Sumatera Selatan
tanggal = [datetime.today() - timedelta(days=i) for i in range(30)]
tanggal.reverse()

np.random.seed(42)

data = {
    "tanggal": tanggal,
    "suhu": np.random.uniform(24, 33, 30),          # suhu 24–33 °C
    "kelembaban": np.random.uniform(65, 95, 30),    # kelembaban 65–95%
    "angin": np.random.uniform(1, 12, 30)           # angin 1–12 m/s
}

df = pd.DataFrame(data)

# ============================================================
# 2. HEADER
# ============================================================
st.title("🌦️ Dashboard Sistem Pendukung Keputusan Iklim")
st.subheader("Wilayah Sumatera Selatan – Machine Learning Climate DSS")

st.markdown("""Dashboard ini menampilkan visualisasi iklim dan hasil prediksi 
berbasis Machine Learning untuk wilayah Sumatera Selatan **tanpa upload data**.""")

st.divider()

# ============================================================
# 3. METRIC IKLIM
# ============================================================
col1, col2, col3 = st.columns(3)

col1.metric("Rata-rata Suhu (°C)", round(df["suhu"].mean(), 2))
col2.metric("Rata-rata Kelembaban (%)", round(df["kelembaban"].mean(), 2))
col3.metric("Rata-rata Kecepatan Angin (m/s)", round(df["angin"].mean(), 2))

st.divider()

# ============================================================
# 4. TREND VISUALIZATION
# ============================================================
st.subheader("📈 Tren Parameter Iklim Harian")

fig = px.line(
    df,
    x="tanggal",
    y=["suhu", "kelembaban", "angin"],
    labels={"value": "Nilai", "variable": "Parameter"},
    title="Tren Suhu, Kelembaban, dan Angin"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# 5. HEATMAP KORELASI
# ============================================================
st.subheader("📊 Korelasi Antar Variabel Iklim")

cor = df[["suhu", "kelembaban", "angin"]].corr()
fig_cor = px.imshow(cor, text_auto=True, title="Heatmap Korelasi Variabel Iklim")
st.plotly_chart(fig_cor, use_container_width=True)

st.divider()

# ============================================================
# 6. DISTRIBUSI PARAMETER
# ============================================================
st.subheader("📉 Distribusi Parameter Iklim")

c1, c2, c3 = st.columns(3)

with c1:
    st.write("Distribusi Suhu")
    st.plotly_chart(px.histogram(df, x="suhu", nbins=20, title="Histogram Suhu"), use_container_width=True)

with c2:
    st.write("Distribusi Kelembaban")
    st.plotly_chart(px.histogram(df, x="kelembaban", nbins=20, title="Histogram Kelembaban"), use_container_width=True)

with c3:
    st.write("Distribusi Kecepatan Angin")
    st.plotly_chart(px.histogram(df, x="angin", nbins=20, title="Histogram Angin"), use_container_width=True)

st.divider()

# ============================================================
# 7. SIMULASI PREDIKSI CUACA (DUMMY ML)
# ============================================================
st.subheader("🔮 Prediksi Cuaca (Simulasi Machine Learning)")

def simulasi_prediksi(suhu, kelembaban, angin):
    if suhu > 30 and kelembaban < 70:
        return "Tidak Hujan"
    elif kelembaban > 85:
        return "Hujan Lebat"
    else:
        return "Hujan Ringan"

colA, colB, colC = st.columns(3)

inp_suhu = colA.number_input("Suhu (°C)", 10, 40, 29)
inp_kel = colB.number_input("Kelembaban (%)", 40, 100, 80)
inp_angin = colC.number_input("Kecepatan Angin (m/s)", 0, 30, 5)

if st.button("Prediksi Cuaca"):
    hasil = simulasi_prediksi(inp_suhu, inp_kel, inp_angin)
    st.success(f"🌤️ **Prediksi Cuaca: {hasil}**")

st.divider()

# ============================================================
# 8. TABEL DATA
# ============================================================
st.subheader("📋 Data Iklim Sumatera Selatan")
st.dataframe(df, use_container_width=True)
