import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from io import BytesIO

# === CONFIG ===
st.set_page_config(
    page_title="DSS Iklim – Sumatera Selatan",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌦️ DSS Iklim — Sumatera Selatan")
st.markdown("Dashboard analisis & prediksi cuaca berdasarkan data iklim wilayah Sumatera Selatan.")

# === AUTO LOAD DATA TANPA UPLOAD ===
DATA_PATH = "SUMSEL.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
    st.success("Data SUMSEL.xlsx berhasil dimuat otomatis!")
except:
    st.error("❌ File SUMSEL.xlsx tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py")
    st.stop()

# Normalisasi kolom
df.columns = (
    df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
)

# Pastikan kolom ada
required = ["tanggal", "curah_hujan", "suhu", "kelembaban", "angin"]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"❌ Kolom wajib hilang: {missing}. Periksa kembali file SUMSEL.xlsx Anda.")
    st.stop()

df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

# === SIDEBAR SETTINGS ===
st.sidebar.header("⚙ Pengaturan")
th_rain = st.sidebar.slider("Threshold Hujan Ekstrem (mm/hari)", 10, 150, 50)

# === DATA & STATISTIK ===
st.subheader("📊 Data dan Statistik Dasar")

with st.expander("Tampilkan Data"):
    st.dataframe(df, height=300)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rata-rata Curah Hujan", f"{df['curah_hujan'].mean():.2f}")
col2.metric("Rata-rata Suhu", f"{df['suhu'].mean():.2f}")
col3.metric("Rata-rata Kelembaban", f"{df['kelembaban'].mean():.2f}")
col4.metric("Rata-rata Angin", f"{df['angin'].mean():.2f}")

# === VISUALISASI ===
st.subheader("📈 Tren Iklim Harian")

fig1 = px.bar(df, x="tanggal", y="curah_hujan", title="Curah Hujan Harian (mm)")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(df, x="tanggal", y="suhu", title="Suhu Harian (°C)")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.line(df, x="tanggal", y="kelembaban", title="Kelembaban Harian (%)")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.line(df, x="tanggal", y="angin", title="Kecepatan Angin Harian (m/s)")
st.plotly_chart(fig4, use_container_width=True)

# === KLASIFIKASI SEDERHANA ===
st.subheader("🌤️ Klasifikasi Cuaca Sederhana")

df["prediksi_simple"] = np.where(df["curah_hujan"] > th_rain, "Hujan", "Cerah/Ringan")

st.dataframe(df[["tanggal", "curah_hujan", "prediksi_simple"]].tail(20))

# === MACHINE LEARNING ===
st.subheader("🤖 Prediksi Cuaca — Random Forest")

df_ml = df.dropna(subset=["suhu", "kelembaban", "angin", "curah_hujan"]).copy()

df_ml["label"] = pd.cut(
    df_ml["curah_hujan"],
    bins=[-1, th_rain, df_ml["curah_hujan"].max()],
    labels=["Cerah/Ringan", "Hujan"],
    include_lowest=True
)

X = df_ml[["suhu", "kelembaban", "angin"]]
y = df_ml["label"]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
except:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.text("📄 Classification Report:")
st.code(classification_report(y_test, y_pred, zero_division=0))

df["prediksi_ml"] = model.predict(
    df[["suhu", "kelembaban", "angin"]].fillna(method="ffill").fillna(method="bfill")
)

st.subheader("🧮 Prediksi Cuaca ML")
st.dataframe(df[["tanggal", "curah_hujan", "prediksi_ml"]].tail(20))

# === DOWNLOAD ===
st.subheader("📥 Unduh Hasil Analisis")

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Hasil")

buffer.seek(0)

st.download_button(
    label="Download Excel",
    data=buffer,
    file_name="hasil_prediksi_iklim_sumsel.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
