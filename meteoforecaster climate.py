import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from io import BytesIO

# --- CONFIG ---
st.set_page_config(
    page_title="DSS Iklim – Sumatera Selatan",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌦️ DSS Iklim — Sumatera Selatan")
st.markdown("Dashboard analisis & prediksi cuaca berdasarkan data iklim lokal.")

# --- SIDEBAR: Upload Data & Pengaturan ---
uploaded_file = st.sidebar.file_uploader("Unggah data Excel (.xlsx)", type=["xlsx"])
st.sidebar.write("📄 Data harus minimal berisi kolom: `tanggal, curah_hujan, suhu, kelembaban, angin`")

# Optional: parameter threshold / settings
st.sidebar.markdown("---")
th_rain = st.sidebar.slider("Threshold Hujan Ekstrem (mm/hari)", min_value=1, max_value=200, value=50)
st.sidebar.markdown("---")

# --- LOAD DATA ---
if not uploaded_file:
    st.warning("Tolong unggah file Excel Anda terlebih dahulu.")
    st.stop()

df = pd.read_excel(uploaded_file)

# Normalisasi nama kolom
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "", regex=True)

required = ["tanggal", "curah_hujan", "suhu", "kelembaban", "angin"]
for col in required:
    if col not in df.columns:
        st.error(f"Kolom wajib `{col}` tidak ditemukan. Periksa nama kolom Anda.")
        st.stop()

df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

# --- TAMPILAN DATA & Statistik ---
st.subheader("📊 Data dan Statistik Dasar")
with st.expander("Tampilkan data mentah"):
    st.dataframe(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rata-rata Curah Hujan", f"{df['curah_hujan'].mean():.2f}")
col2.metric("Rata-rata Suhu", f"{df['suhu'].mean():.2f}")
col3.metric("Rata-rata Kelembaban", f"{df['kelembaban'].mean():.2f}")
col4.metric("Rata-rata Kecepatan Angin", f"{df['angin'].mean():.2f}")

# --- VISUALISASI Tren Iklim ---
st.subheader("📈 Tren Iklim Harian")
fig_rain = px.bar(df, x="tanggal", y="curah_hujan", title="Curah Hujan Harian")
st.plotly_chart(fig_rain, use_container_width=True)

fig_temp = px.line(df, x="tanggal", y="suhu", title="Suhu Harian")
st.plotly_chart(fig_temp, use_container_width=True)

fig_hum = px.line(df, x="tanggal", y="kelembaban", title="Kelembaban Harian")
st.plotly_chart(fig_hum, use_container_width=True)

fig_wind = px.line(df, x="tanggal", y="angin", title="Kecepatan Angin Harian")
st.plotly_chart(fig_wind, use_container_width=True)

# --- Klasifikasi Cuaca sederhana berdasarkan curah hujan & suhu (contoh) ---
st.subheader("🌤️ Klasifikasi Cuaca Sederhana")
df["cuaca_prediksi_simple"] = np.where(df["curah_hujan"] > th_rain, "Hujan", "Cerah / Ringan")
st.dataframe(df[["tanggal", "curah_hujan", "cuaca_prediksi_simple"]].tail(20))

# --- MACHINE LEARNING PREDIKSI CUACA (contoh) ---
st.subheader("🤖 Prediksi Cuaca — Random Forest (Dummy)")

df_ml = df.dropna(subset=["suhu", "kelembaban", "angin", "curah_hujan"])
df_ml["label"] = pd.cut(df_ml["curah_hujan"],
                       bins=[-1, th_rain, df_ml["curah_hujan"].max()],
                       labels=["Cerah/Ringan", "Hujan"],
                       include_lowest=True)

X = df_ml[["suhu", "kelembaban", "angin"]]
y = df_ml["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
st.text("📄 Classification Report:")
st.code(classification_report(y_test, y_pred))

df["prediksi_cuaca_ml"] = model.predict(df[["suhu", "kelembaban", "angin"]].fillna(0))

st.subheader("🧮 Hasil Prediksi Cuaca (ML)")
st.dataframe(df[["tanggal", "curah_hujan", "prediksi_cuaca_ml"]].tail(20))

# --- DOWNLOAD HASIL ---
st.subheader("📥 Unduh Hasil")
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="XlsxWriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Hasil")
    writer.save()
buffer.seek(0)
st.download_button(
    "Download Excel Hasil Analisis & Prediksi",
    data=buffer.getvalue(),
    file_name="hasil_prediksi_iklim_sumsel.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
