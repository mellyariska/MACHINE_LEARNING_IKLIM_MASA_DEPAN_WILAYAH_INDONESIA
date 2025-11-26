# app_streamlit_dss_sumsel.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------- CONFIG ----------
st.set_page_config(page_title="DSS Iklim - Sumatera Selatan", layout="wide")
st.title("🌦️ Decision Support System Iklim — Sumatera Selatan")
st.markdown(
    "Dashboard prediksi & analisis iklim. Data akan dimuat dari file Excel `SUMSEL.xlsx` "
    "yang telah Anda unggah."
)

LOCAL_EXCEL_PATH = "SUMSEL.xlsx"

# ---------- DSS helper functions ----------
def klasifikasi_cuaca(ch, matahari):
    if ch > 20:
        return "Hujan"
    elif ch > 5:
        return "Berawan"
    elif matahari > 4:
        return "Cerah"
    else:
        return "Berawan"

def risiko_kekeringan_score(ch, matahari):
    ch_clamped = np.clip(ch, 0, 200)
    matahari_clamped = np.clip(matahari, 0, 16)
    score = (1 - (ch_clamped / 200)) * 0.7 + (matahari_clamped / 16) * 0.3
    return float(np.clip(score, 0, 1))

def risiko_kekeringan_label(score, thresholds=(0.6, 0.3)):
    high, med = thresholds
    if high < med:
        high, med = med, high
    if score >= high:
        return "Risiko Tinggi"
    elif score >= med:
        return "Risiko Sedang"
    else:
        return "Risiko Rendah"

def hujan_ekstrem_flag(ch, threshold=50):
    return int(ch > threshold)

def compute_weather_index(df):
    eps = 1e-6
    r = df['curah_hujan'].astype(float).values
    r_norm = (r - r.min()) / (r.max() - r.min() + eps)

    t = df['Tavg'].astype(float).values
    comfy_low, comfy_high = 24, 28
    t_dist = np.maximum(0, np.maximum(comfy_low - t, t - comfy_high))
    t_norm = (t_dist - t_dist.min()) / (t_dist.max() - t_dist.min() + eps)

    h = df['kelembaban'].astype(float).values
    hum_dist = np.maximum(0, np.maximum(40 - h, h - 70))
    h_norm = (hum_dist - hum_dist.min()) / (hum_dist.max() - hum_dist.min() + eps)

    w = df['kecepatan_angin'].astype(float).values
    w_norm = (w - w.min()) / (w.max() - w.min() + eps)

    composite = 0.35 * r_norm + 0.25 * t_norm + 0.2 * h_norm + 0.2 * w_norm
    return np.clip(composite, 0, 1)

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_data(local_path=LOCAL_EXCEL_PATH):
    try:
        df = pd.read_excel(local_path, parse_dates=['Tanggal'])
        st.sidebar.success(f"Loaded local Excel: {local_path}")
    except Exception:
        st.sidebar.error("File SUMSEL.xlsx tidak ditemukan!")
        st.stop()

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

    needed = ['curah_hujan', 'Tn', 'Tx', 'Tavg', 'kelembaban', 'matahari', 'kecepatan_angin']
    for col in needed:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.sort_values('Tanggal').reset_index(drop=True)

# Load dataset
data = load_data()

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
extreme_threshold = st.sidebar.number_input("Ambang Hujan Ekstrem (mm/hari)", value=50, min_value=1)
risk_high = st.sidebar.slider("Ambang Risiko Tinggi", 0.0, 1.0, 0.6, 0.01)
risk_med = st.sidebar.slider("Ambang Risiko Sedang", 0.0, 1.0, 0.3, 0.01)
ma_window = st.sidebar.slider("Moving average window (hari)", 1, 60, 7)

# Filter Data
st.sidebar.header("📅 Filter data")
min_date = data['Tanggal'].min().date()
max_date = data['Tanggal'].max().date()
date_range = st.sidebar.date_input("Rentang tanggal", (min_date, max_date))

region = None
if 'Wilayah' in data.columns:
    regions = ['All'] + sorted(data['Wilayah'].unique())
    region = st.sidebar.selectbox("Pilih Wilayah", regions)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (data['Tanggal'] >= start_date) & (data['Tanggal'] <= end_date)

if region and region != "All":
    mask &= (data['Wilayah'] == region)

df = data.loc[mask].copy()
if df.empty:
    st.warning("Tidak ada data pada filter ini — menampilkan seluruh dataset.")
    df = data.copy()

# Derived fields
df['Prediksi Cuaca'] = df.apply(lambda r: klasifikasi_cuaca(r['curah_hujan'], r['matahari']), axis=1)
df['Hujan Ekstrem'] = df['curah_hujan'].apply(lambda x: "Ya" if x > extreme_threshold else "Tidak")
df['extreme_flag'] = df['curah_hujan'].apply(lambda x: hujan_ekstrem_flag(x, extreme_threshold))
df['RiskScore'] = df.apply(lambda r: risiko_kekeringan_score(r['curah_hujan'], r['matahari']), axis=1)
df['RiskLabel'] = df['RiskScore'].apply(lambda s: risiko_kekeringan_label(s, (risk_high, risk_med)))
df['WeatherIndex'] = compute_weather_index(df)
df['Year'] = df['Tanggal'].dt.year
df['Month'] = df['Tanggal'].dt.month

# ---------------- 6. Trend Bulanan ----------------
st.markdown("---")
st.header("6. Tren Kualitas Iklim Bulanan/Tahunan")
m1, m2 = st.columns(2)

with m1:
    years = sorted(df['Year'].unique())
    fig_multi = go.Figure()

    for y in years:
        tmp = df[df['Year'] == y].copy()
        monthly = tmp.groupby(tmp['Tanggal'].dt.month)['curah_hujan'].mean().reset_index()
        fig_multi.add_trace(go.Scatter(
            x=monthly['Tanggal'],
            y=monthly['curah_hujan'],
            mode='lines+markers',
            name=str(y)
        ))

    fig_multi.update_layout(
        title="Monthly Average Rainfall by Year",
        xaxis_title="Month",
        yaxis_title="Rain (mm)"
    )
    st.plotly_chart(fig_multi, use_container_width=True)

with m2:
    df['Rain_MA'] = df['curah_hujan'].rolling(window=ma_window).mean()
    fig_ma = px.line(df, x='Tanggal', y=['curah_hujan', 'Rain_MA'],
                     title=f"Moving Average Rainfall (window={ma_window})")
    st.plotly_chart(fig_ma, use_container_width=True)

# ---------------- Export data ----------------
st.markdown("---")
with st.expander("📁 Lihat & Unduh Data"):
    st.dataframe(df)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Hasil_DSS', index=False)

    buffer.seek(0)

    st.download_button(
        "Unduh Excel Hasil Analisis",
        buffer.getvalue(),
        "hasil_dss_iklim_sumsel.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption("Minimal kolom: Tanggal, curah_hujan, Tn, Tx, Tavg, kelembaban, matahari, kecepatan_angin.")
