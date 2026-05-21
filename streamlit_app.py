import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard Analitik DJPb", layout="wide")
st.title("Dashboard Analitik DJPb")

MODEL_PATH = Path("model/Best_model.pkcls")
DATA_PATH = Path("data/02_realisasi_anggaran_klasifikasi.csv")

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

data = load_data()
model = None
model_error = None
try:
    model = load_model()
except Exception as exc:
    model_error = str(exc)

tab_prediksi, tab_visualisasi = st.tabs(["Menu Prediksi", "Visualisasi"])

with tab_prediksi:
    st.header("Prediksi Realisasi 95%")
    st.markdown("Masukkan data input berikut dan klik tombol untuk melihat hasil prediksi.")

    col1, col2 = st.columns(2)
    with col1:
        jumlah_spm = st.number_input(
            "Jumlah SPM",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            format="%d",
        )
        revisi_dipa = st.number_input(
            "Revisi DIPA",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            format="%d",
        )
        deviasi_rpd_persen = st.number_input(
            "Deviasi RPD (%)",
            min_value=0.00,
            max_value=30.00,
            value=5.50,
            format="%.2f",
        )
    with col2:
        skor_ikpa = st.number_input(
            "Skor IKPA",
            min_value=70.00,
            max_value=100.00,
            value=85.00,
            format="%.2f",
        )
        tipe_satker = st.selectbox(
            "Tipe Satker",
            ["Dekonsentrasi", "Kantor Daerah", "Kantor Pusat", "Tugas Pembantuan"],
        )

    if st.button("Jalankan Prediksi"):
        if model is None:
            st.error("Model tidak dapat dimuat.")
            if model_error:
                st.write(model_error)
        else:
            tipe_mapping = {
                "Dekonsentrasi": [1.0, 0.0, 0.0, 0.0],
                "Kantor Daerah": [0.0, 1.0, 0.0, 0.0],
                "Kantor Pusat": [0.0, 0.0, 1.0, 0.0],
                "Tugas Pembantuan": [0.0, 0.0, 0.0, 1.0],
            }
            tipe_values = tipe_mapping[tipe_satker]
            features = [
                float(jumlah_spm),
                float(revisi_dipa),
                float(deviasi_rpd_persen),
                float(skor_ikpa),
                *tipe_values,
            ]
            X = np.array([features])

            prediction = model.predict(X)
            if isinstance(prediction, tuple):
                prediction_values, probabilities = prediction
            else:
                prediction_values = prediction
                probabilities = None

            label_index = int(prediction_values[0])
            label_value = model.domain.class_var.values[label_index]

            st.success("Prediksi selesai")
            st.write("### Input pengguna")
            st.json(
                {
                    "jumlah_spm": int(jumlah_spm),
                    "revisi_dipa": int(revisi_dipa),
                    "deviasi_rpd_persen": float(deviasi_rpd_persen),
                    "skor_ikpa": float(skor_ikpa),
                    "tipe_satker": tipe_satker,
                }
            )
            st.write("### Hasil prediksi")
            st.metric("Prediksi realisasi 95%", label_value)
            if probabilities is not None and probabilities.shape[1] > 1:
                yes_prob = float(probabilities[0, 1])
                st.write(f"Probabilitas 'Ya': {yes_prob:.2%}")

with tab_visualisasi:
    st.header("Visualisasi Bubble Plot")
    st.markdown(
        "Grafik bubble menunjukkan skor IKPA terhadap deviasi RPD, dengan warna berdasarkan kolom `realisasi_tercapai_95persen`."
    )

    chart = (
        alt.Chart(data)
        .mark_circle(opacity=0.7, size=100)
        .encode(
            x=alt.X(
                "skor_ikpa",
                title="Skor IKPA",
                scale=alt.Scale(domain=[70, 100]),
            ),
            y=alt.Y("deviasi_rpd_persen", title="Deviasi RPD (%)"),
            color=alt.Color(
                "realisasi_tercapai_95persen:N",
                title="Realisasi Tercapai 95%",
            ),
            tooltip=[
                "kode_satker",
                "tipe_satker",
                "skor_ikpa",
                "deviasi_rpd_persen",
                "realisasi_tercapai_95persen",
            ],
        )
        .properties(width=900, height=550)
        .interactive()
    )
    st.altair_chart(chart, width=900)
