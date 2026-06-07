import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import (
    formalisasi_text,
    pos_tag_nlp_id,
    split_review_into_segments,
    ekstraksi_opini_full,
    text_cleaning
)

MAXLEN = 20

# =====================================
# GOOGLE DRIVE FILES
# GANTI ID SESUAI FILE ASLIMU
# =====================================

FILES = {

    "aspect_model_best.keras":
    "1b8Zvx0xNelL4FZZz2erW5dl95o3auV1V",

    "tokenizer_lstm.pkl":
    "1edC3xu63EY_76Rw6njPETfebz8dJtrMZ",

    "aspect_le.pkl":
    "13VuS2aOh9jf22kr5oxm4WgmKsFgQMc-0",

    "sentiment_le.pkl":
    "1VnTQ8ZJqacLqgeokYO5JrWm-a1W2foMi",

    "sentiment_model_biaya_layanan_best.keras":
    "1-WiwtFG6NCZzCKmzt9iyQJ5FlywZ4dne",

    "sentiment_model_fasilitas_dan_infrastruktur_best.keras":
    "1qmcyGUtihOpLRG4N0Q6xnKtUW0joj0F_",

    "sentiment_model_kualitas_pelayanan_medis_dan_staf_best.keras":
    "1HF-UvF1tlmYDBlH2DmxCbcFFOInY4if1",

    "sentiment_model_waktu_tunggu_best.keras":
    "11fIq-xbzRydZ4pI7HdZtNGZhvEeHc2eS"
}

# =====================================
# DOWNLOAD FILE
# =====================================

def download_models():

    for filename, file_id in FILES.items():

        if not os.path.exists(filename):

            url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(
                url,
                filename,
                quiet=False
            )

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_resources():

    download_models()

    aspect_model = load_model(
        "aspect_model_best.keras"
    )

    with open(
        "tokenizer_lstm.pkl",
        "rb"
    ) as f:
        tokenizer = pickle.load(f)

    with open(
        "aspect_le.pkl",
        "rb"
    ) as f:
        aspect_le = pickle.load(f)

    with open(
        "sentiment_le.pkl",
        "rb"
    ) as f:
        sentiment_le = pickle.load(f)

    sentiment_models = {

        "biaya layanan":
        load_model(
            "sentiment_model_biaya_layanan_best.keras"
        ),

        "fasilitas dan infrastruktur":
        load_model(
            "sentiment_model_fasilitas_dan_infrastruktur_best.keras"
        ),

        "kualitas pelayanan medis dan staf":
        load_model(
            "sentiment_model_kualitas_pelayanan_medis_dan_staf_best.keras"
        ),

        "waktu tunggu":
        load_model(
            "sentiment_model_waktu_tunggu_best.keras"
        )
    }

    return (
        aspect_model,
        tokenizer,
        aspect_le,
        sentiment_le,
        sentiment_models
    )

# =====================================
# PREPROCESS
# =====================================

def preprocess_text(text):

    cleaned = text_cleaning(text)

    return cleaned.lower()

# =====================================
# PREDICT ASPECT
# =====================================

def predict_aspect(text):

    processed = preprocess_text(text)

    seq = tokenizer.texts_to_sequences(
        [processed]
    )

    pad = pad_sequences(
        seq,
        maxlen=MAXLEN,
        padding="post"
    )

    pred = aspect_model.predict(
        pad,
        verbose=0
    )

    idx = np.argmax(pred)

    return aspect_le.inverse_transform(
        [idx]
    )[0]

# =====================================
# PREDICT SENTIMENT
# =====================================

def predict_sentiment(
    text,
    aspect
):

    processed = preprocess_text(text)

    seq = tokenizer.texts_to_sequences(
        [processed]
    )

    pad = pad_sequences(
        seq,
        maxlen=MAXLEN,
        padding="post"
    )

    pred = sentiment_models[
        aspect
    ].predict(
        pad,
        verbose=0
    )

    idx = np.argmax(pred)

    return sentiment_le.inverse_transform(
        [idx]
    )[0]

# =====================================
# EKSTRAK OPINI
# =====================================

def extract_opinions(review):

    formal = formalisasi_text(
        review
    )

    pos_tag = pos_tag_nlp_id(
        formal
    )

    segments = split_review_into_segments(
        formal
    )

    opinions, rules = ekstraksi_opini_full(
        pos_tag,
        segments
    )

    if opinions == "Tidak ada ekstraksi":
        return []

    results = []

    for line in opinions.split("\n"):

        if ":" in line:

            results.append(
                line.split(
                    ":",
                    1
                )[1].strip()
            )

    return results

# =====================================
# LOAD RESOURCE
# =====================================

(
    aspect_model,
    tokenizer,
    aspect_le,
    sentiment_le,
    sentiment_models
) = load_resources()

# =====================================
# STREAMLIT UI
# =====================================

st.title(
    "ABSA Rumah Sakit"
)

tab1, tab2 = st.tabs(
    [
        "Input Ulasan",
        "Upload Dataset"
    ]
)

# =====================================
# TAB 1
# =====================================

with tab1:

    review = st.text_area(
        "Masukkan Ulasan"
    )

    if st.button(
        "Analisis"
    ):

        opinions = extract_opinions(
            review
        )

        rows = []

        for op in opinions:

            aspect = predict_aspect(
                op
            )

            sentiment = predict_sentiment(
                op,
                aspect
            )

            rows.append({
                "Extracted Opinion": op,
                "Aspect": aspect,
                "Sentiment": sentiment
            })

        st.dataframe(
            pd.DataFrame(rows)
        )

# =====================================
# TAB 2
# =====================================

with tab2:

    uploaded = st.file_uploader(
        "Upload CSV/XLSX",
        type=[
            "csv",
            "xlsx"
        ]
    )

    if uploaded is not None:

        if uploaded.name.endswith(
            ".csv"
        ):
            df = pd.read_csv(
                uploaded
            )
        else:
            df = pd.read_excel(
                uploaded
            )

        hasil = []

        for review in df["Review"]:

            opinions = extract_opinions(
                str(review)
            )

            for op in opinions:

                aspect = predict_aspect(
                    op
                )

                sentiment = predict_sentiment(
                    op,
                    aspect
                )

                hasil.append({

                    "Review":
                    review,

                    "Extracted Opinion":
                    op,

                    "Aspect":
                    aspect,

                    "Sentiment":
                    sentiment
                })

        hasil_df = pd.DataFrame(
            hasil
        )

        st.dataframe(
            hasil_df
        )
