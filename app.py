import streamlit as st
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Batik Pattern Classifier", layout="centered")

st.title("🎨 Indonesian Batik Pattern Classifier")
st.markdown("Upload a batik fabric image to identify its pattern type.")

@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("models/batik_model.h5")
    with open("models/class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning("Model not found. Please run `notebook.ipynb` first to train and save the model.")

uploaded = st.file_uploader("Upload batik image", type=["jpg", "jpeg", "png"])

if uploaded and model_loaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", width=300)

    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)[0]
    top_idx = np.argsort(preds)[::-1]

    st.subheader("Prediction")
    pred_label = class_names[top_idx[0]].replace("_", " ").title()
    st.markdown(f"### {pred_label}")
    st.markdown(f"Confidence: **{preds[top_idx[0]]*100:.1f}%**")

    st.subheader("Top 5 Probabilities")
    import plotly.express as px
    import pandas as pd
    top5 = pd.DataFrame({
        "Pattern": [class_names[i].replace("_", " ").title() for i in top_idx[:5]],
        "Confidence (%)": [preds[i]*100 for i in top_idx[:5]]
    })
    fig = px.bar(top5, x="Confidence (%)", y="Pattern", orientation="h",
                 color="Confidence (%)", color_continuous_scale="Oranges")
    fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Brief description per class
    descriptions = {
        "parang": "Parang — diagonal lines symbolizing sharpness and continuous motion.",
        "kawung": "Kawung — geometric circles representing purity and justice.",
        "megamendung": "Mega Mendung — cloud motif from Cirebon, symbolizing patience.",
        "truntum": "Truntum — star flowers symbolizing love regrowing.",
        "sidomukti": "Sidomukti — prosperity motif used in royal ceremonies.",
    }
    desc = descriptions.get(class_names[top_idx[0]].lower(), "A traditional Indonesian batik pattern.")
    st.info(desc)

st.markdown("---")
st.caption("Model: MobileNetV2 | Dataset: Indonesian Batik Patterns | Built with TensorFlow + Streamlit")
