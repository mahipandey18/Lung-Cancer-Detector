import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ------------------- LOAD MODEL -------------------
model = tf.keras.models.load_model("lung_cancer_cnn.h5")

# Image size matches your trained model
img_width = 192
img_height = 144

# ------------------- PAGE STYLE -------------------
st.set_page_config(page_title="Lung Cancer Detector", layout="centered")

custom_css = """
<style>
body { background-color: #f0f2f6; }
.title {
    font-size: 45px !important;
    font-weight: 800 !important;
    text-align: center;
    color: white;
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(90deg, #1c92d2, #0bab64);
    box-shadow: 0px 4px 10px rgba(0,0,0,0.25);
}
.upload-box {
    background: rgba(255,255,255,0.7);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 700;
    text-align: center;
    margin-top: 20px;
}
.normal {
    background: #d4edda;
    color: #155724;
    border-left: 7px solid #28a745;
}
.cancer {
    background: #f8d7da;
    color: #721c24;
    border-left: 7px solid #dc3545;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ------------------- PAGE TITLE -------------------
st.markdown("<div class='title'>🫁 Lung Cancer Detection Using CNN</div>", unsafe_allow_html=True)
st.write("### Upload a lung CT scan image below:")

# ------------------- PREDICTION FUNCTION -------------------
def predict_image(img):
    # Resize image to match trained model
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    # 0 = NORMAL, 1 = CANCER
    if pred > 0.5:
        label = "NORMAL"
    else:
        label = "CANCER"

    return label, float(pred)

# ------------------- FILE UPLOADER -------------------
uploaded_file = st.file_uploader("📤 Upload Image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded CT Scan", width=350)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image... please wait ⏳"):
            label, score = predict_image(img)

        if label == "NORMAL":
            st.markdown("<div class='result-box normal'>🟢 NORMAL<br>No cancer detected.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box cancer'>🔴 CANCER<br>Cancer detected! Consult a radiologist.</div>", unsafe_allow_html=True)

        st.write(f"**Confidence Score:** `{score:.4f}`")
