import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("Prediksi Gambar CIFAR-10")

MODEL_PATH = "cnn_cifar10_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("Model tidak ditemukan. Pastikan file .h5 sudah di-upload ke GitHub.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

uploaded_file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", width=250)

    img = image.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = class_names[np.argmax(prediction)]

    st.success(f"Hasil Prediksi: **{label}**")
