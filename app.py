import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Aplikasi Prediksi Gambar CNN")

model = tf.keras.models.load_model("cnn_cifar10_model.h5")

uploaded_file = st.file_uploader(
    "Upload gambar", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input (asli)", use_column_width=True)

    small_img = image.resize((32,32))

    img_array = np.array(small_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    st.success(f"Hasil Prediksi: {np.argmax(prediction)}")

    st.image(small_img.resize((128,128)), caption="Versi 32x32 diperbesar untuk model")
