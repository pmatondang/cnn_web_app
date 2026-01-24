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
    st.image(image, caption="Gambar Input", use_column_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    st.success(f"Hasil Prediksi: {np.argmax(prediction)}")


