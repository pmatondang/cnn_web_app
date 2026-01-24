import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("Aplikasi Prediksi Gambar CNN (CIFAR-10)")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_cifar10_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

uploaded_file = st.file_uploader(
    "Upload gambar", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", use_container_width=True)

    img = image.resize((32, 32))      
    img = np.array(img) / 255.0        
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(
        f"Hasil Prediksi: {class_names[predicted_class]} "
        f"({confidence:.2f}%)"
    )
