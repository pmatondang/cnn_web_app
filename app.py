import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Aplikasi Prediksi Gambar CNN - CIFAR-10")

model = tf.keras.models.load_model("cnn_cifar10_model.h5")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input (asli)", use_column_width=True)

    small_img = image.resize((32, 32))
    img_array = np.array(small_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    proba = np.max(prediction) 
    pred_class = np.argmax(prediction)

    class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

    #st.image(small_img.resize((128,128)), caption="versi 32x32 diperbesar untuk model")

    threshold = 0.5
    if proba < threshold:
        st.warning("Gambar belum ada di dataset CIFAR-10")
    else:
        st.success(f"Hasil Prediksi: {class_names[pred_class]} - (confidence: {proba:.2f})")


