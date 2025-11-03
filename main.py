import streamlit as st
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from picamera2 import Picamera2
import time
import os

# --- Page setup ---
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection System")
st.markdown("Upload an image of a leaf **or capture one** to detect plant diseases.")

# --- Load Model from Hugging Face ---
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="qwertymaninwork/Plant_Disease_Detection_System",
        filename="plant_disease_model.h5"
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- Prediction Function ---
def predict(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_index = np.argmax(preds)
    confidence = np.max(preds) * 100
    return class_index, confidence

# --- Capture Image Section ---
st.header("📸 Capture Image from Camera")

if st.button("Capture Image"):
    st.info("Capturing image... please wait 2 seconds.")
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)  # warm-up time

    image_path = "captured_leaf.jpg"
    picam2.capture_file(image_path)
    picam2.stop()

    st.success("✅ Image captured successfully!")
    st.image(image_path, caption="Captured Image", use_column_width=True)

    # Load image and make prediction
    captured_img = Image.open(image_path)
    class_idx, confidence = predict(captured_img)
    st.subheader(f"🌱 Prediction Result: Class #{class_idx} ({confidence:.2f}% confidence)")

# --- Upload Image Section ---
st.header("🖼️ Upload an Image")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    class_idx, confidence = predict(image)
    st.subheader(f"🌱 Prediction Result: Class #{class_idx} ({confidence:.2f}% confidence)")



