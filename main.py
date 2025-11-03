import streamlit as st
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import time
import subprocess

# Try to import optional camera modules
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Download model
model_path = hf_hub_download(
    repo_id="qwertymaninwork/Plant_Disease_Detection_System",
    filename="trained_model.keras"
)

model = tf.keras.models.load_model(model_path)

# Prediction function
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ("Home", "About", "Disease Recognition", "Capture Image"))

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpg", use_container_width=True)
    st.markdown("""
    🌱 Upload or capture a plant leaf image to detect possible diseases.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("This project uses deep learning to classify 38 types of plant diseases from leaf images.")

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Upload an Image")
    test_image = st.file_uploader("Choose an image:")
    if st.button("Show Image") and test_image:
        st.image(test_image, use_container_width=True)
    if st.button("Detect") and test_image:
        with st.spinner("Analyzing..."):
            result_index = model_prediction(test_image)
            st.success(f"🌿 Detected Disease: {class_name[result_index]}")

# Capture Image Page
elif app_mode == "Capture Image":
    st.header("📸 Capture Image")

    if st.button("Capture from Pi Camera"):
        st.info("Attempting to access camera...")

        if PICAMERA_AVAILABLE:
            st.write("Using Picamera2...")
            picam2 = Picamera2()
            config = picam2.create_still_configuration(main={"size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)
            frame = picam2.capture_array()
            picam2.close()

            if CV2_AVAILABLE:
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_image = Image.fromarray(frame)

        elif CV2_AVAILABLE:
            st.write("Using OpenCV...")
            cap = cv2.VideoCapture(0)
            time.sleep(2)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                st.error("Could not access camera.")
                st.stop()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_image = Image.fromarray(frame)

        else:
            st.warning("Neither Picamera2 nor OpenCV found. Falling back to rpicam-hello.")
            subprocess.run(["rpicam-jpeg", "-o", "captured.jpg"])
            captured_image = Image.open("captured.jpg")

        st.image(captured_image, caption="Captured Image", use_container_width=True)
        captured_image.save("captured.jpg")

        if st.button("Detect Disease"):
            with st.spinner("Processing..."):
                result_index = model_prediction("captured.jpg")
                st.success(f"🌿 Detected Disease: {class_name[result_index]}")

