import streamlit as st 
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image 

# Download MobileNetV2 model from Hugging Face
model_path = hf_hub_download(
    repo_id="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",  # MobileNetV2 plant disease repo
    filename="mobilenet_v2_1.0_224.h5"  # Model file inside repo
)

# Load the MobileNetV2 model
model = tf.keras.models.load_model(model_path)

# TensorFlow Model Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0) / 255.0  # normalize
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ("Home", "About", "Disease Recognition"))

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM (MobileNetV2)")
    image_path = "home_page.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍  
    This version uses **MobileNetV2**, a lightweight CNN optimized for fast and accurate disease detection.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### About Model
    - **Model Used:** MobileNetV2 (Pretrained on ImageNet, fine-tuned for 38 plant disease classes)
    - **Source:** Hugging Face – [linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification)
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image:")

    if st.button("Show Image"):
        st.image(test_image, use_container_width=True)

    if st.button("Detect"):
        with st.spinner("Processing..."):
            result_index = model_prediction(test_image)
            
            # Class names for 38 classes
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)_Powdery_mildew', 'Cherry_(including_sour)_healthy',
                'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)Common_rust',
                'Corn_(maize)_Northern_Leaf_Blight', 'Corn_(maize)_healthy', 'Grape___Black_rot',
                'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange__Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,bell__Bacterial_spot', 'Pepper,bell__healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            
            st.success(f"✅ Detected Disease: {class_name[result_index]}")



