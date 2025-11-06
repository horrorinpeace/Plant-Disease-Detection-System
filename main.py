import streamlit as st 
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image 

# ------------------------------
# Download model weights
# ------------------------------
model_path = hf_hub_download(
    repo_id="qwertymaninwork/Plant_Disease_Detection_System",  # your repo name
    filename="mobilenetv2_plant.keras"  # your actual model file name
)

# ------------------------------
model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)


# ------------------------------
# Load saved weights
# ------------------------------
model.load_weights(model_path)

# ------------------------------
# TensorFlow Model Prediction
# ------------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0) / 255.0  # normalize
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return result_index, confidence

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",("Home","About","Disease Recognition"))

# ------------------------------
# Home Page
# ------------------------------
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System!** 🌿🔍
    
    Upload a leaf image, and our model will detect plant diseases using AI.

    **How It Works:**
    1. Go to the *Disease Recognition* page.
    2. Upload your plant image.
    3. View instant disease detection results.
    """)

# ------------------------------
# About Page
# ------------------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### About Dataset
    This dataset contains healthy and diseased leaf images for:
    - Rice 🌾
    - Wheat 🌾
    - Millet 🌿
    - Sugarcane 🍃
    - Tea 🍵
    - Tomato 🍅
    - Potato 🥔

    **Total Classes:** 30  
    **Model Used:** MobileNetV2  
    **Input Size:** 128×128
    """)

# ------------------------------
# Prediction Page
# ------------------------------
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image:")

    if st.button("Show Image") and test_image:
        st.image(test_image, use_container_width=True)

    if st.button("Detect") and test_image:
        with st.spinner("Processing..."):
            result_index, confidence = model_prediction(test_image)

            class_name = [
                'HEALTHY RICE',
                'RICE BACTERIAL BLIGHT',
                'RICE BROWN SPOT',
                'RICE LEAF SMUT',
                'HEALTHY WHEAT',
                'WHEAT LOOSE SMUT',
                'WHEAT YELLOW RUST',
                'WHEAT BROWN RUST',
                'HEALTHY MILLET',
                'MILLET RUST',
                'MILLET BLAST',
                'HEALTHY SUGARCANE',
                'SUGARCANE YELLOW',
                'SUGARCANE RED ROT',
                'SUGARCANE RUST',
                'HEALTHY TEA LEAF',
                'TEA GREEN MIRID BUG',
                'TEA GRAY BLIGHT',
                'TEA HELOPELITIS',
                'HEALTHY POTATO',
                'POTATO EARLY BLIGHT',
                'POTATO LATE BLIGHT',
                'HEALTHY TOMATO',
                'TOMATO LEAF MOLD',
                'TOMATO MOSAIC VIRUS',
                'TOMATO SEPTORIA LEAF SPOT',
                'HEALTHY RICE',
                'HEALTHY SUGARCANE',
                'HEALTHY TEA LEAF',
                'HEALTHY WHEAT',
            ]

            st.success(f"**Detected:** {class_name[result_index]} ({confidence:.2f}% confidence)")




