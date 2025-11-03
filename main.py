import streamlit as st
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# --------------------------------------------------------
# 🔹 Download the fine-tuned full model (.keras)
# --------------------------------------------------------
model_path = hf_hub_download(
    repo_id="qwertymaninwork/Plant_Disease_Detection_System",  # your Hugging Face repo ID
    filename="fixed_finetuned_model.keras"  # make sure this is the exact filename uploaded
)

# --------------------------------------------------------
# 🔹 Load the complete model safely
# --------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --------------------------------------------------------
# 🔹 Prediction Function
# --------------------------------------------------------
def model_prediction(test_image):
    image = Image.open(test_image).resize((128, 128))
    input_arr = np.array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# --------------------------------------------------------
# 🔹 Sidebar Navigation
# --------------------------------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ("Home", "About", "Disease Recognition"))

# --------------------------------------------------------
# 🔹 Home Page
# --------------------------------------------------------
if app_mode == "Home":
    st.header("🌿 Plant Disease Recognition System")
    st.image("home_page.jpg", use_container_width=True)
    st.markdown("""
    Welcome! Upload a leaf image to detect plant diseases instantly using deep learning.
    """)

# --------------------------------------------------------
# 🔹 About Page
# --------------------------------------------------------
elif app_mode == "About":
    st.header("About Dataset")
    st.markdown("""
    The dataset includes ~87K RGB images of healthy and diseased leaves across 38 classes.
    Training/validation split: 80/20.
    """)

# --------------------------------------------------------
# 🔹 Disease Recognition Page
# --------------------------------------------------------
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image:")

    if st.button("Show Image") and test_image:
        st.image(test_image, use_container_width=True)

    if st.button("Detect") and test_image:
        with st.spinner("Analyzing..."):
            index = model_prediction(test_image)
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                'Apple___healthy', 'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
            st.success(f"Prediction: {class_names[index]}")

