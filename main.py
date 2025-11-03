import streamlit as st
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

# --------------------------------------------------------
# 🔹 Download the fine-tuned model weights from Hugging Face
# --------------------------------------------------------
model_path = hf_hub_download(
    repo_id="qwertymaninwork/Plant_Disease_Detection_System",  # your repo name
    filename="finetuned_model.keras"  # model file name
)

# --------------------------------------------------------
# 🔹 Rebuild the CNN architecture (same as your starting model)
# --------------------------------------------------------
def create_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(38, activation='softmax'))
    return model

# --------------------------------------------------------
# 🔹 Load weights safely (skip mismatched layers if any)
# --------------------------------------------------------
model = create_model()
model.load_weights(model_path, by_name=True, skip_mismatch=True)

# --------------------------------------------------------
# 🔹 Prediction function
# --------------------------------------------------------
def model_prediction(test_image):
    image = Image.open(test_image).resize((128, 128))
    input_arr = np.array(image) / 255.0  # normalize
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
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Upload an image of a plant leaf, and our system will detect if it is healthy or diseased.

    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition* page.
    2. *Analysis:* Our system processes the image using deep learning.
    3. *Results:* Instantly view detected disease name.
    """)

# --------------------------------------------------------
# 🔹 About Page
# --------------------------------------------------------
elif app_mode == "About":
    st.header("About Dataset")
    st.markdown("""
    This dataset consists of around 87K RGB images of healthy and diseased crop leaves, categorized into 38 classes.
    The data is split into training (80%) and validation (20%).
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
        with st.spinner("Processing..."):
            st.write("Detected Disease:")
            result_index = model_prediction(test_image)
            class_name = [
                'Apple___Apple_scab',
                'Apple___Black_rot',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot',
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Raspberry___healthy',
                'Soybean___healthy',
                'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(class_name[result_index])
