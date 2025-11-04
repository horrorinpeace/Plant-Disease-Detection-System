import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image 
import gdown
import os

# Download model from Google Drive (since GitHub has file size limits)
@st.cache_resource
def load_model():
    model_path = "model.h5"
    
    # If model doesn't exist, download it
    if not os.path.exists(model_path):
        with st.spinner("Downloading model for the first time... This may take a moment."):
            # Google Drive direct download link for the model
            url = "https://github.com/AshishSalaskar1/Plant-Leaf-Disease-Detection/raw/master/model.h5"
            
            try:
                # Use gdown to download from Google Drive if it's stored there
                # Otherwise use direct download
                import urllib.request
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.info("Please download the model manually from the GitHub repository and place it in the same folder as this script.")
                return None
    
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

# TensorFlow Model Prediction
def model_prediction(test_image):
    try:
        image = Image.open(test_image)
        image = image.resize((28, 28))  # Changed to 128x128 (check your model's input size)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        input_arr = input_arr / 255.0  # Normalize
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        return result_index, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ("Home", "About", "Disease Recognition"))

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    
    # Check if home_page.jpg exists, otherwise show placeholder
    if os.path.exists("home_page.jpg"):
        st.image("home_page.jpg", use_container_width=True)
    else:
        st.info("📸 Add a 'home_page.jpg' image to display here")
    
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍  
    
    ### Features:
    - **Upload** an image of a plant leaf
    - **Detect** diseases across 38 different plant disease classes
    - **Get instant results** with confidence scores
    
    ### Supported Plants:
    - 🍎 Apple, 🫐 Blueberry, 🍒 Cherry
    - 🌽 Corn, 🍇 Grape, 🍊 Orange
    - 🍑 Peach, 🌶️ Pepper, 🥔 Potato
    - 🍓 Strawberry, 🍅 Tomato, and more!
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### About Dataset
    This model is trained on the **PlantVillage Dataset**, which contains:
    - 📊 **54,305 images** of healthy and diseased plant leaves
    - 🌱 **14 crop species**
    - 🦠 **38 disease classes** (including healthy plants)
    
    ### About Model
    - **Architecture:** Convolutional Neural Network (CNN)
    - **Input Size:** 128x128 pixels
    - **Classes:** 38 plant disease categories
    - **Source:** [GitHub Repository](https://github.com/AshishSalaskar1/Plant-Leaf-Disease-Detection)
    
    ### Disease Types Covered:
    - 🍄 Fungal diseases (17 types)
    - 🦠 Bacterial diseases (4 types)
    - 🧫 Mold diseases (2 types)
    - 🦟 Viral diseases (2 types)
    - 🕷️ Mite-caused diseases (1 type)
    
    ### How It Works:
    1. Upload an image of a plant leaf
    2. The model analyzes the image using deep learning
    3. Get results showing the detected disease and confidence level
    
    ### Note:
    For best results, use clear images of individual leaves with good lighting.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("🔍 Disease Recognition")
    
    if model is None:
        st.error("⚠️ Model failed to load. Please check the error messages above.")
        st.stop()
    
    st.markdown("### Upload a plant leaf image for disease detection")
    
    test_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Show uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(test_image, use_container_width=True)
        
        # Predict button
        if st.button("🔬 Detect Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                result_index, confidence = model_prediction(test_image)
                
                if result_index is not None:
                    # Class names for 38 classes (PlantVillage dataset)
                    class_name = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]
                    
                    detected_class = class_name[result_index]
                    
                    # Format the output nicely
                    plant_name = detected_class.split('___')[0].replace('_', ' ')
                    disease_name = detected_class.split('___')[1].replace('_', ' ')
                    
                    with col2:
                        st.subheader("Detection Results")
                        
                        # Display results with color coding
                        if 'healthy' in disease_name.lower():
                            st.success(f"✅ **Plant:** {plant_name}")
                            st.success(f"✅ **Status:** {disease_name}")
                            st.balloons()
                        else:
                            st.warning(f"🌱 **Plant:** {plant_name}")
                            st.error(f"⚠️ **Disease Detected:** {disease_name}")
                        
                        # Confidence score
                        st.metric("Confidence", f"{confidence:.2f}%")
                        
                        # Progress bar for confidence
                        st.progress(int(confidence))
                        
                        if confidence < 60:
                            st.info("💡 Low confidence. Try uploading a clearer image.")
    else:
        st.info("👆 Please upload an image to get started")
        
        # Show example
        st.markdown("### 📋 Tips for best results:")
        st.markdown("""
        - Use clear, well-lit images
        - Capture individual leaves
        - Avoid blurry or dark images
        - Ensure the leaf fills most of the frame
        """)

