import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained CNN model
MODEL_PATH = 'fashion-mnist-classification/fashion_cnn_trained.h5'
model = load_model(MODEL_PATH)

# Class names for the Fashion MNIST dataset
class_names = ['Áo thun', 'Quần dài', 'Áo len', 'Váy', 'Áo khoác',
               'Guốc', 'Áo sơ mi', 'Giày sneaker', 'Túi', 'Ủng']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to 28x28
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = image_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return image_array, image

# Streamlit app
st.title("Fashion MNIST Model Tester")
st.write("Upload an image to test the model's prediction.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    preprocessed_image, processed_image = preprocess_image(image)

    # Display the preprocessed image
    st.write("### Preprocessed Image:")
    st.image(processed_image, caption="Preprocessed Image (Grayscale, 28x28)", use_container_width=True)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)

    # Display the prediction
    st.write("### Prediction:")
    st.write(f"The model predicts this is a: **{class_names[predicted_class]}**")

    # Display the prediction probabilities as a bar chart
    st.write("### Prediction Probabilities:")
    fig, ax = plt.subplots()
    ax.bar(range(len(class_names)), prediction[0])
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    st.pyplot(fig)