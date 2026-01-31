import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Sign Language Recognition", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.keras")

model = load_model()

st.title("🤟 Sign Language Recognition")
st.write("Upload a hand image to predict the sign")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))  # adjust to your model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"Predicted Sign: {predicted_class}")
    st.image(image, use_container_width=True)
