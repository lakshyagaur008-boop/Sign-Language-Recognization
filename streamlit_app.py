import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Sign Language Recognition")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "sign_language_model.keras",
        compile=False,
        safe_mode=False
    )

model = load_model()

st.title("🤟 Sign Language Recognition")
st.write("Upload an image to predict the sign")

file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))   # change only if your model needs
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    label = np.argmax(preds)

    st.success(f"Predicted Sign: {label}")
    st.image(image, use_container_width=True)
