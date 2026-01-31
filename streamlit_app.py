import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Sign Language Recognition", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.keras")

model = load_model()

def get_mediapipe_hands():
    import mediapipe as mp  # 🚨 delayed import (THIS FIXES THE CRASH)
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

hands = get_mediapipe_hands()

st.title("🤟 Sign Language Recognition")
st.write("Upload a hand image to predict the sign")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    results = hands.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks)
        predicted_class = np.argmax(prediction)

        st.success(f"Predicted Sign: {predicted_class}")
    else:
        st.warning("No hand detected")

    st.image(image, caption="Uploaded Image", use_container_width=True)
