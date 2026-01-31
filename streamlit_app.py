import streamlit as st
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
from PIL import Image

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition")
st.write("Capture a hand sign using your camera and predict it with AI.")

# -------- Load model --------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.keras")

model = load_model()

# -------- MediaPipe --------
mp_holistic = mp.solutions.holistic

CLASS_NAMES = ["Hello", "Thanks", "Yes", "No"]  # adjust if needed

def extract_keypoints(results):
    pose = np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[l.x, l.y, l.z] for l in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# -------- Camera --------
img_file = st.camera_input("📸 Capture a sign")

if img_file:
    image = Image.open(img_file).convert("RGB")
    frame = np.array(image)

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as holistic:

        results = holistic.process(frame)
        keypoints = extract_keypoints(results)
        keypoints = np.expand_dims(keypoints, axis=0)

        prediction = model.predict(keypoints)
        label = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.image(frame, use_column_width=True)
        st.success(f"✅ Prediction: **{label}**")
        st.info(f"Confidence: **{confidence:.2f}**")

st.caption("🚀 Hackathon-ready | Cloud-safe | MediaPipe + TensorFlow")
