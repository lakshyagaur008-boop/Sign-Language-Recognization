import streamlit as st
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition")
st.write("Capture a hand sign using your camera and predict it with AI.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.keras")

model = load_model()

# ---------------- MEDIAPIPE ----------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ---------------- CLASS LABELS ----------------
CLASS_NAMES = ["Hello", "Thanks", "Yes", "No"]  # update if needed

# ---------------- FEATURE EXTRACTION ----------------
def extract_keypoints(results):
    pose = np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[l.x, l.y, l.z] for l in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# ---------------- CAMERA INPUT ----------------
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
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.image(frame, caption="Captured Image", use_column_width=True)
        st.success(f"### ✅ Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}**")

st.markdown("---")
st.caption("🚀 Hackathon-ready Sign Language Recognition Web App")
