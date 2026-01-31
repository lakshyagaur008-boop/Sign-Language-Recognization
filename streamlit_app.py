import streamlit as st
import numpy as np
import cv2
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
st.write("Use your camera to capture a hand sign and predict it using AI.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.keras")

model = load_model()

# ---------------- MEDIAPIPE ----------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ---------------- CLASS LABELS ----------------
# ⚠️ CHANGE this to match your training labels
CLASS_NAMES = [
    "Hello",
    "Thanks",
    "Yes",
    "No"
]

# ---------------- FEATURE EXTRACTION ----------------
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# ---------------- CAMERA INPUT ----------------
img_file = st.camera_input("📸 Capture a hand sign")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as holistic:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        keypoints = extract_keypoints(results)
        keypoints = np.expand_dims(keypoints, axis=0)

        prediction = model.predict(keypoints)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Processed Image", use_column_width=True)

        st.success(f"### ✅ Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}**")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 AI-powered Sign Language Recognition | Hackathon Ready")
