import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# ---------------- CONFIG ----------------
WORDS = ["goodbye", "hello", "no", "please", "yes", "thanks", "sorry"]
SEQUENCE_LENGTH = 10
MODEL_PATH = "sign_language_model.h5"

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Sign Language Detection", layout="centered")
st.title("🤟 Sign Language Detection")
st.write("Live sign language prediction using camera")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_cached()

# ---------------- MEDIAPIPE ----------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ---------------- KEYPOINT EXTRACTION ----------------
# IMPORTANT: EXACTLY 225 features
def extract_keypoints(results):
    pose = (
        np.array([[p.x, p.y, p.z] for p in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 3)
    )

    lh = (
        np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    rh = (
        np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([pose, lh, rh])  # 99 + 63 + 63 = 225

# ---------------- CAMERA INPUT ----------------
st.info("Allow camera access and show a sign clearly")

camera = st.camera_input("📸 Camera")

sequence = deque(maxlen=SEQUENCE_LENGTH)

if camera is not None:
    # Convert Streamlit image to OpenCV format
    bytes_data = camera.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        results = holistic.process(image_rgb)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Debug check (remove later if you want)
        st.caption(f"Keypoints per frame: {len(keypoints)}")

        if len(sequence) == SEQUENCE_LENGTH:
            res = model.predict(
                np.expand_dims(sequence, axis=0),
                verbose=0
            )[0]

            predicted_word = WORDS[np.argmax(res)]
            confidence = float(np.max(res))

            st.success(f"🧠 Prediction: **{predicted_word}**")
            st.progress(confidence)

    st.image(frame, channels="BGR")

else:
    st.warning("Waiting for camera input…")
