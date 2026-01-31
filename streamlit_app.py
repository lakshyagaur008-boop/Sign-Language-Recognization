import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition")
st.write("Capture a hand sign using your camera and get the predicted gesture.")

# -------------------------------
# Load MediaPipe Holistic (cached)
# -------------------------------
@st.cache_resource
def load_holistic():
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

holistic = load_holistic()

# -------------------------------
# Load Trained Model (KERAS format)
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.keras")

model = load_model()

# -------------------------------
# Class Labels (EDIT if needed)
# -------------------------------
CLASS_NAMES = [
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "I LOVE YOU"
]

# -------------------------------
# Feature Extraction Function
# -------------------------------
def extract_keypoints(results):
    pose = (
        np.array([[res.x, res.y, res.z, res.visibility]
                  for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )

    face = (
        np.array([[res.x, res.y, res.z]
                  for res in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(468 * 3)
    )

    lh = (
        np.array([[res.x, res.y, res.z]
                  for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )

    rh = (
        np.array([[res.x, res.y, res.z]
                  for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )

    return np.concatenate([pose, face, lh, rh])

# -------------------------------
# Camera Input (Browser-based)
# -------------------------------
img_file = st.camera_input("📸 Take a photo")

if img_file is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR → RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    # MediaPipe processing
    results = holistic.process(image_rgb)

    # Draw landmarks
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )

    # Display image
    st.image(image, channels="BGR", caption="Captured Image")

    # Extract keypoints & predict
    keypoints = extract_keypoints(results)
    keypoints = np.expand_dims(keypoints, axis=0)

    prediction = model.predict(keypoints, verbose=0)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.subheader("🧠 Prediction")
    st.success(f"**{predicted_class}**")
    st.write(f"Confidence: `{confidence:.2f}`")
