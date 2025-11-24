# =====================================================
# predict_pose_live.py  ✅ Works for image or webcam (CNN+LSTM)
# =====================================================

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import argparse
import os

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
MODEL_PATH = "models/cnn_lstm_yoga_stable.h5"
POSE_CLASSES = ['Bhujangasana', 'Tadasana', 'Vrikshasana', 'Trikonasana', 'WarriorPose']
TIMESTEPS = 10  # same as used in training

# -----------------------------------------------------
# Load trained CNN+LSTM model
# -----------------------------------------------------
print("📦 Loading CNN+LSTM model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -----------------------------------------------------
# Pose correction tips
# -----------------------------------------------------
pose_corrections = {
    "Bhujangasana": "Lift your chest higher and keep your elbows close to the body.",
    "Tadasana": "Straighten your spine and balance your weight evenly on both feet.",
    "Vrikshasana": "Keep your gaze steady and palms together above the head.",
    "Trikonasana": "Open your chest and align both arms in a straight line.",
    "WarriorPose": "Bend your front knee and extend your arms parallel to the floor."
}

# -----------------------------------------------------
# MediaPipe setup
# -----------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# -----------------------------------------------------
# Extract keypoints (x, y, z for 2 landmarks = 6 features)
# -----------------------------------------------------
def extract_keypoints(landmarks):
    # Example: using left shoulder (11) and left hip (23)
    lm = landmarks
    keypoints = np.array([
        lm[11].x, lm[11].y, lm[11].z,
        lm[23].x, lm[23].y, lm[23].z
    ], dtype=np.float32)
    return keypoints

# -----------------------------------------------------
# Predict from Image
# -----------------------------------------------------
def predict_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Unable to load image at {image_path}")
        return

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.6) as pose:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            print("⚠️ No pose detected in image.")
            return

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Prepare sequence
        kp = extract_keypoints(results.pose_landmarks.landmark)
        seq = np.expand_dims(np.repeat([kp], TIMESTEPS, axis=0), axis=0)  # shape (1,10,6)

        # Prediction
        preds = model.predict(seq, verbose=0)
        label_idx = np.argmax(preds)
        confidence = np.max(preds)
        pose_name = POSE_CLASSES[label_idx]

        # Feedback
        correction = pose_corrections.get(pose_name, "Good job! Maintain the posture.")
        print(f"🧘 Detected Pose: {pose_name} ({confidence*100:.2f}%)")
        print(f"💡 Correction Tip: {correction}")

        # Show result
        cv2.putText(image, f"{pose_name} ({confidence*100:.1f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("🧘 Pose Prediction (Image)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# -----------------------------------------------------
# Predict from Webcam (Real-time)
# -----------------------------------------------------
def predict_from_webcam():
    cap = cv2.VideoCapture(0)
    sequence = deque(maxlen=TIMESTEPS)

    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                kp = extract_keypoints(results.pose_landmarks.landmark)
                sequence.append(kp)

                if len(sequence) == TIMESTEPS:
                    X = np.expand_dims(sequence, axis=0)  # (1,10,6)
                    preds = model.predict(X, verbose=0)
                    label_idx = np.argmax(preds)
                    confidence = np.max(preds)
                    pose_name = POSE_CLASSES[label_idx]
                    correction = pose_corrections.get(pose_name, "Hold steady and maintain alignment.")

                    # Display
                    text = f"{pose_name} ({confidence*100:.1f}%)"
                    cv2.rectangle(frame, (10, 10), (480, 80), (0, 0, 0), -1)
                    cv2.putText(frame, text, (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, correction, (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("🧘 Real-Time Yoga Pose Detection (CNN+LSTM)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------
# CLI Arguments
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yoga Pose Prediction using CNN+LSTM Model")
    parser.add_argument("--image", type=str, help="Path to input image (optional)")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for live prediction")
    args = parser.parse_args()

    if args.image:
        predict_from_image(args.image)
    elif args.webcam:
        predict_from_webcam()
    else:
        print(" Please specify either --image <path> or --webcam")
